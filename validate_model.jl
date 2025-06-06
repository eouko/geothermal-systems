using LinearAlgebra
using .Threads
using WriteVTK
using CSV
using DataFrames
using Plots
include("utils.jl")

# Create experiment folder #####################################################
path = "dbhe-array-results/"
rm(path, recursive=true, force=true)
mkpath(path)


# Deep borehole heat exchanger (DBHE) model ####################################
# DBHE type: coaxial
#
# Equations for each DBHE, b:
# 
#     1D element representation of the center fluid temperature, ϕb_c:
#         Cc*∂ϕb_c(t,z)/∂t = (ϕb_a(t,z)-ϕb_c(t,z))/Rac
#                            +m*Cf*∂ϕb_c(t,z)/∂z
#
#     1D element representation of the annulus fluid temperature, ϕb_a:
#         Ca*∂ϕb_a(t,z)/∂t = (ϕb_c(t,z)-ϕb_a(t,z))/Rac
#                           +(ϕb_w(t,z)-ϕb_a(t,z))/Rb
#                           -m*Cf*∂ϕb_a(t,z)/∂z
#
#     Heat flux at the DBHE wall, qb_w:
#         qb_w(t,z) = (ϕb_a(t,z)-ϕb_w(t,z))/(Rb+Rs)
#
#     Boundary and initial conditions:
#         ϕb_c(t,z=0) = ϕb_a(t,z=0)-Q/(m*Cf)
#         ϕb_c(t,z=zb) = ϕb_a(t,z=zb)
#         ϕb_c(t=0,z) = ϕ0(z)
#         ϕb_a(t=0,z) = ϕ0(z)
#         ϕ0(z) = ϕs+gg*z
#
#         z range for borehole b: 1:zb
#
# References:
#    10.1016/j.renene.2024.121963
#    10.1016/j.renene.2021.07.086
#    10.1016/j.energy.2019.05.228
#    10.1016/j.renene.2021.01.036
#    10.1016/j.enbuild.2018.02.013
#
################################################################################
# DBHE array model #############################################################
# 
# Equation for ground model:
#     ρ(x,y,z) c(x,y,z) ∂ϕ(t,x,y,z)/∂t =   ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂x)/∂x
#                                        + ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂y)/∂y
#                                        + ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂z)/∂z
#
# Boundary conditions:
#   Ground upper surface: ks*∂ϕ(t,x,y,z)/∂z = hs*(ϕ-ϕs)
#   Ground lower surface: ϕ(t,x,y,z=zz) = ϕ0(z=zz)
#   Ground lateral sides: ∇ϕ(t,x,y,z),n = 0
#   DBHE walls (1D elements): ϕ(t,x=xb,y=yb,z=1:zb) = ϕb_w(t,z=1:zb)
#
# Coupling of DBHE model (1D) with array model (3D):
#     Temperature at the DBHE wall, ϕb_w, is computed using
#     the heat flux at the DBHE wall, qb_w.
#
# Initial conditions:
#  ϕ(t=0,x,y,z) = ϕ0(z), (x,y,z) ∉ borehole positions.
#
################################################################################


# Geological parameters (10.1016/j.renene.2021.07.086) #########################

# Layer 1: Clay
# Length (m)
dl1 = 636
# Thermal conductivity (W/m.K)
kl1 = 1.8
# Density (kg/m3)
ρl1 = 1780
# Speciﬁc heat capacity (J/kg.K)
Cl1 = 1379
# Diffusion coefficient
D1 = kl1/(ρl1*Cl1)

# Layer 2: Mudstone
# Length (m)
dl2 = 562
# Thermal conductivity (W/m.K)
kl2 = 2.6
# Density (kg/m3)
ρl2 = 2030
# Speciﬁc heat capacity (J/kg.K)
Cl2 = 1450
# Diffusion coefficient
D2 = kl2/(ρl2*Cl2)

# Layer 3: Medium sand
# Length (m)
dl3 = 562
# Thermal conductivity (W/m.K)
kl3 = 3.5
# Density (kg/m3)
ρl3 = 1510
# Speciﬁc heat capacity (J/kg.K)
Cl3 = 1300
# Diffusion coefficient
D3 = kl3/(ρl3*Cl3)

# Layer 4: Stand stone
# Length (m)
dl4 = 1090
# Thermal conductivity (W/m.K)
kl4 = 5.3
# Density (kg/m3)
ρl4 = 2600
# Speciﬁc heat capacity (J/kg.K)
Cl4 = 878
# Diffusion coefficient
D4 = kl4/(ρl4*Cl4)

# Diffusion coefficient
function diff_coeff(z)
    if z < dl1 
        return D1
    elseif z < dl1+dl2
        return D2
    elseif z < dl1+dl2+dl3
        return D3
    else 
        return D4
    end
end

# Ground surface temperature (K). 10.1016/j.renene.2021.07.086.
ϕs = 283.15
# Convective heat transfer coeﬃcient on ground surface (W/(m^22 K)). 10.1016/j.enbuild.2018.02.013
hs = 15
# Thermal conductivity of subsurface (W/(m K)). 10.1016/j.enbuild.2018.02.013.
ks = 2.5
# Geothermal gradient (K/m). 10.1016/j.energy.2019.05.228.
gg = 2/100 
# Ground temperature as a function of depth (K). 10.1016/j.renene.2021.07.086.
ϕ0(z) = ϕs+gg*z

# DBHE parameters ##############################################################

# Borehole depth (m). 10.1016/j.renene.2021.07.086.
dbhe_depth = 2000
# Borehole diameter (m). 10.1016/j.enbuild.2018.02.013
db = 0.28 
# Inner center pipe diameters (m). # 10.1016/j.enbuild.2018.02.013
dco = 0.14 
dci = 0.124
# Outer annulus pipe  diameters (m). 10.1016/j.enbuild.2018.02.013
dao = 0.2
dai = 0.188
# Thermal conductivity of center pipe (W/m.K). 10.1016/j.renene.2021.07.086.
kc = 0.4
# Thermal conductivity of annulus pipe (W/m.K).  10.1016/j.renene.2021.07.086.
ka = 41
# Thermal conductivity of water (W/m.K).
kf = 0.618
# Thermal conductivity of grout (W/m.K). 10.1016/j.renene.2021.07.086.
kg = 1.5
# Speciﬁc heat capacity of water (J/kg.K). 10.1016/j.renene.2021.07.086.
Cf = 4174
# Pipe specific heat capacity (J/(kg⋅K)). 10.1016/j.enbuild.2018.12.006.
Cp = 2100 
# Grout specific heat capacity (J/(kg⋅K)). 10.1016/j.renene.2024.121963.
Cg = 1735 
# Water density (kg/m³). 10.1016/j.renene.2024.121963.
ρf = 998
# Pipe density (kg/m³). 10.1016/j.enbuild.2018.12.006.
ρp = 930
# Grout density (kg/m³). 10.1016/j.renene.2024.121963.
ρg = 2190
# Mass flow rate (kg/s). 10.1016/j.enbuild.2018.02.013
mfr = 11.65

μ_f = 0.000931 # [Pa.s] fluid dynamic viscosity

## Convection coefficients
va = mfr / (ρf * π * (dai^2-dco^2)/4) # velocity of fluid in the annulus (m/s)
hda = (dai - dco) # hydraulic diameter
Re_a = ρf * va * hda / μ_f 
Pr = Cf * μ_f / kf
Nu_a = 0.023 * Re_a^0.8 * Pr^0.4
h3 = Nu_a * kf / hda #convection coefficient btn inner annulus and water

h2=h3 # convection coefficient btn outer wall of center pipe and water

vc = mfr / (ρf*π*dci^2/4) # velocity of fluid in the center (m/s)
Re_c = ρf * vc * dci / μ_f
Nu_c = 0.023 * Re_c^0.8 * Pr^0.4
h1 = Nu_c * kf / dci # convection coefficient btn inner wall of center pipe and water

# Thermal resistance between center and annulus pipe ((K m)/W). 10.1016/j.energy.2019.05.228.
# Rac = 0.08 #R12
Rac = 1/(π*dci*h1) + log(dco/dci)/(2*π*kc) + 1/(π*dco*h2)
# Thermal resistance between annulus and borehole wall ((K m)/W). 10.1016/j.energy.2019.05.228.
# Rb = 0.025 
Rb = 1/(π*dai*h3) + log(dao/dai)/(2*π*ka) + log(db/dao)/(2*π*kg)

# Thermal resistance between the borehole and the surrounding soil ((K m)/W). 10.1016/j.energy.2019.05.228.
Rs = 0.001 
# Thermal capacity of circulating ﬂuid in the annulus, Ca. 10.1016/j.renene.2024.121963.
Ca = (π/4)*(dai^2-dco^2)*ρf*Cf
    +(π/4)*(dao^2-dai^2)*ρp*Cp
    +(π/4)*(db^2-dao^2)*ρg*Cg

# Thermal capacity of circulating ﬂuid in the center, Cc. 10.1016/j.renene.2024.121963.
Cc = (π/4)*(dci^2*ρf*Cf)
    +(π/4)*(dco^2-dci^2)*ρp*Cp

# Heat extraction rate Q (W)
Q = 300_000
# DBHE positions
# xs = [3, 4] # [5]
# ys = [5, 5] # [5]
xs = [5]
ys = [5]
# Number of DBHEs
bb = length(xs)

# Get DBHE index
function dbhe_index(x,y)
    for (b,(xb,yb)) in enumerate(zip(xs,ys))
        r = norm([x,y]-[xb,yb]) # Distance to DBHE center
        if r<dao/2
            return b
        end
    end
end

function dbhe_indexes(b)
    return xs[b], ys[b]
end

# Numerical parameters #########################################################

# Maximum simulation time (s)
sim_time = 3_960_000 # 1100 hours

# Geometry distances (m)
xx = 10 
yy = 10
zz = dl1+dl2+dl3+dl4+30
zzb = dbhe_depth

# dx, dy, dz (m)
dx = db
dy = db
dz = 40
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz
rzb = 0:dz:zzb

# dt using stability condition (check this)
# TODO: verify the stability condition
dtd = (1/(2*maximum([D1,D2,D3,D4]))*(1/dx^2+1/dy^2+1/dz^2)^-1)
#dtc = minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)]) # no convection
#dt = minimum([dtd,dtc])

dt = 10 
println("dt:$dt")

if dt > dtd
    error("dt should not exceed dtd")
end

# No. of time iterations
tt = round(Int,sim_time/dt)
println("tt:$tt")

# Save step
st = 1000 

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)
kkb = round(Int,zzb/dz+1)
nn = ii*jj*kk
println("ii:$ii, jj:$jj, kk:$kk. nn:$nn.")

#### Update functions ####

# Inlet water temperature. 10.1016/j.renene.2021.07.086.
df = CSV.read("./validation_data/inlet-temp-vs-time.csv", DataFrame)
tin = (df."time (h)")*3600 # s
ϕin = df."water_temp (°C)".+273.15 # K

function get_ϕin(t;ϕin=ϕin,tin=tin)
    ind = maximum([findfirst(x->x>=t,tin)-1, 1])
    return ϕin[ind]
end

# Outlet water temperature. 10.1016/j.renene.2021.07.086.
df = CSV.read("./validation_data/outlet-temp-vs-time.csv", DataFrame)
tout = (df."time (h)")*3600 # s
ϕout = df."water_temp (°C)".+273.15 # K

function get_ϕout(t;ϕout=ϕout,tout=tout)
    ind = maximum([findfirst(x->x>=t,tout)-1, 1])
    return ϕout[ind]
end

# Update temperature at the center and annulus of each DBHE, ϕc and ϕa. 1D model.
function update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr,Cf,Ca,Cc,Q)
    for b in 1:bb
        ϕa2[b,1] = get_ϕin(dt*nt) # 10.1016/j.renene.2021.07.086.
        for k in 2:kkb-1
            # Fluid temperature in the annulus of the well
            diff = (ϕc1[b,k]-ϕa1[b,k])/Rac+(ϕbw[b,k]-ϕa1[b,k])/Rb
            conv = -mfr*Cf*(ϕa1[b,k]-ϕa1[b,k-1])/dz
            ϕa2[b,k] = (diff+conv)*dt/Ca+ϕa1[b,k]
            # Fluid temperature in the center of the well
            diff = (ϕa1[b,k]-ϕc1[b,k])/Rac
            conv = +mfr*Cf*(ϕc1[b,k+1]-ϕc1[b,k])/dz
            ϕc2[b,k] = (diff+conv)*dt/Cc+ϕc1[b,k]
        end
        ϕa2[b,kkb] = ϕa2[b,kkb-1]
        ϕc2[b,kkb] = ϕa2[b,kkb]
        ϕc2[b,1] = ϕc2[b,2]
    end
end

# Update heat flux at the wall of each DBHE, qbw. 1D model.
function update_q_dbhe_wall!(qbw,ϕa,ϕbw,bb,kkb,dz,dt,Rb,Rs)
    for b in 1:bb
        for k in 1:kkb
            # TODO: check
            if dz*k < 30 # Depth of insulated section of borehole. 10.1016/j.enbuild.2018.02.013
                qbw[b,k] = 0
            else
                qbw[b,k] = (ϕa[b,k]-ϕbw[b,k])/(Rb+Rs)*20 # TODO: where does the 20 come from??
            end
            #qbw[b,k] = (ϕa[b,k]-ϕbw[b,k])/(Rb+Rs)*20
        end
    end
end

# Update temperature at the borehole walls, ϕbw. 1D model.
function update_ϕ_dbhe_wall!(ϕbw,ϕ,qbw,ka,bb,kkb,dx,dy,dz,dt,ϕa)
    delta = (dx+dy)/2
    for b in 1:bb
        i,j = dbhe_indexes(b)
        for k in 1:kkb
            # TODO: check
            # #q1 = -ka*(ϕ[i+1,j,k]-ϕbw[b,k])/delta
            # #q2 = -ka*(ϕ[i,j+1,k]-ϕbw[b,k])/delta
            # #q3 = ka*(ϕbw[b,k]-ϕ[i-1,j,k])/delta
            # #q4 = ka*(ϕbw[b,k]-ϕ[i,j-1,k])/delta
            # #qbw[b,k] = q1+q2+q3+q4 => ϕbw[b,k]
            ϕbw[b,k] = (ϕ[i+1,j,k]+ϕ[i,j+1,k]+ϕ[i-1,j,k]+ϕ[i,j-1,k])/4+
                       qbw[b,k]*delta/(4*ka)
            ## NEW
            # avg_ϕ = (ϕ[i+1,j,k]+ϕ[i,j+1,k]+ϕ[i-1,j,k]+ϕ[i,j-1,k]) / 4.0
            # # q1 = ka * (avg_ϕ - ϕbw[b,k]) / delta # flow of heat from the surrounding rocks
            # q_in = (avg_ϕ - ϕbw[b,k])/Rs
            # Rconv = 1/(h3*2*π*dai)
            # q_out = (ϕbw[b,k] - ϕa[b,k])/(Rb+Rconv) # flow of heat from the annulus i.e heat loss into the fluid
            # q_net = q_in - q_out 
            # ϕbw[b,k] = ϕbw[b,k] + dt * q_net/(ρp*Cp*π*(dao^2-dai^2)/4)
        end
    end
end

# Update temperature in ground domain. 3D model.
function update_ϕ_ground!(ϕ2,ϕ1,ϕbw,D,dx,dy,dz,dt,ii,jj,kk)
    @threads for k = 2:kk-1
        for j = 2:jj-1
            for i = 2:ii-1
                if D[i,j,k] > 0  # Ground domain
                    # Diffusive term
                    diff = (((D[i+1,j,k]+D[i,j,k])/2*(ϕ1[i+1,j,k]-ϕ1[i,j,k])/dx
                            -(D[i,j,k]+D[i-1,j,k])/2*(ϕ1[i,j,k]-ϕ1[i-1,j,k])/dx)/dx
                           +((D[i,j+1,k]+D[i,j,k])/2*(ϕ1[i,j+1,k]-ϕ1[i,j,k])/dy
                            -(D[i,j,k]+D[i,j-1,k])/2*(ϕ1[i,j,k]-ϕ1[i,j-1,k])/dy)/dy
                           +((D[i,j,k+1]+D[i,j,k])/2*(ϕ1[i,j,k+1]-ϕ1[i,j,k])/dz
                            -(D[i,j,k]+D[i,j,k-1])/2*(ϕ1[i,j,k]-ϕ1[i,j,k-1])/dz)/dz)
                    # Source term
                    source = 0.0
                    # Update temperature
                    ϕ2[i,j,k] = (diff+source)*dt+ϕ1[i,j,k]
                else  # DBHE domain
                    b = dbhe_index(i*dx,j*dy)
                    ϕ2[i,j,k] = ϕbw[b,k]
                end
            end
         end
    end
end

# Update temperature in ground walls. 3D model.
function update_ϕ_ground_bound!(ϕ,ii,jj,kk,dx,dy,dz,hs,ks,ϕs)
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    for j = 2:jj-1
        for i = 2:ii-1
            #10.1016/j.enbuild.2018.02.013
            ϕ[i,j,1] = (ϕ[i,j,2]*ks/dz-ϕs*hs)/(ks/dz-hs)
        end
    end
    nothing
end

# Initial and boundary conditions ##############################################

# Temperature at the center of the DBHEs. 1D model.
ϕc2 = ones(bb,kkb).*ϕ0.(rzb)'
ϕc1 = ones(bb,kkb).*ϕ0.(rzb)'
# Temperature at the annulus of the DBHEs. 1D model.
ϕa2 = ones(bb,kkb).*ϕ0.(rzb)'
ϕa1 = ones(bb,kkb).*ϕ0.(rzb)'
# Temperature at the DBHE wall. 1D model.
ϕbw = ones(bb,kkb).*ϕ0.(rzb)'
# Heat flux at the DBHE wall. 1D model.
qbw = zeros(bb,kkb)
update_q_dbhe_wall!(qbw,ϕa2,ϕbw,bb,kkb,dz,dt,Rb,Rs) # qbw

# Ground temperature. 3D model.
ϕ2 = zeros(ii,jj,kk)
ϕ1 = zeros(ii,jj,kk)
# Diffusion coefficient
D = zeros(ii,jj,kk)
for k in 1:kk
    zk = k*dz
    for j in 1:jj
        yj = j*dy
        for i in 1:ii
            xi = i*dx
            # Ground
            D[i,j,k] = diff_coeff(zk)
            ϕ2[i,j,k] = ϕ0(zk)
            # DBHE
            if zk<=dbhe_depth
                for (xb,yb) in zip(xs,ys)
                    r = norm([xi,yj]-[xb,yb]) # Distance to DBHE center
                    if r<=dao/2
                        D[i,j,k] = 0
                        ϕ2[i,j,k] = ϕ0(zk)
                    end
                end
            end
        end
    end
end
ϕ1 .= ϕ2
save(path,"diff_coeff",D,rx,ry,rz,0)

# Predicted outlet water temperature. 
ϕout_pred = zeros(bb,ceil(Int,tt/st)+1)

# Saved times
ts = collect(0:st*dt:tt*dt)

### Simulation ###

# Run simulation
function run_sim(;to_print=false, compute_energy=false)
    heat_output = 0.0
    for nt = 0:2:tt
        # Save ϕ
        if nt % st == 0
            if to_print
                println("Iteration:$nt, time:$(round(nt*dt/60/60,digits=2))hs, " *
                        "inlet temp:$(round(ϕa1[1,1].-273.15,digits=4))°C, " *
                        "outlet temp:$(round(ϕc1[1,1].-273.15,digits=4))°C")
            end
            ϕout_pred[:,(nt÷st)+1] = ϕc1[:,1]
        end
        # Update ϕ
        update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # 1D: ϕa2,ϕa1,ϕc2,ϕc1
        update_q_dbhe_wall!(qbw,ϕa2,ϕbw,bb,kkb,dz,dt,Rb,Rs) # 1D: qbw
        update_ϕ_dbhe_wall!(ϕbw,ϕ2,qbw,ka,bb,kkb,dx,dy,dz,dt,ϕa2) # 1D: ϕbw
        update_ϕ_ground!(ϕ2,ϕ1,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # 3D: ϕ2,ϕ1
        update_ϕ_ground_bound!(ϕ2,ii,jj,kk,dx,dy,dz,hs,ks,ϕs) # 3D: ϕ2,ϕ1
        # output energy, TODO: modify this to handle the case for changing input temperatures
        if compute_energy
            for well_idx in 1:bb
                heat_output += mfr * dt * Cf * (ϕc2[well_idx,1] - ϕa2[well_idx,1])
            end
        end

        # Update ϕ
        update_ϕ_fluid!(ϕa1,ϕa2,ϕc1,ϕc2,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # 1D: ϕa1,ϕa2,ϕc1,ϕc2
        update_q_dbhe_wall!(qbw,ϕa1,ϕbw,bb,kkb,dz,dt,Rb,Rs) # 1D: qbw
        update_ϕ_dbhe_wall!(ϕbw,ϕ1,qbw,ka,bb,kkb,dx,dy,dz,dt,ϕa1) # 1D: ϕbw
        update_ϕ_ground!(ϕ1,ϕ2,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # 3D: ϕ1,ϕ2
        update_ϕ_ground_bound!(ϕ1,ii,jj,kk,dx,dy,dz,hs,ks,ϕs) # 3D: ϕ1,ϕ2
        if compute_energy
            for well_idx in 1:bb
                heat_output += mfr * dt * Cf * (ϕc1[well_idx,1] - ϕa1[well_idx,1])
            end
        end
    end
    return heat_output
end


### Plots and Validation ###

function plot_validation()
    # Combined validation plot
    plot(ts/3600, get_ϕin.(ts).-273.15,
        label="Inlet temperature [°C]")
    plot!(ts/3600, get_ϕout.(ts).-273.15,
        label="Outlet temperature [°C]")
    plot!(ts/3600, ϕout_pred[1,:].-273.15, # changed from scatter to plot
        label="Predicted outlet temperature [°C]")
    plot!(xlabel="Time [hours]", ylabel="Temperature [°C]", ylims=(5, 40))
    savefig("$path/combined_validation.png")
end

function visualize_simulation()
    # Ground temperature
    rzrev = reverse(-1*rz[1:end-1])
    @views ϕrev = reverse(ϕ1[:,jj÷2,:]'.-273.15, dims=1) # NEW
    heatmap(rx, rzrev, ϕrev, colorbar_title = "Temperature (°C)", colormap=:jet1)
    contour!(rx, rzrev, ϕrev, linewidth = 1, linecolor = :black)
    contour!(xlabel="Distance [m]", ylabel="Depth [m]")
    savefig("$path/ground-temp.png")
end

heat_output = run_sim(to_print=true)
print("Heat Output:", heat_output)
# visualize_simulation()
plot_validation()


## TODO: Correct the validation approach
# First paper: Describe the array system i.e how many wells, layers, duration of data collection etc
# Approach: Input(inlet temperature, flow rate), output(simulated outlet temperature)
# Fix the implementation of validation 
