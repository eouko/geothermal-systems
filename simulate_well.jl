
using Plots

## Main Validation Paper: A slightly inclined deep borehole heat exchanger array behaves better than vertical installation
## Second Paper: Proposing stratiﬁed segmented ﬁnite line source (SS-FLS) method for dynamic 
#                simulation of medium-deep coaxial borehole heat exchanger in multiple ground layers

## Boundary conditions
# -> T_f1(0) = T_in (inlet temp)
# -> T_f1(z_b) = T_f2(z_b) (temp at the bottom of the well) 


### Parameters ###
ϕ0 = 25.0 # [°C] surface temperature
ϕf0 = 20.0 # [°C] initial temperature of the fluid
m = 25/3600 # [kgm^3/s] mass flow rate
d_b = 0.2159 # [m] well diameter
H = 2000 # [m] well depth
d_ao = 0.1778 # [m] annular pipe outer diameter (d1,o)
th_a = 0.0092 # [m] thickness of the annular pipe
d_co = 0.1100 # [m] center pipe outer diameter (d2,o)
th_c = 0.01 # [m] thickness of the center pipe
ρ_f = 998 # [kg/m³] fluid density
λ_f = 0.6 # [W/mK] fluid thermal conductivity
c_f = 4190 # [J/kgK] fluid specific heat
μ_f = 0.000931 # [Pa.s] fluid dynamic viscosity
ρ_g = 2190 # [kg/m³] grout density
λ_g = 0.63 # [W/mK] grout thermal conductivity
c_g = 1735 # [J/kgK] grout specific heat
g = 0.033 # [°C/m] geothermal gradient

### NOTE: The following parameter values are not provided in the validation paper
# Pipe parameters
ρ_p = 7700 # [kg/m³] pipe density
λ_1p = 0.40 # [W/mK] thermal conductivity of the annular(outer) pipe, (from second paper)
λ_2p = 0.40 # [W/mK] thermal conductivity of the center(inner) pipe, (from second paper)
c_p = 420 # [J/kgK] specific heat of the pipe

# Rock parameters (assumed granite)
c_r = 790 # [J/kgK], specific heat of the rock
ρ_r = 2700 # [kg/m³], density of the rock
λ_r =  3.12 # [W/mK], thermal conductivity of the rock

a_r = λ_r/(ρ_r*c_r) # [m²/s] thermal diffusivity of the rock

## Computing the convection heat Coefficients
# Assumptions: Forced convection, turbulent internal flow through a cylinder
# Dittus-Boelter correlation: Nu = 0.023 * Re^0.8 * Pr^0.4, source: https://www.sciencedirect.com/topics/engineering/dittus-boelter-correlation
# Re = ρvD/μ (density, velocity, diameter, dynamic viscosity - in that order): https://www.sciencedirect.com/topics/physics-and-astronomy/reynolds-number
# Pr = cμ/λ (s.h.c, dynamic viscosity, thermal conductivity): source: wikipedia, https://en.wikipedia.org/wiki/Prandtl_number
# Convection Heat Transfer Coefficients: h2 = h3: https://www.sciencedirect.com/science/article/pii/S0017931019321684

###
d_ai = d_ao - 2*th_a # [m] annular pipe inner diameter (d1,i)
d_ci = d_co - 2*th_c # [m] center pipe inner diameter (d2,i)
A1 = π * ((d_ai/2)^2 - (d_co/2)^2) # cross sectional area of annular space
A2 =  π * (d_ci/2)^2 # cross sectional area of center space
C1 = π/4 * (d_ai^2 - d_co^2)*ρ_f*c_f + π/4 * (d_b^2 - d_ao^2)*ρ_g*c_g

# Velocities
v1 = m / (A1 * ρ_f) # velocity in annular space, [ms^-1]
v2 = m / (A2 * ρ_f) # velocity in the center, [ms^-1]

## Convective Heat Transfer Coefficients
# h3: between water and inner wall of the annular pipe
d_ch3 = (d_ai - d_co) # hydraulic diameter
Re3 = ρ_f * v1 * d_ch3 / μ_f 
Pr = c_f * μ_f / λ_f
Nu = 0.023 * Re3^0.8 * Pr^0.4
h3 = Nu * λ_f / d_ch3

# h2: between water and outer wall of center pipe
h2 = h3 

# h1: between water and inner wall of center pipe
d_ch1 = d_ci # hydraulic diameter
Re1 = ρ_f * v2 * d_ch1 / μ_f
Nu = 0.023 * Re1^0.8 * Pr^0.4
h1 = Nu * λ_f/d_ch1
##

R_12 = 1/(π*d_ci*h1) + log(d_co/d_ci)/(2*π*λ_2p) + 1/(π*d_co*h2) 
R_b = 1/(π*d_ai*h3) + log(d_ao/d_ai)/(2*π*λ_1p) + log(d_b/d_ao)/(2*π*λ_g)
C2 = π/4*(d_ci^2)*ρ_f*c_f + π/4 * (d_co^2 - d_ci^2)*ρ_p*c_p

### NOTE: the following values are also not provided in the paper
Δz = 10.0 # [m] 

Δt = 1000
println("Δt = ", Δt)

if Δt >= Δz / max(v1, v2)
    error("CFL condition not met")
end

if H/Δz <= 2
    error("Grid size too small")
end

## Grid Dimensions 

Nz = ceil(Int,H/Δz)

# Well Positions
# well_positions = [(5, 5), (10, 10), (10, 5), (5, 10), (15, 5), (5, 15), (15, 10), (10, 15), (15, 15)] # [(x1, y1),...(xn, yn)]
well_positions = [(5, 5), (10, 10)]

# nw = length(well_positions)

# Initialize the temperature field
function init_temperature(well_positions)
    num_wells = length(well_positions)
    ϕ_a = fill(ϕf0, (num_wells, Nz)) # annular space
    ϕ_c = fill(ϕf0, (num_wells, Nz)) # center space
    ϕ_b = zeros(num_wells, Nz) # well walls
    for k in 1:Nz
        ϕ_b[:, k] .= ϕ0 + g * k * Δz
    end 
    return ϕ_b, ϕ_a, ϕ_c
end

## Heat Transfer Update
function update_temperature!(ϕ_b, ϕ_a, ϕ_c, well_positions, step)

    ϕ_b_new = copy(ϕ_b)
    ϕ_a_new = copy(ϕ_a)
    ϕ_c_new = copy(ϕ_c)

    # Precompute all possible vertical distances once
    all_z = (1:Nz) .* Δz
    v_dists = [abs2(z_source - z) for z_source in all_z, z in all_z]

    τ = step * Δt
    _a = (4*a_r*τ)
    const_factor = Δz/(8*ρ_r*c_r*(π*a_r*τ)^1.5)

    for (idx, (x,y)) in enumerate(well_positions) 
        Threads.@threads for k in 1:Nz            
            if k == 1 # Update the top of the well
                # central pipe
                ϕ_c_new[idx, 1] = ϕ_c[idx, 1] + 
                        ((ϕ_a[idx,1]-ϕ_c[idx,1])/R_12 + 
                        m*c_f*(ϕ_c[idx,2]-ϕ_c[idx,1])/Δz +
                        A2*λ_f*(ϕ_c[idx,3]-2*ϕ_c[idx,2]+ϕ_c[idx,1])/(Δz^2)) *
                        (Δt/C2)

                ## annular space, boundary condition => inlet temperature
                ϕ_a_new[idx,1] = ϕf0
                
            elseif k == Nz # Update the bottom of well
                # Annulus
                ϕ_a_new[idx,Nz] = ϕ_a[idx,Nz] + 
                        ((ϕ_c[idx,Nz]-ϕ_a[idx,Nz])/R_12 + 
                        (ϕ_b[idx,Nz]-ϕ_a[idx,Nz])/R_b - 
                        m*c_f*(ϕ_a[idx,Nz]-ϕ_a[idx,Nz-1])/Δz + 
                        A1*λ_f*(ϕ_a[idx,Nz]-2*ϕ_a[idx,Nz-1]+ϕ_a[idx,Nz-2])/(Δz^2)) * 
                        (Δt/C1)
                # Enforcing boundary condition at the bottom of the well
                ϕ_c_new[idx, Nz] = ϕ_a[idx,Nz]

            else
                ## center space
                ϕ_c_new[idx, k] = ϕ_c[idx, k] + 
                        ((ϕ_a[idx,k]-ϕ_c[idx,k])/R_12 + 
                        # m*c_f*(ϕ_c[idx,k+1]-ϕ_c[idx,k-1])/(2*Δz) + 
                        m*c_f*(ϕ_c[idx,k+1]-ϕ_c[idx,k])/Δz +
                        A2*λ_f*(ϕ_c[idx,k+1]-2*ϕ_c[idx,k]+ϕ_c[idx,k-1])/(Δz^2)) *
                        (Δt/C2)
                ## annular space
                ϕ_a_new[idx,k] = ϕ_a[idx,k] + 
                        ((ϕ_c[idx,k]-ϕ_a[idx,k])/R_12 + 
                        (ϕ_b[idx,k]-ϕ_a[idx,k])/R_b - 
                        # m*c_f*(ϕ_a[idx,k+1]-ϕ_a[idx,k-1])/(2*Δz) + 
                        m*c_f*(ϕ_a[idx,k]-ϕ_a[idx,k-1])/Δz +
                        A1*λ_f*(ϕ_a[idx,k+1]-2*ϕ_a[idx,k]+ϕ_a[idx,k-1])/(Δz^2)) * 
                        (Δt/C1)
            end

            current_v_dist = @view v_dists[:,k]

            for i in eachindex(well_positions) # update all ϕ_b due to the current well
                if i == idx
                    continue
                end
                if k > 1
                    new_x, new_y = well_positions[i] 
                    if abs(new_x - x)<= d_b && abs(new_y - y) <= d_b 
                        continue
                    end

                    r_d = sqrt((x - new_x)^2 + (y - new_y)^2)

                    dist = (current_v_dist .+ r_d^2) ./ _a
                    kernel_val = const_factor * exp.(-dist)
                    q = (ϕ_a[i,:] - ϕ_b[i,:]) / R_b # flux due to water in the current well
                    ϕ_b_new[i,k] += sum(kernel_val .* q) * Δt # TODO: check
                end
            end
        end
        ϕ_b .= ϕ_b_new
        ϕ_a .= ϕ_a_new
        ϕ_c .= ϕ_c_new
    end
end


function run_simulation(simulation_hrs, init_well_positions)

    simulation_time = simulation_hrs * 3600 # seconds

    ## Simulation Loop
    ϕ_b, ϕ_a, ϕ_c = init_temperature(init_well_positions)
    simulation_steps = ceil(Int, simulation_time / Δt)

    ## Heat Output
    function get_heat_output(well_positions)
        heat_output = 0.0
        for step in 1:simulation_steps
            update_temperature!(ϕ_b, ϕ_a, ϕ_c, well_positions, step)
            for idx in eachindex(well_positions)
                Δϕ = ϕ_c[idx, 1] - ϕf0
                heat_output += m * c_f * Δϕ
            end
        end
        println("Heat Output:", heat_output)
        return heat_output
    end
    return get_heat_output
end

st = 1000 # save interval

## Plot output temp.
function plot_outlet_temp(st, well_positions)
    num_points = floor(Int, simulation_steps/st)
    outlet_temp = zeros(length(well_positions), num_points)
    for step in 1:simulation_steps
        update_temperature!(ϕ_b, ϕ_a, ϕ_c, well_positions, step)
        if step % st == 0
            point_idx = Int(step/st)
            for idx in 1:nw
                outlet_temp[idx, point_idx] = ϕ_c[idx, 1]
            end
            # println("Outlet temp:", outlet_temp[1, point_idx])
        end
    end

    time = [st*i/3600.0 for i in 1:num_points]
    plot(time, outlet_temp[1, :], label="Outlet Temp. (Edge well)", xlabel="Time (h)", ylabel="Temp. (°C)", title="Outlet Temperature vs Time")
    plot!(time, outlet_temp[2, :], label="Outlet Temp. (Interior Well)")
    savefig("other_well.png")
end

# simulation_hrs = 2000

# plot_outlet_temp(st, well_positions)

# ans = run_simulation(simulation_hrs, well_positions)(well_positions)
# println("ans:", ans)
