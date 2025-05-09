using Plots; gr()
using Measures
using BenchmarkTools


# Optimization Algorithms
include("optimization_algorithms/differential_evolution.jl")
include("optimization_algorithms/pso_algorithm.jl")
include("optimization_algorithms/simulated_annealing.jl")
include("optimization_algorithms/nelder_mead.jl")
include("optimization_algorithms/nes.jl")
include("optimization_algorithms/bayesian_optimization.jl")

include("./simulate_well.jl")
# include("./utils.jl")

# TODO: get resource usage for the algorithms (add benchmark)

simulation_time = 100 # hours

well_positions = [(2.5, 2.5),(5.0, 5.0), (7.5, 7.5), (10.0, 10.0), (12.5, 12.5), (15.0, 15.0), (17.5, 17.5)]

px = [p[1] for p in well_positions]
py = [p[2] for p in well_positions]

bound_length = length(well_positions)*2
bounds = [(0.0, 20.0) for i in 1:bound_length]

function f(vecx)
    wp = [] # a list of tuples
    half_length = ceil(Int, length(vecx)/2.0)
    for i in 1:half_length
        push!(wp, (vecx[i], vecx[i+half_length]))
    end
    return run_simulation(simulation_time, well_positions)(wp)
end

# Bayesian Optimization

final_config = bayesian_optimization(f, px, py, bounds; 
                        max_iters=20,
                        plot_name="./optimization_plots/bayesian_optimization_ucb.png", 
                        plot_title="Bayesian Optimization", 
                        acq="ucb"
                        )
well_config_name = "optimization_well_configs/bayesian_opt_ucb_well_locations.png"
# # 275.971 s (2870482846 allocations: 1011.46 GiB)
# 199796.123263575

# final_config = bayesian_optimization(f, px, py, bounds; 
#                         max_iters=20,
#                         plot_name="./optimization_plots/bayesian_optimization_ei.png", 
#                         plot_title="Bayesian Optimization",
#                         acq="ei"
#                         )
# well_config_name = "optimization_well_configs/bayesian_opt_ei_well_locations.png"
# # # 279.898 s (2867474850 allocations: 1011.42 GiB)
# # 199796.12326362042

# final_config = bayesian_optimization(f, px, py, bounds; 
#                         max_iters=20,
#                         plot_name="./optimization_plots/bayesian_optimization_poi.png", 
#                         plot_title="Bayesian Optimization",
#                         acq="poi"
#                         )
# well_config_name = "optimization_well_configs/bayesian_opt_poi_well_locations.png"
# # # 281.139 s (2867902368 allocations: 1011.43 GiB)
# # 199796.121160881


# # Differential Evolution Optimization
# final_config = differential_evolution(f, px, py, bounds; 
#                         max_iters=20, 
#                         plot_name="./optimization_plots/de_optim.png", 
#                         plot_title="Differential Evolution Optimization"
#                         )
# well_config_name = "optimization_well_configs/de_well_locations.png"
# # # 525.654 s (5592833468 allocations: 1967.79 GiB)
# # 199796.12181089853

# # Particle Swarm Optimization
# final_config = pso_algorithm(f, px, py, bounds; 
#                 max_iters=20, 
#                 plot_name="./optimization_plots/pso_optim.png", 
#                 plot_title="Particle Swarm Optimization")
# well_config_name = "optimization_well_configs/pso_well_locations.png"
# # # 378.596 s (3961904992 allocations: 1394.40 GiB)
# # 199796.1034


# Simulated Annealing
final_config = simulated_annealing(f, px, py, bounds; 
                    max_iters=20, 
                    plot_name="./optimization_plots/simulated_annealing_optim.png", 
                    plot_title="Simulated Annealing Optimization")
well_config_name = "optimization_well_configs/simulated_annealing_well_locations.png"
# # # 257.839 s (2730111948 allocations: 963.13 GiB)
# # 199796.0108

# # Nelder-Mead
# final_config = nelder_mead(f, px, py, bounds; 
#             max_iters=20, 
#             plot_name="./optimization_plots/nelder_mead_optim.png", 
#             plot_title="Nelder-Mead Optimization")
# well_config_name = "optimization_well_configs/nelder_mead_well_locations.png"
# # # 548.157 s (5460217988 allocations: 1926.25 GiB)
# # 1.997958e+05

# # Natural Evolution Strategies
# final_config = natural_evolution_strategies(f, px, py, bounds; 
#                             max_iters=2, 
#                             plot_name="./optimization_plots/snes_optim.png", 
#                             plot_title="Natural Evolution Strategy Optimization",
#                             nes_type="snes"
#                             )
# well_config_name = "optimization_well_configs/snes_well_locations.png"
# # # 329.723 s (3414680337 allocations: 1203.94 GiB)
# # 199796.1005

# final_config = natural_evolution_strategies(f, px, py, bounds; 
#                             max_iters=2, 
#                             plot_name="./optimization_plots/xnes_optim.png", 
#                             plot_title="Natural Evolution Strategy Optimization",
#                             nes_type="xnes"
#                             )
# well_config_name = "optimization_well_configs/xnes_well_locations.png"
# # # 406.452 s (4227995425 allocations: 1490.61 GiB)
# # 199795.7983

# final_config = natural_evolution_strategies(f, px, py, bounds; 
#                             max_iters=2, 
#                             plot_name="./optimization_plots/dxnes_optim.png", 
#                             plot_title="Natural Evolution Strategy Optimization",
#                             nes_type="dxnes"
#                             )
# well_config_name = "optimization_well_configs/dxnes_well_locations.png"
# # # 403.248 s (4233583368 allocations: 1492.88 GiB)
# # 199795.155

## Plotting well configurations
starting_config = well_positions

k = bounds[1][2]
l = bounds[1][1]

# process final_config to project values outside bounds
final_config = [(max(l, min(k, x[1])), max(l, min(k, x[2]))) for x in final_config]

println(" After refiningfinal_config:", final_config)

# First plot: Starting vs Final
p1 = scatter(getindex.(starting_config, 1), 
            getindex.(starting_config, 2),
            label="Starting", 
            marker=:circle, 
            color=:blue, 
            title="Start vs Final",
            xlims=(0, k), 
            ylims=(0, k),
            xlabel="X [m]",
            ylabel="Y [m]",
            aspect_ratio=:equal,
            )

scatter!(p1, 
        getindex.(final_config, 1), 
        getindex.(final_config, 2),
        label="Final", 
        marker=:circle, 
        color=:red,
        )

plot!(p1, 
    legend=:outerbottom, 
    bottom_margin=-8mm, 
    top_margin=2mm,
    legend_columns=2)

savefig(p1, well_config_name)
println("Start vs Final configuration figure saved at $(well_config_name)")


## CHANGES MADE: Running algorithms till convergence instead of fixed number of iterations



## Some suggestions by Emmanuel
# Median energy output
# Resource Usage by the Algorithms
# Hyperopt for hyperparameter search
