using Plots
using Optim
using BlackBoxOptim
using Optimization, OptimizationBBO

# Julia impl. of NelderMead does not support box constraints
# # https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/nelder_mead

function natural_evolution_strategies(original_f, px, py, bounds; 
                    max_iters=20,
                    plot_name="../jl_plots/nes_optim.png", 
                    plot_title="Natural Evolution Strategy Optimization",
                    nes_type="xnes"
    )

    pxpy = [px..., py...]
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    history = []
    final_config = pxpy
    obj_val = 0.0

    function f(x)
        val = original_f(x)
        push!(history, max(val, obj_val))
        if val >= obj_val
            obj_val = val
            final_config = x
        end
        return -val  
    end

    algorithm = BBO_xnes()
    if nes_type == "xnes"
        algorithm = BBO_xnes()
    elseif nes_type == "snes"
        algorithm = BBO_separable_nes()
    elseif nes_type == "dxnes"
        algorithm = BBO_dxnes()
    else
        error("Invalid NES type. Valid types are xnes, snes, dxnes")
    end

    problem = Optimization.OptimizationProblem((u,_)->f(u), pxpy, lb=lb, ub=ub)
    results = solve(problem, algorithm, maxiters=max_iters)

    println("Best solution found by Natural Evolution Strategies algorithm:", results)

   p =  plot(history; 
        #   title=plot_title, 
          marker=:circle, 
          ylabel="Total Heat Energy Output from Array (J)", 
          xlabel="Objective Function Evaluations", 
        #   label="Heat Energy"
         )  
    savefig(plot_name)
    println("Natural Evolution Strategies Figure saved at $plot_name")
    println(results)
    
    px = final_config[1:lastindex(px)]
    py = final_config[lastindex(px)+1:end]
    return [(px[i], py[i]) for i in 1:lastindex(px)]
end

function test()
    rosenbrock(x) = (1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    px = [0.0]
    py = [0.0]
    natural_evolution_strategies(rosenbrock, px, py, bounds)
end

# test()
