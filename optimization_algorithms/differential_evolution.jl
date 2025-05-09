using Plots
using Optim
using BlackBoxOptim
using Optimization, OptimizationEvolutionary, OptimizationBBO


## Differential Evolution Optimizer
# https://docs.sciml.ai/Optimization/stable/optimization_packages/evolutionary/

function differential_evolution(original_f, px, py, bounds; 
                                max_iters=20,
                                plot_name="../jl_plots/de_optim.png", 
                                plot_title="Differential Evolution Optimization"
    )

    pxpy = [px..., py...]

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

    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    problem = Optimization.OptimizationProblem((u,_)->f(u), pxpy, lb=lb, ub=ub)
    # Default Population size is 50
    # sol = solve(problem, BBO_adaptive_de_rand_1_bin(), maxiters=max_iters; f_calls_limit=max_iters, iterations=max_iters)
    sol = solve(problem, BBO_adaptive_de_rand_1_bin(), maxiters=max_iters; f_calls_limit=max_iters, iterations=max_iters)

    println("Best solution found by DE algorithm:", sol.minimizer)

    plot(history; 
        #   title=plot_title, 
          marker=:circle, 
          ylabel="Total Heat Energy Output from Array (J)", 
          xlabel="Objective Function Evaluations", 
        #   label="Heat Energy"
         )  
    savefig(plot_name)

    println("Differential Evolution Figure saved at $plot_name")
    println("sol:", sol)
    
    px = final_config[1:lastindex(px)]
    py = final_config[lastindex(px)+1:end]
    return [(px[i], py[i]) for i in 1:lastindex(px)]
end

function test()
    rosenbrock(x) = (1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    px = [0.0]
    py = [0.0]
    differential_evolution(rosenbrock, px, py, bounds)
end

# test()
