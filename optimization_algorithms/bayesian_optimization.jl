using Plots
using BayesianOptimization, GaussianProcesses, Distributions
# using LatinHypercubeSampling

# https://soldasim.github.io/BOSS.jl/dev/example/

## References
# https://arxiv.org/pdf/1807.02811
# https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f

function bayesian_optimization(original_f, px, py, bounds; 
                        max_iters=20,
                        plot_name="../jl_plots/de_optim.png", 
                        plot_title="Differential Evolution Optimization",
                        acq="ucb"
                        )

    if length(px) != length(py)
        throw(ArgumentError("px and py must have the same length"))
    end

    pxpy = [px..., py...]

    num_dims = 2 * length(px)

    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    if acq == "ucb"
        acq = UpperConfidenceBound()
    elseif acq == "ei"
        acq = ExpectedImprovement()
    elseif acq == "poi"
        acq = ProbabilityOfImprovement()
    else
        throw(ArgumentError("acq must be ucb, ei, or poi"))
    end
    
    bound_midpoint = (bounds[1][1] + bounds[1][2])/2.0

    model = ElasticGPE(num_dims,
                        mean = MeanConst(bound_midpoint), #0.0
                        kernel = SEArd(zeros(num_dims), 20.0), # 5.0
                        logNoise = -1.0, 
                        ) 

    # set_priors!(model.mean, [Uniform(bounds[1][1], bounds[1][2])]) 
    set_priors!(model.mean, [Normal(1, 2)])

    modeloptimizer = MAPGPOptimizer(every = 1, # 10
                                    noisebounds = [-4, 3],
                                    # kernbounds = [[-1, -1, 0], [4, 4, 10]], # NEW 
                                    # kernbounds = [lb, ub],
                                    maxeval = 10
                                    ) 

    # modeloptimizer = NoModelOptimizer()

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
        return val  
    end

    # warm start
    start_val = [f(pxpy)]
    append!(model, hcat(pxpy), start_val)

    # initializer_iterations = max(max_iters, length(pxpy)) # 5
    initializer_iterations = 4
    initializer = ScaledLHSIterator(lb, ub, initializer_iterations)
    opt = BOpt(f, 
                model, 
                acq, 
                modeloptimizer, 
                lb, 
                ub, 
                repetitions = 1,
                maxiterations = max_iters, 
                sense = Max, 
                acquisitionoptions = (method = :LD_LBFGS, 
                                      restarts = 5, 
                                      maxtime = 0.1, 
                                      maxeval = 100), 
                initializer_iterations=initializer_iterations,
                initializer=initializer,
                )

    result = boptimize!(opt)

    print("Result:", result)

    plot(history, 
        marker=:circle, 
        ylabel="Total Heat Output (J)", 
        xlabel="Objective Function Evaluations")  
    savefig(plot_name)
    px = final_config[1:lastindex(px)]
    py = final_config[lastindex(px)+1:end]
    print("Best Value Obtained:", obj_val)
    return [(px[i], py[i]) for i in 1:lastindex(px)]
end

function test()
    rosenbrock(x) = -((1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    px = [0.0]
    py = [0.0]
    bayesian_optimization(rosenbrock, px, py, bounds)
end

# test()

