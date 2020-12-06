using PyPlot
pygui(true)

include("../rate_model.jl")


function test_f()
    tau=1e-2; a=10; theta=6e-2
    w = 1
    ext = 1
    model = RateModel(Pop(tau, Sigmoid(a, theta)), w, ext)

    # x = 0:0.1:1
    # dxdt = zeros(length(x))
    # for i in 1:length(x)
    #     # dxdt_i = [0]
    #     dxdt[i] = f!(0, x[i], model, 0)
    #     # println(dxdt_i)
    # end
    # plot(x, dxdt)
    simulate(model, [0], 5)
end

function test_simulate()
    T = 0.2; dt = 1e-4; n_iter = Int(T/dt)
    θ = 6e-2; τ = 4e-3; a=10
    w = 1
    ext = 1
    model = RateModel(Pop(τ, Sigmoid(a, θ)), w, ext)
    sol = simulate(model, [0], T)
    t = dt:dt:T
    plot(t, sol(t))
    legend()
end

test_simulate()
