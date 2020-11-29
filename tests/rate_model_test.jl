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

test_f()
