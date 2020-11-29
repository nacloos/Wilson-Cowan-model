using PyPlot
pygui(true)

include("../WC_model.jl")


function test_sigmoid()
    x = -1:0.01:1
    plot(x, act_fn(Sigmoid(1e2, 6e-2))(x))
    # plot(-10:0.1:10, act_fn(-10:0.1:10, Sigmoid(1, 0)))
    # plot(finv(Sigmoid(1, 0))(-0.4:0.01:0.4))
    # plot(-10:0.1:10, fder(Sigmoid(1, 0))(-10:0.1:10))
    # plot(-10:0.1:10, cumsum(fder(Sigmoid(1, 0))(-10:0.1:10))*0.1.-1/2)
end


function test_nullclines()
    tau_E=1.; a_E=1.2; theta_E=2.8
    tau_I=2.; a_I=3.0; theta_I=3.0
    wEE=16; wEI=12; wIE=15; wII=8
    E_ext=1; I_ext=0

    E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
    I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
    wc = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

    E_nullcline, I_nullcline = nullclines(wc)
    x = 0.01:0.01:0.95
    plot(x, E_nullcline(x))
    plot(x, I_nullcline(x))
end


function test_simulate()
    # tau_E=1.; a_E=1.2; theta_E=2.8
    # tau_I=2.; a_I=1.0; theta_I=4.0
    # wEE=9; wEI=4; wIE=13; wII=11
    # E_ext=0; I_ext=0

    # limit cycle
    tau_E=1.; a_E=1.2; theta_E=2.8
    tau_I=2.; a_I=3.0; theta_I=3.0
    wEE=16; wEI=12; wIE=15; wII=8
    E_ext=1; I_ext=0

    E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
    I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
    p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

    x0 = [1., 0.]
    x0 = [0.45, 0.25]


    T = 50
    sol = simulate(p, x0, T)

    t = 0:0.1:T
    plot(t, sol(t)[1,:], label="Excitatory")
    plot(t, sol(t)[2,:], label="Inhibitory")
    legend()
end

# test_sigmoid()
test_nullclines()
# test_simulate()
