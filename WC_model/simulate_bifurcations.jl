using PyPlot
pygui(true)

include("WC_model.jl")


function plot_simulation(wc::WCModel, E_ext_bounds, I_ext, T)

    function E_input(t)
        if t <= T/2
            2*(T/2-t)/T * E_ext_bounds[1] + 2*t/T * E_ext_bounds[2]
        else
            2*(T-t)/T * E_ext_bounds[2] + 2*(t-T/2)/T * E_ext_bounds[1]
        end
    end
    I_input(t) = I_ext

    sol = simulate_time_dependent_input(wc, [E_input, I_input], [0,0], T)

    dt = 0.1
    t = 0:dt:T
    E_inputs = [E_input(ti) for ti in t]
    # plot(t, E_inputs)
    #
    # plot(t, sol(t)[1,:])
    println(size(sol(t)[1,:]))
    println(size(E_inputs))
    # plot(E_inputs, sol(t)[1,:])
    plot(sol[1,:])
end

tau_E=1.; a_E=1.5; theta_E=3.0
tau_I=1.; a_I=1.5; theta_I=3.0
wEE=13; wEI=14; wIE=15; wII=8
E_ext=14.; I_ext=12


E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
wc = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

figure()
# T = 200
# E_ext_bounds = [2, 18]
# I_ext = 12
# plot_simulation(wc, E_ext_bounds, I_ext, T)
#

figure()
T = 200
E_ext_bounds = [2, 6]
I_ext = 1
plot_simulation(wc, E_ext_bounds, I_ext, T)
