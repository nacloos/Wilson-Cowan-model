using Interpolations
using PyPlot
pygui(true)

include("WC_model.jl")


function plot_phase_space(p::WCModel, T)
    E0 = range(0, stop=1, length=10)
    I0 = range(0, stop=1, length=10)

    # E0 = [0.37]
    # I0 = [0.703824656690395]

    for i=1:length(E0), j=1:length(I0)
        sol = simulate(p, [E0[i], I0[j]], T)
        t = range(0, stop=T, length=500)
        plot(sol(t)[1,:], sol(t)[2,:], alpha=0.3, color="cornflowerblue") # TODO: intesecting trajectories!
        # plot(sol[1,:], sol[2,:], alpha=0.3, color="cornflowerblue")
    end
end


function plot_nullclines(p::WCModel)
    E_nullcline, I_nullcline = nullclines(p)
    values = range(0.0, stop=0.99985, length=500)
    I_itp = LinearInterpolation((I_nullcline(values),), values, extrapolation_bc=Flat())
    nullclines_diff = E_nullcline(values) - I_itp(values)
    plot(I_nullcline(values), values, label="\$\\frac{dI}{dt}=0\$")
    plot(values, E_nullcline(values), label="\$\\frac{dE}{dt}=0\$")
    plot(values, nullclines_diff)
    plot(values, I_itp(values))
end


# tau_E=1.; a_E=1.2; theta_E=2.8
# tau_I=2.; a_I=1.0; theta_I=4.0
# wEE=9; wEI=4; wIE=13; wII=11
# E_ext=0; I_ext=0

# tau_E=1.; a_E=1.2; theta_E=2.8
# tau_I=2.; a_I=3.0; theta_I=3.0
# wEE=16; wEI=12; wIE=15; wII=8
# E_ext=1; I_ext=0

tau_E=1.; a_E=3.0; theta_E=3.0
tau_I=1.; a_I=3.0; theta_I=3.0
# wEE=10; wEI=12; wIE=15; wII=8
wEE=10; wEI=14; wIE=18; wII=8
E_ext=8; I_ext=0.2

E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
wc = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

figure()
plot_phase_space(wc, 10.0)
plot_nullclines(wc)
xlabel("E")
ylabel("I")
legend()
