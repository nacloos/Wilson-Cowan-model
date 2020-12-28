using Interpolations
using PyPlot
pygui(true)

include("WC_model.jl")


function plot_phase_space(p::WCModel, T)
    E0 = range(0, stop=1, length=10)
    I0 = range(0, stop=1, length=10)
    # E0 = range(0.05, stop=0.2, length=20)
    # I0 = range(0.9, stop=1, length=20)
    #
    # E0 = [0.1]
    # I0 = [0.96]

    for i=1:length(E0), j=1:length(I0)
        sol = simulate(p, [E0[i], I0[j]], T)
        # t = range(0, stop=T, length=500)
        # plot(sol(t)[1,:], sol(t)[2,:], alpha=0.3, color="cornflowerblue") # TODO: intesecting trajectories!
        plot(sol[1,:], sol[2,:], alpha=0.3, color="cornflowerblue")
    end
end


function plot_nullclines(p::WCModel)
    E_nullcline, I_nullcline = nullclines(p)
    # values = range(0.0, stop=0.99985, length=500)
    values = range(-1/(1+exp(p.I_pop.act.a*p.I_pop.act.θ)), stop=1-1/(1+exp(p.I_pop.act.a*p.I_pop.act.θ)), length=500)
    I_itp = LinearInterpolation((I_nullcline(values),), values, extrapolation_bc=Flat())
    nullclines_diff = E_nullcline(values) - I_itp(values)
    plot(values, E_nullcline(values), label="\$\\frac{dE}{dt}=0\$", color="dimgray")
    plot(I_nullcline(values), values, label="\$\\frac{dI}{dt}=0\$", color="dimgray", linestyle="--")
    # plot(values, nullclines_diff)
    # plot(values, I_itp(values))
end


function plot_limit_cycle()
    tau_E=1.; a_E=1.5; theta_E=3.0
    tau_I=1.; a_I=1.5; theta_I=3.0
    wEE=13; wEI=14; wIE=15; wII=8
    E_ext=6; I_ext=1

    E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
    I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
    global p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

    T = 20
    E0 = 0.1; I0 = 0.01
    E0 = 0.5; I0 = 0.71
    # t = range(0, stop=T, length=1000)
    sol = simulate(p, [E0, I0], T)

    subplot(1, 2, 1)
    plot(sol[1,:], sol[2,:], alpha=0.75, color="cornflowerblue")
    plot_nullclines(p)
    xlabel("E")
    ylabel("I")
    text(0.8, 0.2, "\$E_{ext} = $E_ext\$ ", horizontalalignment="center", color="dimgray", size=12)
    text(0.8, 0.1, "\$I_{ext}= $I_ext\$ ", horizontalalignment="center", color="dimgray", size=12)
    legend()

    subplot(1, 2, 2)
    t = range(0, stop=T, length=size(sol)[2])
    plot(t, sol[1,:], color="cornflowerblue")
    # plot(t, sol[2,:], color="navy")
    xlabel("t")
    ylabel("E")
end


figure(figsize=(10,4), dpi=130)
plot_limit_cycle()


# tau_E=2.; a_E=1.5; theta_E=3.0
# tau_I=2.; a_I=1.5; theta_I=3.0
# wEE=13; wEI=14; wIE=15; wII=8
# E_ext=6.; I_ext=1
#
#
# E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
# I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
# global p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)
#
#
# # I_val = -10:0.01:10
# # plot(I_val, act_fn(p.I_pop.act)(I_val))
#
# figure()
# plot_phase_space(p, 100.0)
# plot_nullclines(p)
# xlabel("E")
# ylabel("I")
# legend()
