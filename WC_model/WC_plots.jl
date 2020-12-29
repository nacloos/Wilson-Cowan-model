using Interpolations
using PyPlot
pygui(true)
using PyCall
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

include("WC_model.jl")
include("bifurcations.jl")

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
    values = range(-1/(1+exp(p.I_pop.act.a*p.I_pop.act.θ)), stop=1-1/(1+exp(p.I_pop.act.a*p.I_pop.act.θ)), length=1000)
    I_itp = LinearInterpolation((I_nullcline(values),), values, extrapolation_bc=Flat())

    x = range(values[1]+1e-8, stop=values[end]-1e-8, length=1000)
    plot(x, E_nullcline(x), label="\$\\frac{dE}{dt}=0\$", color="dimgray")
    plot(x, I_itp(x), label="\$\\frac{dI}{dt}=0\$", color="dimgray", linestyle="--")

    equilibria = find_equilibria(p, (values[1], values[end]))
    for eq in equilibria
        J = jacobian(p, eq[1], eq[2], p.E_ext, p.I_ext)
        fill_style = if (det(J) > 0 && tr(J) < 0) "full"
                     elseif (det(J) > 0 && tr(J) >= 0) "none"
                     else "left" end
        color = if all(abs.(imag(eigvals(J))) .< 1e-10) "cornflowerblue" else "darkorange" end
        plot(eq[1], eq[2], color=color, fillstyle=fill_style, marker="o", markerfacecoloralt="white")
    end
end


function plot_limit_cycle(p::WCModel)
    T = 20
    E0 = 0.5; I0 = 0.71
    # t = range(0, stop=T, length=1000)
    sol = simulate(p, [E0, I0], T)

    subplot(1, 2, 1)
    plot(sol[1,:], sol[2,:], alpha=0.75, color="cornflowerblue")
    plot_nullclines(p)
    xlabel("E")
    ylabel("I")
    text(0.75, 0.2, "\$E_{ext} = $E_ext\$ ", horizontalalignment="center", color="dimgray", size=12)
    text(0.75, 0.1, "\$I_{ext}= $I_ext\$ ", horizontalalignment="center", color="dimgray", size=12)
    legend(loc="upper left")

    subplot(1, 2, 2)
    t = range(0, stop=T, length=size(sol)[2])
    plot(t, sol[1,:], color="cornflowerblue")
    # plot(t, sol[2,:], color="navy")
    xlabel("t")
    ylabel("E")
end



function plot_hysteresis(p::WCModel)
    T = 200
    E_ext_bounds = [2, 18]
    I_ext = 12

    E_input(t) = if (t <= T/2) 2*(T/2-t)/T * E_ext_bounds[1] + 2*t/T * E_ext_bounds[2]
                 else 2*(T-t)/T * E_ext_bounds[2] + 2*(t-T/2)/T * E_ext_bounds[1]
                 end
    I_input(t) = I_ext

    sol = simulate_time_dependent_input(wc, [E_input, I_input], [0,0], T)

    dt = 0.01
    t = 0:dt:T
    E_inputs = [E_input(ti) for ti in t]

    fig = figure(figsize=(5,5), dpi=130)
    grid = matplotlib.pyplot.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    ax = subplot(get(grid, (0,0)))
    plot_nullclines(WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, 6, I_ext))
    text(0.5, 0.9, "\$E_{ext}=6\$", color="dimgray", horizontalalignment="center", transform=ax.transAxes)
    axis("off")

    ax = subplot(get(grid, (0,1)))
    plot_nullclines(WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, 8, I_ext))
    text(0.5, 0.9, "\$E_{ext}=8\$", color="dimgray", horizontalalignment="center", transform=ax.transAxes)
    axis("off")

    ax = subplot(get(grid, (1,0)))
    plot_nullclines(WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, 15, I_ext))
    text(0.5, 0.9, "\$E_{ext}=15\$", color="dimgray", horizontalalignment="center", transform=ax.transAxes)
    xlabel("E")
    ylabel("I")
    axis("off")

    subplot(get(grid, (1,1)))
    E_handle, = plot([], [], label="\$\\frac{dE}{dt}=0\$", color="dimgray")
    I_handle, = plot([], [], label="\$\\frac{dI}{dt}=0\$", color="dimgray", linestyle="--")
    stable_handle, = plot([], [], color="cornflowerblue", fillstyle="full", marker="o", markerfacecoloralt="white", label="Stable equilibrium", linestyle="None")
    saddle_handle, = plot([], [], color="cornflowerblue", fillstyle="left", marker="o", markerfacecoloralt="white", label="Saddle-point", linestyle="None")
    legend(handles=[E_handle, I_handle, stable_handle, saddle_handle], loc="center")
    axis("off")

    figure(figsize=(6,5), dpi=130)
    plot(E_inputs, sol(t)[1,:], color="cornflowerblue")
    xlabel("\$E_{ext}\$")
    ylabel("\$E\$")
    vlines(13.94, minimum(sol[1,:]), maximum(sol[1,:]), color="dimgray", linestyle="dotted")
    vlines(6.6, minimum(sol[1,:]), maximum(sol[1,:]), color="dimgray", linestyle="dotted", label="Saddle-node bifurcation")
    legend()
end



tau_E=1.; a_E=1.5; theta_E=3.0
tau_I=1.; a_I=1.5; theta_I=3.0
wEE=13; wEI=14; wIE=15; wII=8
E_ext=6; I_ext=1

E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
global p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)


# figure(figsize=(10,4), dpi=130)
# plot_limit_cycle(p)
plot_hysteresis(p)





# # I_val = -10:0.01:10
# # plot(I_val, act_fn(p.I_pop.act)(I_val))
#
# figure()
# plot_phase_space(p, 100.0)
# plot_nullclines(p)
# xlabel("E")
# ylabel("I")
# legend()
