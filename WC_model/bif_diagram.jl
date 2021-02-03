using PyPlot
pygui(true)

include("bifurcations.jl")

# directory where you want to save the figures
savefig_path = "figures/"


"""
Plot the saddle-node and Hopf bifurcation curves in the (E_ext, I_ext) plane
"""
function plot_bifurcation_curves(p::WCModel, bounds)
    max_I_ext = 14

    sn_E, sn_I, sn_E_ext, sn_I_ext = saddle_node_curve(p, bounds)

    # manually complete the saddle-node bifuration where it is almost vertical
    sn_E_ext = [sn_E_ext; sn_E_ext[end]]
    sn_I_ext = [sn_I_ext; max_I_ext]

    plot(sn_E_ext[sn_I_ext.<=max_I_ext], sn_I_ext[sn_I_ext.<=max_I_ext], linestyle="dotted", color="dimgray", label="Saddle-node bifurcation")

    hopf_E, hopf_I, hopf_E_ext, hopf_I_ext = hopf_curve(p, bounds)
    plot(hopf_E_ext[hopf_I_ext.<= max_I_ext], hopf_I_ext[hopf_I_ext.<=max_I_ext], color="dimgray", label="Hopf bifurcation")
end

"""
Plot the number and types of equilibria for a grid of values in the (E_ext, I_ext) plane
"""
function plot_equilibria(p::WCModel, bounds)
    Δ = 0.7
    for I_ext=0:Δ:14, E_ext=2:Δ:16
        wc = WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, E_ext, I_ext)
        equilibria = find_equilibria(wc, bounds)

        # set the marker positions depending on the number of equilibria
        if length(equilibria) == 2
            l = 0.05
            marker_pos = [(-sqrt(2)*l, -sqrt(2)*l), (+sqrt(2)*l, +sqrt(2)*l)]
        elseif length(equilibria) == 3
            l = 0.08
            marker_pos = [(-l, -1.3*l/(2*sqrt(5))), (l, -1.3*l/(2*sqrt(5))), (0, sqrt(5)*l-2.5*l/(2*sqrt(5)))]
        else
            marker_pos = [(0, 0)]
        end

        # set the marker shape and color as a function of the type of the equilibrium
        for (i, eq) in enumerate(equilibria)
            J = jacobian(p, eq[1], eq[2], E_ext, I_ext)
            if det(J) > 0
                if tr(J) < 0
                    # stable node
                    fill_style = "full"
                else
                    # instable node
                    fill_style = "none"
                end
            else
                # saddle point
                fill_style = "left"
            end
            if all(abs.(imag(eigvals(J))) .< 1e-10)
                # real eigenvalues
                color = "steelblue"
            else
                # complex conjugate eigenvalues
                # color = "crimson"
                color = "darkorange"
            end

            plot(E_ext+marker_pos[i][1], I_ext+marker_pos[i][2], markersize=4, marker="o",
                 markeredgewidth=0.5, fillstyle=fill_style, color=color)
        end
    end
    xlabel("\$E_{ext}\$")
    ylabel("\$I_{ext}\$")
    axis("equal")
end


# parameters of the two populations
tau_E=1.; a_E=1.5; theta_E=3.0
tau_I=1.; a_I=1.5; theta_I=3.0
wEE=13; wEI=14; wIE=15; wII=8

E_ext=3; I_ext=12
E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

# compute the min and max values of the activation function
min_val = act_fn(p.I_pop.act)(-1e8)
max_val = act_fn(p.I_pop.act)(1e8)
I_bounds = (min_val, max_val)

# make the stability and bifurcation diagram
figure(figsize=(9,7), dpi=130)
plot_bifurcation_curves(p, I_bounds)
plot_equilibria(p, I_bounds)

# make the legend of the plot
sn_handle, = plot([], [], linestyle="dotted", color="dimgray", label="Saddle-node           ")
hopf_handle, = plot([], [], linestyle="-", color="dimgray", label="Andronov-Hopf")
curves_legend = legend(handles=[sn_handle, hopf_handle], title="Bifurcations",
                       bbox_to_anchor=(1.05, 1), loc="upper left")
ax = gca().add_artist(curves_legend)

stable_handle, = plot([], [], linestyle="None", marker="o", markeredgewidth=0.7, fillstyle="full", color="dimgray", label="Stable")
saddle_handle, = plot([], [], linestyle="None", marker="o", markeredgewidth=0.7, fillstyle="left", color="dimgray", label="Saddle point           ")
unstable_handle, = plot([], [], linestyle="None", marker="o", markeredgewidth=0.7, fillstyle="none", color="dimgray", label="Unstable")
eq_legend = legend(handles=[stable_handle, saddle_handle, unstable_handle], title="Equilibria",
       bbox_to_anchor=(1.05, 0.86), loc="upper left")
ax = gca().add_artist(eq_legend)

real_patch = matplotlib.patches.Patch(color="steelblue", label="Real")
complex_patch = matplotlib.patches.Patch(color="darkorange", label="Complex conjugate")
legend(handles=[real_patch, complex_patch], title="Eigenvalues",
       bbox_to_anchor=(1.05, 0.68), loc="upper left")

tight_layout()
savefig(savefig_path*"bifurcation_diagram.png", transparent=true)
savefig(savefig_path*"bifurcation_diagram.pdf")
