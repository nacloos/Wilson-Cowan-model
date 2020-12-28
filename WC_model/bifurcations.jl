using Test
using LinearAlgebra
using Roots
using Interpolations
using PyPlot
pygui(true)

# matplotlib = pyimport("scipy.integrate")

include("WC_model.jl")
include("../activation_fns.jl")


"""
Find the equilibria by computing the roots of the difference between the two nullclines.
"""
function find_equilibria(p::WCModel, bounds)
    E_nullcline, I_nullcline = nullclines(p)
    values = range(bounds[1], stop=bounds[2], length=1000)
    # interpolate to have I as a function of E for the I-nullcline
    I_itp = LinearInterpolation((I_nullcline(values),), values, extrapolation_bc=Flat())
    Δ_nullclines(E) = E_nullcline(E) - I_itp(E)
    equilibria = find_zeros(Δ_nullclines, min_val, max_val)
    # plot(I_nullcline(values), values, label="\$\\frac{dI}{dt}=0\$")
    # plot(values, E_nullcline(values), label="\$\\frac{dE}{dt}=0\$")
    # plot(values, Δ_nullclines(values))
    return [[eq, I_itp(eq)] for eq in equilibria]
end

"""
Compute the Jacobian of the Wilson-Cowan model.
"""
function jacobian(p::WCModel, E::Real, I::Real, E_ext::Real, I_ext::Real)
    τE = p.E_pop.τ; τI = p.I_pop.τ
    Fder_E = fder(p.E_pop.act)(p.wEE*E .- p.wEI*I .+ E_ext)
    Fder_I = fder(p.I_pop.act)(p.wIE*E .- p.wII*I .+ I_ext)

    J = [(p.wEE*Fder_E - 1)/τE -p.wEI*Fder_E/τE;
         p.wIE*Fder_I/τI       -(1+wII*Fder_I)/τI]
    return J
end

function jacobian(p::WCModel, E::Array, I::Array, E_ext::Array, I_ext::Array)
    τE = p.E_pop.τ; τI = p.I_pop.τ
    Fder_E = fder(p.E_pop.act)(p.wEE*E .- p.wEI*I .+ E_ext)
    Fder_I = fder(p.I_pop.act)(p.wIE*E .- p.wII*I .+ I_ext)

    Fder_E = reshape(Fder_E, 1, 1, :); Fder_I = reshape(Fder_I, 1, 1, :)
    J = [(p.wEE*Fder_E .- 1)./τE -p.wEI*Fder_E./τE;
         p.wIE*Fder_I./τI       -(1 .+wII*Fder_I)./τI]
    return J
end


"""
Compute the external inputs I_ext, E_ext so that the system is at equilibrium
for the given E, I.
"""
function ext_input(p::WCModel, E, I, bounds)
    E = E[(I .>= bounds[1]) .& (I .<= bounds[2])] # restricted to values greater than 0 and smaller than 1
    I = I[(I .>= bounds[1]) .& (I .<= bounds[2])]
    E_ext = finv(p.E_pop.act)(E) .- p.wEE*E .+ p.wEI*I
    I_ext = finv(p.I_pop.act)(I) .- p.wIE*E .+ p.wII*I
    return E, I, E_ext, I_ext
end

"""
Compute the curve of equilibria where the determinant of the Jacobian vanishes,
in the (E_ext, I_ext) plane.
"""
function saddle_node_curve(p::WCModel, bounds)
    # E = range(min_val, stop=max_val, length=1000)
    E = range(bounds[1], stop=bounds[2]-0.09*bounds[2], length=1000)
    E = [E; range(bounds[2]-0.09*bounds[2], stop=bounds[2], length=10000)]
    # apply a change of variable to simplify the computation of the determinant
    x = E .+ 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))

    a = p.wEE*(1 .- x).*x .- 1/p.E_pop.act.a
    b = p.wIE*p.wEI*(1 .- x).*x
    ρ = 1 .+ 4 .*a./((p.wII*a .- b).*p.E_pop.act.a)

    I1 = 1/2 .* (1 .+ sqrt.(ρ))
    I2 = 1/2 .* (1 .- sqrt.(ρ))
    # inverse the change of variable to get the activities
    I1 = I1 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))
    I2 = I2 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))

    E1, I1, E_ext1, I_ext1 = ext_input(p, E, I1, bounds)
    # negative values, can be discarded
    # E2, I2, E_ext2, I_ext2 = ext_input(p, E, I2, bounds)

    # a few sanity checks
    for i=1:length(E1)
        J = jacobian(p, E1[i], I1[i], E_ext1[i], I_ext1[i])
        @test isapprox(-E1[i] + act_fn(p.E_pop.act)(p.wEE*E1[i] - p.wEI*I1[i] + E_ext1[i]), 0, atol=1e-10)
        @test isapprox(-I1[i] + act_fn(p.I_pop.act)(p.wIE*E1[i] - p.wII*I1[i] + I_ext1[i]), 0, atol=1e-10)
        @test isapprox(det(J), 0, atol=1e-10)
    end
    return E1, I1, E_ext1, I_ext1
end


function hopf_curve(p::WCModel, bounds)
    E = range(bounds[1], stop=bounds[2], length=1000)
    # apply a change of variable to simplify the computation of the trace
    x = E .+ 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))

    # compute the roots of the trace of the jacobian
    ρ = p.wII^2 .- 4*p.wII.*(p.wEE.*(1 .- x).*x .- 2/p.E_pop.act.a)
    # Δ = 1 - 1/p.wEE*(8 + p.wII)
    # plot(E, ρ)

    E = E[ρ .>= 0]
    ρ = ρ[ρ .>= 0]

    I1 = (p.wII .+ sqrt.(ρ))./(2*p.wII)
    I2 = (p.wII .- sqrt.(ρ))./(2*p.wII)
    # inverse the change of variable to get the activities
    I1 = I1 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))
    I2 = I2 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))

    E1, I1, E_ext1, I_ext1 = ext_input(p, E, I1, bounds)
    # negative values, can be discarded
    # E2, I2, E_ext2, I_ext2 = ext_input(p, E, I2, bounds)

    # remove points where the determinant is negative to keep only hopf bifurcations
    J = jacobian(p, E1, I1, E_ext1, I_ext1)
    D = [det(J[:,:,i]) for i in 1:length(E1)]
    hopf_E = E1[D.>=0]; hopf_I = I1[D.>=0]; hopf_E_ext = E_ext1[D.>=0]; hopf_I_ext = I_ext1[D.>=0]

    # a few sanity checks
    for i=1:length(hopf_E)
        J = jacobian(p, hopf_E[i], hopf_I[i], hopf_E_ext[i], hopf_I_ext[i])
        if det(J) < 0
            println(det(J))
            println("E=$(hopf_E[i]); I=$(hopf_I[i]); E_ext=$(hopf_E_ext[i]); I_ext=$(hopf_I_ext[i])")
        end
        @test isapprox(-hopf_E[i] + act_fn(p.E_pop.act)(p.wEE*hopf_E[i] - p.wEI*hopf_I[i] + hopf_E_ext[i]), 0, atol=1e-10)
        @test isapprox(-hopf_I[i] + act_fn(p.I_pop.act)(p.wIE*hopf_E[i] - p.wII*hopf_I[i] + hopf_I_ext[i]), 0, atol=1e-10)
        @test isapprox(tr(J), 0, atol=1e-10)
    end
    return hopf_E, hopf_I, hopf_E_ext, hopf_I_ext
end


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

function plot_bifurcation_diagram(p::WCModel, bounds)
    Δ = 0.7
    # for I_ext=6:0.5:14, E_ext=13:0.1:16
    # for I_ext=0:Δ:16, E_ext=0:Δ:16
    for I_ext=0:Δ:14, E_ext=2:Δ:16
        wc = WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, E_ext, I_ext)
        equilibria = find_equilibria(wc, bounds)

        if length(equilibria) == 2
            l = 0.05
            marker_pos = [(-sqrt(2)*l, -sqrt(2)*l), (+sqrt(2)*l, +sqrt(2)*l)]
        elseif length(equilibria) == 3
            l = 0.08
            marker_pos = [(-l, -1.3*l/(2*sqrt(5))), (l, -1.3*l/(2*sqrt(5))), (0, sqrt(5)*l-2.5*l/(2*sqrt(5)))]
        else
            marker_pos = [(0, 0)]
        end

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

            # plot(E_ext+i*0.15, I_ext+i*0.15, markersize=4, marker="o", fillstyle=fill_style, color=color)
            plot(E_ext+marker_pos[i][1], I_ext+marker_pos[i][2], markersize=4, marker="o",
                 markeredgewidth=0.5, fillstyle=fill_style, color=color)
        end
    end
    # make the legend
    xlabel("\$E_{ext}\$")
    ylabel("\$I_{ext}\$")
    axis("equal")
end


tau_E=1.; a_E=1.5; theta_E=3.0
tau_I=1.; a_I=1.5; theta_I=3.0
# wEE=10; wEI=12; wIE=15; wII=8
wEE=13; wEI=14; wIE=15; wII=8
# wEE=10; wEI=10; wIE=10; wII=4

# tau_E=2.; a_E=1.0; theta_E=3.0
# tau_I=2.; a_I=1.0; theta_I=3.0
# wEE=16; wEI=12; wIE=15; wII=8

E_ext=7.7; I_ext=7.7
E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
p = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

# eq = find_equilibria(p)
# println(eq)
min_val = act_fn(p.I_pop.act)(-1e8)
max_val = act_fn(p.I_pop.act)(1e8)
I_bounds = (min_val, max_val)

figure(figsize=(9,7), dpi=130)
plot_bifurcation_curves(p, I_bounds)
plot_bifurcation_diagram(p, I_bounds)

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
