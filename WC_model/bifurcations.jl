using Test
using LinearAlgebra
using Roots
using Interpolations
using PyPlot
pygui(true)

include("WC_model.jl")
include("../activation_fns.jl")

# A = [4, 5, 6]
# nodes = ([-1, 2, 3],)
# itp = interpolate(nodes, A, Gridded(Linear()))
# println(itp([2, 2.5]))

"""
Find the equilibria by computing the roots of the difference between the two nullclines
"""
function find_equilibria(p::WCModel)
    E_nullcline, I_nullcline = nullclines(p)
    values = range(0.0, stop=0.9998, length=1000)
    # interpolate to have I as a function of E for the I-nullcline
    I_itp = LinearInterpolation((I_nullcline(values),), values, extrapolation_bc=Flat())
    Δ_nullclines(E) = E_nullcline(E) - I_itp(E)
    equilibria = find_zeros(Δ_nullclines, 0.0, 0.9998)
    # plot(I_nullcline(values), values, label="\$\\frac{dI}{dt}=0\$")
    # plot(values, E_nullcline(values), label="\$\\frac{dE}{dt}=0\$")
    # plot(values, Δ_nullclines(values))

    return [[eq, I_itp(eq)] for eq in equilibria]
end

function jacobian(E, I, E_ext, I_ext, p::WCModel)
    τE = p.E_pop.τ; τI = p.I_pop.τ
    Fder_E = fder(p.E_pop.act)(p.wEE*E - p.wEI*I + E_ext)
    Fder_I = fder(p.I_pop.act)(p.wIE*E - p.wII*I + I_ext)

    J = [(p.wEE*Fder_E - 1)/τE -p.wEI*Fder_E/τE;
         p.wIE*Fder_I/τI       -(1+wII*Fder_I)/τI]

    return J
end

"""
Compute the external inputs I_ext, E_ext so that the system is at equilibrium
for the given E, I
"""
function ext_input(E, I, p::WCModel)
    E = E[(I .> 0) .& (I .< 1)] # restricted to values greater than 0 and smaller than 1
    I = I[(I .> 0) .& (I .< 1)]
    E_ext = finv(p.E_pop.act)(E) .- p.wEE*E .+ p.wEI*I
    I_ext = finv(p.I_pop.act)(I) .- p.wIE*E .+ p.wII*I
    return E, I, E_ext, I_ext
end

function saddle_node_bifurcation(p::WCModel)
    E = range(0, stop=0.9998, length=1000)
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

    E1, I1, E_ext1, I_ext1 = ext_input(E, I1, p)
    plot(E_ext1, I_ext1, color="tab:blue", label="Saddle-node bifurcation")

    E2, I2, E_ext2, I_ext2 = ext_input(E, I2, p)
    # plot(E_ext2, I_ext2)


    i = 500
    println("Predict a saddle-node bifurcation at: E=$(E1[i]); I=$(I1[i]); E_ext=$(E_ext1[i]); I_ext=$(I_ext1[i])")

    for i=1:length(E1)
        J = jacobian(E1[i], I1[i], E_ext1[i], I_ext1[i], p)
        @test isapprox(-E1[i] + act_fn(p.E_pop.act)(p.wEE*E1[i] - p.wEI*I1[i] + E_ext1[i]), 0, atol=1e-10)
        @test isapprox(-I1[i] + act_fn(p.I_pop.act)(p.wIE*E1[i] - p.wII*I1[i] + I_ext1[i]), 0, atol=1e-10)
        @test isapprox(det(J), 0, atol=1e-10)
    end

end


function hopf_bifurcation(p::WCModel)
    # x = E + 1/(1+exp(a theta)) is considered as a curve parameter for E(x), I(x)
    E = range(0, stop=0.9998, length=1000)
    # apply a change of variable to simplify the computation of the trace
    x = E .+ 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))

    # compute the roots of the trace of the jacobian
    ρ = p.wII^2 .- 4*p.wII.*(p.wEE.*(1 .- x).*x .- 2/p.E_pop.act.a)
    # Δ = 1 - 1/p.wEE*(8 + p.wII)
    # plot(E, ρ)

    I1 = (p.wII .+ sqrt.(ρ))./(2*p.wII)
    I2 = (p.wII .- sqrt.(ρ))./(2*p.wII)
    # inverse the change of variable to get the activities
    I1 = I1 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))
    I2 = I2 .- 1/(1+exp(p.E_pop.act.a*p.E_pop.act.θ))
    # plot(E, I1)
    # plot(E, I2)
    # figure()

    E1, I1, E_ext1, I_ext1 = ext_input(E, I1, p)
    plot(E_ext1, I_ext1, color="gray", label="Hopf bifurcation")

    E2, I2, E_ext2, I_ext2 = ext_input(E, I2, p)
    # plot(E_ext2, I_ext2)
    # plot(E, E_ext)
    # plot(E, I_ext)

    i = 300
    println("Predict a Hopf bifurcation at: E=$(E1[i]); I=$(I1[i]); E_ext=$(E_ext1[i]); I_ext=$(I_ext1[i])")

    for i=1:length(E1)
        J = jacobian(E1[i], I1[i], E_ext1[i], I_ext1[i], p)
        if abs(det(J)) < 1e-10
            println(det(J))
            println("E=$(E1[i]); I=$(I1[i]); E_ext=$(E_ext1[i]); I_ext=$(I_ext1[i])")
        end
        @test isapprox(-E1[i] + act_fn(p.E_pop.act)(p.wEE*E1[i] - p.wEI*I1[i] + E_ext1[i]), 0, atol=1e-10)
        @test isapprox(-I1[i] + act_fn(p.I_pop.act)(p.wIE*E1[i] - p.wII*I1[i] + I_ext1[i]), 0, atol=1e-10)
        @test isapprox(tr(J), 0, atol=1e-10)
    end

end


function bifurcation_diagram(p::WCModel)
    # for I_ext=0:1.5:14, E_ext=0:1.5:14
    for I_ext=0:0.5:16, E_ext=0:0.5:16
        wc = WCModel(p.E_pop, p.I_pop, p.wEE, p.wIE, p.wEI, p.wII, E_ext, I_ext)
        equilibria = find_equilibria(wc)
        for (i, eq) in enumerate(equilibria)
            J = jacobian(eq[1], eq[2], E_ext, I_ext, p)
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
                color = "tab:blue"
            else
                # complex conjugate eigenvalues
                color = "crimson"
            end
            plot(E_ext+i*0.15, I_ext+i*0.15, markersize=5, marker="o", fillstyle=fill_style, color=color)
        end
    end
end


# tau_E=1.; a_E=3.0; theta_E=3.0
# tau_I=1.; a_I=3.0; theta_I=3.0
tau_E=1.; a_E=3.0; theta_E=3.0
tau_I=1.; a_I=3.0; theta_I=3.0
# wEE=10; wEI=12; wIE=15; wII=8
wEE=10; wEI=14; wIE=18; wII=8
E_ext=7.7; I_ext=7.7
E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
wc = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

# eq = find_equilibria(wc)
# println(eq)

figure(figsize=(8,8), dpi=130)
saddle_node_bifurcation(wc)
hopf_bifurcation(wc)
bifurcation_diagram(wc)
legend()














# for i=1:100:length(E1)
#     if (p.wII*a .- b)[i] > 0
#         color_inside = "yellowgreen"; color_outside="orange"
#     else
#         color_inside = "orange"; color_outside = "yellowgreen"
#     end
#
#     I_inside = range(I2[i], stop=I1[i], length=50)
#     E_values = fill(E1[i], length(I_inside))
#     _, _, E_ext, I_ext = ext_input(E_values, I_inside, p)
#     scatter(E_ext, I_ext, color=color_inside, marker=".")
#
#     I_outside = range(I1[i], stop=I1[i]+0.1, length=50)
#     _, _, E_ext, I_ext = ext_input(E_values, I_outside, p)
#     scatter(E_ext, I_ext, color=color_outside, marker=".")
#
#     I_outside = range(I2[i], stop=I2[i]-0.1, length=50)
#     _, _, E_ext, I_ext = ext_input(E_values, I_outside, p)
#     scatter(E_ext, I_ext, color=color_outside, marker=".")
# end
