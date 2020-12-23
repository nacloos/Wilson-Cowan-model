using Test
using LinearAlgebra
using PyPlot
pygui(true)

include("WC_model.jl")
include("../activation_fns.jl")

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
    E = range(0, stop=0.999, length=1000)
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
    plot(E_ext1, I_ext1)

    E2, I2, E_ext2, I_ext2 = ext_input(E, I2, p)
    plot(E_ext2, I_ext2)


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
    E = range(0, stop=0.999, length=1000)
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
    plot(E_ext1, I_ext1)

    E2, I2, E_ext2, I_ext2 = ext_input(E, I2, p)
    plot(E_ext2, I_ext2)
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


function plot_ext_plane(p::WCModel)
    for I_ext=0:10, E_ext=0:10
        det = det(jacobian)
        scatter()
    end
end


tau_E=1.; a_E=3.0; theta_E=3.0
tau_I=1.; a_I=3.0; theta_I=3.0
wEE=10; wEI=12; wIE=15; wII=8
E_ext=1; I_ext=0
E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
wc = WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)

figure()
saddle_node_bifurcation(wc)
# hopf_bifurcation(wc)














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
