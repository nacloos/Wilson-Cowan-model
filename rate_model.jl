using DifferentialEquations
using PyPlot
pygui(true)

include("activation_fns.jl")

struct RateModel
    pop::Pop
    w
    ext
end

function f!(dxdt, x, p::RateModel, t)
    act = p.pop.act
	A = x[1]
	# dxdt[1] = 1/p.pop.τ * (-A +act_fn(act)(p.w*A + p.ext))
    f(u) = 1e-3 .*exp.(1/0.004*(u.-act.θ))
	dxdt[1] = 1/p.pop.τ * (-A +f(p.w*A + p.ext))
end


function plot_equilibrium(p::RateModel)
    A = 0:0.001:1
    plt.plot(A, A)
    plt.plot(A, act_fn(p.pop.act)(p.w .* A .+ p.ext))
end


function simulate(p::RateModel, x0, T)
    tspan = (0.0, T)
    prob = ODEProblem(f!, x0, tspan, p)
    sol = solve(prob)
    return sol
end
