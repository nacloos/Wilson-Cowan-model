using DifferentialEquations
using PyPlot
pygui(true)

include("activation_fns.jl")

struct RateModel
    # pop::Pop
    τ
    w
    ext
    f
end

struct RateIntegralModel
    τ
    R
    I
    f
end

function f(x, p::RateModel, dt, iter)
    # act = p.pop.act
	A = x
    F(u) = 1 - exp(-dt*p.f(u))

	dxdt = 1/p.τ * (-A +F(p.w*A + p.ext))
    return dxdt
end


function simulate(p::RateModel, x0, dt, T)
    n_iter = Int(T/dt)
    A = zeros(n_iter)
    A[1] = x0
    dAdt = 0
    for iter in 1:n_iter-1
        A[iter+1] = A[iter] + f(A[iter], p, dt, iter)*dt
    end
    return A
end

# function simulate(p::RateModel, x0, T)
#     tspan = (0.0, T)
#     prob = ODEProblem(f!, x0, tspan, p)
#     sol = solve(prob)
#     return sol
# end

function simulate(p::RateIntegralModel, Δ_abs, dt, T)
    u(t) = p.R*p.I .*(1 .- exp.(-t ./ p.τ))
    ρ(t) = p.f(u(t))

    n_iter = Int(T/dt)
    n = Int(Δ_abs/dt) # number of iterations during the absolute refractory period

    A = zeros(n_iter)
    # A[1:n] .= 1

    for iter in n+1:n_iter
        # A[iter] = ρ(iter*dt)*(1 - sum(A[iter-n:iter-1])*Δ_abs)
        A[iter] = (1 - exp(-dt*ρ(iter*dt)))
    end
    return A
end

function plot_equilibrium(p::RateModel)
    A = 0:0.001:1
    plt.plot(A, A)
    plt.plot(A, act_fn(p.pop.act)(p.w .* A .+ p.ext))
end
