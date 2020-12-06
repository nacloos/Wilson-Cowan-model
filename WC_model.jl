using DifferentialEquations

include("activation_fns.jl")

# TODO time dependent input
struct WCModel
    E_pop::Pop
    I_pop::Pop
    wEE
    wIE
    wEI
    wII
    E_ext
    I_ext
end

struct WCIntegral
    R
    I
    τ
    θ
    Δ_abs
    f
end

function nullclines(p::WCModel)
    E_act = p.E_pop.act
    I_act = p.I_pop.act
    E_nullcline = E -> 1/p.wEI .*(p.wEE .* E .- finv(E_act)(E) .+ p.E_ext)
    I_nullcline = I -> 1/p.wIE .*(p.wII .* I .+ finv(I_act)(I) .- p.I_ext)
    return E_nullcline, I_nullcline
end


function f!(dxdt, x, p::WCModel, t)
    E_act = p.E_pop.act
    I_act = p.I_pop.act
	E, I = x
	dxdt[1] = 1/p.E_pop.τ * (-E + act_fn(E_act)(p.wEE*E - p.wEI*I + p.E_ext)) # dE/dt
	dxdt[2] = 1/p.I_pop.τ * (-I + act_fn(I_act)(p.wIE*E - p.wII*I + p.I_ext)) # dI/dt
end


function simulate(p::WCModel, x0, T)
    tspan = (0.0, T)
    prob = ODEProblem(f!, x0, tspan, p)
    sol = solve(prob)
    return sol
end


# TODO WC integral equation -> require an absolute refractory period that is larger than dt ?!
# TODO still it doesn't oscillate...
function simulate(p::WCIntegral, dt, T)
    u(t) = p.R*p.I .*(1 .- exp.(-t ./ p.τ))
    ρ(t) = p.f(u(t))

    n_iter = Int(T/dt)
    γ = Int(p.Δ_abs/dt) # number of iterations during the absolute refractory period

    A = zeros(n_iter)
    A[1:γ] .= 1

    for iter in n+1:n_iter
        A[iter] = (1 - exp(-dt*ρ(iter*dt))) *(1 - sum(A[iter-γ:iter-1])*p.Δ_abs)
    end
    return A
end
