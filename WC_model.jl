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
    # u(t) = p.R*p.I .*(1 .- exp.(-t ./ p.τ))
    # ρ(t) = p.f(u(t))

    n_iter = Int(T/dt)
    γ = Int(p.Δ_abs/dt) # number of iterations during the absolute refractory period


    h = zeros(n_iter)
    # initially all neurons have just fired at time -dt, they start their refractory period at t=0
    A = zeros(n_iter)

    for iter in 2:n_iter
        h[iter] = h[iter-1] + 1/p.τ*(-h[iter-1] + p.R*p.I)*dt

        if iter > γ
            # A[iter] = f(p.R*p.I) / (1 + Δ_abs*f(p.R*p.I)) * (1 - sum(A[iter-γ+1:iter-1])*p.Δ_abs)
            A[iter] = f(h[iter]) / (1 + Δ_abs*f(h[iter])) * (1 - sum(A[iter-γ+1:iter-1])*p.Δ_abs)
        else
            # initialization, assume zero input current before t=0
            # A[iter] = f(h[iter]) / (1 + Δ_abs*f(h[iter]))
            A[iter] = 0
        end
    end
    return A # population activity starting at time t=0
end
