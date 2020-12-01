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
    f
end

include("activation_fns.jl")
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


function simulate(p::WCIntegral, dt, T)
    u(t) = p.R*p.I*(1 - exp(-t/p.τ))
    ρ(t) = p.f(u(t))

    n_iter = Int(T/dt)

    A = zeros(n_iter)
    # K = n_iter
    K = 1000
    m = zeros((n_iter, K))
    # all neurons have fired juste before time 0
    m[1,1] = 1

    for iter in 2:n_iter
        m[iter,1] = A[iter-1]*dt * exp(-ρ(dt)*dt)
        for k in 2:K
            m[iter,k] = m[iter-1,k-1]*exp(-ρ(k*dt)*dt)
        end
        A[iter] = 1/dt * (1 - sum(m[iter,:]))
    end
    return A
end
