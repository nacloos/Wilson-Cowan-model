
"""
Integral equation for the population activity
"""
struct IntegralEq
    R
    I
    τ
    θ
    f
end

function simulate(p::IntegralEq, Δ_abs, dt, T)
    u(t) = p.R*p.I .*(1 .- exp.(-t ./ p.τ))
    ρ(t) = if (t > Δ_abs) p.f(u(t-Δ_abs)) else 0 end

    n_iter = Int(T/dt)

    A = zeros(n_iter)
    K = 1000
    m = zeros((n_iter, K))
    m[1,1] = 1     # all neurons have fired juste before time 0

    for iter in 2:n_iter
        m[iter,1] = A[iter-1]*dt * exp(-ρ(dt)*dt)
        for k in 2:K
            m[iter,k] = m[iter-1,k-1]*exp(-ρ(k*dt)*dt)
        end
        A[iter] = 1/dt * (1 - sum(m[iter,:]))
    end
    return A
end
