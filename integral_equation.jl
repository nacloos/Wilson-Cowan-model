
"""
Integral equation for the population activity
"""
struct IntegralEq
    ρ
end

function simulate(p::IntegralEq, dt, T)
    n_iter = Int(T/dt)

    A = zeros(n_iter)
    K = 1000
    m = zeros((n_iter, K))
    m[1,1] = 1     # fraction of neurons at time 0 that fired at time -dt

    for iter in 2:n_iter
        m[iter,1] = A[iter-1]*dt * exp(-p.ρ(dt)*dt)
        for k in 2:K
            m[iter,k] = m[iter-1,k-1] * exp(-p.ρ(k*dt)*dt)
        end
        A[iter] = 1/dt * (1 - sum(m[iter,:]))
    end
    return A
end
