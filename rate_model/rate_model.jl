include("../activation_fns.jl")


struct RateModel
    τ   # time constant
    w   # weight
    ext # external input
    f   # activation function
end

"""
Dynamical equation of the rate model
"""
function f(x, p::RateModel, iter)
	A = x
	dxdt = 1/p.τ * (-A +p.f(p.w*A + p.ext[iter]))
    return dxdt
end


function simulate(p::RateModel, x0, dt, T)
    n_iter = Int(T/dt)
    A = zeros(n_iter)
    A[1] = x0
    dAdt = 0
    for iter in 1:n_iter-1
        A[iter+1] = A[iter] + f(A[iter], p, iter)*dt
    end
    return A
end
