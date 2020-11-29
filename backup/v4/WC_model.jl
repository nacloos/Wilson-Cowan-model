using DifferentialEquations

struct Sigmoid
    a::Real
    θ::Real
end

struct Pop
    τ
    act
end

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

# TODO which one is better: return a fn or act_fn(x, sig) ?
act_fn(sig::Sigmoid) = x -> 1 ./ (1 .+ exp.(-sig.a .* (x .- sig.θ))) .- 1/(1 + exp(sig.a*sig.θ))
finv(sig::Sigmoid) = x -> sig.θ .- 1/sig.a .* log.(1 ./(x .+ 1/(1+exp(sig.a*sig.θ))) .- 1)
fder(sig::Sigmoid) = x -> sig.a .*exp.(-sig.a .*(x .- sig.θ))./(1 .+ exp.(-sig.a .*(x .-sig.θ))).^2


function f!(dxdt, x, p::WCModel, t)
    E_act = p.E_pop.act
    I_act = p.I_pop.act
	E, I = x
	dxdt[1] = 1/p.E_pop.τ * (-E + act_fn(E_act)(p.wEE*E - p.wEI*I + p.E_ext)) # dE/dt
	dxdt[2] = 1/p.I_pop.τ * (-I + act_fn(I_act)(p.wIE*E - p.wII*I + p.I_ext)) # dI/dt
end

function nullclines(E_range, I_range, p)
	# param E_range: a range of values of E to compute the corresponding I values on the E-nullcline
	I = 1/p.wEI*(p.wEE*E_range - F_inv(E_range, p.a_E, p.theta_E) + p.I_ext_E) # dE/dt = 0
	E = 1/p.wIE*(p.wII*I_range + F_inv(I_range, p.a_I, p.theta_I) - p.I_ext_I) # dI/dt = 0

	return I, E
end


function simulate(p::WCModel, x0, T)
    tspan = (0.0, T)
    prob = ODEProblem(f!, x0, tspan, p)
    sol = solve(prob)
    return sol
end
