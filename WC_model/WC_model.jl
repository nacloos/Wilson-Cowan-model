using DifferentialEquations
using PyCall

include("../activation_fns.jl")

# produces better trajectories in the phase plane than DifferentialEquations, don't know why
integrate = pyimport("scipy.integrate")


struct Pop
    τ   # time constant of the population
    act # activation function of the population
end

struct WCModel
    E_pop::Pop # excitatory population
    I_pop::Pop # inhibitory population
    wEE
    wIE
    wEI
    wII
    E_ext
    I_ext
end


function nullclines(p::WCModel)
    E_act = p.E_pop.act
    I_act = p.I_pop.act
    E_nullcline = E -> 1/p.wEI .*(p.wEE .* E .- finv(E_act)(E) .+ p.E_ext)
    I_nullcline = I -> 1/p.wIE .*(p.wII .* I .+ finv(I_act)(I) .- p.I_ext)
    return E_nullcline, I_nullcline
end

"""
Dynamical equations of the Wilson-Cowan model
"""
function f(x, t)
    E_act = p.E_pop.act
    I_act = p.I_pop.act
	E, I = x
    dxdt = zeros(2)
	dxdt[1] = 1/p.E_pop.τ * (-E + act_fn(E_act)(p.wEE*E - p.wEI*I + p.E_ext)) # dE/dt
	dxdt[2] = 1/p.I_pop.τ * (-I + act_fn(I_act)(p.wIE*E - p.wII*I + p.I_ext)) # dI/dt
    return dxdt
end


function simulate(p::WCModel, x0, T)
    dt = 1e-3
    t = 0:dt:T
    sol = integrate.odeint(f, x0, t)
    return sol'
end


function f_time_dependent_input!(dxdt, x, params, t)
    p, inputs = params
    E_act = p.E_pop.act
    I_act = p.I_pop.act
	E, I = x
	dxdt[1] = 1/p.E_pop.τ * (-E + act_fn(E_act)(p.wEE*E - p.wEI*I + inputs[1](t))) # dE/dt
	dxdt[2] = 1/p.I_pop.τ * (-I + act_fn(I_act)(p.wIE*E - p.wII*I + inputs[2](t))) # dI/dt
end

function simulate_time_dependent_input(p::WCModel, inputs, x0, T)
    tspan = (0.0, T)
    prob = ODEProblem(f_time_dependent_input!, x0, tspan, (p, inputs))
    sol = solve(prob)
    return sol
end
