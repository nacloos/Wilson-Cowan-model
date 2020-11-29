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
