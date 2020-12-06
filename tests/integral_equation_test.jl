include("../renormalization.jl")
include("../network_weights.jl")
include("../LIF_network.jl")
include("../plotting.jl")
include("../coarse_graining.jl")
include("../integral_equation.jl")

"""
Compare the population activity obtained by smoothing the spikes of a network
of independent LIF neurons with espace noise, the one computed with the integral
equation, for a constant external input.
"""

T = 0.2; dt = 1e-4; n_iter = Int(T/dt)

u_rest = 0; threshold = 6e-2; τ = 4e-3; R = 7e-2

f(u) = 1e-3 .*exp.(1/0.004*(u.-threshold))
Δ_abs = 4e-3
# Δ_abs = 1e-2

neuron = LIFNeuron(u_rest, threshold, τ, R, Δ_abs, f)

n_neurons = 5000
net = SpikingNetwork([neuron for i=1:n_neurons], zeros(n_neurons, n_neurons))


I_ext = 1.8*ones(n_iter)
ext_input = ExternalInput(I_ext, Normal(0, 0))

u0 = zeros(n_neurons)
microstate = simulate(net, ext_input, u0, dt, T)


u(t) = R*I_ext[1] .*(1 .- exp.(-t./τ))
ρ(t) = if (t > Δ_abs+dt) f(u(t-Δ_abs-dt)) else 0 end
integral_eq = IntegralEq(ρ)
A_integral = simulate(integral_eq, dt, T)

println()
println("f(RI): ", f(R*I_ext[1]))

t = dt:dt:T
A = coarse_grain(t, microstate, WindowFilter(1e-3))
# A = coarse_grain(t, microstate, WindowFilter(1.5e-4))

figure()
plot(t, A, label="LIF network")
plot(t, A_integral, label="Integral equation")

plt.hlines(mean(microstate.spike_count)/T, 0, T)
println("Mean firing rate: ", mean(microstate.spike_count)/T)
legend()


# figure()
# plot(t, microstate.u[1,:])
#
# u_test(t) = if (t > Δ_abs+dt) R*I_ext[1] .*(1 .- exp.(-(t-Δ_abs-dt)./τ)) else 0 end
# plot(t, [u_test(t_k) for t_k in t])
# plt.title("Comparison of the evolution of the membrane potential\nbetween LIF neuron and integral equation")
