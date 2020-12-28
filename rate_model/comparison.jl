using PyPlot
pygui(true)

include("../network_weights.jl")
include("../LIF_network.jl")
include("../activation_fns.jl")
include("../coarse_graining.jl")


# T = 0.2
T = 0.2
dt = 1e-4; n_iter = Int(T/dt)
# T = 0.01

u_rest = 0; θ = 6e-2; τ = 4e-3; R = 7e-2
Δ_abs = 0
τ = 1e-4
# f(u) = 1e-3 .*exp.(1/0.004*(u.-θ))

# β = 100
# τ₀ = 1e-3
# f(u) = 1/τ₀ .*exp.(β*(u.-θ))
a = 100
f(u) = act_fn(Sigmoid(a, θ))(R*u) # TODO; change firing prob in LIF_network
# x = 0:0.001:1
# plot(x, f(x))

Δ_abs = 1e-3


neuron = LIFNeuron(u_rest, θ, τ, R, Δ_abs, f)

n_neurons = 1000
J0 = 1000.
W = weights(Homogeneous(J0/n_neurons), FullyConnected(n_neurons, n_neurons))
net = SpikingNetwork([neuron for i=1:n_neurons], W)

ext = 10
I_ext = ext*ones(n_iter)
I_ext[1:n_iter÷2] = 2*ext/T .* collect(dt:dt:T/2)
I_ext[(n_iter÷2)+1:end] = 2*ext.*(T.-collect(T/2+dt:dt:T))/T
# plot(dt:dt:T, I_ext)
ext_input = ExternalInput(I_ext, Normal(0, 0))

u0 = zeros(n_neurons)
state = simulate(net, ext_input, u0, dt, T)


t = dt:dt:T
A = coarse_grain(t, state, WindowFilter(1e-2))

figure(figsize=(12,6))
subplot(2,2,1)
nz_times = state.spike_trains .!= 0
for i in 1:minimum([150, n_neurons])
    spikes = state.spike_trains[i,:][state.spike_trains[i,:] .!= 0]
    eventplot(spikes, lineoffsets=i/2, linelengths=0.4)
end

subplot(2,2,2)
plot(t, A)

subplot(2,2,3)
plot(I_ext, A)
xlabel("\$I^{ext}\$")
ylabel("\$A\$")
