
include("network_weights.jl")
include("network_LIF.jl")
include("plotting.jl")
include("coarse_graining.jl")
# using .NetworkLIF


T = 0.2; dt = 1e-4; n_iter = Int(T/dt)

# u_rest = -6e-2 # -60mV
u_rest = 0; threshold = 6e-2; τ = 1e-2; R = 7e-2
neuron = LIFNeuron(u_rest, threshold, τ, R)

n_neurons = 100
J0 = 100
coupling_proba = 0.5

# W = weights(Homogeneous(J0/n_neurons), FullyConnected(n_neurons, n_neurons
# W = weights(Gaussian(J0/n_neurons, 2e-4/n_neurons), FullyConnected(n_neurons, n_neurons))
W = weights(Homogeneous(J0/(coupling_proba*n_neurons)), Binomial(n_neurons, n_neurons, coupling_proba))
# W = weights(Gaussian(J0/(coupling_proba*n_neurons), 2e-4/n_neurons), Binomial(n_neurons, n_neurons, coupling_proba))

net = SpikingNetwork([neuron for i=1:n_neurons], W)
# plot(net)

I_ext = 1*ones(n_iter)
I_ext[n_iter÷2:end] .= 0
noise_distr = Normal(0, 1e-1)
ext_input = ExternalInput(I_ext, noise_distr)

state = simulate(net, ext_input, dt, T)


figure()
nz_times = state.spike_trains .!= 0
for i in 1:minimum([150, n_neurons])
    spikes = state.spike_trains[i,:][state.spike_trains[i,:] .!= 0]
    eventplot(spikes, lineoffsets=i/2, linelengths=0.4)
end

# plot one neuron
figure()
subplot(2, 1, 1)
t = dt:dt:T

plot(t, state.u[1,:])
scatter(state.spike_trains[1,1:state.spike_count[1]], 0.1*ones(state.spike_count[1]))
axhline(u_rest, linestyle="--")

subplot(2, 1, 2)
plot(t, I_ext)

# A = coarse_grain(t, state, WindowFilter(2e-3))
A = coarse_grain(t, state, GaussianFilter(1e-5))
figure()
plot(t, A)
legend()
