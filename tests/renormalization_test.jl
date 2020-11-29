include("../renormalization.jl")
include("../network_weights.jl")
include("../LIF_network.jl")
include("../plotting.jl")
include("../coarse_graining.jl")
include("../WC_model.jl")
include("../rate_model.jl")


T = 0.2; dt = 1e-4; n_iter = Int(T/dt)

u_rest = 0; threshold = 6e-2; τ = 1e-2; R = 7e-2
# neuron = LIFNeuron(u_rest, threshold, τ, R)

# firing_prob = act_fn(Sigmoid(1e2, threshold))
f(u) = 1e-3 .*exp.(1/0.004*(u .- neuron.threshold))

x = -0.01:0.001:0.07
plot(x, f(x))
plt.show()

neuron = LIFNeuron(u_rest, threshold, τ, R, firing_prob)

n_neurons = 500
J0 = 0
coupling_proba = 1.

W = weights(Homogeneous(J0/(coupling_proba*n_neurons)), Binomial(n_neurons, n_neurons, coupling_proba))

# draw threshold from a Gaussian distribution
# thresholds = rand(Normal(threshold, 5e-3), n_neurons)
# net = SpikingNetwork([LIFNeuron(u_rest, th, τ, R) for th in thresholds], W)

net = SpikingNetwork([neuron for i=1:n_neurons], W)
# plot(net)


I_ext = 2*ones(n_iter)


# wc_model = renormalize(net, WCModel, I_ext)
rate_model = renormalize(net, RateModel, I_ext)
# print(rate_model)
# plt.figure()
# plot_equilibrium(rate_model)
# plt.show()

ext_input = ExternalInput(I_ext, Normal(0, 0))
u0 = zeros(n_neurons)

state = simulate(net, ext_input, u0, dt, T)

x0 = [0., 0.]
sol = simulate(rate_model, x0, T)



t = dt:dt:T
A = coarse_grain(t, state, WindowFilter(1e-2))
# A = coarse_grain(t, state, GaussianFilter(1e-5))

figure()
plot(t, A, label="LIF network")
plot(t, sol(t)[1,:], label="Rate model")

plt.hlines(mean(state.spike_count)/T, 0, T)

legend()



figure()
nz_times = state.spike_trains .!= 0
for i in 1:minimum([150, n_neurons])
    spikes = state.spike_trains[i,:][state.spike_trains[i,:] .!= 0]
    eventplot(spikes, lineoffsets=i/2, linelengths=0.4)
plt.xlim(0, T)
end
