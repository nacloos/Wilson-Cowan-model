include("../renormalization.jl")
include("../network_weights.jl")
include("../LIF_network.jl")
include("../plotting.jl")
include("../coarse_graining.jl")
include("../WC_model.jl")
include("../rate_model/rate_model.jl")


T = 0.2; dt = 1e-4; n_iter = Int(T/dt)

# u_rest = 0; threshold = 6e-2; τ = 1e-2; R = 7e-2
u_rest = 0; threshold = 6e-2; τ = 4e-3; R = 7e-2
f(u) = 1e-3 .*exp.(1/0.004*(u.-threshold))
# Δ_abs = 4e-3
Δ_abs = 0

neuron = LIFNeuron(u_rest, threshold, τ, R, f, Δ_abs)

n_neurons = 50
J0 = 0
coupling_proba = 1.

W = weights(Homogeneous(J0/(coupling_proba*n_neurons)), Binomial(n_neurons, n_neurons, coupling_proba))

# draw threshold from a Gaussian distribution
# thresholds = rand(Normal(threshold, 5e-3), n_neurons)
# net = SpikingNetwork([LIFNeuron(u_rest, th, τ, R) for th in thresholds], W)

net = SpikingNetwork([neuron for i=1:n_neurons], W)
# plot(net)


I_ext = 1.8*ones(n_iter)


ext_input = ExternalInput(I_ext, Normal(0, 0))
u0 = zeros(n_neurons)

state = simulate(net, ext_input, u0, dt, T)


integral_model = WCIntegral(R, I_ext[1], τ, threshold, f)
# A_model = simulate(integral_model, dt, T)
A_model = simulate(integral_model, Δ_abs, dt, T)


# rate_model = renormalize(net, RateModel, I_ext)
rate_model = RateModel(τ, J0, R*I_ext[1], f)
A_rate_model = simulate(rate_model, 0.0, dt, T)

rate_int_model = RateIntegralModel(τ, R, I_ext[1], f)
A_rate_int = simulate(rate_int_model, Δ_abs, dt, T)


println("Coucouc")
println(f(R*I_ext[1]), ", ", 1 - exp(-dt*f(R*I_ext[1])))

t = dt:dt:T
A = coarse_grain(t, state, WindowFilter(1e-3))
# A = coarse_grain(t, state, WindowFilter(1.5e-4))

# convert the estimate of the probability density A(t) in a probability of spike in the discrete time step dt
K = 5
Δt = K*dt
P = zeros(size(A)) # probability of a spike in time interval [t,t*Δt]

for i in K:length(P)
    for k in 0:K-1
        P[i] += A[i-k]
    end
    P[i] *= Δt
end


# A = coarse_grain(t, state, GaussianFilter(1e-5))

figure()
plot(t, A, label="LIF network")
plot(t, P, label="LIF network")
plot(t, A_model, label="Integral model")
plot(t, A_rate_model, label="Rate model")
plot(t, A_rate_int, label="Rate integral model")


plt.hlines(mean(state.spike_count)/T, 0, T)

legend()



figure()
nz_times = state.spike_trains .!= 0
for i in 1:minimum([150, n_neurons])
    spikes = state.spike_trains[i,:][state.spike_trains[i,:] .!= 0]
    eventplot(spikes, lineoffsets=i/2, linelengths=0.4)
plt.xlim(0, T)
end
