using Random, Distributions
Random.seed!(5)

# TODO: SimConfig

abstract type Neuron end
struct LIFNeuron <: Neuron
    u_rest::Real    # mV
    threshold::Real # mV
    τ::Real         # ms
    R::Real         # Ω
    Δ_abs           # ms
    f               # spike intensity
end

"""
Represent the structure of the network
"""
struct SpikingNetwork
    neurons::Vector{Neuron}
    W::Matrix{Float64}
end

"""
Represent the state of the network
"""
struct SpikingNetworkState
    u
    spike_trains
    spike_count
end

struct ExternalInput
    I
    noise::Distribution
end

function spike_alpha(spike_time, t, width=0)
    if t > spike_time + width
        return 0
    else
        return 1
    end
end

"""
Compute the input current of a neuron with input weights `input_weights`,
from the spike trains `spike_trains` of other neurons,
and from the external input `ext`.
"""
function input_current(input_weights, spike_trains, spike_count, ext::ExternalInput, dt, iter)
    I = 0
    for j in 1:length(input_weights)
        if spike_count[j] > 0
            # contribution of the jth input, take only the last spike whose time is spike_count[j]
            I += input_weights[j] * spike_alpha(spike_trains[j, spike_count[j]], iter*dt)
        end
    end

    noise = rand(ext.noise)
    return I + ext.I[iter] + noise
end


"""
Simulate a LIF neuron for a given input current I,
return the updated membrane potential and whether a spike is emitted.
"""
function simulate(neuron::LIFNeuron, u, I, last_spike, dt, iter)
    # check if the neuron is in its refractory period
    if iter*dt <= last_spike + neuron.Δ_abs
        return u, false
    end

    # LIF dynamics
    dudt = 1/neuron.τ*(-u + neuron.R*I)
    next_u = u + dudt*dt

    # probability of emitting one spike in the interval dt, given the membrane potential
    firing_prob = 1 - exp(-dt*neuron.f(next_u))

    if rand() <= firing_prob
        return neuron.u_rest, true
    else
        return next_u, false
    end
end


"""
Simulate a network of LIF neurons
"""
function simulate(net::SpikingNetwork, ext::ExternalInput, u0, dt, T)
    n_iter = Int(T/dt)
    n_neurons = length(net.neurons)

    # intially all neurons have fired once
    state = SpikingNetworkState(zeros(n_neurons, n_iter), zeros(n_neurons, n_iter), ones(Int32, n_neurons))
    state.u[:,1] = u0

    for iter in 1:n_iter-1
        spikes = zeros(n_neurons)
        for i in 1:n_neurons
            neuron = net.neurons[i]
            I = input_current(net.W[i], state.spike_trains, state.spike_count, ext, dt, iter)
            last_spike = state.spike_trains[i,state.spike_count[i]]
            state.u[i,iter+1], spike = simulate(neuron, state.u[i,iter], I, last_spike, dt, iter)

            spikes[i] = spike
        end
        for i in 1:n_neurons
            if spikes[i] == 1
                state.spike_count[i] += 1
                state.spike_trains[i,state.spike_count[i]] = (iter+1)*dt
            end
        end
    end
    return state
end
