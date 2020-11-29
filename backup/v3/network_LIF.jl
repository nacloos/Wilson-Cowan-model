# module NetworkLIF
# export Neuron, simulate_dynamics, pop_actvity
using Random, Distributions
Random.seed!(5)

abstract type Neuron end
struct LIFNeuron <: Neuron
    u_rest::Real    # mV
    threshold::Real # mV
    τ::Real         # ms
    R::Real         # Ω
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

function spike_alpha(spike_time, t, width=1e-2)
    if t > spike_time + width
        return 1
    else
        return 0
    end
end

function input_current(iter, input_weights, spike_trains, spike_count, ext::ExternalInput, dt)
    # compute the input current to a neuron whose incoming weights are given in input_weights
    I = 0
    for j in 1:length(input_weights)
        if spike_count[j] > 0
            I += input_weights[j] * spike_alpha(spike_trains[j, spike_count[j]], iter*dt)
        end
    end
    # gauss_noise = rand(Normal(0, 1e-1))
    noise = rand(ext.noise)
    return I + ext.I[iter] + noise
end

"""
Simulate a LIF neuron for a given input current I
"""
function simulate(neuron::LIFNeuron, u, I)
    du = 1/neuron.τ*(-(u-neuron.u_rest) + neuron.R*I) * dt
    next_u = u + du
    if next_u > neuron.threshold
        return neuron.u_rest, true # emit a spike and reset membrane potential
    else
        return next_u, false
    end
end

"""
Simulate a network of LIF neurons
"""
function simulate(net::SpikingNetwork, ext::ExternalInput, dt, T)
    n_iter = n_iter = Int(T/dt)
    n_neurons = length(net.neurons)

    state = SpikingNetworkState(zeros(n_neurons, n_iter), zeros(n_neurons, n_iter), zeros(Int32, n_neurons))

    for iter in 1:n_iter-1
        for i in 1:n_neurons
            neuron = net.neurons[i]
            I = input_current(iter, net.W[i], state.spike_trains, state.spike_count, ext, dt)
            state.u[i,iter+1], spike = simulate(neuron, state.u[i,iter], I)

            if spike
                state.spike_count[i] += 1
                state.spike_trains[i,state.spike_count[i]] = iter*dt
            end
        end
    end
    return state
end

# end
