# module NetworkLIF
# export Neuron, simulate_dynamics, pop_actvity
using Random, Distributions
Random.seed!(5)

include("activation_fns.jl")

abstract type Neuron end
struct LIFNeuron <: Neuron
    u_rest::Real    # mV
    threshold::Real # mV
    τ::Real         # ms
    R::Real         # Ω
    f
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

function input_current(iter, input_weights, spike_trains, spike_count, ext::ExternalInput, dt)
    # compute the input current to a neuron
    # param input_weights: incoming connections weights
    # param spike_trains: contains the spike trains of all the neurons of the network
    I = 0
    for j in 1:length(input_weights)
        if spike_count[j] > 0
            # contribution of the jth input, take only the last spike whose time is spike_count[j]
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
function simulate(neuron::LIFNeuron, u, I, dt)
    dudt = 1/neuron.τ*(-u + neuron.R*I)
    next_u = u + dudt*dt

    firing_prob = 1 - exp(-dt*neuron.f(next_u))
    # firing_prob = dt*neuron.f(next_u)

    if rand() <= firing_prob
        return neuron.u_rest, true
    else
        return next_u, false
    end

    # if next_u > neuron.threshold
    #     return neuron.u_rest, true # emit a spike and reset membrane potential
    # else
    #     return next_u, false
    # end
end

"""
Simulate a network of LIF neurons
"""
function simulate(net::SpikingNetwork, ext::ExternalInput, u0, dt, T)
    n_iter = n_iter = Int(T/dt)
    n_neurons = length(net.neurons)

    state = SpikingNetworkState(zeros(n_neurons, n_iter), zeros(n_neurons, n_iter), zeros(Int32, n_neurons))
    state.u[:,1] = u0

    for iter in 1:n_iter-1
        spikes = zeros(n_neurons)
        for i in 1:n_neurons
            neuron = net.neurons[i]
            I = input_current(iter, net.W[i], state.spike_trains, state.spike_count, ext, dt)
            state.u[i,iter+1], spike = simulate(neuron, state.u[i,iter], I, dt)

            spikes[i] = spike
        end
        for i in 1:n_neurons
            if spikes[i] == 1
                state.spike_count[i] += 1
                state.spike_trains[i,state.spike_count[i]] = iter*dt
            end
            # TODO don't need to update all membrane potentials before updating the spikes ?
        end
    end
    return state
end


# function simulate(net::SpikingNetwork, ext::ExternalInput, u0, dt, T)
#     n_iter = n_iter = Int(T/dt)
#     n_neurons = length(net.neurons)
#
#     state = SpikingNetworkState(zeros(n_neurons, n_iter), zeros(n_neurons, n_iter), zeros(Int32, n_neurons))
#     state.u[:,1] = u0
#
#     for iter in 1:n_iter-1
#         spikes = zeros(n_neurons)
#         for i in 1:n_neurons
#             neuron = net.neurons[i]
#             state.u[i,iter+1], spike = simulate(neuron, state.u[i,iter], ext.I[1], dt)
#             if spike == 1
#                 state.spike_count[i] += 1
#                 state.spike_trains[i,state.spike_count[i]] = iter*dt
#             end
#         end
#     end
#     return state
# end

# end
