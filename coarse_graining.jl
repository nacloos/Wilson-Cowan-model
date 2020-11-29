

abstract type Filter end
struct GaussianFilter <: Filter
    var::Real
end
struct WindowFilter <: Filter
    width::Real
end

struct Cluster
    clusters::Array
end

function smooth(t, spike_time::Real, filter::GaussianFilter)
    return 1/sqrt(2*pi) .* exp.(-(t.-spike_time).^2 ./ (2*filter.var))
end

function smooth(t, spike_time::Real, filter::WindowFilter)
    return convert(Array{Int8}, spike_time .< t .< spike_time + filter.width)/filter.width
end

"""
Smooth the spike train of a neuron
"""
function smooth(t, spike_train::Array, filter::Filter)
    smoothed = zeros(size(t))
    spikes = spike_train[spike_train .!= 0]
    for spike in spikes
        smoothed .+= smooth(t, spike, filter)
    end
    return smoothed
end

function coarse_grain(t, state::SpikingNetworkState, filter::Filter)
    A = zeros(size(t))
    n_neurons = size(state.spike_trains)[1]

    for j in 1:n_neurons
        A .+= smooth(t, state.spike_trains[j,:], filter)
    end
    A ./= n_neurons
    return A
end


# function coarse_grain(t, state::SpikingNetworkState, ::PCA)
# end



function coarse_grain(cluster::Cluster)
    for c in cluster.clusters
        coarse_grain(c)
    end
end
