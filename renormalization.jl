include("LIF_network.jl")
include("WC_model/WC_model.jl")
include("rate_model.jl")

"""
Take a LIF network and return a Wilson-Cowan model
"""
function renormalize(net::SpikingNetwork, ::Type{WCModel}, ext_input)
    mean_w = mean(net.W)
    mean_τ = mean([neuron.τ for neuron in net.neurons])
    mean_threshold = mean([neuron.threshold for neuron in net.neurons])

    tau_E=mean_τ; a_E=1e2; theta_E=mean_threshold
    tau_I=2.; a_I=3.0; theta_I=3.0
    wEE=mean_w; wEI=0; wIE=0; wII=0
    E_ext=ext_input[1]; I_ext=0

    E_pop = Pop(tau_E, Sigmoid(a_E, theta_E))
    I_pop = Pop(tau_I, Sigmoid(a_I, theta_I))
    return WCModel(E_pop, I_pop, wEE, wIE, wEI, wII, E_ext, I_ext)
end


function renormalize(net::SpikingNetwork, ::Type{RateModel}, ext_input)
    mean_w = mean(net.W)
    mean_τ = mean([neuron.τ for neuron in net.neurons])
    mean_threshold = mean([neuron.threshold for neuron in net.neurons])
    mean_R = mean([neuron.R for neuron in net.neurons])

    a = 1e2
    w = mean_w*length(net.neurons)*mean_R
    ext = mean_R*ext_input[1] # assume stationary input
    pop = Pop(mean_τ, Sigmoid(a, mean_threshold))
    return RateModel(pop, w, ext)
end
