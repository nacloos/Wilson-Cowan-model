using PyPlot
import PyPlot: plot
pygui(true)

include("LIF_network.jl")


function plot(net::SpikingNetwork)
    hist(reshape(net.W, length(net.W)), color="coral")
    plt.title("Weights histogram")
end



# struct Raster
#     spike_times
# end

# function plot(raster::Raster)
# end

#
# function pop_activity()
#     A = pop_actvity(t, spike_times[1:nE,:], 2e-3)
