using PyPlot
using Random, Distributions
pygui(true)

T = 0.3 # s
dt = 1e-3 # s
n_iter = Int(T/dt)

u_rest = -6e-2 # -60mV
threshold = 0
τ = 1e-2 # 10ms
R = 7e-2 # Ω

function spike_alpha(spike_time, t, width=1e-2)
    if t > spike_time + width
        return 1
    else
        return 0
    end
end

function compute_input_current(iter, weights, spike_times, spike_count, I_ext)
    I = 0
    for j in 1:length(weights)
        if spike_count[j] > 0
            I += weights[j] * spike_alpha(spike_times[j,spike_count[j]], iter*dt)
        end
    end
    return I + I_ext
end

# function compute_input_current_fast(iter, weights, spike_times, spike_count, I_ext)
#     n_neurons = length(weights[1,:])
#     I = zeros(n_neurons)
#     for j in 1:n_neurons
#         if spike_count[j] > 0
#             I += weights[:,j] .* spike_alpha(spike_times[j,spike_count[j]], iter*dt)
#         end
#     end
#     return I .+ I_ext
# end


function simulate(weights, I_ext, n_neurons, n_iter)
    spike_times = zeros(n_neurons, n_iter)
    spike_count = zeros(Int32, n_neurons)
    u = zeros(n_neurons, n_iter)

    for iter in 1:n_iter-1
        for i in 1:n_neurons
            I = compute_input_current(iter, weights[i], spike_times, spike_count, I_ext)
            du = 1/τ*(-(u[i,iter]-u_rest) + R*I) * dt
            u[i,iter+1] = u[i,iter] + du
        end

        for i in 1:n_neurons
            if u[i,iter+1] > threshold
                spike_count[i] += 1
                spike_times[i,spike_count[i]] = iter*dt
                u[i,iter+1] = u_rest
            end
        end
    end

    return u, spike_times, spike_count
end
#
# function simulate_fast(weights, I_ext, n_neurons, n_iter)
#     spike_times = zeros(n_neurons, n_iter)
#     spike_count = zeros(Int32, n_neurons)
#     u = zeros(n_neurons, n_iter)
#
#     for iter in 1:n_iter-1
#         I = compute_input_current_fast(iter, weights, spike_times, spike_count, I_ext)
#         du = 1/τ.*(-(u[:,iter].-u_rest) .+ R.*I) .* dt
#         u[:,iter+1] = u[:,iter] .+ du
#
#
#         for i in 1:n_neurons
#             if u[i,iter+1] > threshold
#                 spike_count[i] += 1
#                 spike_times[i,spike_count[i]] = iter*dt
#                 u[i,iter+1] = u_rest
#             end
#         end
#     end
#
#     return u, spike_times, spike_count
# end


# TODO: compute the average population activity E (convolution of alpha with spike times), step function for I_ext

n_neurons = 100
# weights = ones(n_neurons, n_neurons)
weights = rand(MvNormal(2*ones(n_neurons*n_neurons), 0.01*identity(n_neurons*n_neurons)))
weights = reshape(weights, n_neurons, n_neurons)

I_ext = 1
u, spike_times, spike_count = simulate(weights, I_ext, n_neurons, n_iter)
#
# figure()
# eventplot(spike_times, linelengths=0.8)

# plot one neuron
# figure()
# plot(dt:dt:T, u[1,:])
# scatter(spike_times[1,1:spike_count[1]], 0.1*ones(spike_count[1]))
# axhline(u_rest, linestyle="--")
