using PyPlot
using Random, Distributions
using DSP
pygui(true)

T = 0.2 # s
dt = 1e-4 # s
n_iter = Int(T/dt)

# u_rest = -6e-2 # -60mV
u_rest = 0 # -60mV
threshold = 6e-2
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

function simulate(weights, I_ext, n_neurons, n_iter)
    spike_times = zeros(n_neurons, n_iter)
    spike_count = zeros(Int32, n_neurons)
    u = zeros(n_neurons, n_iter)

    for iter in 1:n_iter-1
        for i in 1:n_neurons
            I = compute_input_current(iter, weights[i], spike_times, spike_count, I_ext[iter])
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

function pop_act(t, spike_times, Δt=1e-3)
    function filter(spike_time, t, Δt)
        # return 1/(sqrt(2*pi)*Δt) .* exp.(-(t.-spike_time).^2 ./ (2*Δt^2))
        return convert(Array{Int8}, spike_time .< t .< spike_time + Δt)
    end

    A = zeros(size(t))
    n_neurons = size(spike_times)[1]
    for j in 1:n_neurons
        spikes = spike_times[j,:][spike_times[j,:] .!= 0]
        for spike in spikes
            A .+= filter(spike, t, Δt)
        end
    end
    A ./= n_neurons
    return A
end

# TODO: random network (binomial) with constant weight
n_neurons = 200
# weights = ones(n_neurons, n_neurons)
weights = rand(MvNormal(5*ones(n_neurons*n_neurons), 2e-4*identity(n_neurons*n_neurons)))
weights = reshape(weights, n_neurons, n_neurons)
weights[weights .<= 0] .= 1e-2

I_ext = 2*ones(n_iter)
I_ext[n_iter÷2:end] .= 0
u, spike_times, spike_count = simulate(weights, I_ext, n_neurons, n_iter)

figure()
nz_times = spike_times .!= 0
for i in 1:n_neurons
    spikes = spike_times[i,:][spike_times[i,:] .!= 0]
    eventplot(spikes, lineoffsets=i, linelengths=0.8)
end

# # plot one neuron
# figure()
# subplot(2, 1, 1)
t = dt:dt:T
# plot(t, u[1,:])
# scatter(spike_times[1,1:spike_count[1]], 0.1*ones(spike_count[1]))
# axhline(u_rest, linestyle="--")
#
# subplot(2, 1, 2)
# plot(t, I_ext)

A = pop_act(t, spike_times)
figure()
plot(t, A)

A_filt  = filtfilt(digitalfilter(Lowpass(0.008), Butterworth(7)), A)
plot(t, A_filt)
