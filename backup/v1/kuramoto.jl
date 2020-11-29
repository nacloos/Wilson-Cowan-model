using Random, Distributions, Statistics
using PyPlot, PyCall
animation = pyimport("matplotlib.animation")
pygui(true)

T = 20 # s
dt = 1e-2 # s
n_iter = Int(T/dt)

distr = Exponential(1)
ω = rand(distr, 20)
K = 2

function simulate(n_iter, ω, K)
    n_osc = length(ω)
    θ = 2π*rand(n_iter, n_osc)

    for k in 1:n_iter-1
        order = mean(exp.(im*θ[k,:]))
        r, ψ = abs(order), angle(order)

        dθdt = ω + r*K*sin.(ψ .- θ[k,:])
        θ[k+1,:] = θ[k,:] .+ dθdt.*dt
    end
    return θ
end

function animate(θ)
    z = exp.(im.*θ)
    x, y = real(z), imag(z)

    order = mean(z, dims=2)

	fig = figure(figsize=(7, 7))
    offset = maximum(x)/3
    plt.axis((minimum(x)-offset, maximum(x)+offset, minimum(y)-offset, maximum(y)+offset))

    # plot the circle
    ϕ = 0:0.01:2π
    plt.plot(cos.(ϕ), sin.(ϕ))

    path = scatter(x[1,:], y[1,:], color="tab:blue")
    path_order = scatter(real(order[1]), imag(order[1]), marker="x", color="tab:orange")

	function animate_fun(k)
        path.set_offsets([x[k+1,:] y[k+1,:]])
        path_order.set_offsets([real(order[k+1]) imag(order[k+1])])
		return [path, path_order]
    end
	anim = animation.FuncAnimation(fig, animate_fun, frames=size(θ)[1], interval=0.001, blit=true, repeat=false)
end

θ = simulate(n_iter, ω, K)
animate(θ)
