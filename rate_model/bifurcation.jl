using Roots
using PyPlot
pygui(true)

include("rate_model.jl")

function find_equilibria(p::RateModel, bounds=(0,1))
    fun = x -> -x + p.f(p.w*x .+ p.ext)
    eq = find_zeros(fun, bounds[1], bounds[2])
    return eq
end

function plot_f(p::RateModel)
    x = 0:0.001:1.5
    plot(x, p.f(p.w*x .+ p.ext))
    plot(x, x)
end


function plot_hysteresis()
    ext_values = range(0, stop=0.8, length=100)

    # store the values of ext and the corresponding equilibrium value, when it exists
    stable_eq1 = zeros(length(ext_values))
    stable_eq2 = zeros(length(ext_values))
    unstable_eq = zeros(length(ext_values))

    first_bif_idx = -1 # the index in ext_values corresponding to the first bifurcation occuring as the ext increases, starting from 0.
    second_bif_idx = -1 # set to -1 if tje bifurcation has not yet occured

    for (i, ext) in enumerate(ext_values)
        model = RateModel(τ, w, ext, u->act_fn(Sigmoid(a, θ))(R*u))
        eq = find_equilibria(model)
        # check if encounter a saddle-node bifurcation
        if length(eq) > 1 && first_bif_idx == -1
            first_bif_idx = i
        elseif length(eq) == 1 && first_bif_idx >= 0 && second_bif_idx == -1
            second_bif_idx = i-1 # index just before the bifurcation occurs
        end

        # store the equilibria
        if first_bif_idx == -1
            # first stable equilibrium (the smallest), below the first critical value
            stable_eq1[i] = eq[1]
        else
            if length(eq) > 1
                # between the two critical values
                stable_eq1[i] = eq[1]
                unstable_eq[i] = eq[2]
                stable_eq2[i] = eq[3]
            else
                # above the second critical value
                stable_eq2[i] = eq[1]
            end
        end

        # println(eq)
        # scatter(repeat([ext], length(eq)), eq, color="tab:blue")
    end
    println(first_bif_idx, " ", second_bif_idx)
    plot(ext_values[1:second_bif_idx], stable_eq1[1:second_bif_idx], color="tab:blue", label="Stable equilibrium")
    plot(ext_values[first_bif_idx:second_bif_idx], unstable_eq[first_bif_idx:second_bif_idx], color="tab:blue", linestyle="dotted", label="Unstable equilibrium")
    plot(ext_values[first_bif_idx:end], stable_eq2[first_bif_idx:end], color="tab:blue")

    ylabel("Equilibria")
    xlabel("External input")
    legend()

    critical_ext = (ext_values[first_bif_idx], ext_values[second_bif_idx])
    vlines(critical_ext[1], 0, 1, color="tab:blue", linestyle="--", alpha=0.3)
    vlines(critical_ext[2], 0, 1, color="tab:blue", linestyle="--", alpha=0.3)
end


θ = 6e-2; τ = 4e-3; a=100
R = 7e-2
w = 1
ext = 0.8


# β = 2
# τ₀ = 1e-3
# f(u) = 1/τ₀ .*exp.(β*(u.-θ))

f(x) = act_fn(Sigmoid(a, θ))(R*x)

model = RateModel(τ, w, ext, f)
eq = find_equilibria(model)
println()
println(eq)

plot_f(model)
scatter(eq, model.f(model.w*eq .+ model.ext), marker="x")

figure()
plot_hysteresis()
