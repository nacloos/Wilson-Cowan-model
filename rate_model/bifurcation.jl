using Roots
using PyPlot
pygui(true)

include("rate_model.jl")

function find_equilibria(p::RateModel, bounds=(0,1))
    fun = x -> -x + p.f(p.w*x .+ p.ext)
    eq = find_zeros(fun, bounds[1], bounds[2])
    return eq
end

function plot_act_fn(p::RateModel)
    ext_values = [1, 2.3033033033033035, 4.5, 6.701701701701702, 7.7]
    titles = ["\$I < I_{c_1}\$", "\$I = I_{c_1}\$", "\$I_{c_1} < I < I_{c_2}\$", "\$I = I_{c_2}\$", "\$I < I_{c_2}\$"]
    figure(figsize=(12,3), dpi=130)
    for (i,ext) in enumerate(ext_values)
        subplot(1, length(ext_values), i)
        x = 0:0.001:1.5
        plot(x, x, color="dimgray", linestyle="--", label="\$y = x\$")
        plot(x, p.f(p.w*x .+ ext), color="tab:red", label="\$y = F(wx+I)\$")

        title(titles[i])
        axis("off")
        axis("equal")
        if i == length(ext_values)
            legend(loc="lower right")
        end
    end
    subplots_adjust(left=0.012, right=0.98, wspace=0)
    savefig(savefig_path*"rate_bif.pdf")
    savefig(savefig_path*"rate_bif.png", transparent=true)
end


function plot_hysteresis(p::RateModel)
    ext_values = range(0, stop=13, length=1000)

    # store the values of ext and the corresponding equilibrium value, when it exists
    stable_eq1 = zeros(length(ext_values))
    stable_eq2 = zeros(length(ext_values))
    unstable_eq = zeros(length(ext_values))

    first_bif_idx = -1 # the index in ext_values corresponding to the first bifurcation occuring as the ext increases, starting from 0.
    second_bif_idx = -1 # set to -1 if tje bifurcation has not yet occured

    for (i, ext) in enumerate(ext_values)
        # model = RateModel(τ, w, ext, u->act_fn(Sigmoid(a, θ))(R*u))
        model = RateModel(p.τ, p.w, ext, p.f)
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
    # println(first_bif_idx, " ", second_bif_idx)

    figure(figsize=(6,5), dpi=130)
    plot(ext_values[1:second_bif_idx], stable_eq1[1:second_bif_idx], color="tab:red", label="Stable equilibrium")
    plot(ext_values[first_bif_idx:second_bif_idx], unstable_eq[first_bif_idx:second_bif_idx], color="tab:red", linestyle="--", label="Unstable equilibrium")
    plot(ext_values[first_bif_idx:end], stable_eq2[first_bif_idx:end], color="tab:red")

    ylabel("Equilibria \$A^*\$")
    xlabel("External input \$I\$")

    critical_ext = (ext_values[first_bif_idx], ext_values[second_bif_idx])
    vlines(critical_ext[1], 0, 1, color="dimgray", linestyle="dotted", label="Saddle-node bifurcation")
    vlines(critical_ext[2], 0, 1, color="dimgray", linestyle="dotted")
    display(critical_ext)
    legend()
    savefig(savefig_path*"rate_hysteresis.pdf")
    savefig(savefig_path*"rate_hysteresis.png", transparent=true)
end


function simulate_hystereris(p::RateModel, dt, T)
    n_iter = Int(T/dt)
    max_ext = 0.8
    I_ext = max_ext*ones(n_iter)
    I_ext[1:n_iter÷2] = 2*max_ext/T .* collect(dt:dt:T/2)
    I_ext[(n_iter÷2)+1:end] = 2*max_ext.*(T.-collect(T/2+dt:dt:T))/T

    p = RateModel(p.τ, p.w, I_ext, p.f)
    sol = simulate(p, 0.0, dt, T)
    t = dt:dt:T
    # plot(t, sol)
    plot(I_ext, sol)

end


θ = 6e-2; τ = 4e-3; a=100
θ = 10; a=1.
# R = 7e-2
# w = 1.1
w = 11
ext = 0.8
τ = 1e-4
f(x) = act_fn(Sigmoid(a, θ))(x)


# Δ_abs = 4e-3
# β = 1000
# τ₀ = 1e-3
# f(u) = 1/τ₀ .*exp.(β*(u.-θ))
# F(I) = f(R*I) ./ (1 .+ Δ_abs*f(R*I))


model = RateModel(τ, w, ext, f)

# plot_act_fn(model)

plot_hysteresis(model)

# simulate_hystereris(model, 1e-4, 0.2)
