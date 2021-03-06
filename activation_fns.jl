
"""
Sigmoid activation function
"""
struct Sigmoid
    a::Real
    θ::Real
end

act_fn(sig::Sigmoid) = x -> 1 ./ (1 .+ exp.(-sig.a .* (x .- sig.θ))) .- 1/(1 + exp(sig.a*sig.θ))
finv(sig::Sigmoid) = x -> sig.θ .- 1/sig.a .* log.(1 ./(x .+ 1/(1+exp(sig.a*sig.θ))) .- 1)
fder(sig::Sigmoid) = x -> sig.a .*exp.(-sig.a .*(x .- sig.θ))./(1 .+ exp.(-sig.a .*(x .-sig.θ))).^2
