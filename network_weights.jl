using Random, Distributions
Random.seed!(5)

abstract type Connectivity end
struct FullyConnected <: Connectivity
    n::Integer
    m::Integer
end
struct Binomial <: Connectivity
    n::Integer
    m::Integer
    p::Real
end

# TODO use MultivariateDistribution types instead
abstract type WeightDistribution end
struct Homogeneous <: WeightDistribution
    value::Real
end
struct Gaussian <: WeightDistribution
    mean::Real
    var::Real

end

function weights(weight::Homogeneous, co::FullyConnected)
    return weight.value*ones(co.n,co.n)
end

function weights(weight::Gaussian, co::FullyConnected)
    W = rand(MvNormal(weight.mean*ones(co.n*co.m), weight.var*identity(co.n*co.m)))
    W = reshape(W, co.n, co.m)
    # weights[weights .<= 0] .= 1e-2
    return W
end

# function weights(weight_distr::MultivariateDistribution, co::FullyConnected)
#     W = rand(weight_distr(weight.mean*ones(co.n*co.m), weight.var*identity(co.n*co.m)))
#     W = reshape(W, co.n, co.m)
#     # weights[weights .<= 0] .= 1e-2
#     return W
# end
function weights(weight::WeightDistribution, co::Binomial)
    # create a fully connected network with the proper weight distribution
    W = weights(weight, FullyConnected(co.n, co.m))
    # keep connections with probability p
    for i=1:co.n, j=1:co.m
        if rand() > co.p
            W[i,j] = 0
        end
    end
    return W
end



function inh_exc_homog_fully_connected(wEE, wEI, wIE, wII, nE, nI)
    weights = ones(nE+nI, nE+nI)
    weights[1:nE,1:nE] .= wEE
    weights[1:nE,nE+1:end] .= wEI
    weights[nE+1:end,1:nE] .= wIE
    weights[nE+1:end,nE+1:end] .= wII
    return weights
end

function inh_exc_gauss_fully_connected(wEE, wEI, wIE, wII, nE, nI, w_var)
    weights(Gaussian(wEE, var), FullyConnected(nE, nE))
    weights(Gaussian(wEI, var), FullyConnected(nE, nI))
    weights(Gaussian(wIE, var), FullyConnected(nI, nE))
    weights(Gaussian(wII, var), FullyConnected(nI, nI))


    W = ones(nE+nI, nE+nI)
    W[1:nE,1:nE] = gaussian_fully_connected(wEE, w_var, nE, nE)
    W[1:nE,nE+1:end] = gaussian_fully_connected(wEI, w_var, nE, nI)
    W[nE+1:end,1:nE] = gaussian_fully_connected(wIE, w_var, nI, nE)
    W[nE+1:end,nE+1:end] = gaussian_fully_connected(wII, w_var, nI, nI)
    return W
end

function inh_exc_gauss_binomial(wEE, wEI, wIE, wII, nE, nI, p, w_var)
    weights = ones(nE+nI, nE+nI)
    weights[1:nE,1:nE] = gaussian_binomial(wEE, w_var, nE, nE, p)
    weights[1:nE,nE+1:end] = gaussian_binomial(wEI, w_var, nE, nI, p)
    weights[nE+1:end,1:nE] = gaussian_binomial(wIE, w_var, nI, nE, p)
    weights[nE+1:end,nE+1:end] = gaussian_binomial(wII, w_var, nI, nI, p)
    return weights
end
