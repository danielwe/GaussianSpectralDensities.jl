phase(z::Real) = (z >= zero(z) ? one(z) : -one(z))
phase(z) = exp(im * angle(z))

function partially_hermitize(J, τ::Number)
    ϕ = asin(abs(τ)) / 2
    ph = phase(τ)
    return J * cos(ϕ) + J' * (ph * sin(ϕ))
end

function partially_hermitize(J, T::AbstractVector)
    Tabs = abs.(T)
    ϕ = asin.(Tabs * Tabs') ./ 2
    Tph = phase.(T)
    ph = Tph * transpose(Tph)
    return J .* cos.(ϕ) .+ J' .* ph .* sin.(ϕ)
end

function sample(rng::AbstractRNG, ::Type{E}, ensemble::AbstractMatrixEnsemble) where {E}
    N = length(ensemble.T)
    Jiid = randn(rng, E, N, N) / sqrt(real(E)(N))
    J = partially_hermitize(Jiid, ensemble.T)
    return (ensemble.M + ensemble.L * J * ensemble.R, J)
end

function sample(::Type{E}, ensemble::AbstractMatrixEnsemble) where {E}
    return sample(Random.GLOBAL_RNG, E, ensemble)
end

function sample(rng::AbstractRNG, ensemble::AbstractMatrixEnsemble{R}) where {R}
    return sample(rng, Complex{R}, ensemble)
end

sample(ensemble::AbstractMatrixEnsemble{R}) where {R} = sample(Complex{R}, ensemble)
