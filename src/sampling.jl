phase(z::Real) = (z >= zero(z) ? one(z) : -one(z))
phase(z) = exp(im * angle(z))

function partially_hermitize(J, τ::Number)
    ϕ = asin(abs(τ)) / 2
    return J .* cos(ϕ) .+ J' .* (phase(τ) * sin(ϕ))
end

function partially_hermitize(J, T::AbstractVector)
    ϕ = asin.(abs.(T) .* abs.(transpose(T))) ./ 2
    return J .* cos.(ϕ) .+ J' .* phase.(T) .* phase.(transpose(T)) .* sin.(ϕ)
end

function sample(rng::AbstractRNG, ::Type{E}, ensemble::AbstractMatrixEnsemble) where {E}
    N = length(ensemble.T)
    J = partially_hermitize(randn(rng, E, N, N) / sqrt(real(E)(N)), ensemble.T)
    A = J * ensemble.R
    if applicable(lmul!, ensemble.L, A)
        lmul!(ensemble.L, A)
    else
        A = ensemble.L * A
    end
    A .+= ensemble.M
    return A, J
end

function sample(::Type{E}, ensemble::AbstractMatrixEnsemble) where {E}
    return sample(Random.GLOBAL_RNG, E, ensemble)
end

function sample(rng::AbstractRNG, ensemble::AbstractMatrixEnsemble{R}) where {R}
    return sample(rng, Complex{R}, ensemble)
end

sample(ensemble::AbstractMatrixEnsemble{R}) where {R} = sample(Complex{R}, ensemble)
