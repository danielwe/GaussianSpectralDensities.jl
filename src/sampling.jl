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

function randiid(rng::AbstractRNG, ::Type{E}, N::Integer, randf=randn) where {E}
    randf(rng, E, N, N) / sqrt(real(E)(N))
end

function randiid(::Type{E}, N::Integer, args...) where {E}
    return randiid(Random.GLOBAL_RNG, E, N, args...)
end

randiid(rng::AbstractRNG, N::Integer, args...) = randiid(rng, ComplexF64, N, args...)
randiid(N::Integer, args...) = randiid(ComplexF64, N, args...)

function transform(ensemble::AbstractMatrixEnsemble, iid)
    J = partially_hermitize(iid, ensemble.T)
    A = J * ensemble.R
    if applicable(lmul!, ensemble.L, A)
        lmul!(ensemble.L, A)
    else
        A = ensemble.L * A
    end
    A .+= ensemble.M
    return A, J
end

function randsample(
    rng::AbstractRNG, ::Type{E}, ensemble::AbstractMatrixEnsemble, args...
) where {E}
    return transform(ensemble, randiid(rng, E, length(ensemble.T), args...))
end

function randsample(rng_or_type, ensemble::AbstractMatrixEnsemble, args...) where {E}
    return transform(ensemble, randiid(rng_or_type, length(ensemble.T), args...))
end

function randsample(ensemble::AbstractMatrixEnsemble{R}, args...) where {R}
    return randsample(Complex{R}, ensemble, args...)
end
