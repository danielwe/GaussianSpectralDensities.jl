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

function randpartialherm(rng::AbstractRNG, ::Type{E}, N::Integer, τ::Number, args...) where {E}
    return partially_hermitize(randiid(rng, E, N, args...), τ)
end

function randpartialherm(::Type{E}, N::Integer, τ::Number, args...) where {E}
    return partially_hermitize(randiid(E, N, args...), τ)
end

function randpartialherm(rng::AbstractRNG, N::Integer, τ::E, args...) where {E<:Number}
    return randpartialherm(rng, E, N, τ, args...)
end

randpartialherm(N::Integer, τ::E, args...) where {E<:Number} = randpartialherm(E, N, τ, args...)

function randpartialherm(rng::AbstractRNG, ::Type{E}, T::AbstractVector, args...) where {E}
    return partially_hermitize(randiid(rng, E, length(T), args...), T)
end

function randpartialherm(::Type{E}, T::AbstractVector, args...) where {E}
    return partially_hermitize(randiid(E, length(T), args...), T)
end

function randpartialherm(rng::AbstractRNG, T::AbstractVector, args...)
    return randpartialherm(rng, eltype(T), T, args...)
end

randpartialherm(T::AbstractVector, args...) = randpartialherm(eltype(T), T, args...)

function transform(ensemble::AbstractMatrixEnsemble, J)
    A = J * ensemble.R
    if applicable(lmul!, ensemble.L, A)
        lmul!(ensemble.L, A)
    else
        A = ensemble.L * A
    end
    A .+= ensemble.M
    return A
end

function transform!(ensemble::AbstractMatrixEnsemble, J, A)
    mul!(A, J, ensemble.R)
    lmul!(ensemble.L, A)
    A .+= ensemble.M
    return nothing
end

function randsample(
    rng::AbstractRNG, ::Type{E}, ensemble::AbstractMatrixEnsemble, args...
) where {E}
    J = randpartialherm(rng, E, ensemble.T, args...)
    return transform(ensemble, J), J
end

function randsample(rng_or_type, ensemble::AbstractMatrixEnsemble, args...) where {E}
    J = randpartialherm(rng_or_type, ensemble.T, args...)
    return transform(ensemble, J), J
end

function randsample(ensemble::AbstractMatrixEnsemble{R}, args...) where {R}
    return randsample(Complex{R}, ensemble, args...)
end
