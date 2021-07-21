phase(z::Real) = (z >= zero(z) ? one(z) : -one(z))
phase(z) = cis(angle(z))

function partially_hermitize(J, τ::Number)
    ϕ = asin(abs(τ)) / 2
    return J .* cos(ϕ) .+ J' .* (phase(τ) * sin(ϕ))
end

function partially_hermitize(J, T::AbstractVector)
    ϕ = asin.(abs.(T) .* abs.(transpose(T))) ./ 2
    return J .* cos.(ϕ) .+ J' .* phase.(T) .* phase.(transpose(T)) .* sin.(ϕ)
end

function randiid(rng::AbstractRNG, ::Type{E}, N::Integer, randf=randn) where {E}
    return randf(rng, E, N, N) / sqrt(real(E)(N))
end

function randiid(::Type{E}, N::Integer, args...) where {E}
    return randiid(Random.GLOBAL_RNG, E, N, args...)
end

randiid(rng::AbstractRNG, N::Integer, args...) = randiid(rng, ComplexF64, N, args...)
randiid(N::Integer, args...) = randiid(ComplexF64, N, args...)

function randpartherm(rng::AbstractRNG, ::Type{E}, N::Integer, T, args...) where {E}
    return partially_hermitize(randiid(rng, E, N, args...), T)
end

function randpartherm(rng::AbstractRNG, N::Integer, T, args...)
    return partially_hermitize(randiid(rng, Complex{eltype(T)}, N, args...), T)
end

function randpartherm(::Type{E}, N::Integer, T, args...) where {E}
    return partially_hermitize(randiid(E, N, args...), T)
end

function randpartherm(N::Integer, T, args...)
    return partially_hermitize(randiid(Complex{eltype(T)}, N, args...), T)
end

function transform(ensemble::AbstractMatrixEnsemble, J; hermitize=true)
    if hermitize
        J = partially_hermitize(J, ensemble.T)
    end
    A = J * ensemble.R
    if applicable(lmul!, ensemble.L, A)
        lmul!(ensemble.L, A)
    else
        A = ensemble.L * A
    end
    A .+= ensemble.M
    return A, J
end

function transform!(A, ensemble::AbstractMatrixEnsemble, J; hermitize=true)
    if hermitize
        J = partially_hermitize(J, ensemble.T)
    end
    copy!(A, J)
    if applicable(rmul!, A, ensemble.R)
        rmul!(A, ensemble.R)
    else
        A .= A * ensemble.R
    end
    if applicable(lmul!, ensemble.L, A)
        lmul!(ensemble.L, A)
    else
        A .= ensemble.L * A
    end
    A .+= ensemble.M
    return A, J
end

function randsample(
    rng::AbstractRNG, ::Type{E}, ensemble::AbstractMatrixEnsemble, args...
) where {E}
    return transform(ensemble, randiid(rng, E, length(ensemble.T), args...))
end

function randsample(rng::AbstractRNG, ensemble::AbstractMatrixEnsemble, args...)
return transform(ensemble, randiid(rng, eltype(ensemble), length(ensemble.T), args...))
end

function randsample(::Type{E}, ensemble::AbstractMatrixEnsemble, args...) where {E}
    return transform(ensemble, randiid(E, length(ensemble.T), args...))
end

function randsample(ensemble::AbstractMatrixEnsemble, args...)
    return randsample(eltype(ensemble), ensemble, args...)
end
