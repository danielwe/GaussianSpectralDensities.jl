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

function sample(T, ensemble::AbstractMatrixEnsemble)
    N = length(ensemble.T)
    Jiid = randn(T, N, N) / real(T)(sqrt(N))
    J = partially_hermitize(Jiid, ensemble.T)
    return (ensemble.M + ensemble.L * J * ensemble.R, J)
end

sample(ensemble::AbstractMatrixEnsemble{R}) where {R} = sample(Complex{R}, ensemble)
