module SamplingUtils

export phase, partially_hermitize

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

end
