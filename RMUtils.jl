module RMUtils

export phase, half_phase, partially_hermitize

phase(z::Complex) = exp(im * angle(z))

function phase(z::U) where {U<:Real}
    return z >= 0.0 ? U(1.0) : U(-1.0)
end

half_phase(z::Real) = phase(z)
half_phase(z::Complex) = exp(im * 0.5 * angle(z))

function partially_hermitize(J, τ)
    ϕ = 0.5 * asin(abs(τ))
    ph = half_phase(τ)
    Jθ = (1.0 / ph) * J
    return ph * (Jθ * cos(ϕ) + Jθ' * sin(ϕ))
end

end
