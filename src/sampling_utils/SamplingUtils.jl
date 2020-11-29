module SamplingUtils

export phase, half_phase, partially_hermitize

phase(z) = exp(im * angle(z))
phase(z::Real) = (z >= zero(z) ? one(z) : -one(z))

half_phase(z) = exp(im * angle(z) / 2)
half_phase(z::Real) = phase(z)

function partially_hermitize(J, τ)
    ϕ = asin(abs(τ)) / 2
    ph = half_phase(τ)
    Jθ = conj(ph) * J
    return ph * (Jθ * cos(ϕ) + Jθ' * sin(ϕ))
end

end
