abstract type AbstractSpectralSolution{R<:Real} end

struct BareSpectralSolution{R} <: AbstractSpectralSolution{R}
    μ::MissingOr{R}
    q::MissingOr{Complex{R}}
    γ²::MissingOr{R}
    δ²::MissingOr{R}
    k::MissingOr{R}
    r::MissingOr{Complex{R}}
end

function BareSpectralSolution(ws, objective, z, q)
    return BareSpectralSolution(
        objective.μ,
        q,
        γ²!(ws, objective, z, q),
        δ²!(ws, objective, z, q),
        k!(ws, objective, z, q),
        r!(ws, objective, z, q),
    )
end

function BareSpectralSolution(objective::Fixk)
    return BareSpectralSolution(
        objective.μ, missing, missing, missing, objective.value, missing
    )
end

function BareSpectralSolution(objective::Fixγ²)
    return BareSpectralSolution(
        objective.μ, missing, objective.value, missing, missing, missing
    )
end

function BareSpectralSolution(objective::Fixδ²)
    return BareSpectralSolution(
        objective.μ, missing, missing, objective.value, missing, missing
    )
end

const SpectralTrace{R} = StructVector{BareSpectralSolution{R}}

@generated function empty_trace(::Type{R}) where {R}
    etr = SpectralTrace{R}(undef, 0)
    return :($etr)
end

struct SpectralSolution{R} <: AbstractSpectralSolution{R}
    μ::MissingOr{R}
    q::MissingOr{Complex{R}}
    γ²::MissingOr{R}
    δ²::MissingOr{R}
    k::MissingOr{R}
    r::MissingOr{Complex{R}}
    algorithm::Algorithm{R}
    trace::SpectralTrace{R}
end

function SpectralSolution(sol::BareSpectralSolution, algorithm, trace)
    return SpectralSolution(sol.μ, sol.q, sol.γ², sol.δ², sol.k, sol.r, algorithm, trace)
end

const GlobalSpectralSolution{R} = StructArray{SpectralSolution{R}}
