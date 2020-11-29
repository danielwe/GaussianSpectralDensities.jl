abstract type AbstractSpectralSolution{R<:Real} end

struct BareSpectralSolution{R} <: AbstractSpectralSolution{R}
    μ::MissingOr{R}
    q::MissingOr{Complex{R}}
    γ²::MissingOr{R}
    δ²::MissingOr{R}
    k::MissingOr{R}
end

function BareSpectralSolution(workspace, objective, z, q)
    return BareSpectralSolution(
        objective.μ,
        q,
        γ²!(workspace, objective, z, q),
        δ²!(workspace, objective, z, q),
        k!(workspace, objective, z, q),
    )
end

function BareSpectralSolution(objective::Fixk)
    return BareSpectralSolution(objective.μ, missing, missing, missing, objective.value)
end

function BareSpectralSolution(objective::Fixγ²)
    return BareSpectralSolution(objective.μ, missing, objective.value, missing, missing)
end

function BareSpectralSolution(objective::Fixδ²)
    return BareSpectralSolution(objective.μ, missing, missing, objective.value, missing)
end

const SpectralTrace{R} = StructVector{BareSpectralSolution{R}}
const EMPTY_TRACES = Dict{DataType, SpectralTrace{<:Real}}()

function empty_trace(::Type{R}) where {R}
    if !(R in keys(EMPTY_TRACES))
        EMPTY_TRACES[R] = SpectralTrace{R}(undef, 0)
    end
    return EMPTY_TRACES[R]
end

struct SpectralSolution{R, A<:Algorithm{R}} <: AbstractSpectralSolution{R}
    μ::MissingOr{R}
    q::MissingOr{Complex{R}}
    γ²::MissingOr{R}
    δ²::MissingOr{R}
    k::MissingOr{R}
    algorithm::A
    trace::SpectralTrace{R}
end

function SpectralSolution(sol::BareSpectralSolution, args...)
    return SpectralSolution(sol.μ, sol.q, sol.γ², sol.δ², sol.k, args...)
end

const GlobalSpectralSolution{R, A} = StructArray{SpectralSolution{R, A}}
