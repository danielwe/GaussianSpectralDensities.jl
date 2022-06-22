abstract type AbstractObjective{T<:Real} end
abstract type AbstractSpectralObjective{T} <: AbstractObjective{T} end

struct kObjective{T} <: AbstractObjective{T}
    value::T
end

struct γObjective{T} <: AbstractSpectralObjective{T}
    value::T
end

struct δObjective{T} <: AbstractSpectralObjective{T}
    value::T
end

kObjective() = kObjective(default(:k))
γObjective() = γObjective(default(:γ))

for O in (kObjective, γObjective, δObjective)
    @eval Base.convert(::Type{P}, obj::Q) where {P<:$O,Q<:$O} = P(obj.value)
    @eval Base.convert(::Type{P}, obj::P) where {P<:$O} = obj
end

# adapt(::Type{T}, obj::O) where {T<:Real,O<:AbstractObjective} = convert(O{T}, obj)

_score(value, reference) = one(value) / abs(value - reference)
score(solution, objective::kObjective) = _score(log(solution.k), log(objective.value))
score(solution, objective::γObjective) = _score(solution.γ, objective.value)
score(solution, objective::δObjective) = _score(solution.δ, objective.value)
