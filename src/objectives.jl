abstract type AbstractObjective{T<:Real} end

"""

    TraceObjective(value=DEFAULTS.k)

Set a target value for ``\\tr [QQ^† + γ²I]⁻¹`` when solving averaged resolvent equations
"""
struct TraceObjective{T} <: AbstractObjective{T}
    value::T
end

TraceObjective() = TraceObjective(DEFAULTS.k)

@enumx Regularizers γ δ

"""

    RegularizerObjective(value=DEFAULTS.γ, kind=Regularizers.γ)

Set a target value for a regularizer when solving averaged resolvent equations

The available kinds of regularizers are:

- `Regularizers.γ`: the simple additive regularizer in the denominator ``[QQ^† + γ²I]⁻¹``
- `Regularizers.δ`: refers to the smallest eigenvalue of the entire denominator, such that
    `δ^2 == γ^2 - eigmin(Q * Q')`
"""
struct RegularizerObjective{T} <: AbstractObjective{T}
    value::T
    kind::Regularizers.T
end

RegularizerObjective(value) = RegularizerObjective(value, Regularizers.γ)
RegularizerObjective() = RegularizerObjective(DEFAULTS.γ)

## eltype conversion and adaption
## ==============================

Base.convert(::Type{P}, obj::Q) where {P<:TraceObjective,Q<:TraceObjective} = P(obj.value)

function Base.convert(
    ::Type{P}, obj::Q
) where {P<:RegularizerObjective,Q<:RegularizerObjective}
    return P(obj.value, obj.kind)
end

for O in (TraceObjective, RegularizerObjective)
    @eval Base.convert(::Type{P}, obj::P) where {P<:$O} = obj
    @eval adapt(::Type{T}, obj::$O) where {T<:Real} = convert($O{T}, obj)
end

# ## solution score
# ## ==============
#
# score(solution, obj::TraceObjective) = _score(log(solution.k), log(obj.value))
#
# function score(solution, obj::RegularizerObjective)
#     (obj.kind === Regularizers.γ) && return _score(solution.γ, obj.value)
#     (obj.kind === Regularizers.δ) && return _score(solution.δ, obj.value)
#     error("unknown regularizer")
# end
#
# _score(value, reference) = inv(abs(value - reference))
