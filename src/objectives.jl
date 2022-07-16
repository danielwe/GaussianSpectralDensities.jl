abstract type AbstractObjective{T<:Real} end

"""

    TraceObjective(value)
    TraceObjective(; value=DEFAULTS.kobjective)

Set a target value for ``k = \\tr [QQ^† + γ²I]⁻¹`` when solving averaged resolvent equations
"""
Base.@kwdef struct TraceObjective{T} <: AbstractObjective{T}
    value::T = DEFAULTS.kobjective
end

"""

    RegularizerObjective(value::Regularizer)
    RegularizerObjective(kind, value)
    RegularizerObjective(; kind=nothing, value=DEFAULTS.robjective)

Set a target value for a regularizer when solving averaged resolvent equations

If `kind` is provided, `value` should be a numerical value, otherwise it should be
a `Regularizer`. See [`Regularizer`](@ref) for the available kinds of regularizers.

See [`Regularizer`](@ref) for the available kinds of regularizers.
"""
struct RegularizerObjective{T} <: AbstractObjective{T}
    value::Regularizer{T}
end

RegularizerObjective(kind, value) = RegularizerObjective(Regularizer(kind, value))

function RegularizerObjective(; kind=nothing, value=DEFAULTS.robjective)
    (kind !== nothing) && return RegularizerObjective(kind, value)
    return RegularizerObjective(value)
end

## number type conversion
## ======================

for O in (TraceObjective, RegularizerObjective)
    @eval Base.convert(::Type{P}, obj::Q) where {P<:$O,Q<:$O} = P(obj.value)
end

#=
## solution score
## ==============

score(solution, obj::TraceObjective) = score(log(solution.k), log(obj.value))
score(solution, obj::RegularizerObjective) = score(solution, obj.value)
=#
