abstract type AbstractObjective{F<:Real} end

"""
    TraceObjective(value=DEFAULTS.kobjective)
    TraceObjective{F}(value) where {F<:Real}

Set a target value for ``k = \\tr [QQ^† + γ²I]⁻¹`` when solving averaged resolvent equations
"""
struct TraceObjective{F} <: AbstractObjective{F}
    value::F
end

TraceObjective() = TraceObjective(DEFAULTS.kobjective)

"""
    RegularizerObjective(kind, value)
    RegularizerObjective(value::Regularizer=DEFAULTS.robjective)
    RegularizerObjective{F}(value::Regularizer) where {F<:Real}

Set a target value for a regularizer when solving averaged resolvent equations

If `kind` is provided, `value` should be a numerical value, otherwise it should be
a `Regularizer`. See [`Regularizer`](@ref) for the available kinds of regularizers.

See [`Regularizer`](@ref) for the available kinds of regularizers.
"""
struct RegularizerObjective{F} <: AbstractObjective{F}
    value::Regularizer{F}
end

RegularizerObjective(kind, value) = RegularizerObjective(Regularizer(kind, value))
RegularizerObjective() = RegularizerObjective(DEFAULTS.robjective)

## Conversions
## ===========
for O in (TraceObjective, RegularizerObjective)
    @eval Base.convert(::Type{P}, obj::Q) where {P<:$O,Q<:$O} = P(obj.value)
end

#=
## solution score
## ==============

score(solution, obj::TraceObjective) = score(log(solution.k), log(obj.value))
score(solution, obj::RegularizerObjective) = score(solution, obj.value)
=#
