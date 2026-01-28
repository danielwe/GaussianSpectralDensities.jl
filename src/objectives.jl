#=
module Objectives
# NOTE: I don't think we need this module anymore

using ..Defaults: Defaults
using ..Regularizers: Regularizer

public AbstractObjective, RegularizerObjective, TraceObjective

abstract type AbstractObjective{F<:Real} end

"""
    TraceObjective(value=Defaults.trace_target)
    TraceObjective{F}(value) where {F<:Real}
    TraceObjective{F}(trace_obj::TraceObjective) where {F<:Real}

Set a target value for ``k = \\tr [QQ^† + γ²I]⁻¹`` when solving averaged resolvent equations
"""
struct TraceObjective{F} <: AbstractObjective{F}
    value::F
end

TraceObjective() = TraceObjective(Defaults.trace_target)

TraceObjective{F}(o::TraceObjective) where {F} = TraceObjective{F}(o.value)

## Conversions
## ===========
Base.convert(::Type{P}, o::TraceObjective) where {P<:TraceObjective} = (o isa P) ? o : P(o)

function Base.convert(::Type{AbstractObjective{F}}, o::TraceObjective) where {F}
    return convert(TraceObjective{F}, o)
end

"""
    RegularizerObjective()
    RegularizerObjective(value, type)
    RegularizerObjective{F}(regularizer::Regularizer) where {F<:Real}
    RegularizerObjective{F}(reg_obj::RegularizerObjective) where {F<:Real}

Set a target value for a regularizer when solving averaged resolvent equations

If `type` is provided, `value` should be a numerical value, otherwise it should be
a `Regularizer`. See [`Regularizer`](@ref) for the available regularizer types.

The default value is

```julia
RegularizerObjective(Defaults.regularizer_target_value, Defaults.regularizer_target_type)
```
"""
struct RegularizerObjective{F} <: AbstractObjective{F}
    value::Regularizer{F}
end

RegularizerObjective(value, type) = RegularizerObjective(Regularizer(value, type))

function RegularizerObjective()
    return RegularizerObjective(
        Defaults.regularizer_target_value, Defaults.regularizer_target_type
    )
end

function RegularizerObjective{F}(o::RegularizerObjective) where {F}
    return RegularizerObjective{F}(o.value)
end

## Conversions
## ===========
function Base.convert(::Type{P}, o::RegularizerObjective) where {P<:RegularizerObjective}
    return (o isa P) ? o : P(o)
end

function Base.convert(::Type{AbstractObjective{F}}, o::RegularizerObjective) where {F}
    return convert(RegularizerObjective{F}, o)
end

#=
## solution score
## ==============

score(solution, obj::TraceObjective) = score(log(solution.k), log(obj.value))
score(solution, obj::RegularizerObjective) = score(solution, obj.value)
=#

end  # module Objectives

@testitem "TraceObjective" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.Objectives: AbstractObjective, TraceObjective

    target64 = 1.0
    objective64 = TraceObjective(target64)

    # Traits
    @test objective64 isa TraceObjective{Float64}

    # Defaults
    @test TraceObjective() === TraceObjective(Defaults.trace_target)

    @testset "$F" for F in [Float32, Float64]
        targetF = F(target64)
        objectiveF = TraceObjective(targetF)

        # Floattype inferring constructor
        @test objectiveF isa TraceObjective{F}

        # Floattype enforcing constructor
        @test TraceObjective{F}(target64) === objectiveF
        @test TraceObjective{Float64}(targetF) === objective64

        # Floattype conversions
        @test convert(TraceObjective{F}, objective64) === objectiveF
        @test convert(TraceObjective{Float64}, objectiveF) === objective64
        @test convert(AbstractObjective{F}, objective64) === objectiveF
        @test convert(AbstractObjective{Float64}, objectiveF) === objective64
        @test TraceObjective{F}(objective64) === objectiveF
        @test TraceObjective{Float64}(objectiveF) === objective64
    end
end

@testitem "RegularizerObjective" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.Objectives: AbstractObjective, RegularizerObjective
    using GaussianSpectralDensities.Regularizers: Regularizer

    target_value64, target_type = 0.5, :gap
    target64 = Regularizer(target_value64, target_type)
    objective64 = RegularizerObjective(target64)

    # Traits
    @test objective64 isa RegularizerObjective{Float64}

    # Defaults
    @test RegularizerObjective() === RegularizerObjective(
        Defaults.regularizer_target_value, Defaults.regularizer_target_type
    )

    @testset "$F" for F in [Float32, Float64]
        target_valueF = F(target_value64)
        targetF = Regularizer(target_valueF, target_type)
        objectiveF = RegularizerObjective(targetF)

        # Floattype inferring constructor
        @test objectiveF isa RegularizerObjective{F}
        # - All the signatures
        @test RegularizerObjective(target_valueF, target_type) === objectiveF

        # Floattype enforcing constructor
        @test RegularizerObjective{F}(target64) === objectiveF
        @test RegularizerObjective{Float64}(targetF) === objective64

        # Floattype conversions
        @test convert(RegularizerObjective{F}, objective64) === objectiveF
        @test convert(RegularizerObjective{Float64}, objectiveF) === objective64
        @test convert(AbstractObjective{F}, objective64) === objectiveF
        @test convert(AbstractObjective{Float64}, objectiveF) === objective64
        @test RegularizerObjective{F}(objective64) === objectiveF
        @test RegularizerObjective{Float64}(objectiveF) === objective64
    end
end
=#
