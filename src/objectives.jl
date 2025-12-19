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
Base.convert(::Type{P}, o::Q) where {P<:TraceObjective,Q<:TraceObjective} = P(o)

"""
    RegularizerObjective(value, type)
    RegularizerObjective(regularizer::Regularizer=Defaults.regularizer_target)
    RegularizerObjective{F}(regularizer::Regularizer) where {F<:Real}
    RegularizerObjective{F}(reg_obj::RegularizerObjective) where {F<:Real}

Set a target value for a regularizer when solving averaged resolvent equations

If `type` is provided, `value` should be a numerical value, otherwise it should be
a `Regularizer`. See [`Regularizer`](@ref) for the available regularizer types.
"""
struct RegularizerObjective{F} <: AbstractObjective{F}
    value::Regularizer{F}
end

RegularizerObjective(value, type) = RegularizerObjective(Regularizer(value, type))
RegularizerObjective() = RegularizerObjective(Defaults.regularizer_target)

RegularizerObjective{F}(o::RegularizerObjective) where {F} = RegularizerObjective{F}(o.value)

## Conversions
## ===========
Base.convert(::Type{P}, o::Q) where {P<:RegularizerObjective,Q<:RegularizerObjective} = P(o)

#=
## solution score
## ==============

score(solution, obj::TraceObjective) = score(log(solution.k), log(obj.value))
score(solution, obj::RegularizerObjective) = score(solution, obj.value)
=#

@testitem "TraceObjective" begin
    using GaussianSpectralDensities: Defaults
    trace_target64 = 1.0
    trace_objective64 = TraceObjective(trace_target64)

    # Traits
    @test trace_objective64 isa TraceObjective{Float64}

    # Defaults
    @test TraceObjective() === TraceObjective(Defaults.trace_target)

    @testset "$F" for F in [Float32, Float64]
        trace_targetF = F(trace_target64)
        trace_objectiveF = TraceObjective(trace_targetF)

        # Floattype inferring constructor
        @test trace_objectiveF isa TraceObjective{F}

        # Floattype enforcing constructor
        @test TraceObjective{F}(trace_target64) === trace_objectiveF
        @test TraceObjective{Float64}(trace_targetF) === trace_objective64

        # Floattype conversions
        @test convert(TraceObjective{F}, trace_objective64) === trace_objectiveF
        @test convert(TraceObjective{Float64}, trace_objectiveF) === trace_objective64
        @test TraceObjective{F}(trace_objective64) === trace_objectiveF
        @test TraceObjective{Float64}(trace_objectiveF) === trace_objective64
    end
end

@testitem "RegularizerObjective" begin
    using GaussianSpectralDensities: Defaults, Regularizer

    regularizer_value64, regularizer_type = 0.5, :gap
    regularizer_target64 = Regularizer(regularizer_value64, regularizer_type)
    regularizer_objective64 = RegularizerObjective(regularizer_target64)

    # Traits
    @test regularizer_objective64 isa RegularizerObjective{Float64}

    # Defaults
    @test RegularizerObjective() === RegularizerObjective(Defaults.regularizer_target)

    @testset "$F" for F in [Float32, Float64]
        regularizer_valueF = F(regularizer_value64)
        regularizer_targetF = Regularizer(regularizer_valueF, regularizer_type)
        regularizer_objectiveF = RegularizerObjective(regularizer_targetF)

        # Floattype inferring constructor
        @test regularizer_objectiveF isa RegularizerObjective{F}
        # - All the signatures
        @test RegularizerObjective(regularizer_valueF, regularizer_type) === regularizer_objectiveF

        # Floattype enforcing constructor
        @test RegularizerObjective{F}(regularizer_target64) === regularizer_objectiveF
        @test RegularizerObjective{Float64}(regularizer_targetF) === regularizer_objective64

        # Floattype conversions
        @test convert(RegularizerObjective{F}, regularizer_objective64) === regularizer_objectiveF
        @test convert(RegularizerObjective{Float64}, regularizer_objectiveF) === regularizer_objective64
        @test RegularizerObjective{F}(regularizer_objective64) === regularizer_objectiveF
        @test RegularizerObjective{Float64}(regularizer_objectiveF) === regularizer_objective64
    end
end
