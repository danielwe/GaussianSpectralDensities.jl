module Constraints

using AutoHashEquals: @auto_hash_equals

using ..ArrayTools: ArrayTools

public AbstractConstraint, PTrace

"""
    AbstractConstraint{F<:Real}

Abstract supertype for constraint specifications in point problems.

Concrete subtypes:  [`PTrace`](@ref), [`Regularizer`](@ref) (defined in a separate module)
"""
abstract type AbstractConstraint{F<:Real} end

"""
    PTrace(value)
    PTrace{F}(value) where {F}
    PTrace{F}(ptrace::PTrace) where {F}

Specify a target value for the P-trace.
"""
@auto_hash_equals struct PTrace{F} <: AbstractConstraint{F}
    value::F
end

PTrace{F}(ptrace::PTrace) where {F} = PTrace{F}(ptrace.value)

## Conversions
## ===========
function Base.convert(::Type{PTrace{F}}, ptrace::PTrace) where {F}
    return (ptrace isa PTrace{F}) ? ptrace : PTrace{F}(ptrace)
end

ArrayTools.adaptrealtype(::Type{F}, ptrace::PTrace) where {F} = convert(PTrace{F}, ptrace)

end  # module Constraints

@testitem "PTrace" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Constraints: AbstractConstraint, PTrace

    ptrace64 = PTrace(1.0)

    @test ptrace64 isa AbstractConstraint{Float64}
    @test ptrace64 isa PTrace{Float64}

    @testset "$F" for F in [Float32, Float64]
        ptraceF = PTrace{F}(1.0)

        # Traits
        @test ptraceF isa AbstractConstraint{F}
        @test ptraceF isa PTrace{F}

        # Realtype inferring constructor
        @test PTrace(one(F)) isa PTrace{F}

        # Realtype enforcing constructor
        @test PTrace{F}(1.0) == ptraceF
        @test PTrace{Float64}(PTrace(one(F))) == ptrace64

        # Realtype conversions
        @test PTrace{F}(ptrace64) == ptraceF
        @test PTrace{Float64}(ptraceF) == ptrace64
        @test convert(PTrace{F}, ptrace64) == ptraceF
        @test convert(PTrace{Float64}, ptraceF) == ptrace64
        @test adaptrealtype(F, ptrace64) == ptraceF
        @test adaptrealtype(Float64, ptraceF) == ptrace64
    end
end
