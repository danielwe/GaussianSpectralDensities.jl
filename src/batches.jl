module Batches

using AutoHashEquals: @auto_hash_equals
using StructArrays: StructArray

using ..Problems: AbstractConstraint, PointProblem, ResolventEquation
using ..Spreads: Spread, Spreads
using ..Solutions: PointSolution, ResolventParams, SolutionStatus, Solutions

using ..ArrayTools: adaptrealtype

public BatchProblem, BatchSolution

## BatchProblem
## =============
abstract type AbstractBatch{F,N} <: AbstractArray{F,N} end

Base.IndexStyle(::Type{<:AbstractBatch}) = IndexLinear()
Base.size(d::AbstractBatch) = size(d.zs)

@auto_hash_equals typearg=true struct BatchProblem{
    F,N,E<:ResolventEquation{F},C<:AbstractConstraint{F}
} <: AbstractBatch{PointProblem{F,E,C},N}
    equation::E
    constraint::C
    zs::Array{Complex{F},N}

    function BatchProblem{F}(equation, constraint, zs) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        constraintF = adaptrealtype(F, constraint)
        return new{F,ndims(zs),typeof(eqF),typeof(constraintF)}(eqF, constraintF, zs)
    end
end

function BatchProblem(equation::ResolventEquation{F}, constraint, zs) where {F}
    return BatchProblem{F}(equation, constraint, zs)
end

function Base.getindex(prob::BatchProblem, I)
    return PointProblem(prob.equation, prob.constraint, prob.zs[I])
end

## BatchSolution
## ==============
@auto_hash_equals typearg=true struct BatchSolution{
    F,N,E<:ResolventEquation{F},P<:StructArray{ResolventParams{F},N}
} <: AbstractBatch{PointSolution{F,E},N}
    equation::E
    zs::Array{Complex{F},N}
    params::P
    status::Array{SolutionStatus.T,N}

    function BatchSolution{F}(equation, zs) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        s = size(zs)
        params = StructArray{ResolventParams{F}}(undef, s)
        status = fill(SolutionStatus.not_finite, s)
        return new{F,ndims(zs),typeof(eqF),typeof(params)}(eqF, zs, params, status)
    end
end

BatchSolution(equation::ResolventEquation{F}, zs) where {F} = BatchSolution{F}(equation, zs)

function Base.getindex(sol::BatchSolution{F,N,E}, I) where {F,N,E}
    (; equation, zs, params, status) = sol
    return PointSolution{F,E}(equation, zs[I], params[I], status[I])
end

function Base.setindex!(sol::BatchSolution, ps::PointSolution, I)
    if ps.equation !== sol.equation
        msg = "Cannot assign to BatchSolution using PointSolution for different equation"
        throw(ArgumentError(msg))
    end
    zI = sol.zs[I]
    if ps.z != zI
        msg = "PointSolution z=$(ps.z) does not match z[i] = $zI"
        throw(ArgumentError(msg))
    end
    sol.params[I] = ps.params
    sol.status[I] = ps.status
    return sol
end

## Accessors for BatchSolution
## ----------------------------
## We had to choose between having an abstract solution type to avoid duplicated accessor
## implementations, or subtyping AbstractArray to get full-featured iteration and indexing
## (complete with axes, eachindex, et cetera) for free. We chose the latter.
"""
    issuccess(sol::BatchSolution) -> Vector{Bool}

Return whether the solution status indicates success

A successful solution is finite, self-consistent, and satisfies shiftmin.
"""
function Solutions.issuccess(sol::BatchSolution)
    return map(==(SolutionStatus.success), Solutions.status(sol))
end

"""
    shape(sol::BatchSolution{F}) -> Vector{Complex{F}}

Return the self-consistent shape parameter from the solution.
"""
Solutions.shape(sol::BatchSolution) = sol.params.shape

"""
    ptrace(sol::BatchSolution{F}) -> Vector{F}

Return the trace parameter `ptrace` from the solution.
"""
Solutions.ptrace(sol::BatchSolution) = sol.params.ptrace

"""
    greens(sol::BatchSolution{F}) -> Vector{Complex{F}}

Return the value of the Green's function (resolvent trace) from the solution.
"""
Solutions.greens(sol::BatchSolution) = sol.params.greens

"""
    status(sol::BatchSolution{F}) -> Vector{SolutionStatus.T}

Return the status of the solution.
"""
Solutions.status(sol::BatchSolution) = sol.status

"""
    shift(sol::BatchSolution{F}) -> Vector{F}

Return the spectral shift from the solution.
"""
Spreads.shift(sol::BatchSolution) = sol.params.shift

"""
    gap(sol::BatchSolution{F}) -> Vector{F}

Return the spectral gap from the solution.
"""
Spreads.gap(sol::BatchSolution) = sol.params.gap

"""
    spreadfrom(sol::BatchSolution{F}, type) -> Vector{Spread{F}}

Create a spread parameter of the given `type` from the solution.
"""
function Solutions.spreadfrom(sol::BatchSolution, type)
    return map(spread -> Spread(spread, type), getproperty(sol.params, Symbol(type)))
end

end  # module Batches

@testitem "BatchProblem" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Batches: BatchProblem
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using LinearAlgebra: I

    zs = complex.(0.0:0.5:1.0, (0.0:0.5:1.0)')
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))
    constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))

    prob = BatchProblem(eq, constraint, zs)

    @test prob isa BatchProblem{Float64}
    @test IndexStyle(typeof(prob)) == IndexLinear()
    @test length(prob) == length(zs)

    pp = prob[1]
    @test pp isa PointProblem{Float64}
    @test pp.z == zs[1]
    @test pp.equation == eq
    @test pp.constraint == constraint
end

@testitem "BatchSolution" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Batches: BatchSolution
    using GaussianSpectralDensities.Problems: ResolventEquation
    using GaussianSpectralDensities.Spreads: Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        PointSolution,
        ResolventParams,
        SolutionStatus,
        greens,
        issuccess,
        ptrace,
        shape,
        spreadfrom
    using LinearAlgebra: I

    zs = complex.(0.0:0.5:1.0, (0.0:0.5:1.0)')
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    sol = BatchSolution(eq, zs)

    @test sol isa BatchSolution{Float64}
    @test IndexStyle(typeof(sol)) == IndexLinear()
    @test length(sol) == length(zs)
    @test !any(issuccess(sol))

    params = ResolventParams(0.5 + 0.5im, 0.25, 0.5, 1.0, -0.5 + 2.0im)
    sol.params[1] = params
    sol.status[1] = SolutionStatus.success

    @test sol.params[1] == params
    @test issuccess(sol)[1]

    # setindex! with PointSolution
    z1 = zs[1]
    ps = PointSolution(eq, z1, 1.0, Spread(; shift=1.0))
    sol[1] = ps
    @test sol.params[1] == ps.params
    @test sol.status[1] == ps.status

    # getindex
    @test sol[1] == ps

    # setindex! with mismatched eq or z throws ArgumentError
    eq2 = ResolventEquation(eq.ensemble; shiftmin=0)
    z2 = zs[2]
    ps_wrongA = PointSolution(eq2, z1, 1.0, Spread(; shift=1.0))
    ps_wrongB = PointSolution(eq, z2, 1.0, Spread(; shift=1.0))
    @test_throws ArgumentError sol[1] = ps_wrongA
    @test_throws ArgumentError sol[1] = ps_wrongB

    # accessors
    sol.params.shape .= ComplexF64[0.9 0.8 0.7; 0.6 0.5 0.4; 0.3 0.2 0.1]
    sol.params.ptrace .= [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    sol.params.greens .= ComplexF64[0.9 0.8 0.7; 0.6 0.5 0.4; 0.3 0.2 0.1]
    sol.params.shift .= [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    sol.params.gap .= [0.9 0.8 0.7; 0.6 0.5 0.4; 0.3 0.2 0.1]
    @test shape(sol) == sol.params.shape
    @test ptrace(sol) == sol.params.ptrace
    @test greens(sol) == sol.params.greens
    @test shift(sol) == sol.params.shift
    @test gap(sol) == sol.params.gap
    spread_shift = spreadfrom(sol, :shift)
    spread_gap = spreadfrom(sol, :gap)
    @test length(spread_shift) == length(sol)
    @test length(spread_gap) == length(sol)
    @test all(r -> r isa Spread, spread_shift)
    @test all(r -> r isa Spread, spread_gap)
    @test all(r -> Symbol(r.type) == :shift, spread_shift)
    @test all(r -> Symbol(r.type) == :gap, spread_gap)
    @test [r.val for r in spread_shift] == sol.params.shift
    @test [r.val for r in spread_gap] == sol.params.gap
end
