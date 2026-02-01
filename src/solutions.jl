module Solutions

using AutoHashEquals: @auto_hash_equals

using ..Defaults: approxequal
using ..Problems: PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
using ..Workspaces: Workspace, gap!, greens!, ptrace!, qFP!, shift!

using ..ArrayTools: ArrayTools

public PointSolution, ResolventParams, solves

"""
    ResolventParams(q, shift, gap, ptrace, greens)
    ResolventParams{F}(q, shift, gap, ptrace, greens) where {F}
    ResolventParams{F}(params::ResolventParams) where {F}

Parameter values for a self-consistent resolvent solution.
"""
@auto_hash_equals struct ResolventParams{F<:AbstractFloat}
    q::Complex{F}
    shift::F
    gap::F
    ptrace::F
    greens::Complex{F}
end

function ResolventParams{F}(p::ResolventParams) where {F}
    return ResolventParams{F}(p.q, p.shift, p.gap, p.ptrace, p.greens)
end

## Conversions
## ===========
function Base.convert(::Type{ResolventParams{F}}, p::ResolventParams) where {F}
    return (p isa ResolventParams{F}) ? p : ResolventParams{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, p::ResolventParams) where {F}
    return convert(ResolventParams{F}, p)
end

"""
    PointSolution(equation, z, q, reg; workspace=Workspace(equation))
    PointSolution{F}(equation, z, q, reg, workspace) where {F}
    PointSolution{F}(s::PointSolution) where {F}

Represent a self-consistent solution of an averaged resolvent equation at a point.

# Fields

  - `equation`: resolvent equation defining the shared context
  - `z`: point in the complex plane
  - `params`: derived parameter values (q, shift, gap, ptrace, greens)
  - `success`: whether the derived parameters are finite, self-consistent, and satisfy shiftmin
"""
@auto_hash_equals typearg=true struct PointSolution{F,E<:ResolventEquation{F}}
    equation::E
    z::Complex{F}
    params::ResolventParams{F}
    success::Bool

    function PointSolution{F}(equation, z, q, reg, workspace) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        shiftF = shift!(workspace, eqF, z, q, reg)
        gapF = gap!(workspace, eqF, z, q, reg)
        ptraceF = ptrace!(workspace, eqF, z, q, reg)
        greensF = greens!(workspace, eqF, z, q, reg)
        paramsF = ResolventParams{F}(q, shiftF, gapF, ptraceF, greensF)
        success =
            all(isfinite, (shiftF, gapF, ptraceF, greensF)) &&
            (shiftF >= eqF.shiftmin) &&
            approxequal(qFP!(workspace, eqF, z, q, reg), q)
        return new{F,typeof(eqF)}(eqF, z, paramsF, success)
    end

    function PointSolution{F}(s::PointSolution) where {F}
        # Direct conversion instead of redoing calculations may lead to rounding error
        # differences between solutions that are otherwise the same. This is OK---it was
        # already true for solutions instantiated with :shift vs. :gap regularizers but
        # otherwise the same.
        eqF = convert(ResolventEquation{F}, s.equation)
        return new{F,typeof(eqF)}(eqF, s.z, s.params, s.success)
    end
end

function PointSolution(
    equation::ResolventEquation{F}, z, q, reg; workspace=Workspace(equation)
) where {F}
    return PointSolution{F}(equation, z, q, reg, workspace)
end

## Conversions
## ===========
function Base.convert(::Type{PointSolution{F}}, s::PointSolution) where {F}
    return (s isa PointSolution{F}) ? s : PointSolution{F}(s)
end

function ArrayTools.adaptrealtype(::Type{F}, s::PointSolution) where {F}
    return convert(PointSolution{F}, s)
end

## Consistency
## ===========
function solves(solution::PointSolution, problem::PointProblem)
    solution.success || return false
    (solution.equation === problem.equation) || return false
    (solution.z === problem.z) || return false
    return satisfies(solution.params, problem.constraint)
end

function satisfies(params::ResolventParams, constraint::RegularizerConstraint)
    reg_val = getproperty(params, Symbol(constraint.val.type))
    return approxequal(reg_val, constraint.val.val)
end

function satisfies(params::ResolventParams, constraint::PTraceConstraint)
    return approxequal(params.ptrace, constraint.val)
end

end  # module Solutions

@testitem "ResolventParams" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Solutions: ResolventParams

    q = complex(0.25, -0.125)
    shift = 0.25
    gap = 0.5
    ptrace = 1.0
    greens = -0.5 + 2.0im

    params64 = ResolventParams(q, shift, gap, ptrace, greens)

    @testset "$F" for F in [Float32, Float64]
        paramsF = ResolventParams{F}(q, shift, gap, ptrace, greens)

        # Comparison, courtesy of @auto_hash_equals
        @test paramsF == params64

        # Floattype conversions
        @test ResolventParams{F}(params64) === paramsF
        @test ResolventParams{Float64}(paramsF) === params64
        @test convert(ResolventParams{F}, params64) === paramsF
        @test convert(ResolventParams{Float64}, paramsF) === params64
        @test adaptrealtype(F, params64) === paramsF
        @test adaptrealtype(Float64, paramsF) === params64
    end
end

@testitem "PointSolution" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Defaults: approxequal
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation, update
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution, solves
    using GaussianSpectralDensities.Workspaces:
        Workspace, gap!, greens!, ptrace!, qFP!, shift!

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    ws64 = Workspace(eq64)

    # For an N=1 problem we always have gap * ptrace = 1, so it's easy to define constraints
    # that are satisfied simultaneously
    reg_type = :gap
    reg = Regularizer(0.5, reg_type)

    # We're relying on some very particular parameter tuning here to create a solution we
    # know a priori to be consistent and to survive roundtrip floattype conversion
    z = 0.25 + 0.25im
    q = -0.5 - 0.5im

    solution64 = PointSolution(eq64, z, q, reg; workspace=ws64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(ensemble64)
        eqF = ResolventEquation(ensembleF)
        wsF = Workspace(eqF)

        solutionF = PointSolution(eqF, z, q, reg; workspace=wsF)

        # Type and fields
        @test solutionF isa PointSolution{F}
        @test solutionF.z == z
        @test solutionF.success

        solF_par = solutionF.params
        @test solF_par.q == q

        solF_reg = Regularizer(getproperty(solF_par, reg_type), reg_type)
        @test solF_reg == reg

        @test solF_par.shift == shift!(wsF, eqF, solutionF.z, solF_par.q, solF_reg)
        @test solF_par.gap == gap!(wsF, eqF, solutionF.z, solF_par.q, solF_reg)
        @test solF_par.ptrace == ptrace!(wsF, eqF, solutionF.z, solF_par.q, solF_reg)
        @test solF_par.greens == greens!(wsF, eqF, solutionF.z, solF_par.q, solF_reg)

        @test approxequal(solF_par.q, qFP!(wsF, eqF, solutionF.z, solF_par.q, solF_reg))

        # Actually solves the problems we expect it to
        problem_regF = PointProblem(eqF, RegularizerConstraint(reg, q), z)
        problem_ptraceF = PointProblem(eqF, PTraceConstraint(2.0, q, reg), z)
        @test solves(solutionF, problem_regF)
        @test solves(solutionF, problem_ptraceF)

        # Failure due to inconsistent parameters
        solutionF_inconsistent = PointSolution(eqF, z, zero(q), reg; workspace=wsF)
        @test !solutionF_inconsistent.success

        # Failure due to NaN
        solutionF_nan = PointSolution(eqF, z, oftype(q, NaN), reg; workspace=wsF)
        @test !solutionF_nan.success

        # Failure due to shiftmin (with the given params we know that shift = 0
        eqF_impossible = update(eqF; shiftmin=1.0)
        solutionF_impossible = PointSolution(eqF_impossible, z, q, reg)
        @test !solutionF_impossible.success

        # Floattype conversions
        @test PointSolution{F}(solution64) == solutionF
        @test PointSolution{Float64}(solutionF) == solution64
        @test convert(PointSolution{F}, solution64) == solutionF
        @test convert(PointSolution{Float64}, solutionF) == solution64
        @test adaptrealtype(F, solution64) == solutionF
        @test adaptrealtype(Float64, solutionF) == solution64
    end
end
