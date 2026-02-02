module Solutions

using AutoHashEquals: @auto_hash_equals

using ..Defaults: approxequal
using ..Problems: PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
using ..Spreads: Spread, Spreads
using ..Workspaces: Workspace, gap!, greens!, ptrace!, qFP!, shift!

using ..ArrayTools: ArrayTools

public PointSolution, ResolventParams, greens, issuccess, ptrace, q, solves, spreadfrom

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

## Accessors
## =========
"""
    spreadfrom(params::ResolventParams, type) -> Spread

Create a spread parameter of the given `type` from the parameters.
"""
spreadfrom(params::ResolventParams, type) = Spread(getproperty(params, Symbol(type)), type)

## Conversions
## ===========
function Base.convert(::Type{ResolventParams{F}}, p::ResolventParams) where {F}
    return (p isa ResolventParams{F}) ? p : ResolventParams{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, p::ResolventParams) where {F}
    return convert(ResolventParams{F}, p)
end

## Consistency
## ===========
function satisfies(params::ResolventParams, constraint::SpreadConstraint)
    cspread = constraint.val
    spread = spreadfrom(params, cspread.type)
    return approxequal(spread.val, cspread.val)
end

function satisfies(params::ResolventParams, constraint::PTraceConstraint)
    return approxequal(params.ptrace, constraint.val)
end

"""
    PointSolution(equation, z, q, spread; workspace=Workspace(equation))
    PointSolution{F}(equation, z, q, spread, workspace) where {F}
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

    # Secret constructor that doesn't assert invariants, mainly to support GlobalSolution
    # getindex
    function PointSolution{F,E}(
        equation::E, z, params, success
    ) where {F,E<:ResolventEquation{F}}
        return new{F,E}(equation, z, params, success)
    end

    function PointSolution{F}(equation, z, q, spread, workspace) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        shiftF = shift!(workspace, eqF, z, q, spread)
        gapF = gap!(workspace, eqF, z, q, spread)
        ptraceF = ptrace!(workspace, eqF, z, q, spread)
        greensF = greens!(workspace, eqF, z, q, spread)
        paramsF = ResolventParams{F}(q, shiftF, gapF, ptraceF, greensF)
        # FIXME: we should distinguish between failure by shiftmin (where the parameters are
        # still consistent, it's more of a "you may come this far, but no farther" signpost)
        # and failure by q != qFP, where the parameters are not self-consistent. Should we
        # just manually set them to NaN in the latter case?
        #
        # Or should we move shiftmin out of the equation spec and let
        # success == self-consistent? Setting a shiftmin and declaring "you shall not pass"
        # would then be the purview of the algorithm, not the problem or solution. This
        # makes sense because, as I'm discovering (see FIXME in algorithms.jl), shiftmin is
        # only actually reveltn for SpreadSchedule, not even SymmetrySchedule.
        success =
            all(isfinite, (shiftF, gapF, ptraceF, greensF)) &&
            (shiftF >= eqF.shiftmin) &&
            approxequal(qFP!(workspace, eqF, z, q, spread), q)
        return new{F,typeof(eqF)}(eqF, z, paramsF, success)
    end

    function PointSolution{F}(s::PointSolution) where {F}
        # This direct conversion may result in rounding differences compared to redoing the
        # calculation of derived parameters in the new precision. This is OK---it was
        # already true for otherwise identical solutions instantiated with spread.type being
        # :shift vs. :gap.
        eqF = convert(ResolventEquation{F}, s.equation)
        return new{F,typeof(eqF)}(eqF, s.z, s.params, s.success)
    end
end

function PointSolution(
    equation::ResolventEquation{F}, z, q, spread; workspace=Workspace(equation)
) where {F}
    return PointSolution{F}(equation, z, q, spread, workspace)
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
    issuccess(solution) || return false
    ((solution.equation === problem.equation) && (solution.z == problem.z)) || return false
    return satisfies(solution.params, problem.constraint)
end

## Accessors
## =========
"""
    issuccess(sol::PointSolution) -> Bool

Return whether the solution is finite, self-consistent, and satisfies shiftmin.
"""
issuccess(sol::PointSolution) = sol.success

"""
    q(sol::PointSolution{F}) -> Complex{F}

Return the self-consistency parameter `q` from the solution.
"""
q(sol::PointSolution) = sol.params.q

"""
    ptrace(sol::PointSolution{F}) -> F

Return the trace parameter `ptrace` from the solution.
"""
ptrace(sol::PointSolution) = sol.params.ptrace

"""
    greens(sol::PointSolution{F}) -> Complex{F}

Return the value of the Green's function (resolvent trace) from the solution.
"""
greens(sol::PointSolution) = sol.params.greens

"""
    shift(sol::PointSolution{F}) -> F

Return the spectral shift from the solution.
"""
Spreads.shift(sol::PointSolution) = sol.params.shift

"""
    gap(sol::PointSolution{F}) -> F

Return the spectral gap from the solution.
"""
Spreads.gap(sol::PointSolution) = sol.params.gap

"""
    spreadfrom(sol::PointSolution{F}, type) -> Spread{F}

Create a spread parameter of the given `type` from the solution.
"""
spreadfrom(sol::PointSolution, type) = spreadfrom(sol.params, type)

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
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint, update
    using GaussianSpectralDensities.Spreads: Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        PointSolution, greens, issuccess, ptrace, q, solves, spreadfrom
    using GaussianSpectralDensities.Workspaces:
        Workspace, gap!, greens!, ptrace!, qFP!, shift!

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    ws64 = Workspace(eq64)

    # For an N=1 problem we always have gap * ptrace = 1, so it's easy to define constraints
    # that are satisfied simultaneously
    spread_type = :gap
    spread64 = Spread(0.5, spread_type)
    ptrace64 = 2.0

    # We're relying on some very particular parameter tuning here to create a solution we
    # know a priori to be consistent and to survive roundtrip floattype conversion
    z64 = 0.25 + 0.25im
    q64 = -0.5 - 0.5im

    solution64 = PointSolution(eq64, z64, q64, spread64; workspace=ws64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(ensemble64)
        eqF = ResolventEquation(ensembleF)
        wsF = Workspace(eqF)

        solutionF = PointSolution(eqF, z64, q64, spread64; workspace=wsF)

        # Type and traits
        @test solutionF isa PointSolution{F}
        @test issuccess(solutionF)
        solF_q, solF_spread = q(solutionF), spreadfrom(solutionF, spread_type)
        @test solF_q == q64
        @test solF_spread == spread64

        @test shift(solutionF) == shift!(wsF, eqF, solutionF.z, solF_q, solF_spread)
        @test gap(solutionF) == gap!(wsF, eqF, solutionF.z, solF_q, solF_spread)
        @test ptrace(solutionF) == ptrace!(wsF, eqF, solutionF.z, solF_q, solF_spread)
        @test greens(solutionF) == greens!(wsF, eqF, solutionF.z, solF_q, solF_spread)

        @test approxequal(solF_q, qFP!(wsF, eqF, solutionF.z, solF_q, solF_spread))

        # Actually solves the problems we expect it to
        problem_spreadF = PointProblem(eqF, SpreadConstraint(spread64, q64), z64)
        problem_ptraceF = PointProblem(eqF, PTraceConstraint(ptrace64, q64, spread64), z64)
        @test solves(solutionF, problem_spreadF)
        @test solves(solutionF, problem_ptraceF)

        # Failure due to inconsistent parameters
        solutionF_inconsistent = PointSolution(eqF, z64, zero(q64), spread64; workspace=wsF)
        @test !issuccess(solutionF_inconsistent)

        # Failure due to NaN
        solutionF_nan = PointSolution(eqF, z64, oftype(q64, NaN), spread64; workspace=wsF)
        @test !issuccess(solutionF_nan)

        # Failure due to shiftmin (with the given params we know that shift = 0)
        eqF_impossible = update(eqF; shiftmin=1.0)
        solutionF_impossible = PointSolution(eqF_impossible, z64, q64, spread64)
        @test !issuccess(solutionF_impossible)

        # Floattype conversions
        @test PointSolution{F}(solution64) == solutionF
        @test PointSolution{Float64}(solutionF) == solution64
        @test convert(PointSolution{F}, solution64) == solutionF
        @test convert(PointSolution{Float64}, solutionF) == solution64
        @test adaptrealtype(F, solution64) == solutionF
        @test adaptrealtype(Float64, solutionF) == solution64
    end
end

@testitem "PointSolution accessors" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Spreads: Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        PointSolution, greens, issuccess, ptrace, q, spreadfrom
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(I, I, I, I / 2)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)

    spread_type = :gap
    spread = Spread(0.5, spread_type)
    z = 0.25 + 0.25im
    q_val = -0.5 - 0.5im

    solution = PointSolution(eq, z, q_val, spread; workspace=ws)

    # Test accessors
    @test issuccess(solution) == solution.success
    @test q(solution) == solution.params.q
    @test ptrace(solution) == solution.params.ptrace
    @test greens(solution) == solution.params.greens
    @test shift(solution) == solution.params.shift
    @test gap(solution) == solution.params.gap
    @test spreadfrom(solution, :shift) == Spread(solution.params.shift, :shift)
    @test spreadfrom(solution, :gap) == Spread(solution.params.gap, :gap)
end
