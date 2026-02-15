module Solutions

using AutoHashEquals: @auto_hash_equals

using ..Defaults: approxequal, intheballpark
using ..Problems: PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
using ..Spreads: Spread, Spreads
using ..Workspaces: Workspace, gap!, greens!, ptrace!, shapeFP!, shift!

using ..ArrayTools: ArrayTools

using EnumX: @enumx

public PointSolution,
    ResolventParams,
    SolutionStatus,
    greens,
    issuccess,
    ptrace,
    shape,
    solves,
    spreadfrom,
    status

"""
    ResolventParams(shape, shift, gap, ptrace, greens)
    ResolventParams{F}(shape, shift, gap, ptrace, greens) where {F}
    ResolventParams{F}(params::ResolventParams) where {F}

Parameter values for a self-consistent resolvent solution.
"""
@auto_hash_equals struct ResolventParams{F<:AbstractFloat}
    shape::Complex{F}
    shift::F
    gap::F
    ptrace::F
    greens::Complex{F}
end

function ResolventParams{F}(p::ResolventParams) where {F}
    return ResolventParams{F}(p.shape, p.shift, p.gap, p.ptrace, p.greens)
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
function satisfies(params::ResolventParams{F}, constraint::SpreadConstraint) where {F}
    cspread = constraint.val
    spread = spreadfrom(params, cspread.type)
    return approxequal(F, spread.val, cspread.val)
end

function satisfies(params::ResolventParams{F}, constraint::PTraceConstraint) where {F}
    return approxequal(F, params.ptrace, constraint.val)
end

@enumx SolutionStatus::UInt8 success shift_below_min not_consistent not_intheballpark not_finite

"""
    PointSolution(equation, z, shape, spread; workspace=Workspace(equation))
    PointSolution{F}(equation, z, shape, spread, workspace) where {F}
    PointSolution{F}(s::PointSolution) where {F}

Represent a self-consistent solution of an averaged resolvent equation at a point.

# Fields

  - `equation`: resolvent equation defining the shared context
  - `z`: point in the complex plane
  - `params`: derived parameter values (shape, shift, gap, ptrace, greens)
  - `success`: whether the derived parameters are finite, self-consistent, and satisfy shiftmin
"""
@auto_hash_equals typearg=true struct PointSolution{F,E<:ResolventEquation{F}}
    equation::E
    z::Complex{F}
    params::ResolventParams{F}
    status::SolutionStatus.T

    # Secret constructor that doesn't assert invariants, mainly to support GlobalSolution
    # getindex
    function PointSolution{F,E}(
        equation::E, z, params, status
    ) where {F,E<:ResolventEquation{F}}
        return new{F,E}(equation, z, params, status)
    end

    function PointSolution{F}(equation, z, shape, spread, workspace) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        shiftF = shift!(workspace, eqF, z, shape, spread)
        gapF = gap!(workspace, eqF, z, shape, spread)
        ptraceF = ptrace!(workspace, eqF, z, shape, spread)
        greensF = greens!(workspace, eqF, z, shape, spread)
        paramsF = ResolventParams{F}(shape, shiftF, gapF, ptraceF, greensF)
        status = if !all(isfinite, (shiftF, gapF, ptraceF, greensF))
            SolutionStatus.not_finite
        else
            sFP = shapeFP!(workspace, eqF, z, shape, spread)
            if !intheballpark(F, sFP, shape)
                SolutionStatus.not_intheballpark
            elseif !approxequal(F, sFP, shape)
                SolutionStatus.not_consistent
            elseif shiftF < eqF.shiftmin
                SolutionStatus.shift_below_min
            else
                SolutionStatus.success
            end
        end
        return new{F,typeof(eqF)}(eqF, z, paramsF, status)
    end

    function PointSolution{F}(s::PointSolution) where {F}
        # This direct conversion may result in rounding differences compared to redoing the
        # calculation of derived parameters in the new precision. This is OK---it was
        # already true for otherwise identical solutions instantiated with spread.type being
        # :shift vs. :gap.
        eqF = convert(ResolventEquation{F}, s.equation)
        return new{F,typeof(eqF)}(eqF, s.z, s.params, s.status)
    end
end

function PointSolution(
    equation::ResolventEquation{F}, z, shape, spread; workspace=Workspace(equation)
) where {F}
    return PointSolution{F}(equation, z, shape, spread, workspace)
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

Return whether the solution status indicates success

A successful solution is finite, self-consistent, and satisfies shiftmin.
"""
issuccess(sol::PointSolution) = (status(sol) == SolutionStatus.success)

"""
    shape(sol::PointSolution{F}) -> Complex{F}

Return the self-consistent shape parameter from the solution.
"""
shape(sol::PointSolution) = sol.params.shape

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
    status(sol::PointSolution{F}) -> SolutionStatus.T

Return the status of the solution.
"""
status(sol::PointSolution) = sol.status

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

    shape = complex(0.25, -0.125)
    shift = 0.25
    gap = 0.5
    ptrace = 1.0
    greens = -0.5 + 2.0im

    params64 = ResolventParams(shape, shift, gap, ptrace, greens)

    @testset "$F" for F in [Float32, Float64]
        paramsF = ResolventParams{F}(shape, shift, gap, ptrace, greens)

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
    using GaussianSpectralDensities.Defaults: intheballpark
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint, update
    using GaussianSpectralDensities.Spreads: Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        PointSolution,
        SolutionStatus,
        greens,
        issuccess,
        ptrace,
        shape,
        solves,
        spreadfrom,
        status
    using GaussianSpectralDensities.Workspaces:
        Workspace, gap!, greens!, ptrace!, shapeFP!, shift!

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
    shape64 = -0.5 - 0.5im

    solution64 = PointSolution(eq64, z64, shape64, spread64; workspace=ws64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(ensemble64)
        eqF = ResolventEquation(ensembleF)
        wsF = Workspace(eqF)

        solutionF = PointSolution(eqF, z64, shape64, spread64; workspace=wsF)

        # Type and traits
        @test solutionF isa PointSolution{F}
        @test issuccess(solutionF)
        solF_shape, solF_spread = shape(solutionF), spreadfrom(solutionF, spread_type)
        @test solF_shape == shape64
        @test solF_spread == spread64

        @test shift(solutionF) == shift!(wsF, eqF, solutionF.z, solF_shape, solF_spread)
        @test gap(solutionF) == gap!(wsF, eqF, solutionF.z, solF_shape, solF_spread)
        @test ptrace(solutionF) == ptrace!(wsF, eqF, solutionF.z, solF_shape, solF_spread)
        @test greens(solutionF) == greens!(wsF, eqF, solutionF.z, solF_shape, solF_spread)

        solF_shapeFP = shapeFP!(wsF, eqF, solutionF.z, solF_shape, solF_spread)
        @test intheballpark(F, solF_shape, solF_shapeFP)

        # Actually solves the problems we expect it to
        spreadc = SpreadConstraint(spread64, shape64)
        ptracec = PTraceConstraint(ptrace64, shape64, spread64)
        problem_spreadF = PointProblem(eqF, spreadc, z64)
        problem_ptraceF = PointProblem(eqF, ptracec, z64)
        @test solves(solutionF, problem_spreadF)
        @test solves(solutionF, problem_ptraceF)

        # Failure due to inconsistent parameters
        zshape64 = zero(shape64)
        solutionF_inconsistent = PointSolution(eqF, z64, zshape64, spread64; workspace=wsF)
        @test !issuccess(solutionF_inconsistent)
        @test status(solutionF_inconsistent) in
            (SolutionStatus.not_consistent, SolutionStatus.not_intheballpark)

        # Failure due to NaN
        nanshape64 = oftype(shape64, NaN)
        solutionF_nan = PointSolution(eqF, z64, nanshape64, spread64; workspace=wsF)
        @test !issuccess(solutionF_nan)
        @test status(solutionF_nan) == SolutionStatus.not_finite

        # Failure due to shiftmin (with the given params we know that shift = 0)
        eqF_impossible = update(eqF; shiftmin=1.0)
        solutionF_impossible = PointSolution(eqF_impossible, z64, shape64, spread64)
        @test !issuccess(solutionF_impossible)
        @test status(solutionF_impossible) == SolutionStatus.shift_below_min

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
        PointSolution, SolutionStatus, greens, issuccess, ptrace, shape, spreadfrom, status
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(I, I, I, I / 2)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)

    spread_type = :gap
    spread = Spread(0.5, spread_type)
    z = 0.25 + 0.25im
    shape_val = -0.5 - 0.5im

    solution = PointSolution(eq, z, shape_val, spread; workspace=ws)

    # Test accessors
    @test issuccess(solution) == (solution.status == SolutionStatus.success)
    @test shape(solution) == solution.params.shape
    @test ptrace(solution) == solution.params.ptrace
    @test greens(solution) == solution.params.greens
    @test status(solution) == solution.status
    @test shift(solution) == solution.params.shift
    @test gap(solution) == solution.params.gap
    @test spreadfrom(solution, :shift) == Spread(solution.params.shift, :shift)
    @test spreadfrom(solution, :gap) == Spread(solution.params.gap, :gap)
end
