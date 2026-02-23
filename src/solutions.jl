module Solutions

using AutoHashEquals: @auto_hash_equals

using ..Defaults: approxequal, intheballpark
using ..Problems: PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
using ..Spreads: Spread, Spreads
using ..Workspaces: Workspace, gap!, greens!, ptrace!, shapeFP!, shift!

using ..ArrayTools: ArrayTools

using EnumX: @enumx
using StructArrays: StructArray

public PointSolution,
    BatchSolution,
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

## BatchSolution
## ==============
"""
    BatchSolution{F}(equation, zs)
    BatchSolution(equation::ResolventEquation{F}, zs)

Array of resolvent solutions over a grid of complex points `zs`, sharing a single
`ResolventEquation`. Subtypes `AbstractArray{PointSolution}` for convenient iteration,
indexing, and filtering.

Uses columnar/struct-of-arrays storage: the complex variable `z`, each resolvent parameter,
and the solution status are stored as contiguous, homogeneous arrays; indexing into the
`BatchSolution` constructs a `PointSolution` instance on the fly.

Scalar `getindex` returns a `PointSolution`; non-scalar `getindex` returns a
`BatchSolution` slice. `setindex!` validates that the assigned solution's equation and
z-value match the target slot.
"""
@auto_hash_equals typearg=true struct BatchSolution{
    F,N,E<:ResolventEquation{F},P<:StructArray{ResolventParams{F},N}
} <: AbstractArray{PointSolution{F,E},N}
    equation::E
    zs::Array{Complex{F},N}
    params::P
    status::Array{SolutionStatus.T,N}

    # Secret constructor that accepts data unchecked, mainly to support getindex
    function BatchSolution{F,N,E}(
        equation::E, zs, params, status
    ) where {F,N,E<:ResolventEquation{F}}
        return new{F,N,E,typeof(params)}(equation, zs, params, status)
    end

    function BatchSolution{F}(equation, zs) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        s = size(zs)
        params = StructArray{ResolventParams{F}}(undef, s)
        status = fill(SolutionStatus.not_finite, s)
        return new{F,ndims(zs),typeof(eqF),typeof(params)}(eqF, zs, params, status)
    end
end

BatchSolution(equation::ResolventEquation{F}, zs) where {F} = BatchSolution{F}(equation, zs)

Base.IndexStyle(::Type{<:BatchSolution}) = IndexLinear()
Base.size(sol::BatchSolution) = size(sol.zs)

function Base.getindex(sol::BatchSolution{F,N,E}, I::Integer...) where {F,N,E}
    (; equation, zs, params, status) = sol
    return PointSolution{F,E}(equation, zs[I...], params[I...], status[I...])
end

function Base.getindex(sol::BatchSolution{F,N,E}, I) where {F,N,E}
    (; equation, zs, params, status) = sol
    zsI, paramsI, statusI = zs[I], params[I], status[I]
    return BatchSolution{F,ndims(zsI),E}(equation, zsI, paramsI, statusI)
end

function Base.setindex!(sol::BatchSolution, ps::PointSolution, I::Integer...)
    if ps.equation !== sol.equation
        msg = "Cannot assign to BatchSolution using PointSolution for different equation"
        throw(ArgumentError(msg))
    end
    zI = sol.zs[I...]
    if ps.z != zI
        msg = "PointSolution z=$(ps.z) does not match z[I...] = $zI"
        throw(ArgumentError(msg))
    end
    sol.params[I...] = ps.params
    sol.status[I...] = ps.status
    return sol
end

function Base.setindex!(sol::BatchSolution, s::BatchSolution, I)
    if s.equation !== sol.equation
        msg = "Cannot assign to BatchSolution using BatchSolution for different equation"
        throw(ArgumentError(msg))
    end
    zI = sol.zs[I]
    if s.zs != zI
        msg = "BatchSolution zs=$(s.zs) does not match zs[I] = $zI"
        throw(ArgumentError(msg))
    end
    sol.params[I] = s.params
    sol.status[I] = s.status
    return sol
end

const AnySolution{F} = Union{PointSolution{F},BatchSolution{F}}

## Accessors
## =========
"""
    issuccess(sol::PointSolution) -> Bool
    issuccess(sol::BatchSolution) -> Array{Bool}

Return whether the solution status indicates success.

A successful solution is finite, self-consistent, and satisfies shiftmin.
"""
issuccess(sol::PointSolution) = (status(sol) == SolutionStatus.success)
issuccess(sol::BatchSolution) = map(==(SolutionStatus.success), status(sol))

"""
    shape(sol::PointSolution{F}) -> Complex{F}
    shape(sol::BatchSolution{F}) -> Array{Complex{F}}

Return the self-consistent shape parameter from the solution.
"""
shape(sol::AnySolution) = sol.params.shape

"""
    ptrace(sol::PointSolution{F}) -> F
    ptrace(sol::BatchSolution{F}) -> Array{F}

Return the trace parameter `ptrace` from the solution.
"""
ptrace(sol::AnySolution) = sol.params.ptrace

"""
    greens(sol::PointSolution{F}) -> Complex{F}
    greens(sol::BatchSolution{F}) -> Array{Complex{F}}

Return the value of the Green's function (resolvent trace) from the solution.
"""
greens(sol::AnySolution) = sol.params.greens

"""
    status(sol::PointSolution) -> SolutionStatus.T
    status(sol::BatchSolution) -> Array{SolutionStatus.T}

Return the status of the solution.
"""
status(sol::AnySolution) = sol.status

"""
    shift(sol::PointSolution{F}) -> F
    shift(sol::BatchSolution{F}) -> Array{F}

Return the spectral shift from the solution.
"""
Spreads.shift(sol::AnySolution) = sol.params.shift

"""
    gap(sol::PointSolution{F}) -> F
    gap(sol::BatchSolution{F}) -> Array{F}

Return the spectral gap from the solution.
"""
Spreads.gap(sol::AnySolution) = sol.params.gap

"""
    spreadfrom(sol::PointSolution{F}, type) -> Spread{F}
    spreadfrom(sol::BatchSolution{F}, type) -> Array{Spread{F}}

Create a spread parameter of the given `type` from the solution.
"""
spreadfrom(sol::PointSolution, type) = spreadfrom(sol.params, type)
function spreadfrom(sol::BatchSolution, type)
    return map(v -> Spread(v, type), getproperty(sol.params, Symbol(type)))
end

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
    using GaussianSpectralDensities: MatrixEnsemble, Spread, gap, shift
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Defaults: intheballpark
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint, update
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
    using GaussianSpectralDensities:
        MatrixEnsemble,
        PointProblem,
        ResolventEquation,
        Spread,
        SpreadConstraint,
        gap,
        shift
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

@testitem "BatchSolution" begin
    using GaussianSpectralDensities: MatrixEnsemble, ResolventEquation, Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        BatchSolution,
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

    # non-scalar getindex
    for i in eachindex(sol)
        sol.params[i] = ResolventParams(0.5im * i, 0.25i, 0.5i, 1.0i, -0.5im * i)
        sol.status[i] = isodd(i) ? SolutionStatus.success : SolutionStatus.not_finite
    end
    @testset for index in (
        1:4, [1, 3, 5], [true, false, true, false, true, false, true, false, true]
    )
        sub = sol[index]
        @test sub isa BatchSolution{Float64}
        @test sub.zs == zs[index]
        @test sub.equation === eq
        @test sub.params == sol.params[index]
        @test sub.status == sol.status[index]
        @test length(sub) == length(zs[index])
    end

    # non-scalar setindex!
    sub = sol[1:3]
    for i in eachindex(sub)
        sub.params[i] = ResolventParams(1.0im * i, 0.5i, 1.0i, 2.0i, -1.0im * i)
        sub.status[i] = SolutionStatus.success
    end
    sol[1:3] = sub
    @test sol.params[1:3] == sub.params
    @test sol.status[1:3] == sub.status

    # non-scalar setindex! with mismatched equation
    sub_wrong_eq = BatchSolution(eq2, zs[1:3])
    @test_throws ArgumentError sol[1:3] = sub_wrong_eq

    # non-scalar setindex! with mismatched zs
    sub_wrong_zs = sol[4:6]
    @test_throws ArgumentError sol[1:3] = sub_wrong_zs

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
