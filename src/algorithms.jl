module Algorithms

using AutoHashEquals: @auto_hash_equals

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..Problems: PTraceConstraint, PointProblem, SpreadConstraint, issameproblem, update
using ..Spreads: Spread, SpreadType
using ..Solutions: PointSolution, issuccess, ptrace, shape, solves, spreadfrom
using ..Workspaces: Workspace

using ..ArrayTools: ArrayTools
using ..MathTools: monotonic, predictlinear, samesign

public AbstractAlgorithm,
    AbstractAlgorithmState,
    AbstractSchedule,
    DirectSolve,
    DirectSolveState,
    ScheduleState,
    SpreadSchedule,
    SymmetrySchedule,
    isdone,
    statetype,
    step!

# Since Julia doesn't have traits, and we want _some_ algorithms to have array semantics,
# all of them must
abstract type AbstractAlgorithm{T} <: AbstractVector{T} end
abstract type AbstractAlgorithmState end

Base.IndexStyle(::Type{<:AbstractAlgorithm}) = IndexLinear()

## AbstractAlgorithm interface
## ===========================
## Provided methods:
##
## - isdone(::PointProblem, ::AbstractAlgorithm, ::Nothing)::Bool  (no state no done)
##
## Required methods:
##
## - ArrayTools.adaptrealtype(::Type{<:AbstractFloat}, ::SomeAlgorithm)::SomeAlgorithm
## - nextindex(::SomeAlgorithm, ::Nothing)
## - stepproblem(index, ::PointProblem, ::SomeAlgorithm, ::Nothing)::PointProblem
## - newstate(index, ::PointProblem, ::PointSolution, ::SomeAlgorithm, ::Nothing)::SomeAlgorithmState
## - isdone(::PointProblem, ::SomeAlgorithm, ::SomeAlgorithmState)::Bool
## - statetype(::Type{SomeAlgorithm})::Type{SomeAlgorithmState}
##
## Required unless isdone(::PointProblem, ::SomeAlgorithm, ::SomeAlgorithmState)::Bool is
## always true, i.e., unless the algorithm is guaranteed to finish in a single step:
##
## - nextindex(::SomeAlgorithm, ::SomeAlgorithmState)
## - stepproblem(index, ::PointProblem, ::SomeAlgorithm, ::SomeAlgorithmState)::PointProblem
## - newstate(index, ::PointProblem, ::PointSolution, ::SomeAlgorithm, ::SomeAlgorithmState)::SomeAlgorithmState
##
## Algorithms that implement the indexing or array interface should forward the stateless
## method nextindex(::SomeAlgorithm, ::Nothing) to Base.firstindex.
function nextindex end
function isdone end
function stepproblem end
function newstate end
function statetype end

## Generic stepping implementation
## ===============================
## Provided interface methods:
##
## - step!(::Workspace{F}, ::PointProblem{F}, ::AbstractAlgorithm, ::Nothing)::Tuple{PointSolution{F},AbstractAlgorithmState}
## - step!(::Workspace{F}, ::PointProblem{F}, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Union{Nothing,Tuple{PointSolution{F},AbstractAlgorithmState}}
function step!(
    ws::Workspace{F},
    problem::PointProblem{F},
    alg::AbstractAlgorithm,
    state::Union{Nothing,AbstractAlgorithmState},
) where {F}
    isdone(problem, alg, state) && return nothing
    newindex = nextindex(alg, state)
    newproblem = stepproblem(newindex, problem, alg, state)
    newsolution = eqsolve!(ws, newproblem)
    return newsolution, newstate(newindex, newproblem, newsolution, alg, state)
end

isdone(::PointProblem, ::AbstractAlgorithm, ::Nothing) = false

## Direct algorithm
## ================
"""
    DirectSolve()

Solve the average resolvent equation directly for the given objective

This solution strategy can converge to the wrong branch when ``T ≠ 0``. A scheduling
algorithm is recommended to mitigate this.
"""
struct DirectSolve <: AbstractAlgorithm{Union{}} end
struct DirectSolveState <: AbstractAlgorithmState end

DirectSolve(::DirectSolve) = DirectSolve()

# Traits, conversions
# -------------------
Base.convert(::Type{DirectSolve}, alg::DirectSolve) = alg

# Interface
# ---------
# We have no intention of indexing DirectSolve, but since it is a subtype of AbstractVector
# we need to implement Base.size, otherwise printing would error.
Base.size(::DirectSolve) = (0,)
Base.getindex(alg::DirectSolve, i::Int) = Base.throw_boundserror(alg, (i,)) # XXX: needs test

ArrayTools.adaptrealtype(::Type{F}, alg::DirectSolve) where {F<:AbstractFloat} = alg

nextindex(::DirectSolve, ::Nothing) = nothing
stepproblem(::Nothing, problem::PointProblem, ::DirectSolve, ::Nothing) = problem
function newstate(::Nothing, ::PointProblem, ::PointSolution, ::DirectSolve, ::Nothing)
    return DirectSolveState()
end
isdone(::PointProblem, ::DirectSolve, ::DirectSolveState) = true
statetype(::Type{DirectSolve}) = DirectSolveState

## Schedule algorithms
## ===================
## Provided types:
##
## - ScheduleState{F} with properties problem, solution, index, prevsolution, previndex
##
## Provided interface methods:
##
## - Read-only array interface: Base.size, Base.IndexStyle, Base.getindex
## - nextindex(::AbstractSchedule{F}, ::Nothing)::Int
## - nextindex(::AbstractSchedule{F}, ::ScheduleState{F})::Int
## - newstate(::Int, PointProblem{F}, ::PointSolution{F}, ::AbstractSchedule{F}, ::Nothing)::ScheduleState{F}
## - newstate(::Int, PointProblem{F}, ::PointSolution{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::ScheduleState{F}
## - isdone(::PointProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - statetype(::Type{<:AbstractSchedule{F}})::Type{ScheduleState{F}}
##
## Provided non-interface functions:
##
## - stepproblem_predict(::Int, ::PointProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::PointProblem{F}
## - predictconstraint(::AbstractConstraint{F}, ::PointSolution{F}, ::PointSolution{F})::AbstractConstraint{F}
##
## Required methods:
##
## - Base.convert(::Type{SomeSchedule{F}}, alg::SomeSchedule)::SomeSchedule{F}
## - ArrayTools.adaptrealtype(::Type{F}, ::SomeSchedule)::SomeSchedule{F}
## - stepproblem(::Int, ::PointProblem{F}, ::SomeSchedule{F}, ::Nothing)::PointProblem{F}
## - iscomplete(::PointProblem{F}, ::SomeSchedule{F}, ::ScheduleState{F})::Bool
##
## Required properties:
## - :vals: should be an indexable collection of raw schedule values; while indexing the
##     schedule itself may return an instance of a richer type, indexing this collection
##     should return a plain number
##
## NOTE: The default getindex method is simply Base.getindex(alg, i) = alg.vals[i]. Subtypes
## may # override this if needed as long as the indexing remains linear and the valid
## indices are identical to the linear indices of alg.vals.
##
## NOTE: stepproblem_predict works by calling the stateless stepproblem (the one taking a
## Nothing state, for the initial step) and using predictconstraint to update the initial
## guess based on the last two steps of the algorithm. The provided state-dependent
## implementation of stepproblem just forwards to stepproblem_predict. The stateless method
## _must_ be implemented for each subtype, and the stateful method _may_ be implemented if
## stepproblem_predict is insufficient or appropriate.
##
## NOTE: iscomplete's only concern should be whether the algorithm has stepped all the way
## to the end. Other termination criteria, including whether the most recent intermediate
## step happened upon a successful solution of the final problem, are handled by the
## provided isdone method.
abstract type AbstractSchedule{F<:AbstractFloat,T} <: AbstractAlgorithm{T} end

Base.size(alg::AbstractSchedule) = (length(alg.vals),)
Base.getindex(alg::AbstractSchedule, i::Int) = alg.vals[i]

@auto_hash_equals typearg=true struct ScheduleState{
    F,E,P<:PointProblem{F,E},S<:PointSolution{F,E}
} <: AbstractAlgorithmState
    problem::P
    solution::S
    index::Int
    prevsolution::S
    previndex::Int
end

function stepproblem end
function iscomplete end

nextindex(alg::AbstractSchedule, ::Nothing) = firstindex(alg)
nextindex(::AbstractSchedule{F}, state::ScheduleState{F}) where {F} = (state.index + 1)

function stepproblem(
    index::Int, problem::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    return stepproblem_predict(index, problem, alg, state)
end

function stepproblem_predict(
    index::Int, problem::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    newproblem = stepproblem(index, problem, alg, nothing)
    newconstraint = predictconstraint(
        newproblem.constraint,
        alg.vals[index],
        (alg.vals[state.index], state.solution),
        (alg.vals[state.previndex], state.prevsolution),
    )
    return update(newproblem; constraint=newconstraint)
end

function newstate(
    index::Int,
    problem::PointProblem{F},
    solution::PointSolution{F},
    ::AbstractSchedule{F},
    ::Nothing,
) where {F}
    return ScheduleState(problem, solution, index, solution, index)
end

function newstate(
    index::Int,
    problem::PointProblem{F},
    solution::PointSolution{F},
    ::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    return ScheduleState(problem, solution, index, state.solution, state.index)
end

function isdone(
    problem::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    return (!issuccess(state.solution)) || iscomplete(problem, alg, state)
end

statetype(::Type{<:AbstractSchedule{F}}) where {F} = ScheduleState{F}

# Prediction based on arbitrary knots, e.g., from schedule values
function predictconstraint(
    constraint::SpreadConstraint{F},
    x::F,
    (x1, solution1)::Tuple{F,PointSolution{F}},
    (x2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    shape1, shape2 = shape(solution1), shape(solution2)
    return update(constraint; shape0=predictlinear(x, (x1, shape1), (x2, shape2)))
end

function predictconstraint(
    constraint::PTraceConstraint{F},
    x::F,
    (x1, solution1)::Tuple{F,PointSolution{F}},
    (x2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    shape0 = predictlinear(x, (x1, shape(solution1)), (x2, shape(solution2)))
    spread_type = constraint.spread0.type
    spread1 = spreadfrom(solution1, spread_type)
    spread2 = spreadfrom(solution2, spread_type)
    spread0 = Spread(predictlinear(x, (x1, spread1.val), (x2, spread2.val)), spread_type)
    return update(constraint; shape0, spread0)
end

"""
    SpreadSchedule()
    SpreadSchedule(vals, type)
    SpreadSchedule{F}(vals, type) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the spread parameter

The intended way to use this algorithm is to start from a large (regularized) spread to
ensure convergence to the correct solution branch, and schedule a gradual decay to track
this branch down to the solution.

The parameter `vals` should be a vector of numerical values, and `type` is the type of
spread parameter to use, see [`Spread`](@ref) for details. `vals` must be monotonically
decreasing.

The final step of the algorithm is to solve the true problem using values from the most
recent annealing step as initial guesses for the Newton solve. This either happens at the
end of the prescribed schedule or when the an annealing step intercepts the target
constraint, whichever comes first.

The default value is

```julia
SpreadSchedule(Defaults.spread_schedule_vals, Defaults.spread_schedule_type)
```
"""
@auto_hash_equals typearg=true struct SpreadSchedule{F,U<:AbstractVector{F}} <:
                                      AbstractSchedule{F,Spread{F}}
    vals::U
    type::SpreadType.T

    function SpreadSchedule{F}(vals, type) where {F}
        # map preserves ranges types etc. better than broadcasting
        valsF = (eltype(vals) === F) ? vals : map(F, vals)
        schedule = new{F,typeof(valsF)}(valsF, type)
        # Validate post-instantiation in case the original vals was a single-use iterator
        if !monotonic(Iterators.reverse(schedule.vals))
            throw(ArgumentError("spread schedule must decrease monotonically"))
        end
        return schedule
    end
end

SpreadSchedule(vals, type) = SpreadSchedule{eltype(vals)}(vals, type)

function SpreadSchedule()
    vals, type = Defaults.spread_schedule_vals, Defaults.spread_schedule_type
    return SpreadSchedule(vals, type)
end

SpreadSchedule{F}(alg::SpreadSchedule) where {F} = SpreadSchedule{F}(alg.vals, alg.type)

function Base.convert(::Type{P}, alg::SpreadSchedule) where {F,P<:SpreadSchedule{F}}
    return (alg isa P) ? alg : SpreadSchedule{F}(alg)
end

function ArrayTools.adaptrealtype(::Type{F}, alg::SpreadSchedule) where {F}
    return convert(SpreadSchedule{F}, alg)
end

# Interface
# ---------
Base.getindex(alg::SpreadSchedule, i::Int) = Spread(alg.vals[i], alg.type)

function stepproblem(
    index::Int, problem::PointProblem{F}, alg::SpreadSchedule{F}, ::Nothing
) where {F}
    constraint = SpreadConstraint(alg[index], problem.constraint.shape0)
    return update(problem; constraint)
end

function stepproblem(
    index::Int, problem::PointProblem{F}, alg::SpreadSchedule{F}, state::ScheduleState{F}
) where {F}
    solution1, solution2 = state.solution, state.prevsolution
    if (state.index == lastindex(alg)) ||
        intercepted(problem.constraint, solution1, solution2)
        # Final step: use target problem with updated constraint (predicted init values)
        x, x1, x2 = constraintknots(problem.constraint, solution1, solution2)
        point1, point2 = (x1, solution1), (x2, solution2)
        constraint = predictconstraint(problem.constraint, x, point1, point2)
        return update(problem; constraint)
    end
    return stepproblem_predict(index, problem, alg, state)
end

function iscomplete(
    problem::PointProblem{F}, ::SpreadSchedule{F}, state::ScheduleState{F}
) where {F}
    return issameproblem(state.problem, problem) || solves(state.solution, problem)
end

# Predicate for whether the target was crossed in the most recent step
function intercepted(
    constraint::SpreadConstraint, solution1::PointSolution, solution2::PointSolution
)
    cspreadv, spread1v, spread2v = constraintknots(constraint, solution1, solution2)
    return !samesign(spread1v - cspreadv, spread2v - cspreadv)
end

function intercepted(
    constraint::PTraceConstraint, solution1::PointSolution, solution2::PointSolution
)
    ptracec, ptrace1, ptrace2 = constraintknots(constraint, solution1, solution2)
    return !samesign(ptrace1 - ptracec, ptrace2 - ptracec)
end

# Prediction knots derived from problem constraint
function constraintknots(
    constraint::SpreadConstraint{F},
    solution1::PointSolution{F},
    solution2::PointSolution{F},
) where {F}
    cspread = constraint.val
    spread1 = spreadfrom(solution1, cspread.type)
    spread2 = spreadfrom(solution2, cspread.type)
    return cspread.val, spread1.val, spread2.val
end

function constraintknots(
    constraint::PTraceConstraint{F},
    solution1::PointSolution{F},
    solution2::PointSolution{F},
) where {F}
    return constraint.val, ptrace(solution1), ptrace(solution2)
end

"""
    SymmetrySchedule()
    SymmetrySchedule(vals)
    SymmetrySchedule{F}(vals) where {F<:AbstractFloat}
    SymmetrySchedule{F}(steps::Int) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the degree of symmetry in the ensemble

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``S`` is replaced by ``μS``. The motivation for this is that the averaged resolvent
equations have a unique solution when ``S = 0``, so it makes some sense to solve that
problem first and then track the solution branch while gradually turning on the symmetry.

The caveat is that a given ``z`` may be inside the spectral support at ``μ = 0`` and outside
at ``μ = 1`` or vice versa, in which case the solutions are not analytically connected
(there will be a bifurcation in the solution branches). Granted, we often use analytic
continuation to extend the functional form of the averaged resolvent within the support out
to some margin beyond the boundary such that we can precisely interpolate the boundary
location. This works well for any ``z`` that's inside the support+margin at ``μ = 0``: the
solution branch either survives to ``μ = 1``, or drops beyond the margin (in practical
terms: below shiftmin) and turns into a `success = false` solution as it should. The more
problematic case is if ``z`` is outside the support+margin at ``μ = 0``, such that we fail
to obtain an initial solution, but nevertheless should be inside at ``μ = 1``.

The schedule can be specified as an integer `steps`, which will set the schedule to
`range(0, 1, steps)`. Alternatively, a custom schedule can be provided directly in the form
of an `AbstractVector{<:Real}`, allowing non-uniform step sizes. An error is thrown if the
schedule does not increase monotonically from 0 to 1.

The default value is

```julia
SymmetrySchedule(Defaults.symmetry_schedule_vals)
```
"""
# FIXME: Change abort heuristics. As described in the docstring above, we shouldn't abort
# just because success = false at μ = 0. In fact, since the supports may have complex shapes
# and topologies, we probably shouldn't ever abort: a point can go in and out of the
# support+margin multiple times. If a failed step did converge to finite parameters (which
# is often the case; the point of declaring failure based on shiftmin is to halt a spread
# schedule _before_ it reaches a point where the solver fails to converge, not after), we
# should use them to initialize the next step; if we got NaN, we should revert to the
# initial guesses in the constraint.
@auto_hash_equals typearg=true struct SymmetrySchedule{F,U<:AbstractVector{F}} <:
                                      AbstractSchedule{F,F}
    vals::U

    function SymmetrySchedule{F}(vals) where {F}
        # map preserves ranges types etc. better than broadcasting
        valsF = (eltype(vals) === F) ? vals : map(F, vals)
        schedule = new{F,typeof(valsF)}(valsF)
        # Validate post-instantiation in case the original vals was a single-use iterator
        if (
            (first(schedule.vals) != 0) ||
            (last(schedule.vals) != 1) ||
            !monotonic(schedule.vals)
        )
            throw(ArgumentError("symmetry schedule must increase monotonically 0 to 1"))
        end
        return schedule
    end
end

function SymmetrySchedule{F}(steps::Int) where {F}
    return SymmetrySchedule{F}(range(zero(F), one(F), steps))
end

SymmetrySchedule(vals) = SymmetrySchedule{eltype(vals)}(vals)
SymmetrySchedule() = SymmetrySchedule(Defaults.symmetry_schedule_vals)

SymmetrySchedule{F}(alg::SymmetrySchedule) where {F} = SymmetrySchedule{F}(alg.vals)

# Traits, conversions
# --------------------------------
function Base.convert(::Type{P}, alg::SymmetrySchedule) where {F,P<:SymmetrySchedule{F}}
    return (alg isa P) ? alg : SymmetrySchedule{F}(alg)
end

function ArrayTools.adaptrealtype(::Type{F}, alg::SymmetrySchedule) where {F}
    return convert(SymmetrySchedule{F}, alg)
end

# Interface
# ---------
function stepproblem(
    index::Int, problem::PointProblem{F}, alg::SymmetrySchedule{F}, ::Nothing
) where {F}
    return update(problem; equation=update(problem.equation; sscale=alg[index]))
end

function iscomplete(
    ::PointProblem{F}, alg::SymmetrySchedule{F}, state::ScheduleState{F}
) where {F}
    return (state.index == lastindex(alg))
end

end  # module Algorithms

@testitem "DirectSolve" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms: DirectSolve, DirectSolveState, statetype

    alg = DirectSolve()

    # Traits
    @test alg isa DirectSolve
    @test statetype(DirectSolve) === DirectSolveState

    # Indexing errors
    @test_throws BoundsError alg[1]

    # Conversion
    @test DirectSolve(alg) === DirectSolve()
    @test convert(DirectSolve, alg) === alg
    @test adaptrealtype(Float32, alg) == alg
    @test adaptrealtype(Float64, alg) == alg
end

@testitem "SpreadSchedule" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms: ScheduleState, SpreadSchedule, statetype
    using GaussianSpectralDensities.Defaults: Defaults

    vals64, type = 2.0 .^ (2:-1:-2), :gap
    alg64 = SpreadSchedule(vals64, type)

    # Defaults
    @test SpreadSchedule() ===
        SpreadSchedule(Defaults.spread_schedule_vals, Defaults.spread_schedule_type)

    # Traits
    IndexStyle(typeof(alg64)) == IndexLinear()

    @testset "$F" for F in [Float32, Float64]
        valsF = map(F, vals64)  # map preserves ranges types etc. better than broadcasting
        algF = SpreadSchedule(valsF, type)

        # Floattype inferring constructor
        @test algF isa SpreadSchedule{F}
        @test statetype(SpreadSchedule{F}) === ScheduleState{F}

        # Floattype enforcing constructor
        @test SpreadSchedule{F}(vals64, type) == algF
        @test SpreadSchedule{Float64}(valsF, type) == alg64

        # Floattype conversions
        @test SpreadSchedule{F}(alg64) == algF
        @test SpreadSchedule{Float64}(algF) == alg64
        @test convert(SpreadSchedule{F}, alg64) == algF
        @test convert(SpreadSchedule{Float64}, algF) == alg64
        @test adaptrealtype(F, alg64) == algF
        @test adaptrealtype(Float64, algF) == alg64
    end
end

@testitem "SymmetrySchedule" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms: ScheduleState, SymmetrySchedule, statetype
    using GaussianSpectralDensities.Defaults: Defaults

    steps = 3
    vals64 = range(0.0, 1.0, steps)

    alg64 = SymmetrySchedule(vals64)

    # All the signatures
    @test SymmetrySchedule{Float64}(steps) === alg64

    # Defaults
    @test SymmetrySchedule() === SymmetrySchedule(Defaults.symmetry_schedule_vals)

    # Traits
    IndexStyle(typeof(alg64)) == IndexLinear()

    @testset "$F" for F in [Float32, Float64]
        valsF = map(F, vals64)  # map preserves ranges types etc. better than broadcasting
        algF = SymmetrySchedule(valsF)

        # Floattype inferring constructor
        @test algF isa SymmetrySchedule{F}
        @test statetype(SymmetrySchedule{F}) === ScheduleState{F}

        # Floattype enforcing constructor
        @test SymmetrySchedule{F}(vals64) == algF
        @test SymmetrySchedule{Float64}(valsF) == alg64

        # Upconvert algF properties to capture roundtrip truncation before equality testing
        algF64 = SymmetrySchedule(map(Float64, algF.vals))

        # Floattype conversions
        @test SymmetrySchedule{F}(alg64) == algF
        @test SymmetrySchedule{Float64}(algF) == algF64
        @test convert(SymmetrySchedule{F}, alg64) == algF
        @test convert(SymmetrySchedule{Float64}, algF) == algF64
        @test adaptrealtype(F, alg64) == algF
        @test adaptrealtype(Float64, algF) == alg64
    end
end

@testitem "*Schedule errors" begin
    using GaussianSpectralDensities.Algorithms: SpreadSchedule, SymmetrySchedule

    # Doesn't start at 0
    @test_throws ArgumentError SymmetrySchedule([0.2, 0.6, 1.0])
    @test_throws "0 to 1" SymmetrySchedule([0.2, 0.6, 1.0])
    @test_throws ArgumentError SymmetrySchedule(range(0.2, 1.0, 3))
    @test_throws "0 to 1" SymmetrySchedule(range(0.2, 1.0, 3))
    # Doesn't end at 1
    @test_throws ArgumentError SymmetrySchedule([0.0, 0.4, 0.8])
    @test_throws "0 to 1" SymmetrySchedule([0.0, 0.4, 0.8])
    @test_throws ArgumentError SymmetrySchedule(range(0.0, 0.8, 3))
    @test_throws "0 to 1" SymmetrySchedule(range(0.0, 0.8, 3))
    # Not monotonic
    @test_throws ArgumentError SymmetrySchedule([0.0, 0.67, 0.33, 1.0])
    @test_throws "monotonic" SymmetrySchedule([0.0, 0.67, 0.33, 1.0])
    @test_throws ArgumentError SymmetrySchedule(range(1.0, 0.0, 3))
    @test_throws "monotonic" SymmetrySchedule(range(1.0, 0.0, 3))
    @test_throws ArgumentError SpreadSchedule([1.0, 0.33, 0.67, 0.0], :shift)
    @test_throws "monotonic" SpreadSchedule([1.0, 0.33, 0.67, 0.0], :gap)
    @test_throws ArgumentError SpreadSchedule(range(0.0, 1.0, 3), :shift)
    @test_throws "monotonic" SpreadSchedule(range(0.0, 1.0, 3), :gap)
end

@testitem "DirectSolve/SpreadConstraint" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    constraint64 = SpreadConstraint(; shift=0.25, shape0=(1.0 - im) / 2)

    @testset "$label" for (label, A64, D64) in [
        ("Array", A64data, Diagonal(D64data)),
        ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
        ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
    ]
        ensemble64 = MatrixEnsemble(A64, I, I, D64)
        eq64 = ResolventEquation(ensemble64)
        ws64 = Workspace(eq64)
        problem64 = PointProblem(eq64, constraint64, z64)

        alg = DirectSolve()

        # No state no done
        @test !isdone(problem64, alg, nothing)

        # Stops after one iteration
        solution64, state64 = step!(ws64, problem64, alg, nothing)
        @test nothing === step!(ws64, problem64, alg, state64)

        # Solves the actual target problem
        @test solves(solution64, problem64)
        @test isdone(problem64, alg, state64)
    end
end

@testitem "DirectSolve/PTraceConstraint" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    constraint64 = PTraceConstraint(1.0; shape0=complex(0.1), shift0=1.0)

    ensemble64 = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq64 = ResolventEquation(ensemble64)
    ws64 = Workspace(eq64)
    problem64 = PointProblem(eq64, constraint64, z64)

    alg64 = DirectSolve()

    # No state no done
    @test !isdone(problem64, alg64, nothing)

    # Stops after one iteration
    solution64, state64 = step!(ws64, problem64, alg64, nothing)
    @test nothing === step!(ws64, problem64, alg64, state64)

    # Solves the actual target problem
    @test solves(solution64, problem64)
    @test isdone(problem64, alg64, state64)
end

@testitem "SpreadSchedule/SpreadConstraint" begin
    using GaussianSpectralDensities.Algorithms: SpreadSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Spreads: Spread
    using GaussianSpectralDensities.Solutions: issuccess, shape, solves, spreadfrom
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    shape0_64 = complex(1.0)

    schedule64 = [4.0, 2.0, 1.0, 0.5]
    target_val64 = 0.25

    @testset "Target spread type = $spread_type_target" for spread_type_target in
                                                            [:shift, :gap]
        spread = Spread(target_val64, spread_type_target)
        constraint64 = SpreadConstraint(spread; shape0=shape0_64)

        @testset "Schedule spread type = $spread_type_schedule" for spread_type_schedule in
                                                                    [:shift, :gap]
            @testset "$label" for (label, A64, D64) in [
                ("Array", A64data, Diagonal(D64data)),
                ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
                ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
            ]
                @testset "$F" for F in [Float32, Float64]
                    ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                    eqF = ResolventEquation(ensembleF)
                    wsF = Workspace(eqF)
                    problemF = PointProblem(eqF, constraint64, z64)
                    algF = SpreadSchedule{F}(schedule64, spread_type_schedule)

                    # No state no done
                    @test !isdone(problemF, algF, nothing)

                    # Collect all solutions and states
                    solutions = []
                    states = []
                    result = step!(wsF, problemF, algF, nothing)
                    while !isnothing(result)
                        solution, state = result
                        push!(solutions, solution)
                        push!(states, state)
                        result = step!(wsF, problemF, algF, state)
                    end

                    # In most cases, solutions comprise all schedule steps and the final
                    # step to the target. The exception is (target, schedule) == (:shift,
                    # :gap), in which the solution may be intersected before exhausting the
                    # schedule, triggering an early final step.
                    if (spread_type_target, spread_type_schedule) != (:shift, :gap)
                        @test length(solutions) == length(algF) + 1
                    end

                    scheduleF = algF.vals

                    # Verify integrity of intermediate steps
                    @testset "index = $i" for i in
                                              eachindex(states, solutions)[begin:(end - 1)]
                        @test issuccess(solutions[i])
                        @test states[i].index == i
                        @test states[i].solution === solutions[i]
                        @test !isdone(problemF, algF, states[i])
                        if i > firstindex(solutions)
                            # Each schedule step initialized by linear *polation from the
                            # two previous solutions
                            problem = states[i].problem
                            cshape0 = problem.constraint.shape0
                            i1, i2 = states[i - 1].index, states[i - 1].previndex
                            sol1, sol2 = solutions[i1], solutions[i2]
                            shape1, shape2 = shape(sol1), shape(sol2)
                            x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                            @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))
                        end
                    end

                    # Final step initialized by linear *polation from the two previous
                    # solutions
                    problem = states[end].problem
                    cspread, cshape0 = problem.constraint.val, problem.constraint.shape0
                    i1, i2 = states[end - 1].index, states[end - 1].previndex
                    sol1, sol2 = solutions[i1], solutions[i2]
                    shape1, shape2 = shape(sol1), shape(sol2)
                    spread1 = spreadfrom(sol1, cspread.type)
                    spread2 = spreadfrom(sol2, cspread.type)
                    x, x1, x2 = cspread.val, spread1.val, spread2.val
                    @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))

                    # Final step solves the actual target problem
                    @test solves(solutions[end], problemF)
                    @test isdone(problemF, algF, states[end])
                end
            end
        end
    end
end

@testitem "SpreadSchedule/PTraceConstraint" begin
    using GaussianSpectralDensities.Algorithms: SpreadSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: issuccess, ptrace, shape, solves, spreadfrom
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    shape0_64 = complex(0.1)

    schedule64 = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    target_ptrace64 = 1.0

    @testset "Schedule spread type = $spread_type" for spread_type in [:shift, :gap]
        alg64 = SpreadSchedule(schedule64, spread_type)
        spread0 = alg64[begin]
        constraint64 = PTraceConstraint(target_ptrace64; shape0=shape0_64, spread0)

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
            eqF = ResolventEquation(ensembleF)
            wsF = Workspace(eqF)
            problemF = PointProblem(eqF, constraint64, z64)
            algF = SpreadSchedule{F}(alg64)

            # No state no done
            @test !isdone(problemF, algF, nothing)

            # Collect all solutions and states
            solutions = []
            states = []
            result = step!(wsF, problemF, algF, nothing)
            while !isnothing(result)
                solution, state = result
                push!(solutions, solution)
                push!(states, state)
                result = step!(wsF, problemF, algF, state)
            end

            # Stepping should terminate before exhaustion as ptrace crosses target
            @test length(solutions) < length(algF) + 1

            scheduleF = algF.vals

            # Verify integrity of intermediate steps
            @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                @test issuccess(solutions[i])
                @test states[i].index == i
                @test states[i].solution === solutions[i]
                @test !isdone(problemF, algF, states[i])
                if i > firstindex(solutions)
                    # Each schedule step initialized by linear *polation from the two
                    # previous solutions
                    problem = states[i].problem
                    cshape0 = problem.constraint.shape0
                    i1, i2 = states[i - 1].index, states[i - 1].previndex
                    sol1, sol2 = solutions[i1], solutions[i2]
                    shape1, shape2 = shape(sol1), shape(sol2)
                    x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                    @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))
                end
            end

            # Final step initialized by linear *polation from the two previous solutions
            # Could be either a PTrace or Spread constrained problem
            problem = states[end].problem
            cshape0 = problem.constraint.shape0
            i1, i2 = states[end - 1].index, states[end - 1].previndex
            sol1, sol2 = solutions[i1], solutions[i2]
            shape1, shape2 = shape(sol1), shape(sol2)
            if problem.constraint isa PTraceConstraint
                cspread0 = problem.constraint.spread0
                spread1 = spreadfrom(sol1, cspread0.type)
                spread2 = spreadfrom(sol2, cspread0.type)
                spread1v, spread2v = spread1.val, spread2.val
                x, x1, x2 = problem.constraint.val, ptrace(sol1), ptrace(sol2)
                @test isapproxlinear((x, cspread0.val), (x1, spread1v), (x2, spread2v))
            else
                cspread = problem.constraint.val
                spread1 = spreadfrom(sol1, cspread.type)
                spread2 = spreadfrom(sol2, cspread.type)
                x, x1, x2 = cspread.val, spread1.val, spread2.val
            end
            @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))

            # Final step solves the actual P-trace constraint
            @test solves(solutions[end], problemF)
            @test isdone(problemF, algF, states[end])
        end
    end
end

@testitem "SymmetrySchedule/SpreadConstraint" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Spreads: Spread
    using GaussianSpectralDensities.Solutions: issuccess, shape, solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4

    nsteps = 3

    @testset "Spread type = $spread_type" for spread_type in [:shift, :gap]
        constraint64 = SpreadConstraint(; spread_type => 0.25, shape0=complex(1.0))

        @testset "$label" for (label, A64, D64) in [
            ("Array", A64data, Diagonal(D64data)),
            ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
            ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
        ]
            @testset "$F" for F in [Float32, Float64]
                ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                eqF = ResolventEquation(ensembleF)
                wsF = Workspace(eqF)
                problemF = PointProblem(eqF, constraint64, z64)
                algF = SymmetrySchedule{F}(nsteps)

                # No state no done
                @test !isdone(problemF, algF, nothing)

                # Collect all solutions and states
                solutions = []
                states = []
                result = step!(wsF, problemF, algF, nothing)
                while !isnothing(result)
                    solution, state = result
                    push!(solutions, solution)
                    push!(states, state)
                    result = step!(wsF, problemF, algF, state)
                end

                # solutions comprise schedule steps (no separate final step here)
                @test length(solutions) == length(algF)

                scheduleF = algF.vals

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)
                    @test issuccess(solutions[i])
                    @test states[i].index == i
                    @test states[i].solution === solutions[i]
                    if i < lastindex(solutions)
                        @test !isdone(problemF, algF, states[i])
                    end
                    if i > firstindex(solutions)
                        # Each schedule step initialized by linear *polation from the two
                        # previous solutions
                        problem = states[i].problem
                        cshape0 = problem.constraint.shape0
                        i1, i2 = states[i - 1].index, states[i - 1].previndex
                        sol1, sol2 = solutions[i1], solutions[i2]
                        shape1, shape2 = shape(sol1), shape(sol2)
                        x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                        @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))
                    end
                end

                # Final step solves the actual problem
                @test solves(solutions[end], problemF)
                @test isdone(problemF, algF, states[end])
            end
        end
    end
end

@testitem "SymmetrySchedule/PTraceConstraint" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess, shape, solves, spreadfrom
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    constraint64 = PTraceConstraint(1.0; shape0=complex(0.1), shift0=1.0)

    nsteps = 3

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
        eqF = ResolventEquation(ensembleF)
        wsF = Workspace(eqF)
        problemF = PointProblem(eqF, constraint64, z64)
        algF = SymmetrySchedule{F}(nsteps)

        # No state no done
        @test !isdone(problemF, algF, nothing)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF, nothing)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # solutions comprise schedule steps (no extra final step for symmetry schedule)
        @test length(solutions) == length(algF)

        scheduleF = algF.vals

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)
            @test issuccess(solutions[i])
            @test states[i].index == i
            @test states[i].solution === solutions[i]
            if i < lastindex(solutions)
                @test !isdone(problemF, algF, states[i])
            end
            if i > firstindex(solutions)
                # Each schedule step initialized by linear *polation from the two previous
                # solutions
                problem = states[i].problem
                cshape0, cspread0 = problem.constraint.shape0, problem.constraint.spread0
                i1, i2 = states[i - 1].index, states[i - 1].previndex
                sol1, sol2 = solutions[i1], solutions[i2]
                shape1, shape2 = shape(sol1), shape(sol2)
                spread1 = spreadfrom(sol1, cspread0.type)
                spread2 = spreadfrom(sol2, cspread0.type)
                spread1v, spread2v = spread1.val, spread2.val
                x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                @test isapproxlinear((x, cspread0.val), (x1, spread1v), (x2, spread2v))
                @test isapproxlinear((x, cshape0), (x1, shape1), (x2, shape2))
            end
        end

        # Final step solves the actual problem
        @test solves(solutions[end], problemF)
        @test isdone(problemF, algF, states[end])
    end
end

@testitem "SpreadSchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: SpreadSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Spreads: shift
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    shiftmin = 0

    # z outside Ginibre support => negative shift needed to reach k = 1.0
    z64 = 1.0 + 1.0im
    shape0_64 = complex(0.1)

    alg64 = SpreadSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], :gap)
    constraint64 = PTraceConstraint(1.0; shape0=shape0_64, gap0=4.0)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.0I)
        eqF = ResolventEquation(ensembleF; shiftmin)
        wsF = Workspace(eqF)
        problemF = PointProblem(eqF, constraint64, z64)
        algF = SpreadSchedule{F}(alg64)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF, nothing)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF) + 1

        # Final step failed due to shiftmin violation
        @test !issuccess(solutions[end])
        @test shift(solutions[end]) < eqF.shiftmin
        @test isdone(problemF, algF, states[end])
    end
end

@testitem "SymmetrySchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Spreads: shift
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    shiftmin = 0

    # z inside Ginibre but outside Girko support => negative shift needed to reach k = 1.0
    z64 = 0.25 + 0.75im
    constraint64 = PTraceConstraint(1.0; shape0=complex(0.1), shift0=1.0)

    nsteps = 5

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.75I)
        eqF = ResolventEquation(ensembleF; shiftmin)
        wsF = Workspace(eqF)
        problemF = PointProblem(eqF, constraint64, z64)
        algF = SymmetrySchedule{F}(nsteps)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF, nothing)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF)

        # Final step failed due to shiftmin violation
        @test !issuccess(solutions[end])
        @test shift(solutions[end]) < eqF.shiftmin
        @test isdone(problemF, algF, states[end])
    end
end
