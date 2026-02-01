module Algorithms

using AutoHashEquals: @auto_hash_equals

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..Problems:
    PTraceConstraint, PointProblem, RegularizerConstraint, issameproblem, update
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: PointSolution, issuccess, ptrace, q, regularizer, solves
using ..Workspaces: Workspace

using ..ArrayTools: ArrayTools
using ..MathTools: monotonic, predictlinear, samesign

public AbstractAlgorithm,
    AbstractAlgorithmState,
    AbstractSchedule,
    DirectSolve,
    DirectSolveState,
    RegularizerSchedule,
    ScheduleState,
    SymmetrySchedule,
    isdone,
    statetype,
    step!

abstract type AbstractAlgorithm end
abstract type AbstractAlgorithmState end

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
## Algorithms that implement the indexing interface (Base.getindex, Base.firstindex,
## Base.lastindex, Base.length) should forward the stateless method
## nextindex(::SomeAlgorithm, ::Nothing) to Base.firstindex.
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
struct DirectSolve <: AbstractAlgorithm end
struct DirectSolveState <: AbstractAlgorithmState end

DirectSolve(::DirectSolve) = DirectSolve()

# Traits, conversions
# -------------------
Base.convert(::Type{DirectSolve}, alg::DirectSolve) = alg

# Interface
# ---------
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
## - nextindex(::AbstractSchedule{F}, ::Nothing)::Int
## - nextindex(::AbstractSchedule{F}, ::ScheduleState{F})::Int
## - newstate(::Integer, PointProblem{F}, ::PointSolution{F}, ::AbstractSchedule{F}, ::Nothing)::ScheduleState{F}
## - newstate(::Integer, PointProblem{F}, ::PointSolution{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::ScheduleState{F}
## - isdone(::PointProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - statetype(::Type{<:AbstractSchedule{F}})::Type{ScheduleState{F}}
##
## Provided non-interface functions:
##
## - stepproblem_predict(::Integer, ::PointProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::PointProblem{F}
## - predictconstraint(::AbstractConstraint{F}, ::PointSolution{F}, ::PointSolution{F})::AbstractConstraint{F}
##
## Required methods:
##
## - Indexing interface (Base.getindex, Base.firstindex, Base.lastindex, Base.length)
## - Base.convert(::Type{SomeSchedule{F}}, alg::SomeSchedule)::SomeSchedule{F}
## - ArrayTools.adaptrealtype(::Type{F}, ::SomeSchedule)::SomeSchedule{F}
## - stepproblem(::Integer, ::PointProblem{F}, ::SomeSchedule{F}, ::Nothing)::PointProblem{F}
## - iscomplete(::PointProblem{F}, ::SomeSchedule{F}, ::ScheduleState{F})::Bool
##
## Required properties:
## - :vals: should be an indexable collection of raw schedule values; while indexing the
##     schedule itself may return an instance of a richer type, indexing this collection
##     should return a plain number
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
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end

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
    index::Integer,
    problem::PointProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    return stepproblem_predict(index, problem, alg, state)
end

function stepproblem_predict(
    index::Integer,
    problem::PointProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
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
    index::Integer,
    problem::PointProblem{F},
    solution::PointSolution{F},
    ::AbstractSchedule{F},
    ::Nothing,
) where {F}
    return ScheduleState(problem, solution, index, solution, index)
end

function newstate(
    index::Integer,
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
    constraint::RegularizerConstraint{F},
    x::F,
    (x1, solution1)::Tuple{F,PointSolution{F}},
    (x2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    return update(constraint; q0=predictlinear(x, (x1, q(solution1)), (x2, q(solution2))))
end

function predictconstraint(
    constraint::PTraceConstraint{F},
    x::F,
    (x1, solution1)::Tuple{F,PointSolution{F}},
    (x2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    q0 = predictlinear(x, (x1, q(solution1)), (x2, q(solution2)))
    reg_type = constraint.reg0.type
    reg1 = regularizer(solution1, reg_type)
    reg2 = regularizer(solution2, reg_type)
    reg0 = Regularizer(predictlinear(x, (x1, reg1.val), (x2, reg2.val)), reg_type)
    return update(constraint; q0, reg0)
end

"""
    RegularizerSchedule()
    RegularizerSchedule(vals, type)
    RegularizerSchedule{F}(vals, type) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the regularizer

The intended way to use this algorithm is to start from a large regularizer to ensure
convergence to the correct solution branch, and schedule a gradual decay to approach the
true solution along this branch.

The parameter `vals` should be a vector of numerical values, and `type` is the type of
regularizer, see [`Regularizer`](@ref) for details. An error is thrown if the values do not
monotonically decrease.

The final step of the algorithm is a direct solve of the averaged resolvent problem. This
either happens at the end of the prescribed schedule or when intersecting the target during
the annealing, whichever comes first.

The default value is

```julia
RegularizerSchedule(Defaults.regularizer_schedule_vals, Defaults.regularizer_schedule_type)
```
"""
@auto_hash_equals typearg=true struct RegularizerSchedule{F,U<:AbstractVector{F}} <:
                                      AbstractSchedule{F}
    vals::U
    type::RegularizerType.T

    function RegularizerSchedule{F}(vals, type) where {F}
        # map preserves ranges types etc. better than broadcasting
        valsF = (eltype(vals) === F) ? vals : map(F, vals)
        schedule = new{F,typeof(valsF)}(valsF, type)
        # Validate post-instantiation in case the original vals was a single-use iterator
        if !monotonic(Iterators.reverse(schedule.vals))
            throw(ArgumentError("regularizer schedule must decrease monotonically"))
        end
        return schedule
    end
end

RegularizerSchedule(vals, type) = RegularizerSchedule{eltype(vals)}(vals, type)

function RegularizerSchedule()
    vals, type = Defaults.regularizer_schedule_vals, Defaults.regularizer_schedule_type
    return RegularizerSchedule(vals, type)
end

function RegularizerSchedule{F}(alg::RegularizerSchedule) where {F}
    return RegularizerSchedule{F}(alg.vals, alg.type)
end

function Base.convert(
    ::Type{P}, alg::RegularizerSchedule
) where {F,P<:RegularizerSchedule{F}}
    return (alg isa P) ? alg : RegularizerSchedule{F}(alg)
end

function ArrayTools.adaptrealtype(::Type{F}, alg::RegularizerSchedule) where {F}
    return convert(RegularizerSchedule{F}, alg)
end

# Interface
# ---------
Base.getindex(alg::RegularizerSchedule, i::Integer) = Regularizer(alg.vals[i], alg.type)
Base.firstindex(alg::RegularizerSchedule) = firstindex(alg.vals)
Base.lastindex(alg::RegularizerSchedule) = lastindex(alg.vals)
Base.length(alg::RegularizerSchedule) = length(alg.vals)

function stepproblem(
    index::Integer, problem::PointProblem{F}, alg::RegularizerSchedule{F}, ::Nothing
) where {F}
    constraint = RegularizerConstraint(alg[index], problem.constraint.q0)
    return update(problem; constraint)
end

function stepproblem(
    index::Integer,
    problem::PointProblem{F},
    alg::RegularizerSchedule{F},
    state::ScheduleState{F},
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
    problem::PointProblem{F}, ::RegularizerSchedule{F}, state::ScheduleState{F}
) where {F}
    return issameproblem(state.problem, problem) || solves(state.solution, problem)
end

# Predicate for whether the target was crossed in the most recent step
function intercepted(
    constraint::RegularizerConstraint, solution1::PointSolution, solution2::PointSolution
)
    cregv, reg1v, reg2v = constraintknots(constraint, solution1, solution2)
    return !samesign(reg1v - cregv, reg2v - cregv)
end

function intercepted(
    constraint::PTraceConstraint, solution1::PointSolution, solution2::PointSolution
)
    ptracec, ptrace1, ptrace2 = constraintknots(constraint, solution1, solution2)
    return !samesign(ptrace1 - ptracec, ptrace2 - ptracec)
end

# Prediction knots derived from problem constraint
function constraintknots(
    constraint::RegularizerConstraint{F},
    solution1::PointSolution{F},
    solution2::PointSolution{F},
) where {F}
    creg = constraint.val
    reg1, reg2 = regularizer(solution1, creg.type), regularizer(solution2, creg.type)
    return creg.val, reg1.val, reg2.val
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
    SymmetrySchedule{F}(steps::Integer) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the degree of symmetry in the ensemble

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``T`` is replaced by ``μT``. The idea is that the equations behave better for ``μ = 0``,
i.e, the asymmetric problem, so we solve that first before gradually turning on the
symmetry.

We know that when ``T ∝ I``, the correct solution branch at ``μ = 1`` is analytically
connected to the solution at ``μ = 0``. However, this may not hold for general ``T``, so
symmetry annealing may not always be a viable solution strategy. For example, the topology
of the spectral support may change as a function of ``μ``. This is different from
`RegularizerSchedule`, where the unique solution in the limit of large regularizer is on the
correct branch by definition. Hence, annealing the regularization is usually the preferred
solution strategy.

The schedule can be specified as an integer `steps`, which will set the schedule to
`range(0, 1, steps)`. Alternatively, a custom schedule can be provided directly in the form
of an `AbstractVector{<:Real}`; this can be useful if, for example, you want to take smaller
steps in an unstable region. An error is thrown if the schedule does not increase
monotonically from 0 to 1.

The default value is

```julia
SymmetrySchedule(Defaults.symmetry_schedule_vals)
```
"""
@auto_hash_equals typearg=true struct SymmetrySchedule{F,U<:AbstractVector{F}} <:
                                      AbstractSchedule{F}
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

function SymmetrySchedule{F}(steps::Integer) where {F}
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
Base.getindex(alg::SymmetrySchedule, i::Integer) = alg.vals[i]
Base.firstindex(alg::SymmetrySchedule) = firstindex(alg.vals)
Base.lastindex(alg::SymmetrySchedule) = lastindex(alg.vals)
Base.length(alg::SymmetrySchedule) = length(alg.vals)

function stepproblem(
    index::Integer, problem::PointProblem{F}, alg::SymmetrySchedule{F}, ::Nothing
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

    # Conversion
    @test DirectSolve(alg) === DirectSolve()
    @test convert(DirectSolve, alg) === alg
    @test adaptrealtype(Float32, alg) == alg
    @test adaptrealtype(Float64, alg) == alg
end

@testitem "RegularizerSchedule" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms:
        RegularizerSchedule, ScheduleState, statetype
    using GaussianSpectralDensities.Defaults: Defaults

    vals64, type = 2.0 .^ (2:-1:-2), :gap
    alg64 = RegularizerSchedule(vals64, type)

    # Defaults
    @test RegularizerSchedule() === RegularizerSchedule(
        Defaults.regularizer_schedule_vals, Defaults.regularizer_schedule_type
    )

    @testset "$F" for F in [Float32, Float64]
        valsF = map(F, vals64)  # map preserves ranges types etc. better than broadcasting
        algF = RegularizerSchedule(valsF, type)

        # Floattype inferring constructor
        @test algF isa RegularizerSchedule{F}
        @test statetype(RegularizerSchedule{F}) === ScheduleState{F}

        # Floattype enforcing constructor
        @test RegularizerSchedule{F}(vals64, type) == algF
        @test RegularizerSchedule{Float64}(valsF, type) == alg64

        # Floattype conversions
        @test RegularizerSchedule{F}(alg64) == algF
        @test RegularizerSchedule{Float64}(algF) == alg64
        @test convert(RegularizerSchedule{F}, alg64) == algF
        @test convert(RegularizerSchedule{Float64}, algF) == alg64
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
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, SymmetrySchedule

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
    @test_throws ArgumentError RegularizerSchedule([1.0, 0.33, 0.67, 0.0], :shift)
    @test_throws "monotonic" RegularizerSchedule([1.0, 0.33, 0.67, 0.0], :gap)
    @test_throws ArgumentError RegularizerSchedule(range(0.0, 1.0, 3), :shift)
    @test_throws "monotonic" RegularizerSchedule(range(0.0, 1.0, 3), :gap)
end

@testitem "DirectSolve/Regularizer" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    constraint64 = RegularizerConstraint(; shift=0.25, q0=(1.0 - im) / 2)

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

@testitem "DirectSolve/PTrace" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    constraint64 = PTraceConstraint(1.0; q0=complex(0.1), shift0=1.0)

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

@testitem "RegularizerSchedule/Regularizer" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: issuccess, q, regularizer, solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    q0_64 = complex(1.0)

    schedule64 = [4.0, 2.0, 1.0, 0.5]
    target_val64 = 0.25

    @testset "Target regularizer type = $reg_type_target" for reg_type_target in
                                                              [:shift, :gap]
        kwargs = (; reg_type_target => target_val64, :q0 => q0_64)
        constraint64 = RegularizerConstraint(; kwargs...)

        @testset "Schedule regularizer type = $reg_type_schedule" for reg_type_schedule in
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
                    algF = RegularizerSchedule{F}(schedule64, reg_type_schedule)

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
                    if (reg_type_target, reg_type_schedule) != (:shift, :gap)
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
                            cq0 = problem.constraint.q0
                            i1, i2 = states[i - 1].index, states[i - 1].previndex
                            sol1, sol2 = solutions[i1], solutions[i2]
                            x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                            @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))
                        end
                    end

                    # Final step initialized by linear *polation from the two previous
                    # solutions
                    problem = states[end].problem
                    creg, cq0 = problem.constraint.val, problem.constraint.q0
                    i1, i2 = states[end - 1].index, states[end - 1].previndex
                    sol1, sol2 = solutions[i1], solutions[i2]
                    reg1, reg2 = regularizer(sol1, creg.type), regularizer(sol2, creg.type)
                    x, x1, x2 = creg.val, reg1.val, reg2.val
                    @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))

                    # Final step solves the actual target problem
                    @test solves(solutions[end], problemF)
                    @test isdone(problemF, algF, states[end])
                end
            end
        end
    end
end

@testitem "RegularizerSchedule/PTrace" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess, ptrace, q, regularizer, solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    q0_64 = complex(0.1)

    schedule64 = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    ptrace64_target = 1.0

    @testset "Schedule regularizer type = $reg_type" for reg_type in [:shift, :gap]
        alg64 = RegularizerSchedule(schedule64, reg_type)
        constraint64 = PTraceConstraint(ptrace64_target; q0=q0_64, reg0=alg64[begin])

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
            eqF = ResolventEquation(ensembleF)
            wsF = Workspace(eqF)
            problemF = PointProblem(eqF, constraint64, z64)
            algF = RegularizerSchedule{F}(alg64)

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
                    cq0 = problem.constraint.q0
                    i1, i2 = states[i - 1].index, states[i - 1].previndex
                    sol1, sol2 = solutions[i1], solutions[i2]
                    x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                    @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))
                end
            end

            # Final step initialized by linear *polation from the two previous solutions
            # Could be either a PTrace problem or a Regularizer problem
            problem = states[end].problem
            cq0 = problem.constraint.q0
            i1, i2 = states[end - 1].index, states[end - 1].previndex
            sol1, sol2 = solutions[i1], solutions[i2]
            if problem.constraint isa PTraceConstraint
                creg0 = problem.constraint.reg0
                reg1, reg2 = regularizer(sol1, creg0.type), regularizer(sol2, creg0.type)
                x, x1, x2 = problem.constraint.val, ptrace(sol1), ptrace(sol2)
                @test isapproxlinear((x, creg0.val), (x1, reg1.val), (x2, reg2.val))
            else
                creg = problem.constraint.val
                reg1, reg2 = regularizer(sol1, creg.type), regularizer(sol2, creg.type)
                x, x1, x2 = creg.val, reg1.val, reg2.val
            end
            @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))

            # Final step solves the actual P-trace constraint
            @test solves(solutions[end], problemF)
            @test isdone(problemF, algF, states[end])
        end
    end
end

@testitem "SymmetrySchedule/Regularizer" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess, q, solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4

    nsteps = 3

    @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
        kwargs = (; reg_type => 0.25, :q0 => complex(1.0))
        constraint64 = RegularizerConstraint(; kwargs...)

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
                        cq0 = problem.constraint.q0
                        i1, i2 = states[i - 1].index, states[i - 1].previndex
                        sol1, sol2 = solutions[i1], solutions[i2]
                        x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                        @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))
                    end
                end

                # Final step solves the actual problem
                @test solves(solutions[end], problemF)
                @test isdone(problemF, algF, states[end])
            end
        end
    end
end

@testitem "SymmetrySchedule/PTrace" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess, q, regularizer, solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    constraint64 = PTraceConstraint(1.0; q0=complex(0.1), shift0=1.0)

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
                cq0, creg0 = problem.constraint.q0, problem.constraint.reg0
                i1, i2 = states[i - 1].index, states[i - 1].previndex
                sol1, sol2 = solutions[i1], solutions[i2]
                reg1, reg2 = regularizer(sol1, creg0.type), regularizer(sol2, creg0.type)
                x, x1, x2 = scheduleF[i], scheduleF[i1], scheduleF[i2]
                @test isapproxlinear((x, creg0.val), (x1, reg1.val), (x2, reg2.val))
                @test isapproxlinear((x, cq0), (x1, q(sol1)), (x2, q(sol2)))
            end
        end

        # Final step solves the actual problem
        @test solves(solutions[end], problemF)
        @test isdone(problemF, algF, states[end])
    end
end

@testitem "RegularizerSchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: shift
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    shiftmin = 0

    # z outside Ginibre support => negative shift needed to reach k = 1.0
    z64 = 1.0 + 1.0im
    q0_64 = complex(0.1)

    alg64 = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], :gap)
    constraint64 = PTraceConstraint(1.0; q0=q0_64, gap0=4.0)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.0I)
        eqF = ResolventEquation(ensembleF; shiftmin)
        wsF = Workspace(eqF)
        problemF = PointProblem(eqF, constraint64, z64)
        algF = RegularizerSchedule{F}(alg64)

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
    using GaussianSpectralDensities.Regularizers: shift
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    shiftmin = 0

    # z inside Ginibre but outside Girko support => negative shift needed to reach k = 1.0
    z64 = 0.25 + 0.75im
    constraint64 = PTraceConstraint(1.0; q0=complex(0.1), shift0=1.0)

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
