module Algorithms

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..Problems: PTraceProblem, PointProblem, RegularizerProblem, update
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: PointSolution
using ..Workspaces: Workspace

using ..ArrayTools: ArrayTools
using ..MathTools: monotonic, predictlinear, samesign

public AbstractAlgorithm,
    AbstractAlgorithmState,
    AbstractSchedule,
    DirectSolve,
    RegularizerSchedule,
    SymmetrySchedule,
    isdone,
    statetype,
    step!

abstract type AbstractAlgorithm end
abstract type AbstractAlgorithmState end

## AbstractAlgorithm interface
## ===========================
## Required methods:
##
## - ArrayTools.adaptrealtype(::Type{<:AbstractFloat}, ::SomeAlgorithm)::SomeAlgorithm
## - nextindex(::SomeAlgorithm[, ::SomeAlgorithmState])
## - isdone(::PointProblem, ::SomeAlgorithm, ::SomeAlgorithmState)::Bool
## - steppointproblem(index, ::PointProblem, ::SomeAlgorithm[, ::SomeAlgorithmState])::PointProblem
## - newstate(index, ::PointProblem, ::PointSolution, ::SomeAlgorithm[, ::SomeAlgorithmState])::SomeAlgorithmState
## - statetype(::Type{SomeAlgorithm})::Type{SomeAlgorithmState}
##
## Algorithms that implement the indexing interface (Base.getindex, Base.firstindex,
## Base.lastindex, Base.length) should forward the single-argument method
## nextindex(::SomeAlgorithm) # to Base.firstindex.
function nextindex end
function isdone end
function steppointproblem end
function newstate end
function statetype end

## Generic stepping implementation
## ===============================
## Provided interface methods:
##
## - step!(::Workspace{F}, ::PointProblem{F}[, ::AbstractAlgorithm])::Union{Nothing,Tuple{PointSolution{F},AbstractAlgorithmState}}
function step!(
    ws::Workspace{F},
    pp::PointProblem{F},
    alg::AbstractAlgorithm,
    state::AbstractAlgorithmState...,
) where {F}
    (!isempty(state)) && isdone(pp, alg, state...) && return nothing
    index = nextindex(alg, state...)
    currpp = steppointproblem(index, pp, alg, state...)
    currsolution = eqsolve!(ws, currpp.problem, currpp.z, currpp.init)
    return currsolution, newstate(index, currpp, currsolution, alg, state...)
end

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
nextindex(::DirectSolve, ::DirectSolveState...) = nothing
isdone(::PointProblem, ::DirectSolve, ::DirectSolveState) = true

steppointproblem(::Nothing, pp::PointProblem, ::DirectSolve) = pp

newstate(::Nothing, ::PointProblem, ::PointSolution, ::DirectSolve) = DirectSolveState()

statetype(::Type{DirectSolve}) = DirectSolveState

## Schedule algorithms
## ===================
## Provided types:
##
## - ScheduleState{F} with properties currpp, currsolution, prevsolution, index::Integer
##
## Provided interface methods:
##
## - nextindex(::AbstractSchedule{F}, ::ScheduleState{F})::Int
## - isdone(::PointProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - newstate(::Integer, PointProblem{F}, ::PointSolution{F}, ::AbstractSchedule{F}[, ::ScheduleState{F}])::ScheduleState{F}
## - statetype(::Type{<:AbstractSchedule{F}})::Type{ScheduleState{F}}
##
## Provided non-interface functions:
##
## - steppointproblem_predict(::Integer, ::PointProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::PointProblem{F}
## - updateinit_predict(::PointProblem{F}, ::PointSolution{F}, ::PointSolution{F})::PointProblem{F}
##
## Required methods:
##
## - Indexing interface (Base.getindex, Base.firstindex, Base.lastindex, Base.length)
## - Base.convert(::Type{SomeSchedule{F}}, alg::SomeSchedule)::SomeSchedule{F}
## - ArrayTools.adaptrealtype(::Type{F}, ::SomeSchedule)::SomeSchedule{F}
## - unwrap(::SomeSchedule): should return indexable collection of unwrapped schedule
##     values; while indexing the schedule itself may return an instance of a wrapper,
##     indexing this collection should return a plain number
## - iscomplete(::PointProblem{F}, ::SomeSchedule{F}, ::ScheduleState{F})::Bool
## - steppointproblem(::Integer, ::PointProblem{F}, ::SomeSchedule{F})::PointProblem{F}
##
## NOTE: steppointproblem_predict works by calling the three-argument stepproblem (the
## stateless # version for the initial step) and using updateinit_predit to update the
## initial guess based on the last two steps of the algorithm. The provided four-argument
## implementation of steppointproblem just forwards to steppointproblem_predict. The
## three-argument method _must_ be implemented for each subtype, and the four-argument
## method _may_ be implemented if stepproblempredict is not sufficient.
##
## NOTE: iscomplete should only be concerned with whether the algorithm has reached its
## endpoint. Other termination criteria, including whether the most recent step produced a
## successful solution, are handled by the provided isdone method.
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end

struct ScheduleState{F,P<:PointProblem{F}} <: AbstractAlgorithmState
    currpp::P
    currsolution::PointSolution{F}
    prevsolution::PointSolution{F}
    index::Int
end

function unwrap end
function iscomplete end
function steppointproblem end

nextindex(alg::AbstractSchedule) = firstindex(alg)
nextindex(::AbstractSchedule{F}, state::ScheduleState{F}) where {F} = (state.index + 1)

function isdone(
    pp::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    return (!state.currsolution.success) || iscomplete(pp, alg, state)
end

function steppointproblem(
    index::Integer, pp::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    return steppointproblem_predict(index, pp, alg, state)
end

function steppointproblem_predict(
    index::Integer, pp::PointProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    values = unwrap(alg)
    v, v1, v2 = values[index], values[state.index], values[max(state.index - 1, begin)]
    new_pp = steppointproblem(index, pp, alg)
    return updateinit_predict(new_pp, v, (v1, state.currsolution), (v2, state.prevsolution))
end

function newstate(
    index::Integer,
    currpp::PointProblem{F},
    currsolution::PointSolution{F},
    ::AbstractSchedule{F},
    state::ScheduleState{F}...,
) where {F}
    prevsolution = isempty(state) ? currsolution : first(state).currsolution
    return ScheduleState(currpp, currsolution, prevsolution, index)
end

statetype(::Type{<:AbstractSchedule{F}}) where {F} = ScheduleState{F}

# Prediction using arbitrary knots
# TODO: Would be neater to dispatch on a type instead of a type parameter
function updateinit_predict(
    pp::PointProblem{F,<:RegularizerProblem{F}},
    v,
    (v1, solution1)::Tuple{F,PointSolution{F}},
    (v2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    q1, q2 = solution1.q, solution2.q
    return update(pp; init=(; q=predictlinear(v, (v1, q1), (v2, q2))))
end

function updateinit_predict(
    pp::PointProblem{F,<:PTraceProblem{F}},
    v,
    (v1, solution1)::Tuple{F,PointSolution{F}},
    (v2, solution2)::Tuple{F,PointSolution{F}},
) where {F}
    reg_type = Symbol(pp.init.reg.type)
    regv1 = getproperty(solution1, reg_type)
    regv2 = getproperty(solution2, reg_type)
    q1, q2 = solution1.q, solution2.q
    reg = Regularizer(predictlinear(v, (v1, regv1), (v2, regv2)), reg_type)
    q = predictlinear(v, (v1, q1), (v2, q2))
    return update(pp; init=(; reg, q))
end

# Target-based prediction (knots from problem objective)
function updateinit_predict(
    pp::PointProblem{F,<:RegularizerProblem{F}},
    solution1::PointSolution{F},
    solution2::PointSolution{F},
) where {F}
    v, reg_type = pp.problem.reg.value, Symbol(pp.problem.reg.type)
    v1, v2 = getproperty(solution1, reg_type), getproperty(solution2, reg_type)
    return updateinit_predict(pp, v, (v1, solution1), (v2, solution2))
end

function updateinit_predict(
    pp::PointProblem{F,<:PTraceProblem{F}},
    solution1::PointSolution{F},
    solution2::PointSolution{F},
) where {F}
    v, v1, v2 = pp.problem.ptrace, solution1.ptrace, solution2.ptrace
    return updateinit_predict(pp, v, (v1, solution1), (v2, solution2))
end

"""
    RegularizerSchedule()
    RegularizerSchedule(values, type)
    RegularizerSchedule{F}(values, type) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the regularizer

The intended way to use this algorithm is to start from a large regularizer to ensure
convergence to the correct solution branch, and schedule a gradual decay to approach the
true solution along this branch.

The parameter `values` should be a vector of numerical values, and `type` is the type of
regularizer, see [`Regularizer`](@ref) for details. An error is thrown if the values do not
monotonically decrease.

The final step of the algorithm is a direct solve of the averaged resolvent problem. This
either happens at the end of the prescribed schedule or when intersecting the target during
the annealing, whichever comes first.

The default value is

```julia
RegularizerSchedule(
    Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
)
```
"""
struct RegularizerSchedule{F,U<:AbstractVector{F}} <: AbstractSchedule{F}
    values::U
    type::RegularizerType.T

    function RegularizerSchedule{F}(values, type) where {F}
        # map preserves ranges types etc. better than broadcasting
        valuesF = (eltype(values) === F) ? values : map(F, values)
        schedule = new{F,typeof(valuesF)}(valuesF, type)
        # Validate post-instantiation in case the original values was a single-use iterator
        if !monotonic(Iterators.reverse(schedule.values))
            throw(ArgumentError("regularizer schedule must decrease monotonically"))
        end
        return schedule
    end
end

RegularizerSchedule(values, type) = RegularizerSchedule{eltype(values)}(values, type)

function RegularizerSchedule()
    values, type = Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
    return RegularizerSchedule(values, type)
end

function RegularizerSchedule{F}(alg::RegularizerSchedule) where {F}
    return RegularizerSchedule{F}(unwrap(alg), alg.type)
end

# Traits, comparisons, conversions
# --------------------------------
function Base.:(==)(a::A, b::A) where {A<:RegularizerSchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
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
Base.getindex(alg::RegularizerSchedule, i::Integer) = Regularizer(unwrap(alg)[i], alg.type)
Base.firstindex(alg::RegularizerSchedule) = firstindex(unwrap(alg))
Base.lastindex(alg::RegularizerSchedule) = lastindex(unwrap(alg))
Base.length(alg::RegularizerSchedule) = length(unwrap(alg))

unwrap(alg::RegularizerSchedule) = alg.values

function iscomplete(
    pp::PointProblem{F}, ::RegularizerSchedule{F}, state::ScheduleState{F}
) where {F}
    (pp.problem == state.currpp.problem) && return true
    return issolvedby(pp.problem, state.currsolution)
end

# FIXME: We should be better able to verify that the solution is consistent with the problem
function issolvedby(problem::RegularizerProblem{F}, solution::PointSolution{F}) where {F}
    (; reg) = problem
    return solution.success && (getproperty(solution, Symbol(reg.type)) == reg.value)
end

function issolvedby(problem::PTraceProblem{F}, solution::PointSolution{F}) where {F}
    return solution.success && (solution.ptrace == problem.ptrace)
end

function steppointproblem(
    index::Integer, pp::PointProblem{F}, alg::RegularizerSchedule{F}
) where {F}
    step_problem = RegularizerProblem(pp.problem.eq, alg[index])
    return update(pp; problem=step_problem)
end

function steppointproblem(
    index::Integer,
    pp::PointProblem{F},
    alg::RegularizerSchedule{F},
    state::ScheduleState{F},
) where {F}
    if (state.index == lastindex(alg)) || didintercepttarget(pp.problem, state)
        # Final step: return to target problem with predicted init
        return updateinit_predict(pp, state.currsolution, state.prevsolution)
    end
    return steppointproblem_predict(index, pp, alg, state)
end

function didintercepttarget(
    problem::RegularizerProblem{F}, state::ScheduleState{F}
) where {F}
    regv, reg_type = problem.reg.value, Symbol(problem.reg.type)
    prev_regv = getproperty(state.prevsolution, reg_type)
    curr_regv = getproperty(state.currsolution, reg_type)
    return !samesign(curr_regv - regv, prev_regv - regv)
end

function didintercepttarget(problem::PTraceProblem{F}, state::ScheduleState{F}) where {F}
    prev_ptrace, curr_ptrace = state.prevsolution.ptrace, state.currsolution.ptrace
    return !samesign(curr_ptrace - problem.ptrace, prev_ptrace - problem.ptrace)
end

"""
    SymmetrySchedule()
    SymmetrySchedule(values)
    SymmetrySchedule{F}(values) where {F<:AbstractFloat}
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
SymmetrySchedule(Defaults.symmetry_schedule_values)
```
"""
struct SymmetrySchedule{F,U<:AbstractVector{F}} <: AbstractSchedule{F}
    values::U

    function SymmetrySchedule{F}(values) where {F}
        # map preserves ranges types etc. better than broadcasting
        valuesF = (eltype(values) === F) ? values : map(F, values)
        schedule = new{F,typeof(valuesF)}(valuesF)
        # Validate post-instantiation in case the original values was a single-use iterator
        if (
            (first(schedule.values) != 0) ||
            (last(schedule.values) != 1) ||
            !monotonic(schedule.values)
        )
            throw(ArgumentError("symmetry schedule must increase monotonically 0 to 1"))
        end
        return schedule
    end
end

function SymmetrySchedule{F}(steps::Integer) where {F}
    return SymmetrySchedule{F}(range(zero(F), one(F), steps))
end

SymmetrySchedule(values) = SymmetrySchedule{eltype(values)}(values)
SymmetrySchedule() = SymmetrySchedule(Defaults.symmetry_schedule_values)

function SymmetrySchedule{F}(alg::SymmetrySchedule) where {F}
    return SymmetrySchedule{F}(unwrap(alg))
end

# Traits, comparisons, conversions
# --------------------------------
function Base.:(==)(a::A, b::A) where {A<:SymmetrySchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, alg::SymmetrySchedule) where {F,P<:SymmetrySchedule{F}}
    return (alg isa P) ? alg : SymmetrySchedule{F}(alg)
end

function ArrayTools.adaptrealtype(::Type{F}, alg::SymmetrySchedule) where {F}
    return convert(SymmetrySchedule{F}, alg)
end

# Interface
# ---------
Base.getindex(alg::SymmetrySchedule, i::Integer) = unwrap(alg)[i]
Base.firstindex(alg::SymmetrySchedule) = firstindex(unwrap(alg))
Base.lastindex(alg::SymmetrySchedule) = lastindex(unwrap(alg))
Base.length(alg::SymmetrySchedule) = length(unwrap(alg))

unwrap(alg::SymmetrySchedule) = alg.values

function iscomplete(
    ::PointProblem{F}, alg::SymmetrySchedule{F}, state::ScheduleState{F}
) where {F}
    return (state.index == lastindex(alg))
end

function steppointproblem(
    index::Integer, pp::PointProblem{F}, alg::SymmetrySchedule{F}
) where {F}
    new_eq = update(pp.problem.eq; sscale=alg[index])
    new_problem = update(pp.problem; eq=new_eq)
    return update(pp; problem=new_problem)
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

    values64, type = 2.0 .^ (2:-1:-2), :gap
    alg64 = RegularizerSchedule(values64, type)

    # Defaults
    @test RegularizerSchedule() === RegularizerSchedule(
        Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
    )

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = RegularizerSchedule(valuesF, type)

        # Floattype inferring constructor
        @test algF isa RegularizerSchedule{F}
        @test statetype(RegularizerSchedule{F}) === ScheduleState{F}

        # Floattype enforcing constructor
        @test RegularizerSchedule{F}(values64, type) == algF
        @test RegularizerSchedule{Float64}(valuesF, type) == alg64

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
    using GaussianSpectralDensities.Algorithms:
        ScheduleState, SymmetrySchedule, statetype, unwrap
    using GaussianSpectralDensities.Defaults: Defaults

    steps = 3
    values64 = range(0.0, 1.0, steps)

    alg64 = SymmetrySchedule(values64)

    # All the signatures
    @test SymmetrySchedule{Float64}(steps) === alg64

    # Defaults
    @test SymmetrySchedule() === SymmetrySchedule(Defaults.symmetry_schedule_values)

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = SymmetrySchedule(valuesF)

        # Floattype inferring constructor
        @test algF isa SymmetrySchedule{F}
        @test statetype(SymmetrySchedule{F}) === ScheduleState{F}

        # Floattype enforcing constructor
        @test SymmetrySchedule{F}(values64) == algF
        @test SymmetrySchedule{Float64}(valuesF) == alg64

        # Upconvert algF properties to capture roundtrip truncation before equality testing
        algF64 = SymmetrySchedule(map(Float64, unwrap(algF)))

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

@testitem "DirectSolve/RegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    init64 = (; q=(1.0 - im) / 2)
    reg64 = Regularizer(0.25, :shift)

    @testset "$label" for (label, A64, D64) in [
        ("Array", A64data, Diagonal(D64data)),
        ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
        ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
    ]
        ensemble64 = MatrixEnsemble(A64, I, I, D64)
        eq64 = ResolventEquation(ensemble64)
        ws64 = Workspace(eq64)
        problem64 = RegularizerProblem(eq64, reg64)
        pp64 = PointProblem(problem64, z64, init64)

        alg = DirectSolve()
        solution64, state64 = step!(ws64, pp64, alg)

        # Stops after one iteration
        @test nothing === step!(ws64, pp64, alg, state64)

        # Solves the actual target problem
        @test solution64.success
        @test solution64.z == z64
        @test getproperty(solution64, Symbol(reg64.type)) == reg64.value
        @test isdone(pp64, alg, state64)
    end
end

@testitem "DirectSolve/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    ptrace64_target = 1.0

    ensemble64 = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq64 = ResolventEquation(ensemble64)
    ws64 = Workspace(eq64)
    problem64 = PTraceProblem(eq64, ptrace64_target)
    pp64 = PointProblem(problem64, z64, init64)

    alg64 = DirectSolve()
    solution64, state64 = step!(ws64, pp64, alg64)

    # Stops after one iteration
    @test nothing === step!(ws64, pp64, alg64, state64)

    # Solves the actual target problem
    @test solution64.success
    @test solution64.z == z64
    @test solution64.ptrace ≈ ptrace64_target
    @test isdone(pp64, alg64, state64)
end

@testitem "RegularizerSchedule/RegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4

    schedule_values64 = [4.0, 2.0, 1.0, 0.5]
    target_value64 = 0.25

    @testset "Target regularizer type = $reg_type_target" for reg_type_target in
                                                              [:shift, :gap]
        reg64_target = Regularizer(target_value64, reg_type_target)

        @testset "Schedule regularizer type = $reg_type_schedule" for reg_type_schedule in
                                                                      [:shift, :gap]
            init64 = (;
                reg=Regularizer(schedule_values64[1], reg_type_schedule), q=complex(1.0)
            )

            @testset "$label" for (label, A64, D64) in [
                ("Array", A64data, Diagonal(D64data)),
                ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
                ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
            ]
                @testset "$F" for F in [Float32, Float64]
                    ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                    eqF = ResolventEquation(ensembleF)
                    wsF = Workspace(eqF)
                    problemF = RegularizerProblem(eqF, reg64_target)
                    ppF = PointProblem(problemF, z64, init64)
                    algF = RegularizerSchedule{F}(schedule_values64, reg_type_schedule)

                    # Collect all solutions and states
                    solutions = []
                    states = []
                    result = step!(wsF, ppF, algF)
                    while !isnothing(result)
                        solution, state = result
                        push!(solutions, solution)
                        push!(states, state)
                        result = step!(wsF, ppF, algF, state)
                    end

                    # In most cases, solutions comprise all schedule steps and the final
                    # step to the target. The exception is (target, schedule) == (:shift,
                    # :gap), in which the solution may be intersected before exhausting the
                    # schedule, triggering an early final step.
                    if (reg_type_target, reg_type_schedule) != (:shift, :gap)
                        @test length(solutions) == length(algF) + 1
                    end

                    # Verify integrity of intermediate steps
                    @testset "index = $i" for i in
                                              eachindex(states, solutions)[begin:(end - 1)]
                        @test solutions[i].success
                        @test states[i].index == i
                        @test states[i].currsolution === solutions[i]
                        @test !isdone(ppF, algF, states[i])
                        if i > firstindex(solutions)
                            # Each schedule step initialized by linear *polation from the
                            # two previous solutions
                            pp = states[i].currpp
                            sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                            q0 = pp.init.q
                            values = unwrap(algF)
                            v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                            @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))
                        end
                    end

                    # Final step initialized by linear *polation from the two previous
                    # solutions
                    pp = states[end].currpp
                    sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
                    v, q0 = pp.problem.reg.value, pp.init.q
                    v1 = getproperty(sol1, Symbol(pp.problem.reg.type))
                    v2 = getproperty(sol2, Symbol(pp.problem.reg.type))
                    @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))

                    # Final step solves the actual target problem
                    @test solutions[end].success
                    @test solutions[end].z == z64
                    @test getproperty(solutions[end], reg_type_target) ≈ reg64_target.value
                    @test isdone(ppF, algF, states[end])
                end
            end
        end
    end
end

@testitem "RegularizerSchedule/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    z64 = 0.5 + 0.5im
    q0_64 = complex(0.1)

    schedule_values64 = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    ptrace64_target = 1.0

    @testset "Schedule regularizer type = $reg_type" for reg_type in [:shift, :gap]
        alg64 = RegularizerSchedule(schedule_values64, reg_type)
        init64 = (; reg=alg64[begin], q=q0_64)

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
            eqF = ResolventEquation(ensembleF)
            wsF = Workspace(eqF)
            algF = RegularizerSchedule{F}(alg64)
            problemF = PTraceProblem(eqF, ptrace64_target)
            ppF = PointProblem(problemF, z64, init64)

            # Collect all solutions and states
            solutions = []
            states = []
            result = step!(wsF, ppF, algF)
            while !isnothing(result)
                solution, state = result
                push!(solutions, solution)
                push!(states, state)
                result = step!(wsF, ppF, algF, state)
            end

            # Stepping should terminate before exhaustion as ptrace crosses target
            @test length(solutions) < length(algF) + 1

            # Verify integrity of intermediate steps
            @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                @test solutions[i].success
                @test states[i].index == i
                @test states[i].currsolution === solutions[i]
                @test !isdone(ppF, algF, states[i])
                if i > firstindex(solutions)
                    # Each schedule step initialized by linear *polation from the two
                    # previous solutions
                    pp = states[i].currpp
                    sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                    q0 = pp.init.q
                    values = unwrap(algF)
                    v, v1, v2 = pp.problem.reg.value,
                    values[i - 1],
                    values[max(i - 2, begin)]
                    @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))
                end
            end

            # Final step initialized by linear *polation from the two previous solutions
            # Could be either a trace problem or a regularizer problem
            pp = states[end].currpp
            sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
            if pp.problem isa PTraceProblem
                v, reg0 = pp.problem.ptrace, pp.init.reg
                v1, reg1v = sol1.ptrace, getproperty(sol1, Symbol(reg0.type))
                v2, reg2v = sol2.ptrace, getproperty(sol2, Symbol(reg0.type))
                @test isapproxlinear((v, reg0.value), (v1, reg1v), (v2, reg2v))
            else
                v = pp.problem.reg.value
                v1 = getproperty(sol1, Symbol(pp.problem.reg.type))
                v2 = getproperty(sol2, Symbol(pp.problem.reg.type))
            end
            @test isapproxlinear((v, pp.init.q), (v1, sol1.q), (v2, sol2.q))

            # Final step solves the actual PTraceProblem
            @test solutions[end].success
            @test solutions[end].z == z64
            @test solutions[end].ptrace ≈ ptrace64_target
            @test isdone(ppF, algF, states[end])
        end
    end
end

@testitem "SymmetrySchedule/RegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems:
        PointProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    init64 = (; q=complex(1.0))

    nsteps = 3

    @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
        reg64_target = Regularizer(0.25, reg_type)

        @testset "$label" for (label, A64, D64) in [
            ("Array", A64data, Diagonal(D64data)),
            ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
            ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
        ]
            @testset "$F" for F in [Float32, Float64]
                ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                eqF = ResolventEquation(ensembleF)
                wsF = Workspace(eqF)
                problemF = RegularizerProblem(eqF, reg64_target)
                ppF = PointProblem(problemF, z64, init64)
                algF = SymmetrySchedule{F}(nsteps)

                # Collect all solutions and states
                solutions = []
                states = []
                result = step!(wsF, ppF, algF)
                while !isnothing(result)
                    solution, state = result
                    push!(solutions, solution)
                    push!(states, state)
                    result = step!(wsF, ppF, algF, state)
                end

                # solutions comprise schedule steps (no separate final step here)
                @test length(solutions) == length(algF)

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)
                    @test solutions[i].success
                    @test states[i].index == i
                    @test states[i].currsolution === solutions[i]
                    if i < lastindex(solutions)
                        @test !isdone(ppF, algF, states[i])
                    end
                    if i > firstindex(solutions)
                        # Each schedule step initialized by linear *polation from the two
                        # previous solutions
                        pp = states[i].currpp
                        sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                        values = unwrap(algF)
                        v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                        @test isapproxlinear((v, pp.init.q), (v1, sol1.q), (v2, sol2.q))
                    end
                end

                # Final step solves the actual problem
                @test solutions[end].z == z64
                @test getproperty(solutions[end], reg_type) ≈ reg64_target.value
                @test isdone(ppF, algF, states[end])
            end
        end
    end
end

@testitem "SymmetrySchedule/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    ptrace64_target = 1.0
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    nsteps = 3

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
        eqF = ResolventEquation(ensembleF)
        wsF = Workspace(eqF)
        problemF = PTraceProblem(eqF, ptrace64_target)
        ppF = PointProblem(problemF, z64, init64)
        algF = SymmetrySchedule{F}(nsteps)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, ppF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, ppF, algF, state)
        end

        # solutions comprise schedule steps (no extra final step for symmetry schedule)
        @test length(solutions) == length(algF)

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)
            @test solutions[i].success
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            if i < lastindex(solutions)
                @test !isdone(ppF, algF, states[i])
            end
            if i > firstindex(solutions)
                # Each schedule step initialized by linear *polation from the two previous
                # solutions
                pp = states[i].currpp
                sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                reg0 = pp.init.reg
                reg1v = getproperty(sol1, Symbol(reg0.type))
                reg2v = getproperty(sol2, Symbol(reg0.type))
                values = unwrap(algF)
                v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                @test isapproxlinear((v, reg0.value), (v1, reg1v), (v2, reg2v))
                @test isapproxlinear((v, pp.init.q), (v1, sol1.q), (v2, sol2.q))
            end
        end

        # Final step solves the actual problem
        @test solutions[end].z == z64
        @test solutions[end].ptrace ≈ ptrace64_target
        @test isdone(ppF, algF, states[end])
    end
end

@testitem "RegularizerSchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    shiftmin = 0

    # z outside Ginibre support => negative shift needed to reach k = 1.0
    z64 = 1.0 + 1.0im
    q0_64 = complex(0.1)

    alg64 = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], :gap)
    ptrace64_target = 1.0

    init64 = (; reg=alg64[begin], q=q0_64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.0I)
        eqF = ResolventEquation(ensembleF; shiftmin)
        wsF = Workspace(eqF)
        algF = RegularizerSchedule{F}(alg64)
        problemF = PTraceProblem(eqF, ptrace64_target)
        ppF = PointProblem(problemF, z64, init64)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, ppF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, ppF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF) + 1

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < eqF.shiftmin
        @test isdone(ppF, algF, states[end])
    end
end

@testitem "SymmetrySchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    shiftmin = 0

    # z inside Ginibre but outside Girko support => negative shift needed to reach k = 1.0
    z64 = 0.25 + 0.75im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    nsteps = 5
    ptrace64_target = 1.0

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.75I)
        eqF = ResolventEquation(ensembleF; shiftmin)
        wsF = Workspace(eqF)
        problemF = PTraceProblem(eqF, ptrace64_target)
        ppF = PointProblem(problemF, z64, init64)
        algF = SymmetrySchedule{F}(nsteps)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, ppF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, ppF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF)

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < eqF.shiftmin
        @test isdone(ppF, algF, states[end])
    end
end
