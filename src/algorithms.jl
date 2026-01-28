module Algorithms

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..Problems:
    AbstractResolventProblem, PTraceProblem, RegularizerProblem, issameproblem, update
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: ResolventSolution
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
    step!

abstract type AbstractAlgorithm end
abstract type AbstractAlgorithmState end

## AbstractAlgorithm interface
## ===========================
## Required methods:
##
## - ArrayTools.adaptrealtype(::Type{<:AbstractFloat}, ::SomeAlgorithm)::SomeAlgorithm
## - nextindex(::SomeAlgorithm[, ::SomeAlgorithmState])
## - isdone(::AbstractResolventProblem, ::SomeAlgorithm, ::SomeAlgorithmState)::Bool
## - stepproblem(index, ::AbstractResolventProblem, ::SomeAlgorithm[, ::SomeAlgorithmState])::AbstractResolventProblem
## - newstate(index, ::AbstractResolventProblem, ::ResolventSolution, ::SomeAlgorithm[, ::SomeAlgorithmState])::SomeAlgorithmState
##
## Algorithms that implement the indexing interface (Base.getindex, Base.firstindex,
## Base.lastindex, Base.length) should forward the single-argument method
## nextindex(::SomeAlgorithm) # to Base.firstindex.
function nextindex end
function isdone end
function stepproblem end
function newstate end

## Generic stepping implementation
## ===============================
## Provided interface methods:
##
## - step!(::Workspace{F}, ::AbstractResolventProblem{F}[, ::AbstractAlgorithm])::Union{Nothing,Tuple{ResolventSolution{F},AbstractAlgorithmState}}
function step!(
    ws::Workspace{F},
    problem::AbstractResolventProblem{F},
    alg::AbstractAlgorithm,
    state::AbstractAlgorithmState...,
) where {F}
    (!isempty(state)) && isdone(problem, alg, state...) && return nothing
    index = nextindex(alg, state...)
    currproblem = stepproblem(index, problem, alg, state...)
    currsolution = eqsolve!(ws, currproblem)
    return currsolution, newstate(index, currproblem, currsolution, alg, state...)
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
isdone(::AbstractResolventProblem, ::DirectSolve, ::DirectSolveState) = true

function stepproblem(::Nothing, problem::AbstractResolventProblem, ::DirectSolve)
    return problem
end

function newstate(::Nothing, ::AbstractResolventProblem, ::ResolventSolution, ::DirectSolve)
    return DirectSolveState()
end

## Schedule algorithms
## ===================
## Provided types:
##
## - ScheduleState{F} with properties currproblem, currsolution, prevsolution, index::Integer
##
## Provided interface methods:
##
## - nextindex(::AbstractSchedule{F}, ::ScheduleState{F})::Int
## - isdone(::AbstractResolventProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - stepproblem(::Integer, ::AbstractResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::AbstractResolventProblem{F}
## - newstate(::Integer, AbstractResolventProblem{F}, ::ResolventSolution{F}, ::AbstractSchedule{F}[, ::ScheduleState{F}])::ScheduleState{F}
##
## Provided non-interface functions:
##
## - stepproblempredict(::Integer, ::AbstractResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::AbstractResolventProblem{F}
## - predictinit(::AbstractResolventProblem{F}, ::ResolventSolution{F}, ::ResolventSolution{F})::AbstractResolventProblem{F}
##
## Required methods:
##
## - Indexing interface (Base.getindex, Base.firstindex, Base.lastindex, Base.length)
## - unwrap(::SomeSchedule): should return indexable collection of unwrapped schedule
##     values; while indexing the schedule itself may return an instance of a wrapper,
##     indexing this collection should return a plain number
## - Base.convert(::Type{SomeSchedule{F}}, alg::SomeSchedule)::SomeSchedule{F}
## - ArrayTools.adaptrealtype(::Type{F}, ::SomeSchedule)::SomeSchedule{F}
## - iscomplete(::AbstractResolventProblem{F}, ::SomeSchedule{F}, ::ScheduleState{F})::Bool
## - stepproblem(::Integer, ::AbstractResolventProblem{F}, ::SomeSchedule{F})::AbstractResolventProblem{F}
##
## NOTE: stepproblempredict works by calling the three-argument stepproblem (the stateless
## version for the initial step) and using predictinit to update the initial guess based on
## the state. The provided four-argument implementation of stepproblem just forwards to
## stepproblempredict. The three-argument method _must_ be implemented for each subtype, and
## the four-argument method _may_ be implemented if stepproblempredict is not sufficient.
##
## NOTE: iscomplete should only be concerned with whether the algorithm has reached its
## endpoint. Other termination criteria, including whether the most recent step produced a
## successful solution, are handled by the provided isdone method.
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end

struct ScheduleState{
    F,P<:AbstractResolventProblem{F},S<:ResolventSolution{F}
} <: AbstractAlgorithmState
    currproblem::P
    currsolution::S
    prevsolution::S
    index::Int
end

function unwrap end
function iscomplete end
function stepproblem end

nextindex(alg::AbstractSchedule) = firstindex(alg)
nextindex(::AbstractSchedule{F}, state::ScheduleState{F}) where {F} = (state.index + 1)

function isdone(
    problem::AbstractResolventProblem{F}, alg::AbstractSchedule{F}, state::ScheduleState{F}
) where {F}
    return (!state.currsolution.success) || iscomplete(problem, alg, state)
end

function stepproblem(
    index::Integer,
    problem::AbstractResolventProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    return stepproblempredict(index, problem, alg, state)
end

function newstate(
    index::Integer,
    currproblem::AbstractResolventProblem{F},
    currsolution::ResolventSolution{F},
    ::AbstractSchedule{F},
    state::ScheduleState{F}...,
) where {F}
    prevsolution = isempty(state) ? currsolution : first(state).currsolution
    return ScheduleState(currproblem, currsolution, prevsolution, index)
end

function stepproblempredict(
    index::Integer,
    problem::AbstractResolventProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    values = unwrap(alg)
    v, curr_v, prev_v = values[index], values[index - 1], values[max(index - 2, begin)]
    step_problem = stepproblem(index, problem, alg)
    (; currsolution, prevsolution) = state
    return predictinit(step_problem, v, (curr_v, currsolution), (prev_v, prevsolution))
end

# Arbitrary knots, e.g., unwrap(alg)
function predictinit(
    problem::RegularizerProblem{F},
    v::F,
    (v1, solution1)::Tuple{F,ResolventSolution{F}},
    (v2, solution2)::Tuple{F,ResolventSolution{F}},
) where {F}
    q1, q2 = solution1.q, solution2.q
    return update(problem; init=(; q=predictlinear(v, (v1, q1), (v2, q2))))
end

function predictinit(
    problem::PTraceProblem{F},
    v::F,
    (v1, solution1)::Tuple{F,ResolventSolution{F}},
    (v2, solution2)::Tuple{F,ResolventSolution{F}},
) where {F}
    reg_type = Symbol(problem.init.reg.type)
    regv1, q1 = getproperty(solution1, reg_type), solution1.q
    regv2, q2 = getproperty(solution2, reg_type), solution2.q
    reg = Regularizer(predictlinear(v, (v1, regv1), (v2, regv2)), reg_type)
    q = predictlinear(v, (v1, q1), (v2, q2))
    return update(problem; init=(; reg, q))
end

# Knots derived from problem's objective and corresponding solution values
function predictinit(
    problem::RegularizerProblem{F},
    solution1::ResolventSolution{F},
    solution2::ResolventSolution{F},
) where {F}
    v, reg_type = problem.reg.value, Symbol(problem.reg.type)
    v1, v2 = getproperty(solution1, reg_type), getproperty(solution2, reg_type)
    return predictinit(problem, v, (v1, solution1), (v2, solution2))
end

function predictinit(
    problem::PTraceProblem{F},
    solution1::ResolventSolution{F},
    solution2::ResolventSolution{F},
) where {F}
    v, v1, v2 = problem.ptrace, solution1.ptrace, solution2.ptrace
    return predictinit(problem, v, (v1, solution1), (v2, solution2))
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
    problem::AbstractResolventProblem{F}, ::RegularizerSchedule{F}, state::ScheduleState{F}
) where {F}
    issameproblem(problem, state.currproblem) && return true
    return issolvedby(problem, state.currsolution)
end

function issolvedby(
    problem::RegularizerProblem{F}, solution::ResolventSolution{F}
) where {F}
    (; reg) = problem
    return solution.success && (getproperty(solution, Symbol(reg.type)) == reg.value)
end

function issolvedby(problem::PTraceProblem{F}, solution::ResolventSolution{F}) where {F}
    return solution.success && (solution.ptrace == problem.ptrace)
end

function stepproblem(
    index::Integer, problem::AbstractResolventProblem{F}, alg::RegularizerSchedule{F}
) where {F}
    return RegularizerProblem(problem.eq, alg[index], problem.init)
end

function stepproblem(
    index::Integer,
    problem::AbstractResolventProblem{F},
    alg::RegularizerSchedule{F},
    state::ScheduleState{F},
) where {F}
    if (state.index == lastindex(alg)) || didintercepttarget(problem, state)
        # Break out of the schedule and initialize the final problem based on its objective
        return predictinit(problem, state.currsolution, state.prevsolution)
    end
    return stepproblempredict(index, problem, alg, state)
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
    ::AbstractResolventProblem{F}, alg::SymmetrySchedule{F}, state::ScheduleState{F}
) where {F}
    return (state.index == lastindex(alg))
end

function stepproblem(
    index::Integer, problem::AbstractResolventProblem{F}, alg::SymmetrySchedule{F}
) where {F}
    return update(problem; eq=update(problem.eq; sscale=alg[index] * problem.eq.sscale))
end

end  # module Algorithms

@testitem "DirectSolve" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms: DirectSolve

    alg = DirectSolve()

    # Traits
    @test alg isa DirectSolve

    # Conversion
    @test DirectSolve(alg) === DirectSolve()
    @test convert(DirectSolve, alg) === alg
    @test adaptrealtype(Float32, alg) == alg
    @test adaptrealtype(Float64, alg) == alg
end

@testitem "RegularizerSchedule" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule
    using GaussianSpectralDensities.Defaults: Defaults

    values64, type = 2.0 .^ (2:-1:-2), :gap
    alg64 = RegularizerSchedule(values64, type)

    # Traits
    @test alg64 isa RegularizerSchedule{Float64}

    # Defaults
    @test RegularizerSchedule() === RegularizerSchedule(
        Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
    )

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = RegularizerSchedule(valuesF, type)

        # Floattype inferring constructor
        @test algF isa RegularizerSchedule{F}

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
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, unwrap
    using GaussianSpectralDensities.Defaults: Defaults

    steps = 3
    values64 = range(0.0, 1.0, steps)

    alg64 = SymmetrySchedule(values64)

    # Traits
    @test alg64 isa SymmetrySchedule{Float64}

    # All the signatures
    @test SymmetrySchedule{Float64}(steps) === alg64

    # Defaults
    @test SymmetrySchedule() === SymmetrySchedule(Defaults.symmetry_schedule_values)

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = SymmetrySchedule(valuesF)

        # Floattype inferring constructor
        @test algF isa SymmetrySchedule{F}

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
    using GaussianSpectralDensities.Problems: RegularizerProblem, ResolventEquation
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
        eq64 = ResolventEquation(ensemble64, z64)
        ws64 = Workspace(ensemble64)
        problem64 = RegularizerProblem(eq64, reg64, init64)

        alg = DirectSolve()
        solution64, state64 = step!(ws64, problem64, alg)

        # Stops after one iteration
        @test nothing === step!(ws64, problem64, alg, state64)

        # Solves the actual target problem
        @test solution64.success
        @test solution64.eq === problem64.eq
        @test getproperty(solution64, Symbol(reg64.type)) == reg64.value
        @test isdone(problem64, alg, state64)
    end
end

@testitem "DirectSolve/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    ptrace64_target = 1.0

    ensemble64 = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq64 = ResolventEquation(ensemble64, z64)
    ws64 = Workspace(ensemble64)
    problem64 = PTraceProblem(eq64, ptrace64_target, init64)

    alg64 = DirectSolve()
    solution64, state64 = step!(ws64, problem64, alg64)

    # Stops after one iteration
    @test nothing === step!(ws64, problem64, alg64, state64)

    # Solves the actual target problem
    @test solution64.success
    @test solution64.eq === problem64.eq
    @test solution64.ptrace ≈ ptrace64_target
    @test isdone(problem64, alg64, state64)
end

@testitem "RegularizerSchedule/RegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]
    z64 = -(2.0 + im) / 4
    init64 = (; q=complex(1.0))

    schedule_values64 = [4.0, 2.0, 1.0, 0.5]
    target_value64 = 0.25

    @testset "Target regularizer type = $reg_type_target" for reg_type_target in
                                                              [:shift, :gap]
        reg64_target = Regularizer(target_value64, reg_type_target)

        @testset "Schedule regularizer type = $reg_type_schedule" for reg_type_schedule in
                                                                      [:shift, :gap]
            @testset "$label" for (label, A64, D64) in [
                ("Array", A64data, Diagonal(D64data)),
                ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
                ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
            ]
                @testset "$F" for F in [Float32, Float64]
                    ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                    eqF = ResolventEquation(ensembleF, z64)
                    wsF = Workspace(ensembleF)
                    problemF = RegularizerProblem(eqF, reg64_target, init64)
                    algF = RegularizerSchedule{F}(schedule_values64, reg_type_schedule)

                    # Collect all solutions and states
                    solutions = []
                    states = []
                    result = step!(wsF, problemF, algF)
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

                    # Verify integrity of intermediate steps
                    @testset "index = $i" for i in
                                              eachindex(states, solutions)[begin:(end - 1)]
                        @test solutions[i].success
                        @test states[i].index == i
                        @test states[i].currsolution === solutions[i]
                        @test !isdone(problemF, algF, states[i])
                        if i > firstindex(solutions)
                            # Each schedule step initialized by linear *polation from the
                            # two previous solutions
                            prob = states[i].currproblem
                            sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                            q0 = prob.init.q
                            values = unwrap(algF)
                            v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                            @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))
                        end
                    end

                    # Final step initialized by linear *polation from the two previous
                    # solutions
                    prob = states[end].currproblem
                    sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
                    v, q0 = prob.reg.value, prob.init.q
                    v1 = getproperty(sol1, Symbol(prob.reg.type))
                    v2 = getproperty(sol2, Symbol(prob.reg.type))
                    @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))

                    # Final step solves the actual target problem
                    @test solutions[end].success
                    @test solutions[end].eq === problemF.eq
                    @test getproperty(solutions[end], reg_type_target) ≈ reg64_target.value
                    @test isdone(problemF, algF, states[end])
                end
            end
        end
    end
end

@testitem "RegularizerSchedule/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
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
            eqF = ResolventEquation(ensembleF, z64)
            wsF = Workspace(ensembleF)
            algF = RegularizerSchedule{F}(alg64)
            problemF = PTraceProblem(eqF, ptrace64_target, init64)

            # Collect all solutions and states
            solutions = []
            states = []
            result = step!(wsF, problemF, algF)
            while !isnothing(result)
                solution, state = result
                push!(solutions, solution)
                push!(states, state)
                result = step!(wsF, problemF, algF, state)
            end

            # Stepping should terminate before exhaustion as ptrace crosses target
            @test length(solutions) < length(algF) + 1

            # Verify integrity of intermediate steps
            @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                @test solutions[i].success
                @test states[i].index == i
                @test states[i].currsolution === solutions[i]
                @test !isdone(problemF, algF, states[i])
                if i > firstindex(solutions)
                    # Each schedule step initialized by linear *polation from the two
                    # previous solutions
                    prob = states[i].currproblem
                    sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                    q0 = prob.init.q
                    values = unwrap(algF)
                    v, v1, v2 = prob.reg.value, values[i - 1], values[max(i - 2, begin)]
                    @test isapproxlinear((v, q0), (v1, sol1.q), (v2, sol2.q))
                end
            end

            # Final step initialized by linear *polation from the two previous solutions
            # Could be either a trace problem or a regularizer problem
            prob = states[end].currproblem
            sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
            if prob isa PTraceProblem
                v, reg0 = prob.ptrace, prob.init.reg
                v1, reg1v = sol1.ptrace, getproperty(sol1, Symbol(reg0.type))
                v2, reg2v = sol2.ptrace, getproperty(sol2, Symbol(reg0.type))
                @test isapproxlinear((v, reg0.value), (v1, reg1v), (v2, reg2v))
            else
                v = prob.reg.value
                v1 = getproperty(sol1, Symbol(prob.reg.type))
                v2 = getproperty(sol2, Symbol(prob.reg.type))
            end
            @test isapproxlinear((v, prob.init.q), (v1, sol1.q), (v2, sol2.q))

            # Final step solves the actual PTraceProblem
            @test solutions[end].success
            @test solutions[end].eq === problemF.eq
            @test solutions[end].ptrace ≈ ptrace64_target
            @test isdone(problemF, algF, states[end])
        end
    end
end

@testitem "SymmetrySchedule/RegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: RegularizerProblem, ResolventEquation
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
                eqF = ResolventEquation(ensembleF, z64)
                wsF = Workspace(ensembleF)
                problemF = RegularizerProblem(eqF, reg64_target, init64)
                algF = SymmetrySchedule{F}(nsteps)

                # Collect all solutions and states
                solutions = []
                states = []
                result = step!(wsF, problemF, algF)
                while !isnothing(result)
                    solution, state = result
                    push!(solutions, solution)
                    push!(states, state)
                    result = step!(wsF, problemF, algF, state)
                end

                # solutions comprise schedule steps (no separate final step here)
                @test length(solutions) == length(algF)

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)
                    @test solutions[i].success
                    @test states[i].index == i
                    @test states[i].currsolution === solutions[i]
                    if i < lastindex(solutions)
                        @test !isdone(problemF, algF, states[i])
                    end
                    if i > firstindex(solutions)
                        # Each schedule step initialized by linear *polation from the two
                        # previous solutions
                        prob = states[i].currproblem
                        sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                        values = unwrap(algF)
                        v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                        @test isapproxlinear((v, prob.init.q), (v1, sol1.q), (v2, sol2.q))
                    end
                end

                # Final step solves the actual problem
                @test solutions[end].eq === problemF.eq
                @test getproperty(solutions[end], reg_type) ≈ reg64_target.value
                @test isdone(problemF, algF, states[end])
            end
        end
    end
end

@testitem "SymmetrySchedule/PTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!, unwrap
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathTools: isapproxlinear
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    z64 = 0.5 + 0.5im
    ptrace64_target = 1.0
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    nsteps = 3

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
        eqF = ResolventEquation(ensembleF, z64)
        wsF = Workspace(ensembleF)
        problemF = PTraceProblem(eqF, ptrace64_target, init64)
        algF = SymmetrySchedule{F}(nsteps)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # solutions comprise schedule steps (no extra final step for symmetry schedule)
        @test length(solutions) == length(algF)

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)
            @test solutions[i].success
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            if i < lastindex(solutions)
                @test !isdone(problemF, algF, states[i])
            end
            if i > firstindex(solutions)
                # Each schedule step initialized by linear *polation from the two previous
                # solutions
                prob = states[i].currproblem
                sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                reg0 = prob.init.reg
                reg1v = getproperty(sol1, Symbol(reg0.type))
                reg2v = getproperty(sol2, Symbol(reg0.type))
                values = unwrap(algF)
                v, v1, v2 = values[i], values[i - 1], values[max(i - 2, begin)]
                @test isapproxlinear((v, reg0.value), (v1, reg1v), (v2, reg2v))
                @test isapproxlinear((v, prob.init.q), (v1, sol1.q), (v2, sol2.q))
            end
        end

        # Final step solves the actual problem
        @test solutions[end].eq === problemF.eq
        @test solutions[end].ptrace ≈ ptrace64_target
        @test isdone(problemF, algF, states[end])
    end
end

@testitem "RegularizerSchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
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
        eqF = ResolventEquation(ensembleF, z64; shiftmin)
        wsF = Workspace(ensembleF)
        algF = RegularizerSchedule{F}(alg64)
        initF = problemF = PTraceProblem(eqF, ptrace64_target, init64)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF) + 1

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < eqF.shiftmin
        @test isdone(problemF, algF, states[end])
    end
end

@testitem "SymmetrySchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, isdone, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
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
        eqF = ResolventEquation(ensembleF, z64; shiftmin)
        wsF = Workspace(ensembleF)
        problemF = PTraceProblem(eqF, ptrace64_target, init64)
        algF = SymmetrySchedule{F}(nsteps)

        # Collect all solutions and states
        solutions = []
        states = []
        result = step!(wsF, problemF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # Stepping should terminate before exhaustion due to failure
        @test length(solutions) < length(algF)

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < eqF.shiftmin
        @test isdone(problemF, algF, states[end])
    end
end
