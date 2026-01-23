module Algorithms

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..MathHelpers: monotonic, predictlinear, samesign
using ..Problems:
    AbstractLocalResolventProblem, LocalRegularizerProblem, LocalTraceProblem, update
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: LocalResolventSolution, setfailed
using ..Workspaces: Workspace

public AbstractAlgorithm,
    AbstractSchedule, DirectSolve, RegularizerSchedule, SymmetrySchedule

abstract type AbstractAlgorithm end
abstract type AbstractAlgorithmState end

## Generic iterator implementation
## ===============================
## Implicitly defines the required interface for each algorithm:
## - firstindex(::AbstractAlgorithm)
## - nextindex(::AbstractAlgorithm{F}, ::AbstractAlgorithmState{F})
## - done(::AbstractLocalResolventProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - stepproblem(index, ::AbstractLocalResolventProblem, ::AbstractAlgorithm)::AbstractLocalResolventProblem
## - stepproblem(index, ::AbstractLocalResolventProblem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::AbstractLocalResolventProblem
## - check(::LocalResolventSolution, ::AbstractAlgorithm)::LocalResolventSolution
## - newstate(index, ::LocalResolventSolution, ::AbstractAlgorithm)::AbstractAlgorithmState
## - newstate(index, ::LocalResolventSolution, ::AbstractAlgorithm, ::AbstractAlgorithmState)::AbstractAlgorithmState
function step!(
    ws::Workspace{F}, problem::AbstractLocalResolventProblem{F}, alg::AbstractAlgorithm
) where {F}
    return step_common!(firstindex(alg), ws, problem, alg)
end

function step!(
    ws::Workspace{F},
    problem::AbstractLocalResolventProblem{F},
    alg::AbstractAlgorithm,
    state::AbstractAlgorithmState,
) where {F}
    done(problem, alg, state) && return nothing
    return step_common!(nextindex(alg, state), ws, problem, alg, state)
end

function step_common!(
    index,
    ws::Workspace{F},
    problem::AbstractLocalResolventProblem{F},
    alg::AbstractAlgorithm,
    state::AbstractAlgorithmState...,
) where {F}
    currproblem = stepproblem(index, problem, alg, state...)
    currsolution = check(eqsolve!(ws, currproblem), alg)
    return currsolution, newstate(index, currsolution, alg, state...)
end

# For firstindex, we use a local function with a fallback method such that we don't have to
# add methods to Base for algorithms that are not actually indexable. We still add a
# Base.firstindex method for algorithms that have Base.getindex.
firstindex(alg::AbstractAlgorithm) = Base.firstindex(alg)

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
Base.convert(::Type{AbstractAlgorithm}, alg::DirectSolve) = alg

# Interface
# ---------
firstindex(::DirectSolve) = nothing
nextindex(::DirectSolve, ::DirectSolveState) = nothing
done(::AbstractLocalResolventProblem, ::DirectSolve, ::DirectSolveState) = true

function stepproblem(::Nothing, problem::AbstractLocalResolventProblem, ::DirectSolve)
    return problem
end

check(solution::LocalResolventSolution, ::DirectSolve) = solution

newstate(::Nothing, ::LocalResolventSolution, ::DirectSolve) = DirectSolveState()

## Schedule algorithms
## ===================
## Provided types:
## - ScheduleState{F} with properties currsolution, prevsolution, index::Integer
## Provided methods:
## - nextindex(::AbstractSchedule{F}, ::ScheduleState{F})::Integer
## - stepproblem(::Integer, ::AbstractLocalResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::AbstractLocalResolventProblem{F}
## - check(::LocalResolventSolution{F}, ::AbstractSchedule{F})::LocalResolventSolution{F}
## - newstate(::Integer, ::LocalResolventSolution{F}, ::AbstractSchedule{F})::ScheduleState{F}
## - newstate(::Integer, ::LocalResolventSolution{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::ScheduleState{F}
## Bonus reusable functions:
## - stepproblem_common(::Integer, ::AbstractLocalResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::AbstractLocalResolventProblem{F}
## - predictinit(::AbstractLocalResolventProblem{F}, ::LocalResolventSolution{F}, ::LocalResolventSolution{F})::AbstractLocalResolventProblem{F}
#::Integer, ::AbstractLocalResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::AbstractLocalResolventProblem{F}
## Required interface for each implementation:
## - done(::AbstractLocalResolventProblem{F}, ::AbstractSchedule{F}, ::ScheduleState{F})::Bool
## - stepproblem(::Integer, ::AbstractLocalResolventProblem{F}, ::AbstractSchedule{F})::AbstractLocalResolventProblem{F}
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end

struct ScheduleState{F,S<:LocalResolventSolution{F},P<:LocalResolventSolution{F}} <:
       AbstractAlgorithmState
    currsolution::S
    prevsolution::P
    index::Int
end

function nextindex(::AbstractSchedule{F}, state::ScheduleState{F}) where {F}
    return state.index + 1
end

function stepproblem(
    index::Integer,
    problem::AbstractLocalResolventProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    return stepproblem_common(index, problem, alg, state)
end

function check(solution::LocalResolventSolution{F}, alg::AbstractSchedule{F}) where {F}
    (solution.shift < alg.shiftmin) && return setfailed(solution)
    return solution
end

function newstate(
    index::Integer, currsolution::LocalResolventSolution{F}, ::AbstractSchedule{F}
) where {F}
    return ScheduleState(currsolution, currsolution, index)
end

function newstate(
    index::Integer,
    currsolution::LocalResolventSolution{F},
    ::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    return ScheduleState(currsolution, state.currsolution, index)
end

function stepproblem_common(
    index::Integer,
    problem::AbstractLocalResolventProblem{F},
    alg::AbstractSchedule{F},
    state::ScheduleState{F},
) where {F}
    step_problem = stepproblem(index, problem, alg)
    return predictinit(step_problem, state.currsolution, state.prevsolution)
end

function predictinit(
    problem::LocalRegularizerProblem{F},
    solution1::LocalResolventSolution{F},
    solution2::LocalResolventSolution{F},
) where {F}
    reg_value1, q1 = getproperty(solution1, Symbol(problem.reg.type)), solution1.q
    reg_value2, q2 = getproperty(solution2, Symbol(problem.reg.type)), solution2.q
    init = (; q=predictlinear(problem.reg.value, (reg_value1, q1), (reg_value2, q2)))
    return update(problem; init)
end

function predictinit(
    problem::LocalTraceProblem{F},
    solution1::LocalResolventSolution{F},
    solution2::LocalResolventSolution{F},
) where {F}
    k1, reg1, q1 = solution1.k, solution1.reg, solution1.q
    k2, reg2, q2 = solution2.k, solution2.reg, solution2.q
    reg = predictlinear(problem.k, (k1, reg1), (k2, reg2))
    q = predictlinear(problem.k, (k1, q1), (k2, q2))
    return update(problem; init=(; reg, q))
end

"""
    RegularizerSchedule()
    RegularizerSchedule(values, type, shiftmin=-Inf)
    RegularizerSchedule{F}(values, type, shiftmin=-Inf) where {F<:AbstractFloat}

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

`shiftmin` is a lower bound on the value of the regularizer shift ``γ²``. If the algorithm
reaches a point where `γ² < shiftmin` before reaching the objective, it aborts and the
equations are deemed unsolvable. This is of course redundant when `type=:shift`, but can be
very useful when `type=:gap`. Strictly speaking, the averaged resolvent equations are
derived assuming ``γ² ≥ 0``, suggesting `shiftmin = 0`, but in practice it is useful to
allow analytic continuation to negative values such that the averaged resolvent can be
interpolated between grid points in regions where ``γ²`` vanishes rapidly. However, it's
best to abort before reaching too large negative values, as the annealing may encounter a
bifurcation that annihilates the solution branch, leading to a poorly initialized solver and
rendering interpolation meaningless.

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
    shiftmin::F

    function RegularizerSchedule{F}(values, type, shiftmin=F(-Inf)) where {F}
        # map preserves ranges types etc. better than broadcasting
        values_ = (eltype(values) === F) ? values : map(F, values)
        schedule = new{F,typeof(values_)}(values_, type, shiftmin)
        # Validate post-instantiation in case the original values was a single-use iterator
        if !monotonic(Iterators.reverse(schedule.values))
            throw(ArgumentError("regularizer schedule must decrease monotonically"))
        end
        return schedule
    end
end

RegularizerSchedule(values, args...) = RegularizerSchedule{eltype(values)}(values, args...)

function RegularizerSchedule()
    return RegularizerSchedule(
        Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
    )
end

function RegularizerSchedule{F}(alg::RegularizerSchedule) where {F}
    return RegularizerSchedule{F}(alg.values, alg.type, alg.shiftmin)
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

function Base.convert(::Type{AbstractSchedule{F}}, alg::RegularizerSchedule) where {F}
    return convert(RegularizerSchedule{F}, alg)
end

# Interface
# ---------
Base.getindex(alg::RegularizerSchedule, i::Integer) = Regularizer(alg.values[i], alg.type)
Base.firstindex(alg::RegularizerSchedule) = Base.firstindex(alg.values)
Base.lastindex(alg::RegularizerSchedule) = Base.lastindex(alg.values)

function done(
    problem::LocalTraceProblem{F}, ::RegularizerSchedule{F}, state::ScheduleState{F}
) where {F}
    return (!state.currsolution.success) || (state.currsolution.k == problem.k)
end

function done(
    problem::LocalRegularizerProblem{F}, ::RegularizerSchedule{F}, state::ScheduleState{F}
) where {F}
    (!state.currsolution.success) && return true
    return getproperty(state.currsolution, Symbol(problem.reg.type)) == problem.reg.value
end

function stepproblem(
    index::Integer, problem::AbstractLocalResolventProblem{F}, alg::RegularizerSchedule{F}
) where {F}
    return LocalRegularizerProblem(problem.eq, alg[index], problem.init, problem.qscale)
end

function stepproblem(
    index::Integer,
    problem::AbstractLocalResolventProblem{F},
    alg::RegularizerSchedule{F},
    state::ScheduleState{F},
) where {F}
    if (state.index == lastindex(alg)) || didintercepttarget(problem, state)
        return predictinit(problem, state.currsolution, state.prevsolution)
    end
    return stepproblem_common(index, problem, alg, state)
end

function didintercepttarget(
    problem::LocalRegularizerProblem{F}, state::ScheduleState{F}
) where {F}
    prev_regv = getproperty(state.prevsolution, Symbol(problem.reg.type))
    curr_regv = getproperty(state.currsolution, Symbol(problem.reg.type))
    return !samesign(curr_regv - problem.reg.value, prev_regv - problem.reg.value)
end

function didintercepttarget(
    problem::LocalTraceProblem{F}, state::ScheduleState{F}
) where {F}
    prev_k, curr_k = state.prevsolution.k, state.currsolution.k
    return !samesign(curr_k - problem.k, prev_k - problem.k)
end

"""
    SymmetrySchedule()
    SymmetrySchedule(values, shiftmin=-Inf)
    SymmetrySchedule{F}(steps::Integer, shiftmin=-Inf) where {F<:AbstractFloat}
    SymmetrySchedule{F}(values, shiftmin=-Inf) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the degree of symmetry in the ensemble

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``T`` is replaced by ``μT``. The idea is that the equations behave better for ``μ = 0``,
i.e, the asymmetric problem, so we solve that first before gradually turning on the
symmetry.

We know that when ``T ∝ I``, the correct solution branch at ``μ = 1`` is analytically
connected to the solution at ``μ = 0``, however, this may not hold for general ``T``, so
this may not always be a viable solution strategy. For example, the topology of the spectral
support may change as a function of ``μ``. This is different from `RegularizerSchedule`,
where the unique solution in the limit of large regularizer is on the correct branch by
definition. Hence, annealing the regularization is usually the preferred solution strategy.

The schedule can be specified as an integer `steps`, which will set the schedule to
`range(0, 1, steps)`. Alternatively, a custom schedule can be provided directly in the form
of an `AbstractVector{<:Real}`; this can be useful if, for example, you want to take smaller
steps in an unstable region. An error is thrown if the schedule does not increase
monotonically from 0 to 1.

The meaning of `shiftmin` is the same as in [`RegularizerSchedule`](@ref).

The default value is

```julia
SymmetrySchedule(Defaults.symmetry_schedule_values)
```
"""
struct SymmetrySchedule{F,U<:AbstractVector{F}} <: AbstractSchedule{F}
    values::U
    shiftmin::F

    function SymmetrySchedule{F}(values, shiftmin=F(-Inf)) where {F}
        # map preserves ranges types etc. better than broadcasting
        values_ = (eltype(values) === F) ? values : map(F, values)
        schedule = new{F,typeof(values_)}(values_, shiftmin)
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

function SymmetrySchedule{F}(steps::Integer, args...) where {F}
    return SymmetrySchedule{F}(range(zero(F), one(F), steps), args...)
end

SymmetrySchedule(values, args...) = SymmetrySchedule{eltype(values)}(values, args...)
SymmetrySchedule() = SymmetrySchedule(Defaults.symmetry_schedule_values)

function SymmetrySchedule{F}(alg::SymmetrySchedule) where {F}
    return SymmetrySchedule{F}(alg.values, alg.shiftmin)
end

# Traits, comparisons, conversions
# --------------------------------
function Base.:(==)(a::A, b::A) where {A<:SymmetrySchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, alg::SymmetrySchedule) where {F,P<:SymmetrySchedule{F}}
    return (alg isa P) ? alg : SymmetrySchedule{F}(alg)
end

function Base.convert(::Type{AbstractSchedule{F}}, alg::SymmetrySchedule) where {F}
    return convert(SymmetrySchedule{F}, alg)
end

# Interface
# ---------
Base.getindex(alg::SymmetrySchedule, i::Integer) = alg.values[i]
Base.firstindex(alg::SymmetrySchedule) = Base.firstindex(alg.values)
Base.lastindex(alg::SymmetrySchedule) = Base.lastindex(alg.values)

function done(
    ::AbstractLocalResolventProblem{F}, alg::SymmetrySchedule{F}, state::ScheduleState{F}
) where {F}
    return (!state.currsolution.success) || (state.index == lastindex(alg))
end

function stepproblem(
    index::Integer, problem::AbstractLocalResolventProblem{F}, alg::SymmetrySchedule{F}
) where {F}
    return update(problem; qscale=alg[index] * problem.qscale)
end

end  # module Algorithms

@testitem "DirectSolve" begin
    using GaussianSpectralDensities.Algorithms: AbstractAlgorithm, DirectSolve

    alg64 = DirectSolve()

    # Traits
    @test alg64 isa DirectSolve

    # Conversion
    @test DirectSolve(alg64) === DirectSolve()
    @test convert(DirectSolve, alg64) === alg64
    @test convert(AbstractAlgorithm, alg64) === alg64
end

@testitem "RegularizerSchedule" begin
    using GaussianSpectralDensities.Algorithms: AbstractSchedule, RegularizerSchedule
    using GaussianSpectralDensities.Defaults: Defaults

    values64, type, shiftmin = 2.0 .^ (2:-1:-2), :gap, 0.0
    alg64 = RegularizerSchedule(values64, type, shiftmin)

    # Traits
    @test alg64 isa RegularizerSchedule{Float64}

    # Defaults
    @test RegularizerSchedule() === RegularizerSchedule(
        Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type
    )

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = RegularizerSchedule(valuesF, type, shiftmin)

        # Floattype inferring constructor
        @test algF isa RegularizerSchedule{F}

        # Floattype enforcing constructor
        @test RegularizerSchedule{F}(values64, type, shiftmin) == algF
        @test RegularizerSchedule{Float64}(valuesF, type, shiftmin) == alg64

        # Floattype conversions
        @test convert(RegularizerSchedule{F}, alg64) == algF
        @test convert(AbstractSchedule{F}, alg64) == algF
        @test convert(RegularizerSchedule{Float64}, algF) == alg64
        @test convert(AbstractSchedule{Float64}, algF) == alg64
        @test RegularizerSchedule{F}(alg64) == algF
        @test RegularizerSchedule{Float64}(algF) == alg64
    end
end

@testitem "SymmetrySchedule" begin
    using GaussianSpectralDensities.Algorithms:
        AbstractSchedule, RegularizerSchedule, SymmetrySchedule
    using GaussianSpectralDensities.Defaults: Defaults

    steps = 3
    values64, shiftmin = range(0.0, 1.0, steps), 0.0

    alg64 = SymmetrySchedule(values64, shiftmin)

    # Traits
    @test alg64 isa SymmetrySchedule{Float64}

    # All the signatures
    @test SymmetrySchedule{Float64}(steps, shiftmin) === alg64

    # Defaults
    @test SymmetrySchedule() === SymmetrySchedule(Defaults.symmetry_schedule_values)

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = SymmetrySchedule(valuesF, shiftmin)

        # Floattype inferring constructor
        @test algF isa SymmetrySchedule{F}

        # Floattype enforcing constructor
        @test SymmetrySchedule{F}(values64, shiftmin) == algF
        @test SymmetrySchedule{Float64}(valuesF, shiftmin) == alg64

        # Upconvert algF properties to capture roundtrip truncation before equality testing
        algF64 = SymmetrySchedule(map(Float64, algF.values), algF.shiftmin)

        # Floattype conversions
        @test convert(SymmetrySchedule{F}, alg64) == algF
        @test convert(AbstractSchedule{F}, alg64) == algF
        @test convert(SymmetrySchedule{Float64}, algF) == algF64
        @test convert(AbstractSchedule{Float64}, algF) == algF64
        @test SymmetrySchedule{F}(alg64) == algF
        @test SymmetrySchedule{Float64}(algF) == algF64
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

@testitem "DirectSolve/LocalRegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation
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
        eq64 = LocalResolventEquation(ensemble64, z64)
        ws64 = Workspace(ensemble64)
        problem64 = LocalRegularizerProblem(eq64, reg64, init64)

        alg64 = DirectSolve()
        solution64, state64 = step!(ws64, problem64, alg64)

        # Stops after one iteration
        @test nothing === step!(ws64, problem64, alg64, state64)

        # Solves the actual target problem
        @test solution64.success
        @test solution64.problem.eq == problem64.eq
        @test solution64.problem.reg == problem64.reg
        @test solution64.problem.qscale == problem64.qscale
        @test getproperty(solution64, Symbol(reg64.type)) == reg64.value
        @test done(problem64, alg64, state64)
    end
end

@testitem "DirectSolve/LocalTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    z64 = 0.5 + 0.5im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    k64_target = 1.0

    ensemble64 = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq64 = LocalResolventEquation(ensemble64, z64)
    ws64 = Workspace(ensemble64)
    problem64 = LocalTraceProblem(eq64, k64_target, init64)

    alg64 = DirectSolve()
    solution64, state64 = step!(ws64, problem64, alg64)

    # Stops after one iteration
    @test nothing === step!(ws64, problem64, alg64, state64)

    # Solves the actual target problem
    @test solution64.success
    @test solution64.problem.eq == problem64.eq
    # We don't check solution64.problem.k == problem64.k because it may happen
    # that we land on the solution during a regular step and never take the final,
    # in which case solution64.problem would be a LocalRegularizerProblem, not a
    # LocalTraceProblem, and thus won't have a .k property.
    @test solution64.problem.qscale == problem64.qscale
    @test solution64.k ≈ k64_target
    @test done(problem64, alg64, state64)
end

@testitem "RegularizerSchedule/LocalRegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation
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
                    eqF = LocalResolventEquation(ensembleF, z64)
                    wsF = Workspace(ensembleF)
                    problemF = LocalRegularizerProblem(eqF, reg64_target, init64)
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
                        @test length(solutions) == length(algF.values) + 1
                    end

                    # Verify integrity of intermediate steps
                    @testset "index = $i" for i in
                                              eachindex(states, solutions)[begin:(end - 1)]
                        @test solutions[i].success
                        @test states[i].index == i
                        @test states[i].currsolution === solutions[i]
                        @test !done(problemF, algF, states[i])
                        if i > firstindex(solutions)
                            # Each schedule step initialized by linear *polation from the
                            # two previous solutions
                            prob = solutions[i].problem
                            sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                            reg, q0 = prob.reg, prob.init.q
                            reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                            reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                            @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))
                        end
                    end

                    # Final step initialized by linear *polation from the two previous
                    # solutions
                    prob = solutions[end].problem
                    sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
                    reg, q0 = prob.reg, prob.init.q
                    reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                    reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                    @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))

                    # Final step solves the actual target problem
                    @test solutions[end].success
                    @test solutions[end].problem.eq == problemF.eq
                    @test solutions[end].problem.reg == problemF.reg
                    @test solutions[end].problem.qscale == problemF.qscale
                    @test getproperty(solutions[end], reg_type_target) ≈ reg64_target.value
                    @test done(problemF, algF, states[end])
                end
            end
        end
    end
end

@testitem "RegularizerSchedule/LocalTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    z64 = 0.5 + 0.5im
    q0_64 = complex(0.1)

    schedule_values64 = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    k64_target = 1.0

    @testset "Schedule regularizer type = $reg_type" for reg_type in [:shift, :gap]
        alg64 = RegularizerSchedule(schedule_values64, reg_type)

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
            eqF = LocalResolventEquation(ensembleF, z64)
            wsF = Workspace(ensembleF)
            algF = RegularizerSchedule{F}(alg64)
            problemF = LocalTraceProblem(eqF, k64_target, (; reg=algF[begin], q=q0_64))

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

            # Stepping should terminate before exhaustion as k crosses target
            @test length(solutions) < length(algF.values) + 1

            # Verify integrity of intermediate steps
            @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                @test solutions[i].success
                @test states[i].index == i
                @test states[i].currsolution === solutions[i]
                @test !done(problemF, algF, states[i])
                if i > firstindex(solutions)
                    # Each schedule step initialized by linear *polation from the two
                    # previous solutions
                    prob = solutions[i].problem
                    sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                    reg, q0 = prob.reg, prob.init.q
                    reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                    reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                    @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))
                end
            end

            # Final step initialized by linear *polation from the two previous solutions
            # Could be either a trace problem or a regularizer problem
            prob = solutions[end].problem
            sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
            q0, q1, q2 = prob.init.q, sol1.q, sol2.q
            if prob isa LocalTraceProblem
                reg0 = prob.init.reg
                k1, reg1v = sol1.k, getproperty(sol1, Symbol(reg0.type))
                k2, reg2v = sol2.k, getproperty(sol2, Symbol(reg0.type))
                @test isapproxlinear((prob.k, reg0.value), (k1, reg1v), (k2, reg2v))
                @test isapproxlinear((prob.k, q0), (k1, q1), (k2, q2))
            else
                reg1v = getproperty(sol1, Symbol(prob.reg.type))
                reg2v = getproperty(sol2, Symbol(prob.reg.type))
                @test isapproxlinear((prob.reg.value, q0), (reg1v, q1), (reg2v, q2))
            end

            # Final step solves the actual LocalTraceProblem
            @test solutions[end].success
            @test solutions[end].problem.eq == problemF.eq
            # We don't check solutions[end].problem.k == problemF.k because it may happen
            # that we land on the solution during a regular step and never take the final,
            # in which case solutions[end].problem would be a LocalRegularizerProblem, not a
            # LocalTraceProblem, and thus won't have a .k property.
            @test solutions[end].problem.qscale == problemF.qscale
            @test solutions[end].k ≈ k64_target
            @test done(problemF, algF, states[end])
        end
    end
end

@testitem "SymmetrySchedule/LocalRegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation
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
                eqF = LocalResolventEquation(ensembleF, z64)
                wsF = Workspace(ensembleF)
                problemF = LocalRegularizerProblem(eqF, reg64_target, init64)
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
                @test length(solutions) == length(algF.values)

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                    @test solutions[i].success
                    @test states[i].index == i
                    @test states[i].currsolution === solutions[i]
                    @test !done(problemF, algF, states[i])
                    if i > firstindex(solutions)
                        # Each schedule step initialized by linear *polation from the two
                        # previous solutions
                        prob = solutions[i].problem
                        sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                        reg, q0 = prob.reg, prob.init.q
                        reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                        reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                        @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))
                    end
                end

                # Final step initialized by linear *polation from the two previous solutions
                prob = solutions[end].problem
                sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
                reg, q0 = prob.reg, prob.init.q
                reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))

                # Final step solves the actual problem
                @test solutions[end].success
                @test solutions[end].problem.eq == problemF.eq
                @test solutions[end].problem.reg == problemF.reg
                @test solutions[end].problem.qscale == problemF.qscale
                @test getproperty(solutions[end], reg_type) ≈ reg64_target.value
                @test done(problemF, algF, states[end])
            end
        end
    end
end

@testitem "SymmetrySchedule/LocalTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    z64 = 0.5 + 0.5im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    nsteps = 3
    k64_target = 1.0

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
        eqF = LocalResolventEquation(ensembleF, z64)
        wsF = Workspace(ensembleF)
        problemF = LocalTraceProblem(eqF, k64_target, init64)
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
        @test length(solutions) == length(algF.values)

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
            @test solutions[i].success
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            @test !done(problemF, algF, states[i])
            if i > firstindex(solutions)
                # Each schedule step initialized by linear *polation from the two previous
                # solutions
                prob = solutions[i].problem
                sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                k, reg0, q0 = prob.k, prob.init.reg, prob.init.q
                k1, reg1v, q1 = sol1.k, getproperty(sol1, Symbol(reg0.type)), sol1.q
                k2, reg2v, q2 = sol2.k, getproperty(sol2, Symbol(reg0.type)), sol2.q
                @test isapproxlinear((k, reg0.value), (k1, reg1v), (k2, reg2v))
                @test isapproxlinear((k, q0), (k1, q1), (k2, q2))
            end
        end

        # Final step initialized by linear *polation from the two previous solutions
        # Could be either a trace problem or a regularizer problem
        prob = solutions[end].problem
        sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
        k, reg0, q0 = prob.k, prob.init.reg, prob.init.q
        k1, reg1v, q1 = sol1.k, getproperty(sol1, Symbol(reg0.type)), sol1.q
        k2, reg2v, q2 = sol2.k, getproperty(sol2, Symbol(reg0.type)), sol2.q
        @test isapproxlinear((k, reg0.value), (k1, reg1v), (k2, reg2v))
        @test isapproxlinear((k, q0), (k1, q1), (k2, q2))

        # Final step solves the actual problem
        @test solutions[end].success
        @test solutions[end].problem.eq == problemF.eq
        # We don't check solutions[end].problem.k == problemF.k because it may happen
        # that we land on the solution during a regular step and never take the final,
        # in which case solutions[end].problem would be a LocalRegularizerProblem, not a
        # LocalTraceProblem, and thus won't have a .k property.
        @test solutions[end].problem.qscale == problemF.qscale
        @test solutions[end].k ≈ k64_target
        @test done(problemF, algF, states[end])
    end
end

@testitem "RegularizerSchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    # z outside Ginibre support => negative shift needed to reach k = 1.0
    z64 = 1.0 + 1.0im
    q0_64 = complex(0.1)

    alg64 = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], :gap, 0)
    k64_target = 1.0

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.0I)
        eqF = LocalResolventEquation(ensembleF, z64)
        wsF = Workspace(ensembleF)
        algF = RegularizerSchedule{F}(alg64)
        problemF = LocalTraceProblem(eqF, k64_target, (; reg=algF[begin], q=q0_64))

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
        @test length(solutions) < length(algF.values) + 1

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
            @test solutions[i].success
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            @test !done(problemF, algF, states[i])
            if i > firstindex(solutions)
                # Each schedule step initialized by linear *polation from the two previous
                # solutions
                prob = solutions[i].problem
                sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                reg, q0 = prob.reg, prob.init.q
                reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
                reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
                @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))
            end
        end

        # Final step initialized by linear *polation from the two previous solutions
        # Shouldn't be a trace problem because we aborted before reaching the final step
        prob = solutions[end].problem
        sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
        reg, q0 = prob.reg, prob.init.q
        reg1v, q1 = getproperty(sol1, Symbol(reg.type)), sol1.q
        reg2v, q2 = getproperty(sol2, Symbol(reg.type)), sol2.q
        @test isapproxlinear((reg.value, q0), (reg1v, q1), (reg2v, q2))

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < algF.shiftmin
        @test done(problemF, algF, states[end])
    end
end

@testitem "SymmetrySchedule shiftmin abort" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.MathHelpers: isapproxlinear
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    # z inside Ginibre but outside Girko support => negative shift needed to reach k = 1.0
    z64 = 0.25 + 0.75im
    init64 = (; reg=Regularizer(1.0, :shift), q=complex(0.1))

    nsteps = 5
    k64_target = 1.0

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.75I)
        eqF = LocalResolventEquation(ensembleF, z64)
        wsF = Workspace(ensembleF)
        problemF = LocalTraceProblem(eqF, k64_target, init64)
        algF = SymmetrySchedule{F}(nsteps, 0)

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
        @test length(solutions) < length(algF.values)

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
            @test solutions[i].success
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            @test !done(problemF, algF, states[i])
            if i > firstindex(solutions)
                # Each schedule step is initialized by linear *polation from the two
                # previous solutions
                prob = solutions[i].problem
                sol1, sol2 = solutions[i - 1], solutions[max(i - 2, begin)]
                k, reg0, q0 = prob.k, prob.init.reg, prob.init.q
                k1, reg1v, q1 = sol1.k, getproperty(sol1, Symbol(reg0.type)), sol1.q
                k2, reg2v, q2 = sol2.k, getproperty(sol2, Symbol(reg0.type)), sol2.q
                @test isapproxlinear((k, reg0.value), (k1, reg1v), (k2, reg2v))
                @test isapproxlinear((k, q0), (k1, q1), (k2, q2))
            end
        end

        # Final step initialized by linear *polation from the two previous solutions
        prob = solutions[end].problem
        sol1, sol2 = solutions[end - 1], solutions[max(end - 2, begin)]
        k, reg0, q0 = prob.k, prob.init.reg, prob.init.q
        k1, reg1v, q1 = sol1.k, getproperty(sol1, Symbol(reg0.type)), sol1.q
        k2, reg2v, q2 = sol2.k, getproperty(sol2, Symbol(reg0.type)), sol2.q
        @test isapproxlinear((k, reg0.value), (k1, reg1v), (k2, reg2v))
        @test isapproxlinear((k, q0), (k1, q1), (k2, q2))

        # Final step failed due to shiftmin violation
        @test !solutions[end].success
        @test solutions[end].shift < algF.shiftmin
        @test done(problemF, algF, states[end])
    end
end
