module Algorithms

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..MathHelpers: interpolate, samesign
using ..Problems:
    AbstractLocalResolventProblem, LocalRegularizerProblem, LocalTraceProblem, update
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: LocalResolventSolution
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
## - done(problem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::Bool
## - step_problem(index, problem, ::AbstractAlgorithm)::AbstractLocalResolventProblem
## - step_problem(index, problem, ::AbstractAlgorithm, ::AbstractAlgorithmState)::AbstractLocalResolventProblem
## - newstate(index, currsolution, ::AbstractAlgorithm)::AbstractAlgorithmState
## - newstate(index, currsolution, ::AbstractAlgorithm, ::AbstractAlgorithmState)::AbstractAlgorithmState
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
    currsolution = eqsolve!(ws, currproblem)
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

DirectSolve(::DirectSolve) = DirectSolve()

# Traits, conversions
# -------------------
Base.convert(::Type{DirectSolve}, alg::DirectSolve) = alg
Base.convert(::Type{AbstractAlgorithm}, alg::DirectSolve) = alg

# Indexing interface
# ------------------
firstindex(::DirectSolve) = nothing

# State
# -----
struct DirectSolveState <: AbstractAlgorithmState end

nextindex(::DirectSolve, ::DirectSolveState) = nothing
done(::AbstractLocalResolventProblem, ::DirectSolve, ::DirectSolveState) = true

function stepproblem(::Nothing, problem::AbstractLocalResolventProblem, ::DirectSolve)
    return problem
end

newstate(::Nothing, ::LocalResolventSolution, ::DirectSolve) = DirectSolveState()

## Schedule algorithms
## ===================
## Required interface for each algorithm:
## - (state::AbstractScheduleState).index::Integer
## Provided methods:
## - nextindex(::AbstractSchedule{F}, ::AbstractScheduleState{F})::Integer
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end
abstract type AbstractScheduleState{F<:AbstractFloat} <: AbstractAlgorithmState end

function nextindex(::AbstractSchedule{F}, state::AbstractScheduleState{F}) where {F}
    return state.index + 1
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
regularizer, see [`Regularizer`](@ref) for details.

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
        return new{F,typeof(values_)}(values_, type, shiftmin)
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

# Indexing interface
# ------------------
Base.getindex(alg::RegularizerSchedule, i::Integer) = Regularizer(alg.values[i], alg.type)
Base.firstindex(alg::RegularizerSchedule) = Base.firstindex(alg.values)
Base.lastindex(alg::RegularizerSchedule) = Base.lastindex(alg.values)

# State
# -----
struct RegularizerScheduleState{
    F,S<:LocalResolventSolution{F},P<:LocalResolventSolution{F}
} <: AbstractScheduleState{F}
    currsolution::S
    prevsolution::P
    index::Int
end

function done(
    problem::LocalTraceProblem{F},
    ::RegularizerSchedule{F},
    state::RegularizerScheduleState{F},
) where {F}
    return state.currsolution.k == problem.k
end

function done(
    problem::LocalRegularizerProblem{F},
    ::RegularizerSchedule{F},
    state::RegularizerScheduleState{F},
) where {F}
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
    state::RegularizerScheduleState{F},
) where {F}
    if (state.index == lastindex(alg)) || didintercepttarget(problem, state)
        return finalproblem(problem, state.currsolution, state.prevsolution)
    end
    q = state.currsolution.q
    return LocalRegularizerProblem(problem.eq, alg[index], (; q), problem.qscale)
end

function newstate(
    index::Integer, currsolution::LocalResolventSolution{F}, ::RegularizerSchedule{F}
) where {F}
    return RegularizerScheduleState(currsolution, currsolution, index)
end

function newstate(
    index::Integer,
    currsolution::LocalResolventSolution{F},
    ::RegularizerSchedule{F},
    state::RegularizerScheduleState{F},
) where {F}
    return RegularizerScheduleState(currsolution, state.currsolution, index)
end

function didintercepttarget(
    problem::LocalRegularizerProblem{F}, state::RegularizerScheduleState{F}
) where {F}
    prev_regv = getproperty(state.prevsolution, Symbol(problem.reg.type))
    curr_regv = getproperty(state.currsolution, Symbol(problem.reg.type))
    return !samesign(curr_regv - problem.reg.value, prev_regv - problem.reg.value)
end

function didintercepttarget(
    problem::LocalTraceProblem{F}, state::RegularizerScheduleState{F}
) where {F}
    prev_k, curr_k = state.prevsolution.k, state.currsolution.k
    return !samesign(curr_k - problem.k, prev_k - problem.k)
end

function finalproblem(
    problem::LocalRegularizerProblem{F},
    currsolution::LocalResolventSolution{F},
    prevsolution::LocalResolventSolution{F},
) where {F}
    curr_regv, curr_q = getproperty(currsolution, Symbol(problem.reg.type)), currsolution.q
    prev_regv, prev_q = getproperty(prevsolution, Symbol(problem.reg.type)), prevsolution.q
    q = interpolate(problem.reg.value, (prev_regv, prev_q), (curr_regv, curr_q))
    return update(problem; init=(; q))
end

function finalproblem(
    problem::LocalTraceProblem{F},
    currsolution::LocalResolventSolution{F},
    prevsolution::LocalResolventSolution{F},
) where {F}
    curr_k, curr_reg, curr_q = currsolution.k, currsolution.reg, currsolution.q
    prev_k, prev_reg, prev_q = prevsolution.k, prevsolution.reg, prevsolution.q
    reg = interpolate(problem.k, (prev_k, prev_reg), (curr_k, curr_reg))
    q = interpolate(problem.k, (prev_k, prev_q), (curr_k, curr_q))
    return update(problem; init=(; reg, q))
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
        if (first(values) != 0) || (last(values) != 1) || !monotonic(values)
            msg = "symmetry schedule must increase monotonically from 0 to 1"
            throw(ArgumentError(msg))
        end
        # map preserves ranges types etc. better than broadcasting
        values_ = (eltype(values) === F) ? values : map(F, values)
        return new{F,typeof(values_)}(values_, shiftmin)
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

# Indexing interface
# ------------------
Base.getindex(alg::SymmetrySchedule, i::Integer) = alg.values[i]
Base.firstindex(alg::SymmetrySchedule) = Base.firstindex(alg.values)
Base.lastindex(alg::SymmetrySchedule) = Base.lastindex(alg.values)

# Helpers
# -------
function monotonic(x)
    (length(x) == 1) && return true
    tail = @view x[(begin + 1):end]
    return (first(x) < first(tail)) && monotonic(tail)
end

# State
# -----
struct SymmetryScheduleState{F,S<:LocalResolventSolution{F}} <: AbstractScheduleState{F}
    currsolution::S
    index::Int
end

function done(
    ::AbstractLocalResolventProblem{F},
    alg::SymmetrySchedule{F},
    state::SymmetryScheduleState{F},
) where {F}
    return state.index == lastindex(alg)
end

function stepproblem(
    index::Integer, problem::AbstractLocalResolventProblem{F}, alg::SymmetrySchedule{F}
) where {F}
    return update(problem; qscale=alg[index] * problem.qscale)
end

function stepproblem(
    index::Integer,
    problem::AbstractLocalResolventProblem{F},
    alg::SymmetrySchedule{F},
    state::SymmetryScheduleState{F},
) where {F}
    init = (; reg=state.currsolution.reg, q=state.currsolution.q)
    return update(problem; init, qscale=alg[index] * problem.qscale)
end

function newstate(
    index::Integer,
    currsolution::LocalResolventSolution{F},
    ::SymmetrySchedule{F},
    ::SymmetryScheduleState{F}...,
) where {F}
    return SymmetryScheduleState(currsolution, index)
end

# function abort(
#     alg::AbstractSchedule{F}, currsolution::LocalResolventSolution{F}
# ) where {F}
#     return (currsolution.shift < alg.shiftmin)
# end

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

@testitem "SymmetrySchedule errors" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule

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
end

@testitem "stepping DirectSolve" begin
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
    q0_64 = (1.0 - im) / 2
    reg_shift64 = Regularizer(0.25, :shift)

    @testset "$label" for (label, A64, D64) in [
        ("Array", A64data, Diagonal(D64data)),
        ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
        ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
    ]
        ensemble64 = MatrixEnsemble(A64, I, I, D64)
        eq64 = LocalResolventEquation(ensemble64, z64)
        ws64 = Workspace(ensemble64)
        problem64 = LocalRegularizerProblem(eq64, reg_shift64, (; q=q0_64))

        alg64 = DirectSolve()
        solution64, state64 = step!(ws64, problem64, alg64)
        @test done(problem64, alg64, state64)
        @test nothing === step!(ws64, problem64, alg64, state64)
        @test solution64.problem.reg == problem64.reg
        @test getproperty(solution64, Symbol(reg_shift64.type)) == reg_shift64.value
    end
end

@testitem "stepping RegularizerSchedule with LocalRegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, done, step!
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
    q0_64 = complex(1.0)

    schedule_values64 = [4.0, 2.0, 1.0, 0.5]

    @testset "Regularizer type = $type" for type in [:shift, :gap]
        target_reg64 = Regularizer(0.25, type)

        @testset "$label" for (label, A64, D64) in [
            ("Array", A64data, Diagonal(D64data)),
            ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
            ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
        ]
            @testset "$F" for F in [Float32, Float64]
                ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                eqF = LocalResolventEquation(ensembleF, z64)
                wsF = Workspace(ensembleF)
                problemF = LocalRegularizerProblem(eqF, target_reg64, (; q=q0_64))
                algF = RegularizerSchedule{F}(schedule_values64, type)

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

                # solutions comprise schedule steps + final step to target
                @test length(solutions) == length(algF.values) + 1

                # All solutions should succeed
                @test all(s -> s.success, solutions)

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                    @test states[i].index == i
                    @test states[i].currsolution === solutions[i]
                    @test !done(problemF, algF, states[i])
                    if i > firstindex(solutions)
                        # Each schedule step uses previous solution's q as initial guess
                        @test solutions[i].problem.init.q == solutions[i - 1].q
                    end
                end

                # Final step solves the actual target problem
                @test done(problemF, algF, states[end])
                @test getproperty(solutions[end], type) ≈ target_reg64.value
            end
        end
    end
end

@testitem "stepping RegularizerSchedule with LocalTraceProblem" begin
    using GaussianSpectralDensities.Algorithms: RegularizerSchedule, done, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I

    z64 = 0.5 + 0.5im
    q0_64 = complex(0.1)

    target_k64 = 1.0
    alg64 = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], :shift)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(0.0I, I, I, 0.5I)
        eqF = LocalResolventEquation(ensembleF, z64)
        wsF = Workspace(ensembleF)
        algF = RegularizerSchedule{F}(alg64)
        problemF = LocalTraceProblem(eqF, target_k64, (; reg=algF[begin], q=q0_64))

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

        # All solutions should succeed
        @test all(s -> s.success, solutions)

        # Verify integrity of intermediate steps
        @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
            @test states[i].index == i
            @test states[i].currsolution === solutions[i]
            @test !done(problemF, algF, states[i])
            if i > firstindex(solutions)
                # Each schedule step uses previous solution's q as initial guess
                @test solutions[i].problem.init.q == solutions[i - 1].q
            end
        end

        # Final step solves the actual LocalTraceProblem
        @test done(problemF, algF, states[end])
        @test solutions[end].k ≈ target_k64
    end
end

@testitem "stepping SymmetrySchedule with LocalRegularizerProblem" begin
    using GaussianSpectralDensities.Algorithms: SymmetrySchedule, done, step!
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
    q0_64 = complex(1.0)

    nsteps = 3

    @testset "Regularizer type = $type" for type in [:shift, :gap]
        target_reg64 = Regularizer(0.25, type)

        @testset "$label" for (label, A64, D64) in [
            ("Array", A64data, Diagonal(D64data)),
            ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
            ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
        ]
            @testset "$F" for F in [Float32, Float64]
                ensembleF = MatrixEnsemble{F}(A64, I, I, D64)
                eqF = LocalResolventEquation(ensembleF, z64)
                wsF = Workspace(ensembleF)
                problemF = LocalRegularizerProblem(eqF, target_reg64, (; q=q0_64))
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

                # All solutions should succeed
                @test all(s -> s.success, solutions)

                # Verify integrity of intermediate steps
                @testset "index = $i" for i in eachindex(states, solutions)[begin:(end - 1)]
                    @test states[i].index == i
                    @test states[i].currsolution === solutions[i]
                    @test !done(problemF, algF, states[i])
                    if i > firstindex(solutions)
                        # Each schedule step uses previous solution's q as initial guess
                        @test solutions[i].problem.init.q == solutions[i - 1].q
                    end
                end

                # Final step solves the actual problem
                @test done(problemF, algF, states[end])
                @test getproperty(solutions[end], type) ≈ target_reg64.value
            end
        end
    end
end

# @testitem "stepping RegularizerSchedule abort" begin
#     using GaussianSpectralDensities.Algorithms: RegularizerSchedule, abort, step!
#     using GaussianSpectralDensities.Ensembles: MatrixEnsemble
#     using GaussianSpectralDensities.Problems:
#         LocalRegularizerProblem, LocalResolventEquation
#     using GaussianSpectralDensities.Regularizers: Regularizer
#     using GaussianSpectralDensities.Workspaces: Workspace
#
#     using LinearAlgebra: I
#
#     ensemble = MatrixEnsemble{Float64}(I, I, I, I)
#     eq = LocalResolventEquation(ensemble, 0.0im)
#     ws = Workspace(ensemble)
#
#     # Schedule with shiftmin that will be violated
#     schedule_values = [4.0, 2.0, 1.0, 0.5]
#     shiftmin = 1.0
#     alg = RegularizerSchedule(schedule_values, :shift, shiftmin)
#
#     target_reg = Regularizer(0.1, :shift)
#     problem = LocalRegularizerProblem(eq, target_reg, complex(0.1))
#
#     solution, state = step!(ws, problem, alg)
#
#     # When solution's shift goes below shiftmin, abort should return true
#     # We test the abort function directly with a solution that has low shift
#     @test abort(alg, solution) == (solution.shift < shiftmin)
# end
#
# @testitem "isdone and didpasstarget" begin
#     using GaussianSpectralDensities.Algorithms: isdone, didpasstarget
#     using GaussianSpectralDensities.Ensembles: MatrixEnsemble
#     using GaussianSpectralDensities.Problems:
#         LocalRegularizerProblem, LocalResolventEquation, LocalTraceProblem
#     using GaussianSpectralDensities.Regularizers: Regularizer
#     using GaussianSpectralDensities.Solutions: LocalResolventSolution
#
#     using LinearAlgebra: I
#
#     ensemble = MatrixEnsemble{Float64}(I, I, I, I)
#     eq = LocalResolventEquation(ensemble, 0.0im)
#
#     @testset "LocalRegularizerProblem type=$type" for type in [:shift, :gap]
#         target_value = 1.0
#         reg = Regularizer(target_value, type)
#         problem = LocalRegularizerProblem(eq, reg, complex(0.1))
#
#         # Create mock solutions with different shift/gap values
#         make_solution =
#             (shift, gap) -> LocalResolventSolution(
#                 problem, true, complex(0.1), reg, shift, gap, 1.0, complex(0.5)
#             )
#
#         if type === :shift
#             # isdone: solution.shift == target
#             @test isdone(problem, make_solution(target_value, 2.0))
#             @test !isdone(problem, make_solution(target_value + 0.1, 2.0))
#
#             # didpasstarget: solution.shift <= target (overshot or hit)
#             @test didpasstarget(problem, make_solution(target_value, 2.0))
#             @test didpasstarget(problem, make_solution(target_value - 0.1, 2.0))
#             @test !didpasstarget(problem, make_solution(target_value + 0.1, 2.0))
#         else  # :gap
#             # isdone: solution.gap == target
#             @test isdone(problem, make_solution(0.5, target_value))
#             @test !isdone(problem, make_solution(0.5, target_value + 0.1))
#
#             # didpasstarget: solution.gap <= target
#             @test didpasstarget(problem, make_solution(0.5, target_value))
#             @test didpasstarget(problem, make_solution(0.5, target_value - 0.1))
#             @test !didpasstarget(problem, make_solution(0.5, target_value + 0.1))
#         end
#     end
#
#     @testset "LocalTraceProblem" begin
#         target_k = 1.0
#         reg = Regularizer(0.5, :shift)
#         problem = LocalTraceProblem(eq, target_k, reg, complex(0.1))
#
#         make_solution =
#             k -> LocalResolventSolution(
#                 problem, true, complex(0.1), reg, 0.5, 0.5, k, complex(0.5)
#             )
#
#         # isdone: solution.k == target
#         @test isdone(problem, make_solution(target_k))
#         @test !isdone(problem, make_solution(target_k + 0.1))
#         @test !isdone(problem, make_solution(target_k - 0.1))
#
#         # didpasstarget: solution.k >= target
#         @test didpasstarget(problem, make_solution(target_k))
#         @test didpasstarget(problem, make_solution(target_k + 0.1))
#         @test !didpasstarget(problem, make_solution(target_k - 0.1))
#     end
# end
