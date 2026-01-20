module Algorithms

using ..Defaults: Defaults
using ..EquationSolvers: eqsolve!
using ..MathHelpers: samesign
using ..Problems: AbstractLocalResolventProblem, LocalRegularizerProblem, LocalTraceProblem
using ..Regularizers: Regularizer, RegularizerType
using ..Solutions: LocalResolventSolution
using ..Workspaces: Workspace

public AbstractAlgorithm,
    AbstractSchedule, DirectSolve, RegularizerSchedule, SymmetrySchedule, initiallength

abstract type AbstractAlgorithm end
abstract type AbstractAlgorithmState end

"""
    DirectSolve()

Solve the average resolvent equation directly for the given objective

This solution strategy can converge to the wrong branch when ``T ≠ 0``. A scheduling
algorithm is recommended to mitigate this.
"""
struct DirectSolve <: AbstractAlgorithm end
struct DirectSolveState <: AbstractAlgorithmState end

DirectSolve(::DirectSolve) = DirectSolve()

## Traits, conversions
## ===================
Base.length(::DirectSolve) = 0
initiallength(::DirectSolve) = 0
Base.convert(::Type{DirectSolve}, alg::DirectSolve) = alg
Base.convert(::Type{AbstractAlgorithm}, alg::DirectSolve) = alg

# Type tree
# =========
abstract type AbstractSchedule{F<:AbstractFloat} <: AbstractAlgorithm end
abstract type AbstractScheduleState{F<:AbstractFloat} <: AbstractAlgorithmState end

## Traits
## ======
Base.length(alg::AbstractSchedule) = length(alg.values)

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

struct RegularizerScheduleState{
    F,SP<:LocalResolventSolution{F},SC<:LocalResolventSolution{F}
} <: AbstractScheduleState{F}
    index::Int
    previoussolution::SP
    currentsolution::SC
    final::Bool
    done::Bool
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

## Traits, comparisons, conversions
## ================================
initiallength(::RegularizerSchedule) = 0

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

## Indexing interface
## ==================
Base.getindex(alg::RegularizerSchedule, i) = Regularizer(alg.values[i], alg.type)
Base.firstindex(alg::RegularizerSchedule) = firstindex(alg.values)
Base.lastindex(alg::RegularizerSchedule) = lastindex(alg.values)

"""
    SymmetrySchedule()
    SymmetrySchedule(values, initialalg=DirectSolve(), shiftmin=-Inf)
    SymmetrySchedule{F}(
        steps::Integer, initialalg=DirectSolve(), shiftmin=-Inf
    ) where {F<:AbstractFloat}
    SymmetrySchedule{F}(
        values, initialalg=DirectSolve(), shiftmin=-Inf
    ) where {F<:AbstractFloat}

Solve the average resolvent equation by annealing the degree of symmetry in the ensemble

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``T`` is replaced by ``μT``. The idea is that the equations behave better for ``μ = 0``,
i.e, the asymmetric problem, so we solve that first before gradually turning on the
symmetry.

We're not aware of a guarantee that the correct solution branch at ``μ = 1`` is analytically
connected to the solution at ``μ = 0``, except when ``T ∝ I``, so this may not always be a
viable solution strategy. For example, the topology of the spectral support may change as a
function of ``μ``. This is different from `RegularizerSchedule`, where the surviving branch
in the limit of large regularizer is the correct branch by definition. Hence, annealing the
regularization is usually the preferred solution strategy.

The schedule can be specified as an integer `steps`, which will set the schedule to
`range(0, 1, steps)`. Alternatively, a custom schedule can be provided directly in the form
of an `AbstractVector{<:Real}`; this can be useful if, for example, you want to take smaller
steps in an unstable region. An error is thrown if the schedule does not increase
monotonically from 0 to 1.

`initialalg` holds the algorithm that will be used to find the solution at ``μ = 0``. The
default `DirectSolve` usually suffices here, but you can also use a `RegularizerSchedule`.

The meaning of `shiftmin` is the same as in [`RegularizerSchedule`](@ref).

The default value is

```julia
SymmetrySchedule(Defaults.symmetry_schedule_values)
```
"""
struct SymmetrySchedule{
    F,A<:Union{DirectSolve,RegularizerSchedule{F}},U<:AbstractVector{F}
} <: AbstractSchedule{F}
    values::U
    initialalg::A
    shiftmin::F

    function SymmetrySchedule{F}(
        values, initialalg=DirectSolve(), shiftmin=F(-Inf)
    ) where {F}
        if (first(values) != 0) || (last(values) != 1) || !monotonic(values)
            msg = "symmetry schedule must increase monotonically from 0 to 1"
            throw(ArgumentError(msg))
        end
        # map preserves ranges types etc. better than broadcasting
        values_ = (eltype(values) === F) ? values : map(F, values)
        initialalg_ = if initialalg isa AbstractSchedule
            convert(AbstractSchedule{F}, initialalg)
        else
            initialalg
        end
        return new{F,typeof(initialalg_),typeof(values_)}(values_, initialalg_, shiftmin)
    end
end

struct SymmetryScheduleState{
    F,
    S<:LocalResolventSolution{F},
    IP<:Union{DirectSolve,LocalRegularizerProblem{F}},
    IS<:Union{DirectSolveState,RegularizerScheduleState{F}},
} <: AbstractScheduleState{F}
    index::Int
    currentsolution::S
    final::Bool
    done::Bool
    initial::Bool
    initialproblem::IP
    initialstate::IS
end

function SymmetrySchedule{F}(steps::Integer, args...) where {F}
    return SymmetrySchedule{F}(range(zero(F), one(F), steps), args...)
end

SymmetrySchedule(values, args...) = SymmetrySchedule{eltype(values)}(values, args...)
SymmetrySchedule() = SymmetrySchedule(Defaults.symmetry_schedule_values)

function SymmetrySchedule{F}(alg::SymmetrySchedule) where {F}
    return SymmetrySchedule{F}(alg.values, alg.initialalg, alg.shiftmin)
end

## Traits, comparisons, conversions
## ================================
initiallength(alg::SymmetrySchedule) = length(alg.initialalg)

function Base.:(==)(a::A, b::A) where {A<:SymmetrySchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, alg::SymmetrySchedule) where {F,P<:SymmetrySchedule{F}}
    return (alg isa P) ? alg : SymmetrySchedule{F}(alg)
end

function Base.convert(::Type{AbstractSchedule{F}}, alg::SymmetrySchedule) where {F}
    return convert(SymmetrySchedule{F}, alg)
end

## Helpers
## =======
function monotonic(x)
    (length(x) == 1) && return true
    tail = @view x[(begin + 1):end]
    return (first(x) < first(tail)) && monotonic(tail)
end

## Implementation
## ==============
function step!(
    ws::Workspace{F}, problem::AbstractLocalResolventProblem{F}, ::DirectSolve
) where {F}
    return (eqsolve!(ws, problem), DirectSolveState())
end

function step!(
    ::Workspace{F}, ::AbstractLocalResolventProblem{F}, ::DirectSolve, ::DirectSolveState
) where {F}
    return nothing
end

function step!(
    ws::Workspace{F},
    problem::AbstractLocalResolventProblem{F},
    alg::RegularizerSchedule{F},
    state::Union{Nothing,RegularizerScheduleState{F}}=nothing,
) where {F}
    (!isnothing(state)) && state.done && return nothing

    index = isnothing(state) ? firstindex(alg) : state.index + 1

    if (!isnothing(state)) && state.final
        fproblem = finalproblem(problem, state)
        solution = eqsolve!(ws, fproblem)
        return (
            solution,
            RegularizerScheduleState(index, state.currentsolution, solution, false, true),
        )
    end

    step_q0 = isnothing(state) ? problem.q0 : state.currentsolution.q
    stepproblem = LocalRegularizerProblem(problem.eq, alg[index], step_q0, problem.qscale)
    stepsolution = eqsolve!(ws, stepproblem)
    prevsolution = isnothing(state) ? stepsolution : state.currentsolution
    done = isdone(problem, stepsolution)

    final = if done
        # TODO: hit in tests?
        false
    elseif (index == lastindex(alg))
        true
    else
        didpasstarget(problem, stepsolution, prevsolution)
    end

    return (
        stepsolution,
        RegularizerScheduleState(index, prevsolution, stepsolution, final, done),
    )
end

function finalproblem(
    problem::LocalRegularizerProblem{F}, state::AbstractScheduleState{F}
) where {F}
    # Linear interpolation between prev and curr, weighted by distance from target
    prev, curr = state.previoussolution, state.currentsolution
    target_regv, reg_type = problem.reg.value, Symbol(problem.reg.type)
    prev_regv, curr_regv = getproperty(prev, reg_type), getproperty(curr, reg_type)
    final_q0 = if curr_regv == prev_regv
        # TODO: hit in tests
        (prev.q + curr.q) / 2
    else
        dcurr_regv, dprev_regv = curr_regv - target_regv, target_regv - prev_regv
        ((prev.q * dcurr_regv) + (curr.q * dprev_regv)) / (curr_regv - prev_regv)
    end
    return LocalRegularizerProblem(problem.eq, problem.reg, final_q0, problem.qscale)
end

function finalproblem(
    problem::LocalTraceProblem{F}, state::AbstractScheduleState{F}
) where {F}
    # Linear interpolation between prev and curr, weighted by distance from target
    prev, curr = state.previoussolution, state.currentsolution
    final_reg0, final_q0 = if curr.k == prev.k
        # TODO: hit in tests
        ((prev.reg + curr.reg) / 2, (prev.q + curr.q) / 2)
    else
        dcurr_k, dprev_k = curr.k - problem.k, problem.k - prev.k
        dk = curr.k - prev.k
        (
            ((prev.reg * dcurr_k) + (curr.reg * dprev_k)) / dk,
            ((prev.q * dcurr_k) + (curr.q * dprev_k)) / dk,
        )
    end
    return LocalTraceProblem(problem.eq, problem.k, final_reg0, final_q0, problem.qscale)
end

function isdone(
    problem::LocalTraceProblem{F}, stepsolution::LocalResolventSolution{F}
) where {F}
    return stepsolution.k == problem.k
end

function isdone(
    problem::LocalRegularizerProblem{F}, stepsolution::LocalResolventSolution{F}
) where {F}
    return getproperty(stepsolution, Symbol(problem.reg.type)) == problem.reg.value
end

function didpasstarget(
    problem::LocalTraceProblem{F},
    stepsolution::LocalResolventSolution{F},
    prevsolution::LocalResolventSolution{F},
) where {F}
    return !samesign(stepsolution.k - problem.k, prevsolution.k - problem.k)
end

function didpasstarget(
    problem::LocalRegularizerProblem{F},
    stepsolution::LocalResolventSolution{F},
    prevsolution::LocalResolventSolution{F},
) where {F}
    target_regv, reg_type = problem.reg.value, Symbol(problem.reg.type)
    prev_regv = getproperty(prevsolution, reg_type)
    step_regv = getproperty(stepsolution, reg_type)
    return !samesign(step_regv - target_regv, prev_regv - target_regv)
end

# function abort(
#     alg::AbstractSchedule{F}, currentsolution::LocalResolventSolution{F}
# ) where {F}
#     return (currentsolution.shift < alg.shiftmin)
# end

#=
## initial values
## ==============

# TODO: Is δ0 what we need a function for? Is γ∞ better? # function q∞!(problem, γ∞, ws)
δ0(alg::AbstractRegularizerSchedule) = first(alg.schedule)
δ0(alg::μSchedule) = δ0(alg.initialalg)
δ0(alg) = alg.δ0

q0!(problem, alg::AbstractRegularizerSchedule, ws) = q∞!(problem, δ0(alg), ws)
q0!(problem, alg::μSchedule, ws) = q0!(problem, alg.initialalg, ws)
q0!(problem, alg, ws) = alg.q0

# TODO: This seems superfluous
initialalg(alg::μSchedule, q0, d0) = alg.initialalg
initialalg(alg, q0, δ0) = DirectSolve(q0, δ0)

## stepping behavior
## =================

# TODO: seems overcomplicated, compare OrdinaryDiffEq `perform_step`
step_objective(obj, alg::δSchedule, i) = δObjective(alg.schedule[i])
step_objective(obj, alg::γSchedule, i) = γObjective(alg.schedule[i])

# TODO: that ain't working---does μ need to be in the objective after all? Look for better
# solution in solve.jl
function step_objective(obj::O, algorithm::μSchedule, i) where {O}
    return O(obj.value, algorithm.schedule[i])
end

# TODO: lots to go through here, should look at solve.jl first
# function invalid(sol, ::DirectSolve, ::Any=nothing)
#     return sol.δ² < zero(sol.δ²)
# end
#
# invalid(step_sol, ::Rampδ², γ_²) = ((step_sol.q === missing) || (step_sol.γ² > γ_²))
# invalid(step_sol, ::Ramp, ::Any) = (step_sol.q === missing)
#
# done(step_sol, ::Rampγ², objective::Fixγ²) = (step_sol.γ² == objective.value)
# done(step_sol, ::Rampδ², objective::Fixδ²) = (step_sol.δ² == objective.value)
# done(step_sol, ::Rampμ, objective) = (step_sol.μ == objective.μ)
# done(::Any, ::Ramp, ::Any) = false
#
# final(step_sol, ::Downramp, objective::Fixk) = (step_sol.k >= objective.value)
# final(step_sol, ::Downramp, objective::Fixγ²) = (step_sol.γ² <= objective.value)
# final(step_sol, ::Downramp, objective::Fixδ²) = (step_sol.δ² <= objective.value)
# final(step_sol, ::Rampμ, objective) = (step_sol.μ > objective.μ)
# final(::Any, ::Ramp, ::Any) = false
#
# abort(step_sol, algorithm::Union{Rampδ², Rampμ}) = (step_sol.γ² < algorithm.γ²min)
# abort(::Any, ::Ramp) = false
=#

end  # module Algorithms

@testitem "DirectSolve" begin
    using GaussianSpectralDensities.Algorithms:
        AbstractAlgorithm, DirectSolve, initiallength

    alg64 = DirectSolve()

    # Traits
    @test alg64 isa DirectSolve
    @test length(alg64) == 0
    @test initiallength(alg64) == 0

    # Conversion
    @test DirectSolve(alg64) === DirectSolve()
    @test convert(DirectSolve, alg64) === alg64
    @test convert(AbstractAlgorithm, alg64) === alg64
end

@testitem "RegularizerSchedule" begin
    using GaussianSpectralDensities.Algorithms:
        AbstractSchedule, RegularizerSchedule, initiallength
    using GaussianSpectralDensities.Defaults: Defaults

    values64, type, shiftmin = 2.0 .^ (2:-1:-2), :gap, 0.0
    alg64 = RegularizerSchedule(values64, type, shiftmin)

    # Traits
    @test alg64 isa RegularizerSchedule{Float64}
    @test length(alg64) == length(values64)
    @test initiallength(alg64) == 0

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
        AbstractSchedule, RegularizerSchedule, SymmetrySchedule, initiallength
    using GaussianSpectralDensities.Defaults: Defaults

    steps = 3
    values64, shiftmin = range(0.0, 1.0, steps), 0.0
    initialalg = RegularizerSchedule()

    alg64 = SymmetrySchedule(values64, initialalg, shiftmin)

    # Traits
    @test alg64 isa SymmetrySchedule{Float64,<:RegularizerSchedule{Float64}}
    @test length(alg64) == length(values64)
    @test initiallength(alg64) == length(initialalg)

    # All the signatures
    @test SymmetrySchedule{Float64}(steps, initialalg, shiftmin) === alg64

    # Defaults
    @test SymmetrySchedule() === SymmetrySchedule(Defaults.symmetry_schedule_values)

    @testset "$F" for F in [Float32, Float64]
        valuesF = map(F, values64)  # map preserves ranges types etc. better than broadcasting
        algF = SymmetrySchedule(valuesF, initialalg, shiftmin)

        # Floattype inferring constructor
        @test algF isa SymmetrySchedule{F,<:RegularizerSchedule{F}}

        # Floattype enforcing constructor
        @test SymmetrySchedule{F}(values64, initialalg, shiftmin) == algF
        @test SymmetrySchedule{Float64}(valuesF, initialalg, shiftmin) == alg64

        # Upconvert algF properties to capture roundtrip truncation before equality testing
        algF64 = SymmetrySchedule(map(Float64, algF.values), algF.initialalg, algF.shiftmin)

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
    using GaussianSpectralDensities.Algorithms: DirectSolve, step!
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
    q64 = (1.0 - im) / 2
    reg_shift64 = Regularizer(0.25, :shift)

    @testset "$label" for (label, A64, D64) in [
        ("Array", A64data, Diagonal(D64data)),
        ("SArray", SMatrix{2,2}(A64data), Diagonal(SVector{2}(D64data))),
        ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data)),
    ]
        ensemble64 = MatrixEnsemble(A64, I, I, D64)
        eq64 = LocalResolventEquation(ensemble64, z64)
        ws64 = Workspace(ensemble64)
        problem64 = LocalRegularizerProblem(eq64, reg_shift64, q64)

        alg64 = DirectSolve()
        solution64, state64 = step!(ws64, problem64, alg64)
        @test nothing === step!(ws64, problem64, alg64, state64)
        @test solution64.problem.reg == problem64.reg
        @test getproperty(solution64, Symbol(reg_shift64.type)) == reg_shift64.value
    end
end

@testitem "stepping RegularizerSchedule" begin
    using GaussianSpectralDensities.Algorithms:
        RegularizerSchedule, RegularizerScheduleState, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: LocalResolventSolution
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
                problemF = LocalRegularizerProblem(eqF, target_reg64, q0_64)
                algF = RegularizerSchedule{F}(schedule_values64, type)

                # Collect all solutions and states
                solutions = LocalResolventSolution{F}[]
                states = RegularizerScheduleState{F}[]
                result = step!(wsF, problemF, algF)
                while !isnothing(result)
                    solution, state = result
                    push!(solutions, solution)
                    push!(states, state)
                    result = step!(wsF, problemF, algF, state)
                end

                # Schedule steps + final step to target
                nschedule = length(schedule_values64)
                @test length(solutions) == nschedule + 1

                # All solutions should succeed
                @test all(s -> s.success, solutions)

                # Schedule steps: index increases, not done yet
                for i in 1:nschedule
                    @test states[i].index == i
                    @test states[i].currentsolution === solutions[i]
                    @test !states[i].done
                    if i < nschedule
                        @test !states[i].final
                    end
                end

                # Last schedule step should be final (schedule exhausted)
                @test states[nschedule].final

                # Each schedule step uses previous solution's q as initial guess
                for i in 2:nschedule
                    @test solutions[i].problem.q0 == solutions[i - 1].q
                end

                # Final step: solving for the actual target problem
                final_solution = solutions[end]
                final_state = states[end]
                @test !final_state.final
                @test final_state.done
                @test final_solution.problem.reg == problemF.reg
                @test getproperty(final_solution, type) == target_reg64.value
            end
        end
    end
end

@testitem "stepping RegularizerSchedule with LocalTraceProblem" begin
    using GaussianSpectralDensities.Algorithms:
        RegularizerSchedule, RegularizerScheduleState, step!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: LocalResolventSolution
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
        problemF = LocalTraceProblem(eqF, target_k64, algF[begin], q0_64)

        solutions = LocalResolventSolution{F}[]
        states = RegularizerScheduleState{F}[]
        result = step!(wsF, problemF, algF)
        while !isnothing(result)
            solution, state = result
            push!(solutions, solution)
            push!(states, state)
            result = step!(wsF, problemF, algF, state)
        end

        # All solutions should succeed
        @test all(s -> s.success, solutions)

        # Find where didpasstarget triggered (first state with final=true)
        final_schedule_idx = findfirst(s -> s.final, states)

        # nothing means a regular step landed directly on "done". We want to avoid this to
        # ensure we're exercising the final step codepath.
        @test !isnothing(final_schedule_idx)

        # There should only be one more step after reaching final
        @test final_schedule_idx + 1 == lastindex(solutions)

        # Verify final was set because k crossed target, not schedule exhaustion
        @test final_schedule_idx < length(algF)

        # Final step solves the actual LocalTraceProblem
        final_solution = solutions[end]
        final_state = states[end]
        @test !final_state.final
        @test final_state.done
        @test final_solution.k == target_k64
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
