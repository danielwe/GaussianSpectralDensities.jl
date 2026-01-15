module Algorithms

using ..Defaults: Defaults
using ..Problems: LocalRegularizerProblem
using ..Solutions: LocalResolventSolution

public AbstractAlgorithm, DirectSolve, AbstractSchedule, RegularizerSchedule,
    SymmetrySchedule, initiallength

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
RegularizerSchedule(Defaults.regularizer_schedule_values, Defaults.regularizer_schedule_type)
```
"""
struct RegularizerSchedule{F,U<:AbstractVector{F}} <: AbstractSchedule{F}
    # TODO: since `type` is not in the type domain, we should store a values vector and a
    # type separately. The current implementation won't reject a mixed-type schedule.
    values::U
    type::Symbol
    shiftmin::F

    function RegularizerSchedule{F}(values, type, shiftmin=F(-Inf)) where {F}
        # map preserves ranges types etc. better than broadcasting
        values_ = (eltype(values) === F) ? values : map(F, values)
        return new{F,typeof(values_)}(values_, type, shiftmin)
    end
end

struct RegularizerScheduleState{F,S<:LocalResolventSolution{F}} <: AbstractScheduleState{F}
    index::Int
    solution::S
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

function Base.convert(::Type{P}, alg::RegularizerSchedule) where {F,P<:RegularizerSchedule{F}}
    return (alg isa P) ? alg : RegularizerSchedule{F}(alg)
end

function Base.convert(::Type{AbstractSchedule{F}}, alg::RegularizerSchedule) where {F}
    return convert(RegularizerSchedule{F}, alg)
end

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

    function SymmetrySchedule{F}(values, initialalg=DirectSolve(), shiftmin=F(-Inf)) where {F}
        if (first(values) != 0) || (last(values) != 1) || !monotonic(values)
            throw(ArgumentError("symmetry schedule must increase monotonically from 0 to 1"))
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
    solution::S
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
    tail = @view x[(begin+1):end]
    return (first(x) < first(tail)) && monotonic(tail)
end


#=
## Implementation
## ==============
step!(ws, problem, ::DirectSolve) = (eqsolve!(ws, problem), DirectSolveState())
step!(ws, problem, ::DirectSolve, ::DirectSolveState) = nothing

function step!(ws, problem, alg::RegularizerSchedule, state=problem)
    _done(state) && return nothing
    index, problem = _stepproblem(state, alg)
    stepsolution = eqsolve!(ws, problem)
    # TODO: handle final and done
    return stepsolution, RegularizerScheduleState(
        index, stepsolution, final(problem, stepsolution), done(problem, stepsolution)
    )
end

function _stepproblem(problem::AbstractLocalResolventProblem, alg::RegularizerSchedule)
    # TODO: use asymptotic q0 either here or as default value in LocalRegularizerProblem
    i = firstindex(alg.schedule)
    return i, LocalRegularizerProblem(problem.eq, alg.schedule[i], problem.q0, problem.μ)
end

function _stepproblem(state::RegularizerScheduleState, alg::RegularizerSchedule)
    current = state.currentsolution
    (; eq, μ) = current.problem
    i = state.index + 1
    return i, LocalRegularizerProblem(eq, alg.schedule[i], current.q, μ)
end

_done(::Any) = false
_done(state::DirectSolveState) = true
_done(state::AbstractScheduleState) = state.done

done(problem::LocalTraceProblem, stepsolution) = (stepsolution.k == problem.k)

function done(problem::LocalRegularizerProblem, stepsolution)
    (problem.reg.kind === Regularizers.γ) && return stepsolution.γ² == problem.reg.value
    (problem.reg.kind === Regularizers.δ) && return stepsolution.δ² == problem.reg.value
end

final(problem::LocalTraceProblem, stepsolution) = (stepsolution.k >= problem.k)

function final(problem::LocalRegularizerProblem, stepsolution)
    (problem.reg.kind === Regularizers.γ) && return stepsolution.γ² <= problem.reg.value
    (problem.reg.kind === Regularizers.δ) && return stepsolution.δ² <= problem.reg.value
end

abort(alg::AbstractSchedule, stepsolution) = (stepsolution.γ² < alg.γ²min)
=#

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
    using GaussianSpectralDensities.Algorithms: AbstractAlgorithm, DirectSolve, initiallength

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
    using GaussianSpectralDensities.Algorithms: AbstractSchedule, RegularizerSchedule,
        initiallength
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
    using GaussianSpectralDensities.Algorithms: AbstractSchedule, RegularizerSchedule,
        SymmetrySchedule, initiallength
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
