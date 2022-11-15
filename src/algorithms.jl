"""
    DirectSolve()

Solve the average resolvent equation directly for the given objective

This solution strategy can converge to the wrong branch when ``T ≠ 0``. A scheduling
algorithm is recommended to mitigate this.
"""
struct DirectSolve end

Base.length(::DirectSolve) = 0
adapt(::Type{F}, alg::DirectSolve) where {F} = alg

abstract type AbstractSchedule{F} end

## Traits
## ======
Base.length(alg::AbstractSchedule) = length(alg.schedule)

## Union rather than supertype to make DirectSolve <: AbstractAlgorithm{F} for any F
const AbstractAlgorithm{F} = Union{DirectSolve,<:AbstractSchedule{F}}

"""
    RegularizerSchedule()
    RegularizerSchedule(kind, schedule, γ²min)
    RegularizerSchedule(schedule::AbstractVector{<:Regularizer}, γ²min)
    RegularizerSchedule{F}(schedule::AbstractVector{<:Regularizer}, γ²min) where {F<:Real}

Solve the average resolvent equation by annealing the regularization

The intended way to use this algorithm is to start from a large regularizer to ensure
convergence to the correct solution branch, and let it decay such as to gradually approach
the true solution along this branch.

If `kind` is provided, `schedule` should be a vector of numerical values, otherwise it
should be a vector of `Regularizer`s. See [`Regularizer`](@ref) for the available kinds of
regularizers.

The final step of the algorithm is a direct solve of the averaged resolvent problem. This
either happens at the end of the prescribed schedule or when intersecting the target during
the annealing, whichever comes first.

`γ²min` is a lower bound on the value of ``γ²``. If the algorithm reaches a point where
`γ^2 < γ²min` before reaching the objective, it is aborted and the equations are deemed
unsolvable. This is of course redundant when `kind=:γ`, but can be very useful when
`kind=:δ`. Strictly speaking, the averaged resolvent equations are derived assuming ``γ²
≥ 0``, suggesting `γ²min = 0`, but in practice it is useful to allow analytic continuation
to negative values such that the averaged resolvent can be interpolated between grid points
in regions where ``γ²`` vanishes rapidly. However, it's best to abort before reaching too
large negative values, as the annealing may encounter a bifurcation that annihilates the
solution branch, rendering the solver slow and unstable and interpolation meaningless.

Default values:
- `DEFAULTS.regschedule`, an exponentially decaying sequence of ``δ``-regularizers
- `DEFAULTS.γ²min`
"""
struct RegularizerSchedule{F,U<:AbstractVector{Regularizer{F}}} <: AbstractSchedule{F}
    schedule::U
    γ²min::F

    function RegularizerSchedule{F}(schedule, γ²min) where {F}
        if eltype(schedule) !== Regularizer{F}
            schedule = adapt.(F, schedule)
        end
        return new{F,typeof(schedule)}(schedule, γ²min)
    end
end

function RegularizerSchedule(schedule::AbstractVector{Regularizer{F}}, γ²min) where {F}
    return RegularizerSchedule{F}(schedule, γ²min)
end

function RegularizerSchedule(kind, schedule, γ²min)
    return RegularizerSchedule(Regularizer.(kind, schedule), γ²min)
end

RegularizerSchedule() = RegularizerSchedule(DEFAULTS.regschedule, DEFAULTS.γ²min)

## Comparisons, conversions
## ========================
function Base.:(==)(a::A, b::A) where {A<:RegularizerSchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(
    ::Type{P}, alg::Q
) where {F,P<:RegularizerSchedule{F},Q<:RegularizerSchedule}
    (alg isa P) && return alg
    return RegularizerSchedule{F}(alg.schedule, alg.γ²min)
end

"""
    SymmetrySchedule()
    SymmetrySchedule(steps::Integer, args...)
    SymmetrySchedule(schedule::AbstractVector{<:Real}, γ²min, initialalg=DirectSolve())
    SymmetrySchedule{F}(schedule, γ²min, initialalg) where {F<:Real}

Solve the average resolvent equation by annealing the degree of symmetry in the ensemble

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``T`` is replaced by ``μT``. The idea is that the equations behave better for ``μ = 0``,
i.e, the asymmetric problem, so we solve that first before gradually turning on the
symmetry.

There is no guarantee that the correct solution branch at ``μ = 1`` is analytically
connected to the solution at ``μ = 0``, so this may not always be a viable solution
strategy. For example, the topology of the spectral support may change as a function of
``μ``. This is different from `RegularizerSchedule`, where the surviving branch in the limit
of large regularizer is the correct branch by definition. Hence, annealing the
regularization is usually the preferred solution strategy.

The schedule can be specified as an integer `steps`, which will set the schedule to
`range(0, 1, steps)`. Alternatively, a custom schedule can be provided directly in the form
of an `AbstractVector{<:Real}`; this can be useful if, for example, you want to take smaller
steps in an unstable region. An error is thrown if the schedule does not increase
monotonically from 0 to 1.

`initialalg` holds the algorithm that will be used to find the solution at ``μ = 0``. The
default `DirectSolve` usually suffices here, but you can also use a `RegularizerSchedule`.

The meaning of `γ²min` is the same as in [`RegularizerSchedule`](@ref).

Default values:
- `DEFAULTS.μschedule`, a linear range from 0 to 1
- `DEFAULTS.γ²min`
"""
struct SymmetrySchedule{
    F,A<:Union{DirectSolve,<:RegularizerSchedule{F}},U<:AbstractVector{F}
} <: AbstractSchedule{F}
    schedule::U
    γ²min::F
    initialalg::A

    function SymmetrySchedule{F}(schedule, γ²min, initialalg) where {F}
        if (first(schedule) != 0) || (last(schedule) != 1) || !_monotonic(schedule)
            msg = "symmetry schedule must increase monotonically from 0 to 1"
            throw(ArgumentError(msg))
        end
        schedule = adapt(F, schedule)
        initialalg = adapt(F, initialalg)
        return new{F,typeof(initialalg),typeof(schedule)}(schedule, γ²min, initialalg)
    end
end

function SymmetrySchedule(
    schedule::AbstractVector{F}, γ²min, initialalg=DirectSolve()
) where {F}
    return SymmetrySchedule{F}(schedule, γ²min, initialalg)
end

function SymmetrySchedule(steps::Integer, γ²min::F, args...) where {F}
    return SymmetrySchedule(range(zero(F), one(F), steps), γ²min, args...)
end

SymmetrySchedule() = SymmetrySchedule(DEFAULTS.μschedule, DEFAULTS.γ²min)

## Traits, comparisons, conversions
## ================================
Base.length(alg::SymmetrySchedule) = length(alg.initialalg) + length(alg.schedule)

function Base.:(==)(a::A, b::A) where {A<:SymmetrySchedule}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(
    ::Type{P}, alg::Q
) where {F,P<:SymmetrySchedule{F},Q<:SymmetrySchedule}
    (alg isa P) && return alg
    return SymmetrySchedule{F}(alg.schedule, alg.γ²min, alg.initialalg)
end

## Helpers
## =======
function _monotonic(x)
    (length(x) == 1) && return true
    tail = @view(x[(begin + 1):end])
    return (first(x) < first(tail)) && _monotonic(tail)
end

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
