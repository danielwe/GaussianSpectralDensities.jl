abstract type AbstractAlgorithm{F<:Real} end
abstract type AbstractSchedule{F} <: AbstractAlgorithm{F} end

## Traits
## ======
Base.length(alg::AbstractAlgorithm) = 0
Base.length(alg::AbstractSchedule) = length(alg.schedule)

"""
    DirectSolve()
    DirectSolve(q0, reg0)
    DirectSolve{F}(q0, reg0) where {F<:Real}

Solve the average resolvent equation directly for the given objective

This solution strategy can converge to the wrong branch when ``T ≠ 0``. A scheduling
algorithm is recommended to mitigate this.

`q0` sets the initial value for ``q = \\tr TQ^†(QQ^† + γ²I)⁻¹`` in the nonlinear solver.
`reg0` sets the initial value for one of the regularizers ``γ²,δ²``, see
[`Regularizer`](@ref).

Default values: `DEFAULTS.q0, DEFAULTS.reg0`.
"""
struct DirectSolve{F} <: AbstractAlgorithm{F}
    q0::Complex{F}
    reg0::Regularizer{F}
end

DirectSolve() = DirectSolve(DEFAULTS.q0, DEFAULTS.reg0)

## Conversions
## ===========
Base.convert(::Type{P}, alg::Q) where {P<:DirectSolve,Q<:DirectSolve} = P(alg.q0, alg.reg0)

"""
    RegularizerSchedule()
    RegularizerSchedule(kind, schedule, γ²min)
    RegularizerSchedule(schedule::AbstractVector{<:Regularizer}, γ²min)
    RegularizerSchedule{F}(schedule::AbstractVector{<:Regularizer}, γ²min) where {F<:Real}

Solve the average resolvent equation along a schedule for a regularizer

The intended way to use this algorithm is to start from a large value to ensure convergence
to the correct branch of the equation, and decay gradually from there to stay on this branch
while approaching the final objective.

If `kind` is provided, `schedule` should be a vector of numerical values, otherwise it
should be a vector of `Regularizer`s. See [`Regularizer`](@ref) for the available kinds of
regularizers.

The last step of the algorithm is to solve directly for the actual objective. This either
happens at the end of the prescribed schedule or when the objective is reached or exceeded
during the course of the schedule, whichever comes first.

`γ²min` is a lower bound on the value of ``γ²``. If the algorithm reaches a point where
`γ^2 < γ²min` before reaching the objective, it is aborted and the equations are deemed
unsolvable. This is of course redundant when `kind=:γ`, but can be very useful when
`kind=:δ`. Strictly speaking, the equations are derived assuming ``γ² ≥ 0``, suggesting
`γ²min = 0`, but in practice it is useful to allow analytic continuation to small negative
values to support interpolation between grid points in regions where ``γ²`` vanishes
rapidly. At the same time, it's often best to abort before reaching too large negative
values, where the solver usually gets slow and unstable as the solution branch is
annihilated in a bifurcation.

Default values:
- `DEFAULTS.rschedule`, an exponentially decaying sequence of ``δ``-regularizers
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

RegularizerSchedule() = RegularizerSchedule(DEFAULTS.rschedule, DEFAULTS.γ²min)

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
    SymmetrySchedule(initialalg)
    SymmetrySchedule(initialalg, steps::Integer, γ²min)
    SymmetrySchedule(initialalg, schedule::AbstractVector{<:Real}, γ²min)
    SymmetrySchedule{F}(initialalg, schedule::AbstractVector{<:Real}, γ²min) where {F<:Real}

Solve the average resolvent equation along a schedule for the degree of symmetry

What we're scheduling here is a prefactor ``μ ∈ [0, 1]`` such that the symmetry parameter
``T`` is replaced by ``μT``. The idea is that the equations behaved better for ``μ = 0``,
i.e, the asymmetric problem, so we solve that first before turning on the symmetry
gradually.

There is no guarantee that the desired solution branch at ``μ = 1`` is analytically
connected to the solution at ``μ = 0``, so this may not always be a viable solution
strategy. For example, the topology of the spectral support may change as a function of
``μ``. This is unlike `RegularizerSchedule`, where the surviving branch in the limit of
large regularizer is the desired one by definition, so a scheduling a regularizer is usually
the preferred solution strategy.

If `schedule` is an integer, the schedule is set to `range(0, 1, schedule)`. Otherwise,
a custom schedule can be supplied, for example if you want to reduce the step size in an
unstable region. An error is thrown if the schedule does not increase monotonically from
0 to 1.

`initialalg` holds the algorithm that will be used to find the solution at ``μ = 0``.
A `DirectSolve` usually suffices here, but you can also use a `RegularizerSchedule`.

The meaning of `γ²min` is the same as in [`RegularizerSchedule`](@ref).

Default values:
- `DEFAULTS.μschedule`, a linear range from 0 to 1
- `DEFAULTS.γ²min`
"""
struct SymmetrySchedule{
    F,A<:Union{DirectSolve{F},RegularizerSchedule{F}},U<:AbstractVector{F}
} <: AbstractSchedule{F}
    initialalg::A
    schedule::U
    γ²min::F

    function SymmetrySchedule{F}(initialalg, schedule, γ²min) where {F}
        if (first(schedule) != 0) || (last(schedule) != 1) || !_monotonic(schedule)
            msg = "symmetry schedule must increase monotonically from 0 to 1"
            throw(ArgumentError(msg))
        end
        initialalg = adapt(F, initialalg)
        schedule = adapt(F, schedule)
        return new{F,typeof(initialalg),typeof(schedule)}(initialalg, schedule, γ²min)
    end
end

function SymmetrySchedule(initialalg::AbstractAlgorithm{F}, schedule, γ²min) where {F}
    return SymmetrySchedule{F}(initialalg, schedule, γ²min)
end

function SymmetrySchedule(initialalg::AbstractAlgorithm{F}, steps::Integer, γ²min) where {F}
    return SymmetrySchedule{F}(initialalg, range(zero(F), one(F), steps), γ²min)
end

function SymmetrySchedule(initialalg)
    return SymmetrySchedule(initialalg, DEFAULTS.μschedule, DEFAULTS.γ²min)
end

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
    return SymmetrySchedule{F}(alg.initialalg, alg.schedule, alg.γ²min)
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
