abstract type AbstractAlgorithm{F<:Real} end
abstract type AbstractSchedule{F} <: AbstractAlgorithm{F} end

"""

    DirectSolve(q0, δ0)
    DirectSolve(; q0=DEFAULTS.q0, δ0=DEFAULTS.δ0)
    DirectSolve{F}(q0, δ0)
    DirectSolve{F}(; q0=DEFAULTS.q0, δ0=DEFAULTS.δ0)

Solve the average resolvent equation directly for the given objective

This solution strategy can converge to the wrong branch when ``T ≠ 0``. A scheduling
algorithm is recommended to mitigate this.

`q0` sets the initial value for ``q = \\tr TQ^†(QQ^† + γ²I)⁻¹`` in the nonlinear solver.
`δ0` sets the initial value for the regularizer ``δ`` explained in [`Regularizer`](@ref).

All constructors support an optional type parameter `F` to enforce the floating point type.
"""
Base.@kwdef struct DirectSolve{F} <: AbstractAlgorithm{F}
    q0::Complex{F} = DEFAULTS.q0
    δ0::F = DEFAULTS.δ0
end

function Base.:(==)(dsa::DirectSolve, dsb::DirectSolve)
    return (dsa.q0 == dsb.q0) && (dsa.δ0 == dsb.δ0)
end

"""

    RegularizerSchedule(schedule::AbstractVector{<:Regularizer}, γ²min)
    RegularizerSchedule(kind, schedule, γ²min)
    RegularizerSchedule(; kind=nothing, schedule=DEFAULTS.rschedule, γ²min=DEFAULTS.γ²min)
    RegularizerSchedule{F}(schedule::AbstractVector{<:Regularizer}, γ²min)
    RegularizerSchedule{F}(kind, schedule, γ²min)
    RegularizerSchedule{F}(; kind=nothing, schedule=DEFAULTS.rschedule, γ²min=DEFAULTS.γ²min)

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
values, where the solver usually gets slow and unstable due to annihilation of the solution
branch.

The default schedule is an exponentially decaying sequence of ``δ``-values, given by
`DEFAULTS.rschedule`.

All constructors support an optional type parameter `F` to enforce the floating point type.
"""
struct RegularizerSchedule{F,U<:AbstractVector{Regularizer{F}}} <: AbstractSchedule{F}
    schedule::U
    γ²min::F
end

function RegularizerSchedule{F}(schedule, γ²min) where {F}
    if eltype(schedule) !== Regularizer{F}
        schedule = convert.(Regularizer{F}, schedule)
    end
    return RegularizerSchedule{F,typeof(schedule)}(schedule, γ²min)
end

function RegularizerSchedule{F}(kind, schedule, γ²min) where {F}
    schedule = Regularizer{F}.(kind, schedule)
    return RegularizerSchedule{F,typeof(schedule)}(schedule, γ²min)
end

function RegularizerSchedule{F}(
    ; kind=nothing, schedule=DEFAULTS.rschedule, γ²min=DEFAULTS.γ²min
) where {F}
    (kind !== nothing) && return RegularizerSchedule{F}(kind, schedule, γ²min)
    return RegularizerSchedule{F}(schedule, γ²min)
end

function RegularizerSchedule(kind, schedule, γ²min)
    return RegularizerSchedule(Regularizer.(kind, schedule), γ²min)
end

function RegularizerSchedule(
    ; kind=nothing, schedule=DEFAULTS.rschedule, γ²min=DEFAULTS.γ²min
)
    (kind !== nothing) && return RegularizerSchedule(kind, schedule, γ²min)
    return RegularizerSchedule(schedule, γ²min)
end

# constructor with explicit number type
function Base.:(==)(rsa::RegularizerSchedule, rsb::RegularizerSchedule)
    return (rsa.schedule == rsb.schedule) && (rsa.γ²min == rsb.γ²min)
end

"""

    SymmetrySchedule(schedule::Integer, γ²min, initialalg)
    SymmetrySchedule(schedule, γ²min, initialalg)
    SymmetrySchedule(
        ; schedule=DEFAULTS.μschedule, γ²min=DEFAULTS.γ²min, initialalg=DirectSolve(),
    )
    SymmetrySchedule{F}(schedule::Integer, γ²min, initialalg)
    SymmetrySchedule{F}(schedule, γ²min, initialalg)
    SymmetrySchedule{F}(
        ; schedule=DEFAULTS.μschedule, γ²min=DEFAULTS.γ²min, initialalg=DirectSolve(),
    )

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

The meaning of `γ²min` is the same as in [`RegularizerSchedule`](@ref).

`initialalg` holds the algorithm that will be used to find the solution at ``μ = 0``.
A `DirectSolve` usually suffices here, but you can also use a `RegularizerSchedule`.

The default schedule is a linear range from from 0 to 1 specified by `DEFAULTS.μschedule`.

All constructors support an optional type parameter `F` to enforce the floating point type.
"""
Base.@kwdef struct SymmetrySchedule{
    F,U<:AbstractVector{F},A<:Union{DirectSolve{F},RegularizerSchedule{F}}
} <: AbstractSchedule{F}
    schedule::U = DEFAULTS.μschedule
    γ²min::F = DEFAULTS.γ²min
    initialalg::A = DirectSolve()

    function SymmetrySchedule{F}(schedule, γ²min, initialalg) where {F}
        if (first(schedule) != 0) || (last(schedule) != 1) || !_monotonic(schedule)
            msg = "symmetry schedule must increase monotonically from 0 to 1"
            throw(ArgumentError(msg))
        end
        if eltype(schedule) !== F
            schedule = convert.(F, schedule)
        end
        initialalg = adapt(F, initialalg)
        return new{F,typeof(schedule),typeof(initialalg)}(schedule, γ²min, initialalg)
    end
end

function SymmetrySchedule{F}(schedule::Integer, args...) where {F}
    return SymmetrySchedule{F}(range(convert(F, 0), convert(F, 1), schedule), args...)
end

function SymmetrySchedule{F}(
    ; schedule=DEFAULTS.μschedule, γ²min=DEFAULTS.γ²min, initialalg=DirectSolve()
) where {F}
    return SymmetrySchedule{F}(schedule, γ²min, initialalg)
end

function SymmetrySchedule(
    schedule::AbstractVector{F},
    γ²min::F,
    initialalg::Union{DirectSolve{F},RegularizerSchedule{F}}
) where {F}
    return SymmetrySchedule{F}(schedule, γ²min, initialalg)
end

function SymmetrySchedule(schedule::Integer, args...)
    return SymmetrySchedule(range(0, 1, schedule), args...)
end

function Base.:(==)(ssa::SymmetrySchedule, ssb::SymmetrySchedule)
    return (
        (ssa.schedule == ssb.schedule)
        && (ssa.γ²min == ssb.γ²min)
        && (ssa.initialalg == ssb.initialalg)
    )
end

function _monotonic(x)
    (length(x) == 1) && return true
    tail = @view(x[(begin + 1):end])
    return (first(x) < first(tail)) && _monotonic(tail)
end

## traits
## ======

"""

    length(alg::AbstractAlgorithm)

Number of steps in the algorithm's schedule
"""
Base.length(alg::AbstractAlgorithm) = 0
Base.length(alg::AbstractSchedule) = length(alg.schedule)
Base.length(alg::SymmetrySchedule) = length(alg.schedule) + length(alg.initialalg)

## number type conversion and adaption
## ===================================

Base.convert(::Type{P}, alg::Q) where {P<:DirectSolve,Q<:DirectSolve} = P(alg.q0, alg.δ0)

function Base.convert(
    ::Type{P}, alg::Q
) where {P<:RegularizerSchedule,Q<:RegularizerSchedule}
    return P(alg.schedule, alg.γ²min)
end

function Base.convert(::Type{P}, alg::Q) where {P<:SymmetrySchedule,Q<:SymmetrySchedule}
    return P(alg.schedule, alg.γ²min, alg.initialalg)
end

for A in (DirectSolve, RegularizerSchedule, SymmetrySchedule)
    @eval adapt(::Type{F}, alg::$A) where {F} = convert($A{F}, alg)
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
