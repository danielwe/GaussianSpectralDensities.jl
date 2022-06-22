abstract type AbstractAlgorithm{T<:Real} end
abstract type AbstractRamp{T} <: AbstractAlgorithm{T} end
abstract type AbstractDownramp{T} <: AbstractRamp{T} end

struct γRamp{T} <: AbstractDownramp{T}
    values::Vector{T}
end

γRamp() = γRamp(default(:downramp))
γRamp(base, startexp, stopexp) = γRamp(base.^(startexp:-1:stopexp))

Base.convert(::Type{P}, ramp::Q) where {P<:γRamp,Q<:γRamp} = P(ramp.values)
Base.convert(::Type{P}, ramp::P) where {P<:γRamp} = ramp

# adapt(::Type{T}, ramp::γRamp) where {T<:Real} = convert(γRamp{T}, ramp)

struct Rampδ²{R} <: Downramp{R}
    ramp::Vector{R}
    γ²min::R
end

Rampδ²() = Rampδ²(default(:r²ramp))
Rampδ²(ramp) = Rampδ²(ramp, eltype(ramp)(default(:γ²min)))
Rampδ²(base, startexp, stopexp, args...) = Rampδ²(base.^(startexp:-1:stopexp), args...)

function adapt(::Type{R}, algorithm::Rampδ²) where {R<:Real}
    return Rampδ²(R.(algorithm.ramp), R(algorithm.γ²min))
end

adapt(::Type{R}, algorithm::Rampδ²{R}) where {R} = algorithm

struct Rampμ{R, A<:Algorithm{R}} <: Ramp{R}
    initial_algorithm::A
    ramp::Vector{R}
    γ²min::R
end

Rampμ() = Rampμ(StraightShot())

function Rampμ(initial_algorithm::Algorithm{R}) where {R}
    return Rampμ(initial_algorithm, R.(default(:μramp)))
end

function Rampμ(initial_algorithm, ramp)
    return Rampμ(initial_algorithm, ramp, eltype(ramp)(default(:γ²min)))
end

function Rampμ(initial_algorithm::Algorithm{R}, n::Integer, args...) where {R}
    return Rampμ(initial_algorithm, collect(LinRange(zero(R), one(R), n + 1)), args...)
end

function adapt(::Type{R}, algorithm::Rampμ) where {R<:Real}
    return Rampμ(
        adapt(R, algorithm.initial_algorithm), R.(algorithm.ramp), R(algorithm.γ²min)
    )
end

adapt(::Type{R}, algorithm::Rampμ{R}) where {R} = algorithm

struct StraightShot{R} <: Algorithm{R}
    q₀::Complex{R}
    δ₀²::R
end

StraightShot() = StraightShot(default(:q₀))
StraightShot(q₀::C) where {C} = StraightShot(q₀, real(C)(default(:δ₀²)))

function adapt(::Type{R}, algorithm::StraightShot) where {R<:Real}
    return StraightShot(Complex{R}(algorithm.q₀), R(algorithm.δ₀²))
end

adapt(::Type{R}, algorithm::StraightShot{R}) where {R} = algorithm

Base.length(algorithm::Rampμ) = length(algorithm.initial_algorithm) + length(algorithm.ramp)
Base.length(algorithm::Downramp) = length(algorithm.ramp)
Base.length(::Algorithm) = 0

# TODO: Is δ₀² what we need a function for? Is γ∞² better? # function q∞!(problem, γ∞², ws)
δ₀²(algorithm::Downramp) = algorithm.ramp[1]
δ₀²(algorithm::Rampμ) = δ₀²(algorithm.initial_algorithm)
δ₀²(algorithm) = algorithm.δ₀²

q₀!(problem, algorithm::Downramp, ws) = q∞!(problem, δ₀²(algorithm), ws)
q₀!(problem, algorithm::Rampμ, ws) = q₀!(problem, algorithm.initial_algorithm, ws)
q₀!(::Any, algorithm, ::Any) = algorithm.q₀

initial_algorithm(algorithm::Rampμ, ::Any, ::Any) = algorithm.initial_algorithm
initial_algorithm(algorithm, q₀, δ₀²) = StraightShot(q₀, δ₀²)

function step_objective(objective, algorithm::Rampδ², i)
    return Fixδ²(algorithm.ramp[i])
end

function step_objective(objective, algorithm::Rampγ², i)
    return Fixγ²(algorithm.ramp[i])
end

# TODO: that ain't working---does μ need to be in the objective after all?
function step_objective(objective::O, algorithm::Rampμ, i) where {O}
    return O(objective.value, algorithm.ramp[i])
end

function invalid(sol, ::StraightShot, ::Any=nothing)
    return sol.δ² < zero(sol.δ²)
end

invalid(step_sol, ::Rampδ², γ_²) = ((step_sol.q === missing) || (step_sol.γ² > γ_²))
invalid(step_sol, ::Ramp, ::Any) = (step_sol.q === missing)

done(step_sol, ::Rampγ², objective::Fixγ²) = (step_sol.γ² == objective.value)
done(step_sol, ::Rampδ², objective::Fixδ²) = (step_sol.δ² == objective.value)
done(step_sol, ::Rampμ, objective) = (step_sol.μ == objective.μ)
done(::Any, ::Ramp, ::Any) = false

final(step_sol, ::Downramp, objective::Fixk) = (step_sol.k >= objective.value)
final(step_sol, ::Downramp, objective::Fixγ²) = (step_sol.γ² <= objective.value)
final(step_sol, ::Downramp, objective::Fixδ²) = (step_sol.δ² <= objective.value)
final(step_sol, ::Rampμ, objective) = (step_sol.μ > objective.μ)
final(::Any, ::Ramp, ::Any) = false

abort(step_sol, algorithm::Union{Rampδ², Rampμ}) = (step_sol.γ² < algorithm.γ²min)
abort(::Any, ::Ramp) = false
