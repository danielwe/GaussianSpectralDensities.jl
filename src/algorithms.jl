abstract type Algorithm{R<:Real} end
abstract type Ramp{R} <: Algorithm{R} end
abstract type Downramp{R} <: Ramp{R} end

struct Rampγ²{R} <: Downramp{R}
    ramp::Vector{R}
end

Rampγ²() = Rampγ²(default(:r²ramp))
Rampγ²(base, startexp, stopexp) = Rampγ²(base.^(startexp:-1:stopexp))

function adapt(::Type{R}, algorithm::Rampγ²) where {R<:Real}
    return Rampγ²(R.(algorithm.ramp))
end

adapt(::Type{R}, algorithm::Rampγ²{R}) where {R<:Real} = algorithm

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

adapt(::Type{R}, algorithm::Rampδ²{R}) where {R<:Real} = algorithm

struct Rampμ{R, A<:Algorithm{R}} <: Ramp{R}
    initial_algorithm::A
    ramp::Vector{R}
    γ²min::R
end

Rampμ() = Rampμ(default(:μInitialAlgorithm))

function Rampμ(initial_algorithm::Algorithm{R}) where {R}
    return Rampμ(initial_algorithm, R.(default(:μramp)))
end

function Rampμ(initial_algorithm, ramp)
    return Rampμ(initial_algorithm, ramp, eltype(ramp)(default(:γ²min)))
end

function Rampμ(initial_algorithm::Algorithm{R}, n::Integer, args...) where {R}
    return Rampμ(initial_algorithm, collect(range(R(0.0), R(1.0); length=(n + 1))), args...)
end

function adapt(::Type{R}, algorithm::Rampμ) where {R<:Real}
    return Rampμ(
        adapt(R, algorithm.initial_algorithm), R.(algorithm.ramp), R(algorithm.γ²min)
    )
end

adapt(::Type{R}, algorithm::Rampμ{R}) where {R<:Real} = algorithm

struct StraightShot{R} <: Algorithm{R}
    q₀::Complex{R}
    δ₀²::R
end

StraightShot() = StraightShot(default(:q₀))
StraightShot(q₀) = StraightShot(q₀, real(typeof(q₀))(default(:δ₀²)))

function adapt(::Type{R}, algorithm::StraightShot) where {R<:Real}
    return StraightShot(Complex{R}(algorithm.q₀), R(algorithm.δ₀²))
end

adapt(::Type{R}, algorithm::StraightShot{R}) where {R<:Real} = algorithm

Base.length(algorithm::Rampμ) = length(algorithm.initial_algorithm) + length(algorithm.ramp)
Base.length(algorithm::Downramp) = length(algorithm.ramp)
Base.length(::Algorithm) = 0

q₀!(workspace, algorithm::Downramp, z) = q∞!(workspace, δ₀²(algorithm), z)
q₀!(workspace, algorithm::Rampμ, z) = q₀!(workspace, algorithm.initial_algorithm, z)
q₀!(::Any, algorithm, z) = algorithm.q₀

δ₀²(algorithm::Downramp) = algorithm.ramp[begin]
δ₀²(algorithm::Rampμ) = δ₀²(algorithm.initial_algorithm)
δ₀²(algorithm) = algorithm.δ₀²

initial_algorithm(algorithm::Rampμ, ::Any, ::Any) = algorithm.initial_algorithm
initial_algorithm(algorithm, q₀, δ₀²) = StraightShot(q₀, δ₀²)

function step_objective(objective::Objective{QNormal}, algorithm::Rampδ², i) where {QNormal}
    return Fixδ²{QNormal}(algorithm.ramp[i], objective.μ)
end

function step_objective(objective::Objective{QNormal}, algorithm::Rampγ², i) where {QNormal}
    return Fixγ²{QNormal}(algorithm.ramp[i], objective.μ)
end

function step_objective(objective::O, algorithm::Rampμ, i) where {O}
    return O(objective.value, algorithm.ramp[i])
end

function invalid_step(::Rampδ², step_solution, γ_²)
    return (step_solution.q === missing) || (step_solution.γ² > γ_²)
end

invalid_step(::Any, step_solution, ::Any) = (step_solution.q === missing)

done(objective::Fixγ², ::Rampγ², step_solution) = (step_solution.γ² == objective.value)
done(objective::Fixδ², ::Rampδ², step_solution) = (step_solution.δ² == objective.value)
done(objective, ::Rampμ, step_solution) = (step_solution.μ == objective.μ)
done(::Any, ::Any, ::Any) = false

final(objective::Fixk, ::Downramp, step_solution) = (step_solution.k >= objective.value)
final(objective::Fixγ², ::Downramp, step_solution) = (step_solution.γ² <= objective.value)
final(objective::Fixδ², ::Downramp, step_solution) = (step_solution.δ² <= objective.value)
final(objective, ::Rampμ, step_solution) = (step_solution.μ > objective.μ)
final(::Any, ::Any, ::Any) = false

function abort(algorithm::Union{Rampδ², Rampμ}, step_solution)
    return (step_solution.γ² < algorithm.γ²min)
end

abort(::Any, ::Any) = false
