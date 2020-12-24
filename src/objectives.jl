abstract type Objective{QNormal, R<:Real} end
abstract type Fixr²{QNormal, R} <: Objective{QNormal, R} end

struct Fixk{QNormal, R} <: Objective{QNormal, R}
    value::R
    μ::R
end

struct Fixγ²{QNormal, R} <: Fixr²{QNormal, R}
    value::R
    μ::R
end

struct Fixδ²{QNormal, R} <: Fixr²{QNormal, R}
    value::R
    μ::R
end

Fixk{QNormal}(k::R, μ::R) where {QNormal, R} = Fixk{QNormal, R}(k, μ)
Fixγ²{QNormal}(γ²::R, μ::R) where {QNormal, R} = Fixγ²{QNormal, R}(γ², μ)
Fixδ²{QNormal}(δ²::R, μ::R) where {QNormal, R} = Fixδ²{QNormal, R}(δ², μ)

Fixk{QNormal}(k::R) where {QNormal, R} = Fixk{QNormal, R}(k, R(default(:μ)))
Fixγ²{QNormal}(γ²::R) where {QNormal, R} = Fixγ²{QNormal, R}(γ², R(default(:μ)))
Fixδ²{QNormal}(δ²::R) where {QNormal, R} = Fixδ²{QNormal, R}(δ², R(default(:μ)))

Fixk(args...) = Fixk{default(:QNormal)}(args...)
Fixγ²(args...) = Fixγ²{default(:QNormal)}(args...)
Fixδ²(args...) = Fixδ²{default(:QNormal)}(args...)

Fixk{QNormal}() where {QNormal} = Fixk{QNormal}(default(:k))
Fixγ²{QNormal}() where {QNormal} = Fixγ²{QNormal}(default(:γ²))

function adapt(::Type{R}, objective::Fixk{QNormal}) where {R<:Real, QNormal}
    return Fixk{QNormal, R}(R(objective.value), R(objective.μ))
end

function adapt(::Type{R}, objective::Fixγ²{QNormal}) where {R<:Real, QNormal}
    return Fixγ²{QNormal, R}(R(objective.value), R(objective.μ))
end

function adapt(::Type{R}, objective::Fixδ²{QNormal}) where {R<:Real, QNormal}
    return Fixδ²{QNormal, R}(R(objective.value), R(objective.μ))
end

adapt(::Type{R}, objective::Fixk{QNormal, R}) where {R<:Real, QNormal} = objective
adapt(::Type{R}, objective::Fixγ²{QNormal, R}) where {R<:Real, QNormal} = objective
adapt(::Type{R}, objective::Fixδ²{QNormal, R}) where {R<:Real, QNormal} = objective

score(value, reference) = one(value) / abs(value - reference)
score(solution, objective::Fixk) = score(log(solution.k), log(objective.value))
score(solution, objective::Fixγ²) = score(solution.γ², objective.value)
score(solution, objective::Fixδ²) = score(solution.δ², objective.value)
