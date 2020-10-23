abstract type Algorithm end

struct FixK <: Algorithm
    rsqs::Vector{Float64}
end

struct FixΓ <: Algorithm
    rsqs::Vector{Float64}
end

FixK(rbase, startexp, stopexp) = FixK(rbase.^(startexp:-1:stopexp))
FixΓ(rbase, startexp, stopexp) = FixΓ(rbase.^(startexp:-1:stopexp))
FixK() = FixK(DEFAULT_ALGORITHM_ARGS...)
FixΓ() = FixΓ(DEFAULT_ALGORITHM_ARGS...)

const DEFAULT_ALGORITHM_ARGS = (1.5, 18, -18)
const DEFAULT_ALGORITHM = FixK()
