function default(x)
    (x === :normalizeQ) && return false
    (x === :verbosity) && return 0
    (x === :store_trace) && return false
    (x === :k) && return 1.0
    (x === :γ) && return 0.0
    (x === :μ) &&  return 1.0
    (x === :downramp) && return 2.0.^(4:-0.5:-4)
    (x === :μramp) && return collect(LinRange(0.0, 1.0, 9))
    (x === :γ²min) && return -Inf
    (x === :q0) && return complex(Float64(relstep(Float32)))
    (x === :δ0) && return 1.0
    (x === :margin) && return 0.1
    throw(DomainError(x, "no default value for $x"))
end
