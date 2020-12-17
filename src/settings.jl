function default(x)
    if x === :QNormal; return false; end
    if x === :verbosity; return 0; end
    if x === :store_trace; return false; end
    if x === :k; return 1.0; end
    if x === :γ²; return 0.0; end
    if x === :μ;  return 1.0; end
    if x === :r²ramp; return 2.0.^(8:-1:-8); end
    if x === :μramp; return collect(range(0.0, 1.0; length=9)); end
    if x === :γ²min; return -1.0; end
    if x === :q₀; return complex(Float64(relstep(Float32))); end
    if x === :δ₀²; return 1.0; end
    error("no default value for $x")
end
