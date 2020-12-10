struct NewtonException <: Exception
    msg::String
end

function newton(f, j, x; maxiters=999)
    fx = f(x)
    if approxzero(fx)
        return x, 0
    end
    for i in 1:maxiters
        x = x - j(x, fx) \ fx
        fx = f(x)
        if approxzero(fx)
            return x, i
        end
    end
    throw(NewtonException("maxiters=$maxiters exceeded"))
end

function newton(f, x; kwargs...)
    j(x, fx) = finite_difference_jacobian(f, x, fx)
    return newton(f, j, x; kwargs...)
end

function finite_difference_jacobian(f, x)
    return finite_difference_jacobian(f, x, f(x))
end

@generated function finite_difference_jacobian(f, x::SVector{L}, fx) where {L}
    ex = Expr(:call, :hcat)
    for i in 1:L
        push!(ex.args, :(finite_difference_jacobian_column(f, x, fx, $i)))
    end
    return ex
end

function finite_difference_jacobian_column(f, x::SVector{L}, fx, i) where {L}
    xidxi = x[i] + step(x[i])
    xdx = SVector{L}(j == i ? xidxi : x[j] for j in 1:L)
    return (f(xdx) - fx) / (xidxi - x[i])  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
end

function step(x)
    if abs2(x) > one(x)
        return relstep(x) * x
    end
    if x < zero(x)
        return -relstep(x)
    end
    return relstep(x)
end

approxzero(v) = (sqrt(sum(abs2, v) / length(v)) < atol(v))

atol(x) = relstep(x)

relstep(x) = sqrt(eps(eltype(x)))
