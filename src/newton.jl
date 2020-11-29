struct NewtonException <: Exception
    msg::String
end

function newton(f, x; maxiters=500)
    fx = f(x)
    if approxzero(fx)
        return x, 0
    end
    for i in 1:maxiters
        Jx = finite_difference_jacobian(f, x, fx)
        x = x - Jx \ fx
        fx = f(x)
        if approxzero(fx)
            return x, i
        end
    end
    throw(NewtonException("maxiters=$maxiters exceeded"))
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
    dxi = (x[i] + step(x[i])) - x[i]  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
    xdx = SVector{L}(j == i ? x[j] + dxi : x[j] for j in 1:L)
    return (f(xdx) - fx) * (one(dxi) / dxi)
end

function step(x)
    if abs2(x) > one(x)
        return atol(x) * x
    end
    if x < zero(x)
        return -atol(x)
    end
    return atol(x)
end

approxzero(v) = (sqrt(sum(abs2, v) / length(v)) < atol(v))

atol(x) = sqrt(eps(eltype(x)))
