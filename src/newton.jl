struct NewtonException <: Exception
    msg::String
end

"""
    newton(f, x; atol=<see text>, maxiters=999)
    newton(f, jac, x; atol=<see text>, maxiters=999)

Find a zero of `f` using Newton's method starting at `x`

`jac` must be a function with signature `jac(x, f(x))` that returns the jacobian of `f` at
`x` in a type that supports `ldiv`, such that `jac(x, f(x)) \\ f(x)` works.

If `jac` is not provided, it is estimated using finite differences. The finite difference
implementation currently supports `x isa Real` and `x isa SVector` from `StaticArrays.jl`.

The keyword argument `atol` sets the tolerance for what's considered a scalar zero. For
a vector `y` this is scaled by `sqrt(length(y))` and compared with the 2-norm, i.e., the
criterion for `f(x)` to be considered zero is `norm(f(x)) < atol * sqrt(length(f(x)))`. The
default value is `atol=sqrt(eps(T) * oneunit(T)) / 2` where `T = eltype(x)` (the `oneunit`
ensures dimensional consistency in case `T` is a dimensionful type).

A `NewtonException` is thrown after `maxiters` iterations without convergence.
"""
function newton(f, x; kwargs...)
    jac = let f=f
        jac(x, fx) = finite_difference_jacobian(f, x, fx)
    end
    return newton(f, jac, x; kwargs...)
end

function newton(f, jac, x; atol=_atol(eltype(x)), maxiters=999)
    fx = f(x)
    _approxzero(fx, atol) && return (x, 0)
    for i in 1:maxiters
        x = x - jac(x, fx) \ fx
        fx = f(x)
        _approxzero(fx, atol) && return (x, i)
    end
    throw(NewtonException("failed to converge after maxiters=$maxiters iterations"))
end

_approxzero(y, atol) = (sum(abs2, y) < length(y) * atol^2)
_atol(T) = sqrt(eps(T) * oneunit(T)) / 2  # oneunit for dimensional consistency

"""
    finite_difference_jacobian(f, x::Real, fx=f(x))
    finite_difference_jacobian(f, x::SVector, fx=f(x))

Approximate the jacobian of `f` at `x` using finite differences

For `x isa Real` this approximates the derivative. For `x isa SVector` it
returns an `SMatrix` approximating the jacobian.

There are no hard constraints on the return type of `f`, but the methods are written with
`Real` and `SVector` in mind and are not optimized for other cases such as `Vector`.
"""
function finite_difference_jacobian(f, x::Real, fx=f(x))
    xinc = x + _increment(x)
    return (f(xinc) - fx) / (xinc - x)  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
end

function finite_difference_jacobian(f, x::SVector, fx=f(x))
    T = typeof(oneunit(eltype(fx)) / oneunit(eltype(x)))  # oneunit for dimensional consistency
    jac = MMatrix{length(fx),length(x),T}(undef)
    mx = MVector(x)
    for i in eachindex(x)
        mx[i] = x[i] + _increment(x[i])
        xinc = SVector(mx)
        jac[:, i] = (f(xinc) - fx) / (xinc[i] - x[i])  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
        mx[i] = x[i]
    end
    return SMatrix(jac)
end

function _increment(x::T) where {T<:Real}
    if (x > oneunit(T)) || (x < -oneunit(T))  # oneunit for dimensional consistency
        return _relativeincrement(T) * x
    elseif x < zero(T)
        return -_relativeincrement(T) * oneunit(T)
    else
        return _relativeincrement(T) * oneunit(T)
    end
end

_relativeincrement(T) = sqrt(eps(T) / oneunit(T))  # oneunit for dimensional consistency
