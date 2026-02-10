module Newton

using StaticArrays: MMatrix, MVector, SMatrix, SVector

using ..Defaults: Defaults

# TODO: Try with Enzyme.jl, it might actually be able to diff through the Workspace craze

public NewtonException, finite_difference_jacobian, newton

struct NewtonException <: Exception
    msg::String
end

"""
    newton(f, x; atol=<see text>, maxiters=63)
    newton(f, jac, x; atol=<see text>, maxiters=63)

Find a zero of `f` using Newton's method starting at `x`

`jac` must be a function with signature `jac(x, f(x))` that returns the jacobian of `f` at
`x` in a type that supports `ldiv`, such that `jac(x, f(x)) \\ f(x)` works.

If `jac` is not provided, it is estimated using finite differences. The finite difference
implementation currently supports `x isa Real` and `x isa SVector` from `StaticArrays.jl`.

The keyword argument `atol` sets the tolerance for what's considered a scalar zero. For
a vector `y` this is scaled by `sqrt(length(y))` and compared with the 2-norm, i.e., the
criterion for `f(x)` to be considered zero is `norm(f(x)) < atol * sqrt(length(f(x)))`. The
default value is `atol=eps(T)^0.75 * oneunit(T)^0.25` where `T = eltype(x)` (the `oneunit`
ensures dimensional consistency in case `T` is a dimensionful type).

A `NewtonException` is thrown after `maxiters` iterations without convergence.
"""
function newton(f, x; kwargs...)
    jac = let f = f
        jac(x, fx) = finite_difference_jacobian(f, x, fx)
    end
    return newton(f, jac, x; kwargs...)
end

function newton(f, jac, x; atol=Defaults.tol(eltype(x)) / 10, maxiters=63)
    fx = f(x)
    approx_zero(fx, atol) && return (x, 0)
    for i in 1:maxiters
        x = x - jac(x, fx) \ fx
        fx = f(x)
        approx_zero(fx, atol) && return (x, i)
    end
    throw(NewtonException("failed to converge after maxiters=$maxiters iterations"))
end

approx_zero(y, atol) = (sqrt(sum(abs2, y) / length(y)) < atol)

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
    xinc = x + increment(x)
    return (f(xinc) - fx) / (xinc - x)  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
end

function finite_difference_jacobian(f, x::SVector, fx=f(x))
    T = typeof(oneunit(eltype(fx)) / oneunit(eltype(x)))  # oneunit for dimensional consistency
    jac = MMatrix{length(fx),length(x),T}(undef)
    mx = MVector(x)
    for i in eachindex(x)
        mx[i] = x[i] + increment(x[i])
        xinc = SVector(mx)
        jac[:, i] = (f(xinc) - fx) / (xinc[i] - x[i])  # Numerical recipes accuracy trick https://twitter.com/willkurt/status/1330183861452541953
        mx[i] = x[i]
    end
    return SMatrix(jac)
end

increment(x::T) where {T<:Real} =
    if (x > oneunit(T)) || (x < -oneunit(T))  # oneunit for dimensional consistency
        return relative_increment(T) * x
    elseif x < zero(T)
        return -relative_increment(T) * oneunit(T)
    else
        return relative_increment(T) * oneunit(T)
    end

relative_increment(T) = sqrt(eps(T) / oneunit(T))  # oneunit for dimensional consistency

end  # module Newton

@testitem "newton Real" begin
    using GaussianSpectralDensities.Newton: finite_difference_jacobian, newton

    using LinearAlgebra: norm

    rtol_exponent = 2 // 5
    atol_exponent = 3 // 4

    f(x) = cos(x) - x^3
    df(x, _) = -sin(x) - 3x^2

    x0_64 = 0.5

    @testset "$F" for F in [Float32, Float64]
        x0_F = F(x0_64)
        rtol = eps(F)^rtol_exponent
        atol = eps(F)^atol_exponent

        # finite_difference_jacobian
        fx0_F = f(x0_F)
        dfx0_F = df(x0_F, fx0_F)
        @test finite_difference_jacobian(f, x0_F, fx0_F) ≈ dfx0_F rtol = rtol
        @test finite_difference_jacobian(f, x0_F) ≈ dfx0_F rtol = rtol

        # newton
        @test norm(f(first(newton(f, x0_F)))) < atol
        @test norm(f(first(newton(f, df, x0_F)))) < atol
    end
end

@testitem "newton SVector" begin
    using GaussianSpectralDensities.Newton: finite_difference_jacobian, newton

    using LinearAlgebra: norm
    using StaticArrays: SMatrix, SVector, Size

    rtol_exponent = 2 // 5
    atol_exponent = 3 // 4

    g(x) = (x .* (x .^ 2 .- 1)) .+ (reverse(x) ./ 2)

    function jg(x::SVector{L}, _) where {L}
        jac = fill!(similar(x, Size(L, L)), 0)
        for i in 1:L
            jac[i, i] = 3x[i]^2 - 1
            jac[L - i + 1, i] = 0.5
        end
        return SMatrix(jac)
    end

    x0_64 = SVector(0.3, -0.7)

    @testset "$F" for F in [Float32, Float64]
        x0_F = map(F, x0_64)
        rtol = eps(F)^rtol_exponent
        atol = eps(F)^atol_exponent

        # finite_difference_jacobian
        gx0_F = g(x0_F)
        jgx0_F = jg(x0_F, gx0_F)
        @test finite_difference_jacobian(g, x0_F, gx0_F) ≈ jgx0_F rtol = rtol
        @test finite_difference_jacobian(g, x0_F) ≈ jgx0_F rtol = rtol

        # newton
        N = length(x0_F)
        @test norm(g(first(newton(g, x0_F)))) < atol * sqrt(N)
        @test norm(g(first(newton(g, jg, x0_F)))) < atol * sqrt(N)
    end
end

@testitem "Newton errors" begin
    using GaussianSpectralDensities.Newton: NewtonException, newton
    f(x) = cos(x) - x^3
    @test_throws NewtonException newton(f, 0.5; maxiters=0)
end
