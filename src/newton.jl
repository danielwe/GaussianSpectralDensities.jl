module Newton

using StaticArrays: MMatrix, MVector, SMatrix, SVector

using ..Defaults: Defaults

public finite_difference_jacobian, increment, newton

"""
    newton(f, x0; atol=<see text>, maxiters=512)
    newton(f, jac, x0; atol=<see text>, maxiters=512)

Find a zero of `f` using Newton's method starting at `x0`

`jac` must be a function with signature `jac(x, f(x))` that returns the jacobian of `f` at
`x` in a type that supports `ldiv`, such that `jac(x, f(x)) \\ f(x)` works.

If `jac` is not provided, it is estimated using finite differences. The finite difference
implementation currently supports `x::Real` and `x::SVector` from `StaticArrays.jl`.

The keyword argument `atol` sets the tolerance for what's considered a scalar zero. For
a vector `y` this is scaled by `sqrt(length(y))` and compared with the 2-norm, i.e., the
criterion for `f(x)` to be considered zero is `norm(f(x)) < atol * sqrt(length(f(x)))`. The
default value is `atol=sqrt(eps(eltype(x))) / 4`.

Returns `(x, i, residual, converged)`, where `x` is the final state, `i::Int` is the number
of iterations, `residual` is the final residual, i.e., `norm(f(x)) / sqrt(length(f(x)))`,
and `converged::Bool` is `true` if the iterations converged.
"""
function newton(f, x0; kwargs...)
    jac = let f = f
        jac(x, fx) = finite_difference_jacobian(f, x, fx)
    end
    return newton(f, jac, x0; kwargs...)
end

function newton(f, jac, x0; atol=Defaults.atol_newton(eltype(x0)), maxiters=512)
    i = 0
    x = x0
    fx, residual, converged = apply_and_check(f, x, atol)
    while (!converged) && isfinite(residual) && (i < maxiters)
        i += 1
        x = x - jac(x, fx) \ fx
        fx, residual, converged = apply_and_check(f, x, atol)
    end
    return (x, i, residual, converged)
end

function apply_and_check(f, x, atol)
    fx = f(x)
    residual = scaled_norm(fx)
    return fx, residual, (residual < atol)
end

scaled_norm(y) = sqrt(sum(abs2, y) / length(y))

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

function increment(x::Real)
    inc = sqrt(eps(one(x))) * oneunit(x) / 4  # empirically gives the most accurate derivatives
    return (x < zero(x)) ? -inc : inc
end

end  # module Newton

@testitem "newton Real" begin
    using GaussianSpectralDensities: Defaults
    using GaussianSpectralDensities.Newton: finite_difference_jacobian, newton

    using LinearAlgebra: norm

    tol_exponent = 1 // 3

    f(x) = cos(x) - x^3
    df(x, _) = -sin(x) - 3x^2

    x0_64 = 0.5

    @testset "$F" for F in [Float32, Float64]
        x0_F = F(x0_64)
        tol = eps(F)^tol_exponent

        # finite_difference_jacobian
        fx0_F = f(x0_F)
        dfx0_F = df(x0_F, fx0_F)
        dfx0_fd = finite_difference_jacobian(f, x0_F, fx0_F)
        @test dfx0_fd isa typeof(fx0_F / x0_F)
        @test dfx0_fd ≈ dfx0_F rtol = tol
        @test finite_difference_jacobian(f, x0_F) ≈ dfx0_F rtol = tol

        # Negative point exercises the negative increment
        xneg_F = -x0_F
        fxneg_F = f(xneg_F)
        dfxneg_F = df(xneg_F, fxneg_F)
        @test finite_difference_jacobian(f, xneg_F, fxneg_F) ≈ dfxneg_F rtol = tol
        @test finite_difference_jacobian(f, xneg_F) ≈ dfxneg_F rtol = tol

        # newton, with and without explicit jacobian
        @testset "$label" for (label, jac) in
                              [("finite difference", nothing), ("explicit", df)]
            result = (jac === nothing) ? newton(f, x0_F) : newton(f, jac, x0_F)
            x, i, residual, converged = result
            @test x isa typeof(x0_F)
            @test converged
            @test 0 < i < 512
            @test residual ≈ norm(f(x))  # scalar objective => scale factor is 1
            @test residual < Defaults.atol_newton(F)
        end

        # maxiters and residual threshold work as expected
        c = F(0.5)
        fconst(_) = c
        x, i, residual, converged = newton(fconst, x0_F; atol=c, maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == c
        @test !converged

        x, i, residual, converged = newton(fconst, x0_F; atol=nextfloat(c), maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == c
        @test converged

        default_atol = Defaults.atol_newton(F)
        x, i, residual, converged = newton(x -> default_atol, x0_F; maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == default_atol
        @test !converged

        x, i, residual, converged = newton(x -> prevfloat(default_atol), x0_F; maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == prevfloat(default_atol)
        @test converged
    end
end

@testitem "newton SVector" begin
    using GaussianSpectralDensities: Defaults
    using GaussianSpectralDensities.Newton: finite_difference_jacobian, newton

    using LinearAlgebra: norm
    using StaticArrays: SMatrix, SVector, Size

    tol_exponent = 1 // 3

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
        tol = eps(F)^tol_exponent

        # finite_difference_jacobian
        gx0_F = g(x0_F)
        jgx0_F = jg(x0_F, gx0_F)
        jgx0_fd = finite_difference_jacobian(g, x0_F, gx0_F)
        @test jgx0_fd isa typeof(gx0_F ./ x0_F')
        @test jgx0_fd ≈ jgx0_F rtol = tol
        @test finite_difference_jacobian(g, x0_F) ≈ jgx0_F rtol = tol

        # newton, with and without explicit jacobian
        @testset "$label" for (label, jac) in
                              [("finite difference", nothing), ("explicit", jg)]
            result = (jac === nothing) ? newton(g, x0_F) : newton(g, jac, x0_F)
            x, i, residual, converged = result
            @test x isa typeof(x0_F)
            @test converged
            @test 0 < i < 512
            @test residual ≈ norm(g(x)) / sqrt(length(g(x)))
            @test residual < Defaults.atol_newton(F)
        end

        # residual is normalized, and maxiters and residual threshold work as expected
        c = F(0.5)
        h(_) = SVector{2,F}(c, c)
        x, i, residual, converged = newton(h, x0_F; atol=c, maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == F(0.5)
        @test !converged

        x, i, residual, converged = newton(h, x0_F; atol=nextfloat(c), maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == F(0.5)
        @test converged

        default_atol = Defaults.atol_newton(F)
        default_atolv = SVector{2,F}(default_atol, default_atol)
        x, i, residual, converged = newton(x -> default_atolv, x0_F; maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == default_atol
        @test !converged

        x, i, residual, converged = newton(x -> prevfloat.(default_atolv), x0_F; maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual == prevfloat(default_atol)
        @test converged
    end
end

@testitem "Newton failure flags" begin
    using GaussianSpectralDensities.Newton: newton

    using LinearAlgebra: norm
    using StaticArrays: SVector

    f(x) = cos(x) - x^3
    g(x) = (x .* (x .^ 2 .- 1)) .+ (reverse(x) ./ 2)

    @testset "$F" for F in [Float32, Float64]
        x0_F = F(0.5)

        # maxiters=0 returns a failure flag
        x, i, residual, converged = newton(f, x0_F; maxiters=0)
        @test x == x0_F
        @test i == 0
        @test residual ≈ norm(f(x0_F))
        @test isfinite(residual)
        @test !converged

        # maxiters=1 performs one iteration and still reports failure here
        x, i, residual, converged = newton(f, x0_F; maxiters=1)
        @test x != x0_F
        @test i == 1
        @test residual ≈ norm(f(x))
        @test isfinite(residual)
        @test !converged

        # Nonfinite initial residual stops immediately
        @testset "f(x0) = $bad" for bad in (F(NaN), F(Inf))
            x, i, residual, converged = newton(x -> bad, x0_F)
            @test x == x0_F
            @test i == 0
            @test !isfinite(residual)
            @test !converged
        end

        # Nonfinite residual produced during iteration stops immediately
        nonfinitef(x) = x < 1 ? x - 1 : oftype(x, NaN)
        unitjac(x, fx) = oneunit(fx) / oneunit(x)
        x, i, residual, converged = newton(nonfinitef, unitjac, zero(F))
        @test x == F(1.0)
        @test i == 1
        @test isnan(residual)
        @test !converged

        # Initial point is already a root: converges without iterating
        linearf(x) = x - oftype(x, 0.5)
        x, i, residual, converged = newton(linearf, F(0.5))
        @test x == F(0.5)
        @test i == 0
        @test residual == zero(F)
        @test converged

        # Linear function converges after one iteration
        x, i, residual, converged = newton(linearf, unitjac, zero(F); maxiters=1)
        @test x == F(0.5)
        @test i == 1
        @test residual == zero(F)
        @test converged

        # Failure flag and nonfinite residual for vectors
        x0v_F = SVector{2,F}(0.3, -0.7)
        xv, i, residual, converged = newton(g, x0v_F; maxiters=0)
        @test xv == x0v_F
        @test i == 0
        @test residual ≈ norm(g(xv)) / sqrt(length(g(xv)))
        @test !converged

        @testset "g(x0v) = SVector($bad, 0)" for bad in (F(NaN), F(Inf))
            xv, i, residual, converged = newton(x -> SVector{2,F}(bad, 0.0), x0v_F)
            @test xv == x0v_F
            @test i == 0
            @test !isfinite(residual)
            @test !converged
        end
    end
end
