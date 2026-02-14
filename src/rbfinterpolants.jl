module RBFInterpolants

using LinearAlgebra: Symmetric, norm
using LogExpFunctions: xlogx
using StaticArrays: SMatrix, SVector

public RBFInterpolant, ThinPlateSpline, append!, push!

struct ThinPlateSpline end

(::ThinPlateSpline)(r::T) where {T<:Real} = r * xlogx(r)
(phi::ThinPlateSpline)(v::SVector) = phi(norm(v))

mutable struct RBFInterpolant{F<:AbstractFloat,N,Phi,P<:SMatrix{N,3,F}}
    phi::Phi
    x::Vector{SVector{2,F}}
    y::Vector{SVector{N,F}}
    coeffs::Vector{SVector{N,F}}
    poly::P
end

function RBFInterpolant(
    phi::Phi, x::AbstractVector{SVector{2,F}}, y::AbstractVector{SVector{N,F}}
) where {F<:AbstractFloat,N,Phi}
    xv, yv = convert(Vector{SVector{2,F}}, x), convert(Vector{SVector{N,F}}, y)
    coeffs, poly = rbf_coefficients(phi, xv, yv)
    return RBFInterpolant{F,N,Phi,typeof(poly)}(phi, xv, yv, coeffs, poly)
end

function (rbf::RBFInterpolant{F})(pt::SVector{2,F}) where {F}
    result = rbf.poly * SVector(oneunit(F), pt...)
    for (index, xi) in pairs(rbf.x)
        result += rbf.coeffs[index] * rbf.phi(pt - xi)
    end
    return result
end

function push!(
    rbf::RBFInterpolant{F,N}, x_new::SVector{2,F}, y_new::SVector{N,F}
) where {F,N}
    any(xi -> xi == x_new, rbf.x) && throw(ArgumentError("collision with existing point"))
    Base.push!(rbf.x, x_new)
    Base.push!(rbf.y, y_new)
    rbf.coeffs, rbf.poly = rbf_coefficients(rbf.phi, rbf.x, rbf.y)
    return rbf
end

function append!(
    rbf::RBFInterpolant{F,N},
    xs_new::AbstractVector{SVector{2,F}},
    ys_new::AbstractVector{SVector{N,F}},
) where {F,N}
    length(xs_new) == length(ys_new) || throw(ArgumentError("x/y length mismatch"))

    existing = Set(rbf.x)
    seen = Set{eltype(xs_new)}()
    for xi in xs_new
        (xi in existing) && throw(ArgumentError("collision with existing point"))
        (xi in seen) && throw(ArgumentError("collision among new points"))
        Base.push!(seen, xi)
    end

    Base.append!(rbf.x, xs_new)
    Base.append!(rbf.y, ys_new)
    rbf.coeffs, rbf.poly = rbf_coefficients(rbf.phi, rbf.x, rbf.y)
    return rbf
end

function rbf_coefficients(
    phi::Phi, x::Vector{SVector{2,F}}, y::Vector{SVector{N,F}}
) where {F<:AbstractFloat,N,Phi}
    n = length(x)
    n == length(y) || throw(ArgumentError("x/y length mismatch"))

    system = Matrix{F}(undef, n + 3, n + 3)
    for j in 1:n
        system[j, j] = zero(F)
        xj = x[j]
        for i in (j + 1):n
            xi = x[i]
            phi_r = phi(xj - xi)
            system[i, j] = phi_r
            system[j, i] = phi_r
        end
    end

    for j in 1:n
        system[(n + 1):end, j] = SVector(one(F), x[j]...)
    end

    system[1:n, (n + 1):end] = system[(n + 1):end, 1:n]'
    system[(n + 1):end, (n + 1):end] .= zero(F)

    rhs = Matrix{F}(undef, N, n + 3)
    for j in 1:n
        rhs[:, j] = y[j]
    end
    rhs[:, (n + 1):end] .= zero(F)

    solution = rhs / Symmetric(system)

    coeffs = similar(y)
    for j in 1:n
        coeffs[j] = solution[:, j]
    end

    poly_coeffs = SMatrix{N,3,F}(solution[:, (n + 1):end])
    return coeffs, poly_coeffs
end

end  # module RBFInterpolants

@testitem "ThinPlateSpline" begin
    using GaussianSpectralDensities.RBFInterpolants: ThinPlateSpline

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        @test phi(zero(F)) === zero(F)
        @test phi(one(F)) === zero(F)
        @test phi(F(2)) === F(4) * log(F(2))
        @test phi(F(0.5)) === F(0.25) * log(F(0.5))
    end
end

@testitem "RBFInterpolant constructor and evaluation" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x = SVector{2,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
        y = SVector{1,F}[(1,), (2,), (4,), (8,)]

        rbf = RBFInterpolant(phi, x, y)
        @test rbf isa RBFInterpolant{F,1,ThinPlateSpline}

        # Exact interpolation at sample points
        @testset "point $i" for i in eachindex(x)
            @test rbf(x[i]) ≈ y[i] atol = Defaults.tol(F)
        end
    end
end

@testitem "RBFInterpolant vector-valued output" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x = SVector{2,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
        y = SVector{3,F}[(1, 2, 4), (2, 4, 8), (4, 8, 16), (8, 16, 32)]

        rbf = RBFInterpolant(phi, x, y)
        @test rbf isa RBFInterpolant{F,3,ThinPlateSpline}

        @testset "point $i" for i in eachindex(x)
            @test rbf(x[i]) ≈ y[i] atol = Defaults.tol(F)
        end
    end
end

@testitem "push! produces same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = SVector{2,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{2,F}[(1, 2), (4, 8), (16, 32)]

        x_new = SVector{2,F}(F(0.5), F(0.5))
        y_new = SVector{2,F}(3, 6)

        # Build incrementally
        rbf_push = RBFInterpolant(phi, copy(x_init), copy(y_init))
        push!(rbf_push, x_new, y_new)

        # Build from scratch with all points
        x_all = vcat(x_init, [x_new])
        y_all = vcat(y_init, [y_new])
        rbf_full = RBFInterpolant(phi, x_all, y_all)

        # Coefficients should match
        @test rbf_push.coeffs ≈ rbf_full.coeffs
        @test rbf_push.poly ≈ rbf_full.poly

        # Evaluation should match at a test point
        pt = SVector{2,F}(F(0.25), F(0.75))
        @test rbf_push(pt) ≈ rbf_full(pt)
    end
end

@testitem "append! produces same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = SVector{2,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{1,F}[(1,), (2,), (4,)]

        xs_new = SVector{2,F}[(0.5, 0.5), (1, 1)]
        ys_new = SVector{1,F}[(8,), (16,)]

        # Build incrementally
        rbf_append = RBFInterpolant(phi, copy(x_init), copy(y_init))
        append!(rbf_append, xs_new, ys_new)

        # Build from scratch with all points
        x_all = vcat(x_init, xs_new)
        y_all = vcat(y_init, ys_new)
        rbf_full = RBFInterpolant(phi, x_all, y_all)

        @test rbf_append.coeffs ≈ rbf_full.coeffs
        @test rbf_append.poly ≈ rbf_full.poly

        pt = SVector{2,F}(F(0.25), F(0.75))
        @test rbf_append(pt) ≈ rbf_full(pt)
    end
end

@testitem "push! collision detection" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y))

    # Collision with existing point
    @test_throws Exception push!(rbf, SVector{2,Float64}(0.0, 0.0), SVector{1,Float64}(99))
    @test_throws Exception push!(rbf, SVector{2,Float64}(1.0, 0.0), SVector{1,Float64}(99))
end

@testitem "append! collision detection" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y))

    # Collision with existing point
    @test_throws Exception append!(
        rbf, SVector{2,Float64}[(0.5, 0.5), (0.0, 0.0)], SVector{1,Float64}[(8,), (99,)]
    )

    # Collision among new points
    rbf2 = RBFInterpolant(phi, copy(x), copy(y))
    @test_throws Exception append!(
        rbf2, SVector{2,Float64}[(0.5, 0.5), (0.5, 0.5)], SVector{1,Float64}[(8,), (16,)]
    )
end

@testitem "interpolation convergence" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector
    using Statistics: mean

    phi = ThinPlateSpline()
    F = Float64

    # Known smooth function: f(x, y) = cos(pi*x/2) * cos(pi*y/2)
    # on [-1, 1] x [-1, 1], evaluated on a dense grid
    testfun(x, y) = cos(F(pi) * x / 2) * cos(F(pi) * y / 2)

    mean_errors = F[]
    max_errors = F[]
    eval_n = 9
    eval_points = SVector{2,F}[]
    for i in 0:(eval_n - 1), j in 0:(eval_n - 1)
        xi = F(-1) + F(2) * i / (eval_n - 1)
        yj = F(-1) + F(2) * j / (eval_n - 1)
        push!(eval_points, SVector{2,F}(xi, yj))
    end

    for n in [4, 8, 16]
        xs = SVector{2,F}[]
        ys = SVector{1,F}[]
        for i in 0:(n - 1), j in 0:(n - 1)
            xi = F(-1) + F(2) * i / (n - 1)
            yj = F(-1) + F(2) * j / (n - 1)
            push!(xs, SVector{2,F}(xi, yj))
            push!(ys, SVector{1,F}(testfun(xi, yj)))
        end
        rbf = RBFInterpolant(phi, xs, ys)
        errors = F[]
        for pt in eval_points
            exact_val = testfun(pt...)
            err = abs(rbf(pt)[1] - exact_val)
            push!(errors, err)
        end
        push!(mean_errors, mean(errors))
        push!(max_errors, maximum(errors))
    end

    # Error should decrease as sample density increases
    @test mean_errors[2] < mean_errors[1]
    @test mean_errors[3] < mean_errors[2]
    @test max_errors[2] < max_errors[1]
    @test max_errors[3] < max_errors[2]
    # Finest grid should be quite accurate on average
    @test mean_errors[3] < 1e-3
end
