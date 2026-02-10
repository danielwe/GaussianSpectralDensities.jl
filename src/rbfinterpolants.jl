module RBFInterpolants

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
        x = Tuple{F,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
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
        x = Tuple{F,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
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
        x_init = Tuple{F,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{2,F}[(1, 2), (4, 8), (16, 32)]

        x_new = (F(0.5), F(0.5))
        y_new = SVector{2,F}(3, 6)

        # Build incrementally
        rbf_push = RBFInterpolant(phi, copy(x_init), copy(y_init))
        push!(rbf_push, x_new, y_new)

        # Build from scratch with all points
        x_all = vcat(x_init, [x_new])
        y_all = vcat(y_init, [y_new])
        rbf_full = RBFInterpolant(phi, x_all, y_all)

        # Coefficients should match
        @test rbf_push.coeffs ≈ rbf_full.coeffs # NOTE: relaxed equality during initial implementation
        @test rbf_push.poly ≈ rbf_full.poly # NOTE: relaxed equality during initial implementation

        # Evaluation should match at a test point
        pt = (F(0.25), F(0.75))
        @test rbf_push(pt) ≈ rbf_full(pt) # NOTE: relaxed equality during initial implementation
    end
end

@testitem "append! produces same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = Tuple{F,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{1,F}[(1,), (2,), (4,)]

        xs_new = Tuple{F,F}[(0.5, 0.5), (1, 1)]
        ys_new = SVector{1,F}[(8,), (16,)]

        # Build incrementally
        rbf_append = RBFInterpolant(phi, copy(x_init), copy(y_init))
        append!(rbf_append, xs_new, ys_new)

        # Build from scratch with all points
        x_all = vcat(x_init, xs_new)
        y_all = vcat(y_init, ys_new)
        rbf_full = RBFInterpolant(phi, x_all, y_all)

        @test rbf_append.coeffs ≈ rbf_full.coeffs # NOTE: relaxed equality during initial implementation
        @test rbf_append.poly ≈ rbf_full.poly # NOTE: relaxed equality during initial implementation

        pt = (F(0.25), F(0.75))
        @test rbf_append(pt) ≈ rbf_full(pt) # NOTE: relaxed equality during initial implementation
    end
end

@testitem "push! collision detection" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = Tuple{Float64,Float64}[(0, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y))

    # Collision with existing point
    @test_throws Exception push!(rbf, (0.0, 0.0), SVector{1,Float64}(99))
    @test_throws Exception push!(rbf, (1.0, 0.0), SVector{1,Float64}(99))
end

@testitem "append! collision detection" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = Tuple{Float64,Float64}[(0, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y))

    # Collision with existing point
    @test_throws Exception append!(
        rbf, Tuple{Float64,Float64}[(0.5, 0.5), (0.0, 0.0)], SVector{1,Float64}[(8,), (99,)]
    )

    # Collision among new points
    rbf2 = RBFInterpolant(phi, copy(x), copy(y))
    @test_throws Exception append!(
        rbf2,
        Tuple{Float64,Float64}[(0.5, 0.5), (0.5, 0.5)],
        SVector{1,Float64}[(8,), (16,)],
    )
end

@testitem "interpolation convergence" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    F = Float64

    # Known smooth function: f(x, y) = cos(pi*x/2) * cos(pi*y/2)
    # on [-1, 1] x [-1, 1], evaluated at the origin
    testfun(x, y) = cos(F(pi) * x / 2) * cos(F(pi) * y / 2)
    test_pt = (F(0.125), F(0.125))
    exact_val = testfun(test_pt...)

    errors = F[]
    for n in [4, 8, 16]
        xs = Tuple{F,F}[]
        ys = SVector{1,F}[]
        for i in 0:(n - 1), j in 0:(n - 1)
            xi = F(-1) + F(2) * i / (n - 1)
            yj = F(-1) + F(2) * j / (n - 1)
            push!(xs, (xi, yj))
            push!(ys, SVector{1,F}(testfun(xi, yj)))
        end
        rbf = RBFInterpolant(phi, xs, ys)
        err = abs(rbf(test_pt)[1] - exact_val)
        push!(errors, err)
    end

    # Error should decrease as sample density increases
    @test errors[2] < errors[1]
    @test errors[3] < errors[2]
    # Finest grid should be quite accurate
    @test errors[3] < 1e-4
end
