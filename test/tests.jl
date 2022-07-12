using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt
using GaussianSpectralDensities: newton, finite_difference_jacobian, NewtonException

using LinearAlgebra
using StaticArrays

@testset "GaussianSpectralDensities.jl" begin
    @testset "ensembles" begin
        @testset "MatrixEnsemble" begin
            N = 2
            @test MatrixEnsemble(N, I, I, I, I).N == 2

            D32 = 0.5f0I(N)
            D64 = 0.5I(N)
            @testset "$args" for (args, e32, e64) in (
                ("M=D, L=I, R=I, T=I", map(D -> MatrixEnsemble(D, I, I, I), (D32, D64))...),
                ("M=I, L=D, R=I, T=I", map(D -> MatrixEnsemble(I, D, I, I), (D32, D64))...),
                ("M=I, L=I, R=D, T=I", map(D -> MatrixEnsemble(I, I, D, I), (D32, D64))...),
                ("M=I, L=I, R=I, T=D", map(D -> MatrixEnsemble(I, I, I, D), (D32, D64))...),
            )
                @test e32.N == N
                @test e64.N == N
                @test e32 isa MatrixEnsemble{Float32}
                @test e64 isa MatrixEnsemble{Float64}
            end

            @test_throws ArgumentError MatrixEnsemble(I, I, I, I)
            @test_throws DimensionMismatch MatrixEnsemble(N + 1, D64, I, I, I)
            @test_throws DimensionMismatch MatrixEnsemble(D64, I, I, 0.5I(N + 1))
            @test_throws ArgumentError MatrixEnsemble(I, I, I, Matrix(D64))
            @test_throws ArgumentError MatrixEnsemble(I, I, I, -D64)
        end
    end
    @testset "objectives" begin
        @test TraceObjective().value === DEFAULTS[:k]
        @test RegularizerObjective().value === DEFAULTS[:γ]
        @test RegularizerObjective().kind === Regularizers.γ
        @test RegularizerObjective(0.0).kind === Regularizers.γ

        k32, k64 = map(TraceObjective, (1.0f0, 1.0))
        γ32, γ64 = map(RegularizerObjective, (0.0f0, 0.0))
        δ32, δ64 = map(δ -> RegularizerObjective(δ, Regularizers.δ), (0.5f0, 0.5))

        @test k32.value === 1.0f0
        @test γ32.value === 0.0f0
        @test δ32.value === 0.5f0
        @test γ32.kind === Regularizers.γ
        @test δ32.kind === Regularizers.δ

        @test k64.value === 1.0
        @test γ64.value === 0.0
        @test δ64.value === 0.5
        @test γ64.kind === Regularizers.γ
        @test δ64.kind === Regularizers.δ

        @test convert(TraceObjective{Float32}, k64) === k32
        @test convert(RegularizerObjective{Float32}, γ64) === γ32
        @test convert(RegularizerObjective{Float32}, δ64) === δ32

        @test convert(TraceObjective{Float64}, k64) === k64
        @test convert(RegularizerObjective{Float64}, γ64) === γ64
        @test convert(RegularizerObjective{Float64}, δ64) === δ64

        @test adapt(Float32, k64) === k32
        @test adapt(Float32, γ64) === γ32
        @test adapt(Float32, δ64) === δ32

        @test adapt(Float64, k64) === k64
        @test adapt(Float64, γ64) === γ64
        @test adapt(Float64, δ64) === δ64
    end
    @testset "newton" begin
        @testset "scalar" begin
            f(x) = cos(x) - x^3
            df(x, _) = -sin(x) - 3x^2
            x0 = 0.5

            # finite_difference_jacobian
            fx0 = f(x0)
            dfx0 = df(x0, fx0)
            rtol = eps()^0.4
            @test finite_difference_jacobian(f, x0, fx0) ≈ dfx0 rtol=rtol
            @test finite_difference_jacobian(f, x0) ≈ dfx0 rtol=rtol

            # newton
            atol = sqrt(eps()) / 2
            @test norm(f(first(newton(f, x0)))) < atol
            @test norm(f(first(newton(f, df, x0)))) < atol

            @test_throws NewtonException newton(f, x0; maxiters=0)
            @test_throws NewtonException newton(f, df, x0; maxiters=0)
        end
        @testset "SVector" begin
            g(x) = (x .* (x.^2 .- 1)) + (reverse(x) ./ 2)

            function jg(x::SVector{L}, _) where {L}
                jac = fill!(similar(x, Size(L, L)), 0)
                for i in 1:L
                    jac[i, i] = 3x[i]^2 - 1
                    jac[L - i + 1, i] = 0.5
                end
                return SMatrix(jac)
            end

            x0 = SVector(0.3, -0.7)

            # finite_difference_jacobian
            gx0 = g(x0)
            jgx0 = jg(x0, gx0)
            rtol = eps()^0.4
            @test finite_difference_jacobian(g, x0, gx0) ≈ jgx0 rtol=rtol
            @test finite_difference_jacobian(g, x0) ≈ jgx0 rtol=rtol

            # newton
            atol = sqrt(eps()) / 2
            normalizednorm(y) = norm(y) / sqrt(length(y))
            @test normalizednorm(g(first(newton(g, x0)))) < atol
            @test normalizednorm(g(first(newton(g, jg, x0)))) < atol

            @test_throws NewtonException newton(g, x0; maxiters=0)
            @test_throws NewtonException newton(g, jg, x0; maxiters=0)
        end
    end
end
