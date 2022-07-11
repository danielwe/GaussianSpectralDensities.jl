using GaussianSpectralDensities
using GaussianSpectralDensities: newton, finite_difference_jacobian, NewtonException

using LinearAlgebra
using StaticArrays

@testset "GaussianSpectralDensities.jl" begin
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
