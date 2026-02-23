module Samples

using LinearAlgebra: Diagonal, UniformScaling, eigvals
using Random: AbstractRNG, default_rng

using ..ArrayTools: DiagonalOrUniform
using ..Ensembles: MatrixEnsemble

public expand, partially_hermitize, sample_matrix, sample_eigenvalues

function expand(e::MatrixEnsemble, ncopies::Int)
    ncopies >= 1 || throw(ArgumentError("ncopies must be positive"))
    n = size(e, 1)
    N = n * ncopies

    function expand_field(A)
        A isa UniformScaling && return A
        if A isa Diagonal
            return Diagonal(repeat(parent(A), ncopies))
        end
        result = zeros(eltype(A), N, N)
        for k in 1:ncopies
            rows = (k - 1) * n .+ (1:n)
            result[rows, rows] = A
        end
        return result
    end

    return (expand_field(e.M), expand_field(e.L), expand_field(e.R), expand_field(e.S))
end

function partially_hermitize(X, S::DiagonalOrUniform)
    if S isa UniformScaling
        s = S.λ
        phi = asin(s * s) / 2
        cos_phi = cos(phi)
        sin_phi = sin(phi)
        return X .* cos_phi .+ adjoint(X) .* sin_phi
    end

    s = parent(S)
    result = similar(X)
    for j in axes(X, 2), i in axes(X, 1)
        phi = asin(s[i] * s[j]) / 2
        result[i, j] = X[i, j] * cos(phi) + conj(X[j, i]) * sin(phi)
    end
    return result
end

function sample_matrix(
    e::MatrixEnsemble{F}, N::Int; rng::AbstractRNG=default_rng()
) where {F}
    n = size(e, 1)
    if N < 1 || N % n != 0
        throw(ArgumentError("N must be a positive multiple of the ensemble size"))
    end
    ncopies = N ÷ n

    M_N, L_N, R_N, S_N = expand(e, ncopies)
    X = randn(rng, Complex{F}, N, N) / sqrt(F(N))
    J = partially_hermitize(X, S_N)
    A = M_N + L_N * J * R_N
    return A
end

function sample_eigenvalues(
    e::MatrixEnsemble{F}, N::Int; rng::AbstractRNG=default_rng()
) where {F}
    return eigvals(sample_matrix(e, N; rng))
end

end  # module Samples

@testitem "expand: UniformScaling passes through" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I, UniformScaling

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, I / 2)
        M_N, L_N, R_N, S_N = expand(e, 5)
        @test M_N isa UniformScaling
        @test L_N isa UniformScaling
        @test R_N isa UniformScaling
        @test S_N isa UniformScaling
        @test M_N === e.M
        @test L_N === e.L
        @test R_N === e.R
        @test S_N === e.S
    end
end

@testitem "expand: AbstractMatrix produces block-diagonal" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I, UniformScaling

    @testset "$F" for F in [Float32, Float64]
        A = Complex{F}[1 2; 3 4]
        L = Complex{F}[2 1; -1 3]
        R = Complex{F}[1 0; 0 2]
        e = MatrixEnsemble{F}(A, L, R, I / 2)
        ncopies = 3
        M_N, L_N, R_N, S_N = expand(e, ncopies)

        n = size(e, 1)
        N = n * ncopies

        @test size(M_N) == (N, N)
        @test size(L_N) == (N, N)
        @test size(R_N) == (N, N)
        @test S_N isa UniformScaling

        @testset "block $k" for k in 1:ncopies
            rows = (k - 1) * n .+ (1:n)
            @test M_N[rows, rows] == e.M
            @test L_N[rows, rows] == e.L
            @test R_N[rows, rows] == e.R
        end

        @testset "off-diagonal block ($j, $k)" for k in 1:ncopies, j in 1:ncopies
            j == k && continue
            rows_j = (j - 1) * n .+ (1:n)
            rows_k = (k - 1) * n .+ (1:n)
            @test all(iszero, M_N[rows_j, rows_k])
            @test all(iszero, L_N[rows_j, rows_k])
            @test all(iszero, R_N[rows_j, rows_k])
        end
    end
end

@testitem "expand: Diagonal produces block-diagonal Diagonal" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I

    @testset "$F" for F in [Float32, Float64]
        d = F[0.25, 0.75]
        e = MatrixEnsemble{F}(I, I, I, Diagonal(d))
        ncopies = 4
        M_N, L_N, R_N, S_N = expand(e, ncopies)

        @test S_N isa Diagonal
        @test size(S_N) == (length(d) * ncopies, length(d) * ncopies)
        @test parent(S_N) == repeat(parent(e.S), ncopies)
    end
end

@testitem "expand: mixed matrix types" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I, UniformScaling

    @testset "$F" for F in [Float32, Float64]
        M = Complex{F}[1 0; 0 -1]
        L = Complex{F}[1 1; 0 1]
        e = MatrixEnsemble{F}(M, L, I, Diagonal(F[0.3, 0.7]))
        ncopies = 2
        M_N, L_N, R_N, S_N = expand(e, ncopies)

        n = size(e, 1)
        N = n * ncopies

        @test size(M_N) == (N, N)
        @test size(L_N) == (N, N)
        @test R_N isa UniformScaling
        @test S_N isa Diagonal
        @test size(S_N) == (N, N)
    end
end

@testitem "expand: ncopies = 1 returns single-block result" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I

    A = ComplexF64[1 2; 3 4]
    e = MatrixEnsemble(A, I, I, Diagonal([0.5, 0.5]))
    M_N, L_N, R_N, S_N = expand(e, 1)

    @test M_N == e.M
    @test L_N == e.L
    @test R_N == e.R
    @test parent(S_N) == parent(e.S)
end

@testitem "expand: errors on ncopies < 1" begin
    using GaussianSpectralDensities.Samples: expand
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I

    e = MatrixEnsemble(I, I, I, I / 2)
    @test_throws ArgumentError expand(e, 0)
    @test_throws ArgumentError expand(e, -1)
end

@testitem "partially_hermitize: S = 0 returns X unchanged" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: I

    @testset "$F" for F in [Float32, Float64]
        X = Complex{F}[1 2 3; 4 5 6; 7 8 9]
        result = partially_hermitize(X, F(0) * I)
        @test result ≈ X
    end
end

@testitem "partially_hermitize: S = I returns Hermitian matrix" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: I

    @testset "$F" for F in [Float32, Float64]
        X = Complex{F}[1 2; 3 4]
        result = partially_hermitize(X, one(F) * I)
        @test result ≈ result'
    end
end

@testitem "partially_hermitize: correct size and element type" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: Diagonal, I

    @testset "$F" for F in [Float32, Float64]
        N = 5
        X = randn(Complex{F}, N, N)

        @testset "scalar S" begin
            result = partially_hermitize(X, F(0.5) * I)
            @test size(result) == (N, N)
            @test eltype(result) === Complex{F}
        end

        @testset "diagonal S" begin
            S = Diagonal(F[0.1, 0.2, 0.3, 0.4, 0.5])
            result = partially_hermitize(X, S)
            @test size(result) == (N, N)
            @test eltype(result) === Complex{F}
        end
    end
end

@testitem "partially_hermitize: scalar vs diagonal S consistency" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: Diagonal, I

    @testset "$F" for F in [Float32, Float64]
        N = 4
        X = randn(Complex{F}, N, N)
        s = F(0.6)
        result_scalar = partially_hermitize(X, s * I)
        result_diag = partially_hermitize(X, Diagonal(fill(s, N)))
        @test result_scalar ≈ result_diag
    end
end

@testitem "partially_hermitize: real-valued X" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: I

    @testset "$F" for F in [Float32, Float64]
        X = F[1 2; 3 4]
        result = partially_hermitize(X, F(0.5) * I)
        @test size(result) == (2, 2)
    end
end

@testitem "partially_hermitize: diagonal formula (4x4)" begin
    using GaussianSpectralDensities.Samples: partially_hermitize

    using LinearAlgebra: Diagonal

    @testset "$F" for F in [Float32, Float64]
        X = Complex{F}[
            1 2 3 4;
            5 6 7 8;
            9 10 11 12;
            13 14 15 16
        ]
        s = F[0.2, 0.4, 0.6, 0.8]
        S = Diagonal(s)

        result = partially_hermitize(X, S)

        expected = similar(X)
        for j in axes(X, 2), i in axes(X, 1)
            phi = asin(s[i] * s[j]) / 2
            expected[i, j] = X[i, j] * cos(phi) + X[j, i] * sin(phi)
        end

        @test result ≈ expected
    end
end

@testitem "sample_matrix: return type and size" begin
    using GaussianSpectralDensities.Samples: sample_matrix
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I
    using Random: default_rng, seed!

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, F(0) * I)
        N = 10
        A = sample_matrix(e, N; rng=seed!(default_rng(), 42))
        @test A isa Matrix{Complex{F}}
        @test size(A) == (N, N)
    end
end

@testitem "sample_matrix: errors on invalid N" begin
    using GaussianSpectralDensities.Samples: sample_matrix
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I

    e_scalar = MatrixEnsemble(I, I, I, 0.5I)

    @test_throws ArgumentError sample_matrix(e_scalar, 0)
    @test_throws ArgumentError sample_matrix(e_scalar, -1)

    A = ComplexF64[1 0; 0 1]
    e_matrix = MatrixEnsemble(A, I, I, 0.5I)
    @test_throws ArgumentError sample_matrix(e_matrix, 3)
end

@testitem "sample_matrix: deterministic with seeded rng" begin
    using GaussianSpectralDensities.Samples: sample_matrix
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I
    using Random: default_rng, seed!

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, F(0) * I)
        N = 10

        A1 = sample_matrix(e, N; rng=seed!(default_rng(), 123))
        A2 = sample_matrix(e, N; rng=seed!(default_rng(), 123))
        @test A1 == A2

        A3 = sample_matrix(e, N; rng=seed!(default_rng(), 456))
        @test A1 != A3
    end
end

@testitem "sample_matrix: S = I produces Hermitian J" begin
    using GaussianSpectralDensities.Samples: sample_matrix
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I
    using Random: default_rng, seed!

    e = MatrixEnsemble(0.0I, I, I, 1.0I)
    A = sample_matrix(e, 20; rng=seed!(default_rng(), 42))
    # With M=0, L=R=I, S=I: A = J, and J should be Hermitian
    @test A ≈ A'
end

@testitem "sample_eigenvalues: consistent with sample_matrix" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues, sample_matrix
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I, eigvals
    using Random: default_rng, seed!

    e = MatrixEnsemble(0.0I, I, I, 0.5I)
    N = 10

    A = sample_matrix(e, N; rng=seed!(default_rng(), 99))
    eigs = sample_eigenvalues(e, N; rng=seed!(default_rng(), 99))
    @test eigs ≈ eigvals(A)
end

@testitem "sample_eigenvalues: return type and length" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I
    using Random: default_rng, seed!

    rng = seed!(default_rng(), 42)

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, F(0) * I)
        n = size(e, 1)
        N = 10 * n
        eigs = sample_eigenvalues(e, N; rng=rng)
        @test eigs isa Vector{Complex{F}}
        @test length(eigs) == N
    end
end

@testitem "sample_eigenvalues: errors on invalid N" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I

    e_scalar = MatrixEnsemble(I, I, I, 0.5I)

    @test_throws ArgumentError sample_eigenvalues(e_scalar, 0)
    @test_throws ArgumentError sample_eigenvalues(e_scalar, -1)

    A = ComplexF64[1 0; 0 1]
    e_matrix = MatrixEnsemble(A, I, I, 0.5I)
    n = size(e_matrix, 1)
    @test n == 2

    @test_throws ArgumentError sample_eigenvalues(e_matrix, 3)
    @test_throws ArgumentError sample_eigenvalues(e_matrix, 5)
end

@testitem "sample_eigenvalues: deterministic with seeded rng" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I
    using Random: default_rng, seed!

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, F(0) * I)
        N = 10

        eigs1 = sample_eigenvalues(e, N; rng=seed!(default_rng(), 123))
        eigs2 = sample_eigenvalues(e, N; rng=seed!(default_rng(), 123))
        @test eigs1 == eigs2

        eigs3 = sample_eigenvalues(e, N; rng=seed!(default_rng(), 456))
        @test eigs1 != eigs3
    end
end

@testitem "sample_eigenvalues: works with matrix-valued ensemble" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I
    using Random: default_rng, seed!

    rng = seed!(default_rng(), 99)

    @testset "$F" for F in [Float32, Float64]
        M = Complex{F}[1 0; 0 -1]
        L = Complex{F}[2 0; 0 1]
        S = Diagonal(F[0.3, 0.7])
        e = MatrixEnsemble{F}(M, L, I, S)
        n = size(e, 1)
        N = 5 * n
        eigs = sample_eigenvalues(e, N; rng=rng)
        @test eigs isa Vector{Complex{F}}
        @test length(eigs) == N
    end
end

@testitem "sample_eigenvalues: works with all-matrix ensemble" begin
    using GaussianSpectralDensities.Samples: sample_eigenvalues
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal
    using Random: default_rng, seed!

    rng = seed!(default_rng(), 77)

    @testset "$F" for F in [Float32, Float64]
        M = Complex{F}[0 1; -1 0]
        L = Complex{F}[1 0; 0 2]
        R = Complex{F}[2 0; 0 1]
        S = Diagonal(F[0.5, 0.5])
        e = MatrixEnsemble{F}(M, L, R, S)
        n = size(e, 1)
        N = 4 * n
        eigs = sample_eigenvalues(e, N; rng=rng)
        @test eigs isa Vector{Complex{F}}
        @test length(eigs) == N
    end
end

@testitem "sample_eigenvalues: Ginibre ensemble eigenvalues near unit circle" begin
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.Samples: sample_eigenvalues

    using LinearAlgebra: I
    using Random: default_rng, seed!

    rng = seed!(default_rng(), 2024)
    e = MatrixEnsemble(0.0I, I, I, 0.0I)
    N = 200
    eigs = sample_eigenvalues(e, N; rng=rng)

    radii = abs.(eigs)
    @test maximum(radii) < 2.0
    @test count(r -> r <= 1.0, radii) > N ÷ 2
    mean_radius = sum(radii) / N
    @test 0.3 < mean_radius < 1.0
end

@testitem "sample_eigenvalues: elliptic ensemble elongation" begin
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.Samples: sample_eigenvalues

    using LinearAlgebra: I
    using Random: default_rng, seed!
    using Statistics: std

    rng_ginibre = seed!(default_rng(), 1000)
    rng_elliptic = seed!(default_rng(), 1000)

    e_ginibre = MatrixEnsemble(0.0I, I, I, 0.0I)
    e_elliptic = MatrixEnsemble(0.0I, I, I, 0.8I)

    N = 500
    eigs_ginibre = sample_eigenvalues(e_ginibre, N; rng=rng_ginibre)
    eigs_elliptic = sample_eigenvalues(e_elliptic, N; rng=rng_elliptic)

    real_std_ginibre = std(real.(eigs_ginibre))
    imag_std_ginibre = std(imag.(eigs_ginibre))
    ratio_ginibre = real_std_ginibre / imag_std_ginibre

    real_std_elliptic = std(real.(eigs_elliptic))
    imag_std_elliptic = std(imag.(eigs_elliptic))
    ratio_elliptic = real_std_elliptic / imag_std_elliptic

    @test ratio_ginibre < ratio_elliptic
    @test ratio_elliptic > 1.5
end

@testitem "sample_eigenvalues: N = n (single copy)" begin
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.Samples: sample_eigenvalues

    using LinearAlgebra: I
    using Random: default_rng, seed!

    @testset "$F" for F in [Float32, Float64]
        e = MatrixEnsemble{F}(I, I, I, F(0) * I)
        n = size(e, 1)
        eigs = sample_eigenvalues(e, n; rng=seed!(default_rng(), 42))
        @test eigs isa Vector{Complex{F}}
        @test length(eigs) == n
    end
end

@testitem "sample_eigenvalues: default rng" begin
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.Samples: sample_eigenvalues

    using LinearAlgebra: I

    e = MatrixEnsemble(I, I, I, 0.0I)
    eigs = sample_eigenvalues(e, 10)
    @test eigs isa Vector{ComplexF64}
    @test length(eigs) == 10

    e32 = MatrixEnsemble{Float32}(I, I, I, 0.0f0 * I)
    eigs32 = sample_eigenvalues(e32, 10)
    @test eigs32 isa Vector{ComplexF32}
    @test length(eigs32) == 10
end
