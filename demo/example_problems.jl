using GaussianSpectralDensities
using LinearAlgebra

function diagonal_roots_of_unity(N, root, σ, τ::T) where {T}
    diagonal = exp(im * T(2pi / root)).^(0 : (N - 1))
    return MatrixEnsemble(Diagonal(diagonal), one(T) * I, σ * I, τ)
end

function bidiagonal_roots_of_unity(N, root, w::S, σ, τ::T, blocksize=nothing) where {S,T}
    if blocksize === nothing
        blocksize = N
    end
    subdiagonal = zeros(S, N - 1)
    diagonal = exp(im * S(2pi / root)).^(0 : (N - 1))
    superdiagonal = fill(w, blocksize - 1)
    if blocksize !== N
        superdiagonal = [
            repeat([superdiagonal; 0], (N - 1) ÷ blocksize);
            superdiagonal[1 : ((N - 1) % blocksize)]
        ]
    end
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(T) * I, σ * I, τ
    )
end

function jordan(N, λ, σ, τ::T, blocksize=nothing) where {T}
    if blocksize === nothing
        blocksize = N
    end
    subdiagonal = zeros(typeof(λ), N - 1)
    diagonal = fill(λ, N)
    superdiagonal = ones(typeof(λ), blocksize - 1)
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(real(T)) * I, σ * I, τ
    )
end

function circulant_from_jordan(N, λ, σ, τ, blocksize=nothing)
    if blocksize === nothing
        blocksize = N
    end
    jproblem = jordan(N, λ, σ, τ, blocksize)
    M = Array(jproblem.M)
    for j in 1 : blocksize : (N - blocksize + 1)
        M[j + blocksize - 1, j] = 1
    end
    return MatrixEnsemble(M, jproblem.L, jproblem.R, jproblem.T)
end

function annihilation(N, σ, τ::T, blocksize=nothing) where {T}
    if blocksize === nothing
        blocksize = N
    end
    subdiagonal = zeros(T, N - 1)
    diagonal = zeros(T, N)
    superdiagonal = T.(sqrt.(1 : (blocksize - 1)) / sqrt(blocksize))
    if blocksize !== N
        superdiagonal = [
            repeat([superdiagonal; 0], (N - 1) ÷ blocksize);
            superdiagonal[1 : ((N - 1) % blocksize)]
        ]
    end
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(T) * I, σ * I, τ
    )
end

function toeplitz_from_annihilation(N, σ, τ, blocksize=nothing)
    if blocksize === nothing
        blocksize = N
    end
    aproblem = annihilation(N, σ, τ, blocksize)
    M = Array(aproblem.M)
    for j in 1 : blocksize : (N - blocksize + 1)
        M[j + blocksize - 1, j] = 1
    end
    return MatrixEnsemble(M, aproblem.L, aproblem.R, aproblem.T)
end
