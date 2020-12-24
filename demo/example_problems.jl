using GaussianSpectralDensities
using LinearAlgebra

function diagonal_roots_of_unity(N, root, σ, T)
    V = real(eltype(T))
    diagonal = exp(im * V(2pi / root)).^(0 : (N - 1))
    return MatrixEnsemble(Diagonal(diagonal), one(V) * I, σ * I, T)
end

function bidiagonal_roots_of_unity(N, root, w::U, σ, T, blocksize=nothing) where {U}
    if blocksize === nothing
        blocksize = N
    end
    V = real(eltype(T))
    subdiagonal = zeros(U, N - 1)
    diagonal = exp(im * U(2pi / root)).^(0 : (N - 1))
    superdiagonal = fill(w, blocksize - 1)
    if blocksize !== N
        superdiagonal = [
            repeat([superdiagonal; 0], (N - 1) ÷ blocksize);
            superdiagonal[1 : ((N - 1) % blocksize)]
        ]
    end
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(V) * I, σ * I, T
    )
end

function jordan(N, λ::U, σ, T, blocksize=nothing) where {U}
    if blocksize === nothing
        blocksize = N
    end
    V = real(eltype(T))
    subdiagonal = zeros(U, N - 1)
    diagonal = fill(λ, N)
    superdiagonal = ones(U, blocksize - 1)
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(V) * I, σ * I, T
    )
end

function circulant_from_jordan(N, λ, σ, T, blocksize=nothing)
    if blocksize === nothing
        blocksize = N
    end
    jproblem = jordan(N, λ, σ, T, blocksize)
    M = Array(jproblem.M)
    for j in 1 : blocksize : (N - blocksize + 1)
        M[j + blocksize - 1, j] = 1
    end
    return MatrixEnsemble(M, jproblem.L, jproblem.R, jproblem.T)
end

function annihilation(N, σ, T, blocksize=nothing)
    if blocksize === nothing
        blocksize = N
    end
    U = real(eltype(T))
    subdiagonal = zeros(U, N - 1)
    diagonal = zeros(U, N)
    superdiagonal = U.(sqrt.(1 : (blocksize - 1)) / sqrt(blocksize))
    if blocksize !== N
        superdiagonal = [
            repeat([superdiagonal; 0], (N - 1) ÷ blocksize);
            superdiagonal[1 : ((N - 1) % blocksize)]
        ]
    end
    return MatrixEnsemble(
        Tridiagonal(subdiagonal, diagonal, superdiagonal), one(U) * I, σ * I, T
    )
end

function toeplitz_from_annihilation(N, σ, T, blocksize=nothing)
    if blocksize === nothing
        blocksize = N
    end
    aproblem = annihilation(N, σ, T, blocksize)
    M = Array(aproblem.M)
    for j in 1 : blocksize : (N - blocksize + 1)
        M[j + blocksize - 1, j] = 1
    end
    return MatrixEnsemble(M, aproblem.L, aproblem.R, aproblem.T)
end
