using ComplexSpectralDensities
using LinearAlgebra
using SparseArrays

const ZI = complex(1.0) * I

function diagonal_roots_of_unity(N, root, σ, τ)
    diagonal = exp(im * 2pi / root).^(0 : (N - 1))
    return MatrixEnsemble(Diagonal(diagonal), ZI, σ * ZI, fill(sqrt(τ), N))
end

function bidiagonal_roots_of_unity(N, blocksize, root, w, σ, τ)
    diagonal = exp(im * 2pi / root).^(0 : (N - 1))
    block_superdiagonal = fill(complex(w), blocksize - 1)
    superdiagonal = [
        repeat([block_superdiagonal; 0.0], (N - 1) ÷ blocksize);
        block_superdiagonal[1 : ((N - 1) % blocksize)]
    ]
    return MatrixEnsemble(
        Bidiagonal(diagonal, superdiagonal, :U), ZI, σ * ZI, fill(sqrt(τ), N)
    )
end

function jordan(N, blocksize, λ, σ, τ)
    diagonal = fill(complex(λ), N)
    block_superdiagonal = ones(ComplexF64, blocksize - 1)
    superdiagonal = [
        repeat([block_superdiagonal; 0.0], (N - 1) ÷ blocksize);
        block_superdiagonal[1 : ((N - 1) % blocksize)]
    ]
    return MatrixEnsemble(
        Bidiagonal(diagonal, superdiagonal, :U), ZI, σ * ZI, fill(sqrt(τ), N)
    )
end

function circulant_from_jordan(N, blocksize, λ, σ, τ)
    jproblem = jordan(N, blocksize, λ, σ, τ)
    M = dropzeros!(sparse(jproblem.M))
    for j in 1 : blocksize : (N - blocksize + 1)
        M[j + blocksize - 1, j] = 1.0
    end
    return MatrixEnsemble(M, jproblem.L, jproblem.R, jproblem.T)
end
