"""

    _factorize(A::Hermitian{E}; shift=zero(E)) where {E}
    _factorize(A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0)

Find the Cholesky factorization of `A + shift * I` with Hermitian `A` and real `shift`
"""

function _factorize(A::Hermitian{E}; shift=zero(E)) where {E}
    iszero(shift) && cholesky(A)
    _check_shift(shift)
    return cholesky(A + convert(E, shift) * I)
end

_factorize(A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...) = cholesky(A; kwargs...)

"""

    _factorize(A::Union{<:Diagonal{E},UniformScaling{E}}; shift=zero(E)) where {E}

Return `A + shift * I`
"""
function _factorize(A::DiagonalOrUniform{E}; shift=zero(E)) where {E}
    return A + convert(E, shift) * I
end

"""

    _factorize!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    _factorize(
        F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0
    )
    _factorize!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}

Factorize `A + shift * I` into an existing factorization instance

The factorization object `F` is updated in-place and returned. The appropriate type of `F`
is the return type of `_factorize(A)`; that is, appropriate usage may look as follows:

Note: Whenever an error is thrown, for example if `F::Cholesky` but `A + shift * I` is not
positive definite, the factorization object `F` may have been left in an inconsistent and
unrecoverable state. This is the price of in-place, allocation-free execution. For a safe
alternative, use `_factorize` instead.

```julia
F = _factorize(A)
# Do stuff with F, which holds a factorization of A
F = _factorize!(F, A; shift=1)`
# Do more stuff with F, which now holds a factorization of A + I
```
"""
function _factorize!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    copyto!(F.factors, A)
    if !iszero(shift)
        _check_shift(shift)
        _shift_diag!(F.factors, convert(E, shift))
    end
    _, info = LAPACK.potrf!(F.uplo, F.factors)
    (info == 0) || throw(PosDefException(info))
    return F
end

function _factorize!(
    F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...
)
   cholesky!(F, A; kwargs...)
   return F
end

function _factorize!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}
    copyto!(F.diag, A.diag)
    if shift != 0
        _shift_diag!(F, convert(E, shift))
    end
    return F
end

"""

    _factorize!(F::UniformScaling{E}, A::UniformScaling{E}; kwargs...) where {E}

Return `A + shift * I`

Ignores `F` and forwards the call to `_factorize(A; kwargs...)`. This provides a generic
factorization updating API that also covers immutable factorization types.
"""
function _factorize!(F::UniformScaling{E}, A::UniformScaling{E}; kwargs...) where {E}
    return _factorize(A; kwargs...)
end

"""

    _ldiv!(A, B)
    _ldiv!(A::SuiteSparse.CHOLMOD.Factor, B)

Left divide `A` into `B` in-place such that `B` is overwritten by `A \\ B`

Provides a custom implementation for `A::SuiteSparse.CHOLMOD.Factor`, falls back to
LinearAlgebra.ldiv! otherwise.
"""
_ldiv!(A, B) = ldiv!(A, B)

function _ldiv!(A::SuiteSparse.CHOLMOD.Factor, B)
    L = LowerTriangular(sparse(A.L))
    U = L'
    p = A.p
    for c in eachcol(B)
        invpermute!(ldiv!(U, ldiv!(L, permute!(c, p))), p)
        # c[p] .= ldiv!(U, ldiv!(L, c[p]))  # Out-of-place permutation may perform better
    end
end

"""

    _inv(F)
    _inv(F::Cholesky)
    _inv(F::SuiteSparse.CHOLMOD.Factor)

Compute the inverse of a matrix from a factorization

The factorization object `F` should equal `_factorize(A)` where `A` is the matrix to invert.

This function falls back to `inv` on types for which it is not specialized.
"""
_inv(F) = inv(F)
_inv(F::Cholesky) = Hermitian(inv(F), Symbol(F.uplo))

function _inv(F::SuiteSparse.CHOLMOD.Factor)
    id = spdiagm(ones(eltype(F), size(F, 1)))
    return Hermitian(Matrix(F \ id))
end

"""

    _inv!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    _inv!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}
    _inv!(Ai::Diagonal{E}, F::Diagonal{E})

Compute the inverse of a matrix from a factorization, filling an existing array

The factorization object `F` should equal `_factorize(A)` where `A` is the matrix to invert.
The appropriate type of `Ai` is the return type of `_inv(_factorize(A))`. Thus, appropriate
usage may look as follows:

```julia
F = _factorize(A)
Ai = _inv(F)
# Do stuff with Ai, which holds the inverse of A
F = _factorize!(F, A; shift=1)`
Ai = _inv!(Ai, F)
# Do more stuff with Ai, which now holds the inverse of A + I
```
"""
function _inv!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    (Ai.uplo == F.uplo) || throw(ArgumentError("must have Ai.uplo == F.uplo"))
    copyto!(Ai.data, F.factors)
    LAPACK.potri!(Ai.uplo, Ai.data)
    return Ai
end

function _inv!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}
    copyto!(Ai.data, I)
    _ldiv!(F, Ai.data)
    return Ai
end

function _inv!(Ai::Diagonal{E}, F::Diagonal{E}) where {E}
    Ai.diag .= inv.(F.diag)
    return Ai
end

"""

    _inv!(Ai::UniformScaling{E}, F::UniformScaling{E}) where {E}

Compute the inverse of `F`

Ignores `Ai` and forwards the call to `_inv(F)`. This provides a inverse updating updating
API that also covers immutable inverse types.
"""
_inv!(Ai::UniformScaling{E}, F::UniformScaling{E}) where {E} = _inv(F)

_check_shift(shift) = (!iszero(imag(shift)) && throw(DomainError("shift must be real")))

function _shift_diag!(A, shift)
    @inbounds for i in diagind(A)
        A[i] += shift
    end
    return nothing
end
