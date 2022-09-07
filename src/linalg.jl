## Positive semidefinite matrix multiplication
## ===========================================
"""
    _psdmul!!(H, A)

Compute `A * A'`, overwriting and returning `H` if possible as per `_psdmul!!(H, A, α, β)`
"""
_psdmul!!(H, A) = _psdmul!!(H, A, true, false)

"""
    _psdmul!!(H::Hermitian, A, α, β)
    _psdmul!!(H::Diagonal{<:Real}, A::Diagonal, α, β)

Overwrite `H` with `α * A * A' + β * H`, return `H`
"""

function _psdmul!!(H::Hermitian, A, α, β)
    mul!(H.data, A, A', α, β)
    return H
end

function _psdmul!!(H::Diagonal{<:Real}, A::Diagonal, α, β)
    H.diag .= α .* abs2.(A.diag) .+ β .* H.diag
    return H
end

"""
    _psdmul!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    _psdmul!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)
    _psdmul!!(H::UniformScaling{<:Real}, A::UniformScaling, α, β)

Return `α * A * A' + β * H`
"""
function _psdmul!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    return Hermitian(α * A * A' + β * H.data, H.uplo)
end

function _psdmul!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)
    return Diagonal(α * abs2.(A.diag) + β * H.diag)
end

function _psdmul!!(H::UniformScaling{<:Real}, A::UniformScaling, α, β)
    return UniformScaling(α * abs2(A.λ) + β * H.λ)
end

## Factorization and inversion
## ===========================
"""
    _factorize(A::Hermitian{E}; shift=zero(E)) where {E}

Get the Cholesky factorization of `A + shift * I` with Hermitian `A` and real `shift`
"""
function _factorize(A::Hermitian{E}; shift=zero(E)) where {E}
    if (A.data isa StaticMatrix) && (A.uplo == 'L')  # Work around StaticArrays bug
        A = Hermitian(A.data', :U)  # https://github.com/JuliaArrays/StaticArrays.jl/issues/1075
    end
    iszero(shift) && cholesky(A)
    _check_shift(shift)
    return cholesky(Hermitian(A.data + convert(E, shift) * I, Symbol(A.uplo)))
end

"""
    _factorize(A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0)

Get the LDLt factorization of `A + shift * I` with Hermitian `A` and real `shift`
"""
_factorize(A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...) = ldlt(A; kwargs...)

"""

    _factorize(A::Union{<:Diagonal{E},UniformScaling{E}}; shift=zero(E)) where {E}

Return `A + shift * I`
"""
function _factorize(A::DiagonalOrUniform{E}; shift=zero(E)) where {E}
    return A + convert(E, shift) * I
end

"""
    _factorize!!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    _factorize!!(
        F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0
    )
    _factorize!!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}

Factorize `A + shift * I` into an existing factorization instance `F`; return `F`

The factorization object `F` is updated in-place and returned. The appropriate type of `F`
is the return type of `_factorize(A)`; that is, appropriate usage may look as follows:

```julia
F = _factorize(A)
# Do stuff with F, which holds a factorization of A
F = _factorize!(F, A; shift=1)`
# Do more stuff with F, which now holds a factorization of A + I
```

Note: Whenever an error is thrown, for example if `F::Cholesky` but `A + shift * I` is not
positive definite, the factorization object `F` may be left in an inconsistent and
unrecoverable state. This is the price of in-place, allocation-free execution. For a safe
alternative, use `_factorize` instead.
"""
function _factorize!!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    copyto!(F.factors, A)
    if !iszero(shift)
        _check_shift(shift)
        _shift_diag!(F.factors, convert(E, shift))
    end
    _, info = LAPACK.potrf!(F.uplo, F.factors)
    (info == 0) || throw(PosDefException(info))
    return F
end

function _factorize!!(
    F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...
)
   return ldlt!(F, A; kwargs...)
end

function _factorize!!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}
    copyto!(F.diag, A.diag)
    if shift != 0
        _shift_diag!(F, convert(E, shift))
    end
    return F
end

"""
    _factorize!!(F::UniformScaling{E}, A::UniformScaling{E}; kwargs...) where {E}
    _factorize!!(F::Cholesky{E,<:SMatrix}, A::Hermitian{E,<:SMatrix}; kwargs...) where {E}
    _factorize!!(F::Diagonal{E,<:SMatrix}, A::Diagonal{E,<:SMatrix}; kwargs...) where {E}

Return `A + shift * I`

Ignores `F` and forwards the call to `_factorize(A; kwargs...)`. This provides a generic
factorization updating interface that also covers immutable factorization types.
"""
function _factorize!!(F::UniformScaling{E}, A::UniformScaling{E}; kwargs...) where {E}
    return _factorize(A; kwargs...)
end

function _factorize!!(
    F::Cholesky{E,<:SMatrix}, A::Hermitian{E,<:SMatrix}; kwargs...
) where {E}
    return _factorize(A; kwargs...)
end

function _factorize!!(
    F::Diagonal{E,<:SVector}, A::Diagonal{E,<:SVector}; kwargs...
) where {E}
    return _factorize(A; kwargs...)
end

"""
    _ldiv!!(A, B)

Left-divide `A` into `B` in-place such that `B` is overwritten by `A \\ B`, and return B
"""
_ldiv!!(A, B) = ldiv!(A, B)

"""
    _ldiv!!(A, B::_Immutable)
    _ldiv!!(A::SuiteSparse.CHOLMOD.Factor, B)

Return `A \\ B`

This provides a generic left division interface that also covers cases where the most
efficient or only option is out-of-place division.
"""
_ldiv!!(A, B::_Immutable) = A \ B
_ldiv!!(A::SuiteSparse.CHOLMOD.Factor, B) = A \ B

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
    _inv!!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    _inv!!(Ai::Diagonal{E}, F::Diagonal{E})

Compute the inverse of a matrix from a factorization, filling an existing array

The factorization object `F` should equal `_factorize(A)` where `A` is the matrix to invert.
The appropriate type of `Ai` is the return type of `_inv(_factorize(A))`. Thus, appropriate
usage may look as follows:

```julia
F = _factorize(A)
Ai = _inv(F)
# Do stuff with Ai, which holds the inverse of A
F = _factorize!!(F, A; shift=1)`
Ai = _inv!!(Ai, F)
# Do more stuff with Ai, which now holds the inverse of A + I
```
"""
function _inv!!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    (Ai.uplo == F.uplo) || throw(ArgumentError("must have Ai.uplo == F.uplo"))
    copyto!(Ai.data, F.factors)
    LAPACK.potri!(Ai.uplo, Ai.data)
    return Ai
end

function _inv!!(Ai::Diagonal{E}, F::Diagonal{E}) where {E}
    Ai.diag .= inv.(F.diag)
    return Ai
end

"""
    _inv!!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}

Compute the inverse of a matrix from a factorization, using an existing array as workspace

!!! note
    `Ai` will not contain the inverse at the end; it is used as a working array only. The
    inverse is the function's return value.
"""
function _inv!!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}
    copyto!(Ai.data, I)
    return Hermitian(_ldiv!!(F, Ai.data), Symbol(Ai.uplo))
end

"""
    _inv!!(::UniformScaling{E}, F::UniformScaling{E}) where {E}
    _inv!!(::Hermitian{E,<:SMatrix}, F::Cholesky{E,<:SMatrix}) where {E}
    _inv!!(::Diagonal{E,<:SVector}, F::Cholesky{E,<:SVector}) where {E}

Compute the inverse of `F`

The call is forwarded to `_inv(F)`. This provides a inverse updating interface that also
covers immutable inverse types.
"""
_inv!!(::UniformScaling{E}, F::UniformScaling{E}) where {E} = _inv(F)
_inv!!(::Hermitian{E,<:SMatrix}, F::Cholesky{E,<:SMatrix}) where {E} = _inv(F)
_inv!!(::Diagonal{E,<:SVector}, F::Diagonal{E,<:SVector}) where {E} = _inv(F)

_check_shift(shift) = iszero(imag(shift)) || throw(DomainError("shift must be real"))

function _shift_diag!(A, shift)
    @inbounds for i in diagind(A)
        A[i] += shift
    end
    return nothing
end

## Smallest eigenvalue
## ==================
"""
    _eigmin!!(F, A)

Find the smallest eigenvalue of `A`, possibly via a factorization into `F`

The factorization object `F` may be updated in-place or remain unused, depending on the type
of `A`. The appropriate type of `F` is the return type of `_factorize(A)`.
"""
function _eigmin!!(F, A::Hermitian)
    schur, history = partialschur(_invmap!(F, A); nev=1, which=LM(), tol=_eigmin_reltol(A))
    (history.nconverged >= 1) || error("smallest eigenvalue not found: did not converge")
    return inv(real(first(schur.eigenvalues)))
end

_eigmin!!(F, A::Diagonal{<:Real}) = minimum(A.diag)
_eigmin!!(F, A::UniformScaling{<:Real}) = A.λ
_eigmin!!(F, A::_Immutable) = eigmin(A)

_eigmin_reltol(::Type{T}) where {T} = eps(real(T))^(2//3)
_eigmin_reltol(x) = _eigmin_reltol(eltype(x))

function _invmap!(F::Cholesky{E}, A::Hermitian{E,<:StridedMatrix}) where {E}
    _factorize!!(F, A)
    return LinearMap{E}(
        (y, x) -> ldiv!(y, F, x),
        size(A, 2);
        ismutating=true,
        ishermitian=true,
        isposdef=true,
    )
end

function _invmap!(F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC})
    _factorize!!(F, A)
    return LinearMap{eltype(F)}(
        x -> F \ x, size(A, 2); ismutating=false, ishermitian=true, isposdef=true
    )
end
