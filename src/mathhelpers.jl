module MathHelpers

using Adapt: WrappedArray
using ArnoldiMethod: LM, partialschur
using LinearAlgebra: I, Bidiagonal, Cholesky, Diagonal, Hermitian, PosDefException,
    SymTridiagonal, Tridiagonal, UniformScaling, LAPACK, cholesky, diagind, dot, eigmin,
    ldiv!, ldlt, ldlt!, mul!, tr
using LinearMaps: LinearMap
using MappedArrays: mappedarray
using SparseArrays: SparseMatrixCSC, spdiagm
using StaticArrays: SArray, SMatrix, SVector, StaticMatrix, @SVector
using SuiteSparse: SuiteSparse

export RealOrComplex
export AbstractMatrixOrUniform, DiagonalOrUniform, AnySArray, AnySMatrix
export mapdiag, alldiag, inunitdisc, phase, converteltype, sizedmatrix

@static if VERSION >= v"1.11.0-DEV.469"
    # `public ldiv rdiv` without getting parse errors in 1.10
    eval(Expr(
        :public,
        :ldiv,
        :rdiv,
        :decompose,
        :decompose!!,
        :eigmin!!,
        :invert,
        :invert!!,
        :ldiv!!,
        :muladd!!,
        :muladd_diag!!,
        :tr_normalized,
        :hsdot_normalized,
        :psdmul,
        :psdmul!!,
    ))
end

## Type utilities
## ==============

# Number type unions
const RealOrComplex{T<:Real} = Union{T,Complex{T}}

# Array/operator type unions
const AbstractMatrixOrUniform{T} = Union{AbstractMatrix{T},UniformScaling{T}}
const DiagonalOrUniform{T} = Union{<:Diagonal{T},UniformScaling{T}}

"""
    AnySArray
    AnySMatrix

Union of plain and wrapped `SArray`s
"""
const AnySArray{T,N} = Union{
    <:SArray{<:Any,T,N},
    <:WrappedArray{T,N,<:SArray,<:SArray},  # Last parameter should be SArray{<:Any,T,N} after Adapt.jl#51
    <:Hermitian{T,<:SArray{<:Any,T,N}},  # Adapt.jl#51
}
const AnySMatrix{T} = AnySArray{T,2}

# Exported helpers
mapdiag(f, A::Diagonal) = Diagonal(f.(A.diag))
mapdiag(f, A::UniformScaling) = UniformScaling(f(A.λ))

alldiag(f, A::Diagonal) = all(f, A.diag)
alldiag(f, A::UniformScaling) = f(A.λ)

inunitdisc(a::Number) = (abs2(a) <= 1)

phase(a::Number) = (a == 0) ? one(a) : a / abs(a)  # like sign except when a == 0

function converteltype(::Type{E}, A) where {E}
    (E === eltype(A)) && return A
    return convert.(E, A)
end

converteltype(::Type{E}, A::Diagonal) where {E} = Diagonal(converteltype(E, A.diag)) # Work around lack of structure-preserving broadcast with SArray storage
converteltype(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

sizedmatrix(A::UniformScaling, ::UniformScaling) = Diagonal(SVector{1}(A.λ))
sizedmatrix(A::UniformScaling, T::AnySMatrix) = Diagonal(@SVector fill(A.λ, size(T, 1)))
sizedmatrix(A::UniformScaling, T::AbstractMatrix) = A(size(T, 1))
sizedmatrix(A::AbstractMatrix, ::AbstractMatrixOrUniform) = A

# Public helpers
ldiv(L, A) = L \ A
ldiv(L::AnySMatrix, A::UniformScaling) = L \ SMatrix{size(L, 1),size(L, 1)}(A)  # should file issue
ldiv(L::Diagonal{<:Any,<:SArray}, A::UniformScaling) = L \ sizedmatrix(A, L)
ldiv(L::SparseMatrixCSC, A::UniformScaling) = L \ sizedmatrix(A, L)

rdiv(A, R) = A / R
rdiv(A::UniformScaling, R::AnySMatrix) = SMatrix{size(R, 2),size(R, 2)}(A) / R  # should file issue
rdiv(A::UniformScaling, R::Diagonal{<:Any,<:SArray}) = sizedmatrix(A, R') / R
rdiv(A::UniformScaling, R::SparseMatrixCSC) = sizedmatrix(A, R') / R

## Array unary operations
## ======================
"""
   neg(A)

Return a non-allocating representation of `-A`

Uses `-` for static arrays and `MappedArrays` otherwise, and preserves structured matrix
types.
"""
neg(A) = mappedarray(-, A)
neg(A::SArray) = -A
neg(A::Diagonal) = Diagonal(neg(A.diag))
neg(A::Bidiagonal) = Bidiagonal(neg(A.dv), neg(A.ev), Symbol(A.uplo))
neg(A::SymTridiagonal) = SymTridiagonal(neg(A.dv), neg(A.ev))
neg(A::Tridiagonal) = Tridiagonal(neg(A.dl), neg(A.d), neg(A.du))

## Array muladd
## ============
"""
    muladd!!(C, λ::Number, A, B)

Overwrite ``C`` with ``λA + B` and return ``C``
"""
function muladd!!(C, λ::Number, A, B)
    C .= muladd.(λ, A, B)
    return C
end

"""
    muladd!!(::AnySArray, λ, A, B)

Return ``λA + B``
"""
muladd!!(::AnySArray, λ::Number, A, B) = muladd(λ, A, B)

"""
    muladd_diag!!(C, λ::Number, A, B)

Overwrite the diagonal of ``C`` with `λ * diag(A) + diag(B)`, and return ``C``
"""
function muladd_diag!!(C, λ::Number, A, B)
    diagview(C) .= muladd.(λ, diagview(A), diagview(B))
    return C
end

"""
    muladd_diag!!(C::AnySMatrix, λ::Number, A::Union{Diagonal,UniformScaling}, B)

Return ``λA + B``

This is intended as an out-of-place equivalent of the in-place ``_muladd_diag!!``
implementation. This only makes sense when ``C`` and ``B`` only differ on the diagonal,
which is the case of interest in this package.
"""
muladd_diag!!(::AnySMatrix, λ::Number, A::DiagonalOrUniform, B) = muladd(λ, A, B)

# Array muladd helpers
# --------------------
"""
    diagview(A)

Return a view of the diagonal of the matrix ``A``
"""
diagview(A) = view(A, diagind(A))
diagview(A::Diagonal) = A.diag
diagview(A::Bidiagonal) = A.dv
diagview(A::SymTridiagonal) = A.dv
diagview(A::Tridiagonal) = A.d
diagview(A::UniformScaling) = A.λ

## Traces and trace inner products
## ===============================
"""
    tr_normalized(A)

Compute the normalized trace `tr(A) / size(A, 2)`
"""
tr_normalized(A) = tr(A) / size(A, 2)

"""
    hsdot_normalized(A, B)

Compute the normalized Hilbert-Schmidt inner product `tr(A'B) / size(B, 2)`
"""
hsdot_normalized(A, B) = dot(A, B) / size(B, 2)

"""
    hsdot_normalized(T::Diagonal, A, B)
    hsdot_normalized(T::UniformScaling, A, B)

Compute the normalized trace `tr(T * A'B) / size(B, 2)`

For positive definite `T` this can be seen as a generalized Hilbert-Schmidt inner product
"""
function hsdot_normalized(T::Diagonal, A, B)
    @boundscheck begin
        if (axes(A) != axes(B)) || (axes(T.diag, 1) != axes(A, 2))
            throw(ArgumentError("matrix axes don't match"))
        end
    end
    out = zero(promote_type(eltype(T), eltype(A), eltype(B)))
    @inbounds for j in axes(A, 2)
        out += T.diag[j] * dot(view(A, :, j), view(B, :, j))
    end
    return out / length(T.diag)
end

hsdot_normalized(T::UniformScaling, A, B) = T.λ * hsdot_normalized(A, B)

## Positive semidefinite matrix multiplication
## ===========================================
"""
    psdmul(A)

Compute `A * A'` and return as an appropriate type
"""
psdmul(A) = Hermitian(A * A')
psdmul(A::Diagonal) = Diagonal(abs2.(A.diag))

"""
    psdmul!!(H, A)

Compute `A * A'`, overwriting and returning `H` if possible as per `psdmul!!(H, A, α, β)`
"""
psdmul!!(H, A) = psdmul!!(H, A, true, false)

"""
    psdmul!!(H::Hermitian, A, α, β)
    psdmul!!(H::Diagonal{<:Real}, A::Diagonal, α, β)

Overwrite `H` with `α * A * A' + β * H`, return `H`
"""
function psdmul!!(H::Hermitian, A, α, β)
    mul!(H.data, A, A', α, β)
    return H
end

function psdmul!!(H::Diagonal{<:Real}, A::Diagonal, α, β)
    H.diag .= α .* abs2.(A.diag) .+ β .* H.diag
    return H
end

"""
    psdmul!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    psdmul!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)

Return `α * A * A' + β * H`
"""
function psdmul!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    return Hermitian(α * A * A' + β * H.data, H.uplo)
end

function psdmul!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)
    return Diagonal(α * abs2.(A.diag) + β * H.diag)
end

## Decomposition
## =============
"""
    decompose(A::Hermitian{E}; shift=zero(E)) where {E}

Get the Cholesky factorization of `A + shift * I` with Hermitian `A` and real `shift`
"""
function decompose(A::Hermitian{E}; shift=zero(E)) where {E}
    if (A.data isa StaticMatrix) && (A.uplo == 'L')  # Work around StaticArrays bug
        A = Hermitian(A.data', :U)  # https://github.com/JuliaArrays/StaticArrays.jl/issues/1075
    end
    iszero(shift) && cholesky(A)
    _check_shift(shift)
    return cholesky(Hermitian(A.data + convert(E, shift) * I, Symbol(A.uplo)))
end

"""
    decompose(A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0)

Get the LDLt factorization of `A + shift * I` with Hermitian `A` and real `shift`
"""
decompose(A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...) = ldlt(A; kwargs...)

"""

    decompose(A::Diagonal{E}; shift=zero(E)) where {E}

Return `A + shift * I`
"""
function decompose(A::Diagonal{E}; shift=zero(E)) where {E}
    return A + convert(E, shift) * I
end

"""
    decompose!!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    decompose!!(
        F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0
    )
    decompose!!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}

Factorize `A + shift * I` into an existing factorization instance `F`; return `F`

The factorization object `F` is updated in-place and returned. The appropriate type of `F`
is the return type of `decompose(A)`; that is, appropriate usage may look as follows:

```julia
F = decompose(A)
# Do stuff with F, which holds a factorization of A
F = decompose!(F, A; shift=1)`
# Do more stuff with F, which now holds a factorization of A + I
```

Note: Whenever an error is thrown, for example if `F::Cholesky` but `A + shift * I` is not
positive definite, the factorization object `F` may be left in an inconsistent and
unrecoverable state. This is the price of in-place, allocation-free execution. For a safe
alternative, use `decompose` instead.
"""
function decompose!!(F::Cholesky{E}, A::Hermitian{E}; shift=zero(E)) where {E}
    copyto!(F.factors, A)
    if !iszero(shift)
        _check_shift(shift)
        _shift_diag!(F.factors, convert(E, shift))
    end
    _, info = LAPACK.potrf!(F.uplo, F.factors)
    (info == 0) || throw(PosDefException(info))
    return F
end

function decompose!!(
    F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...
)
    return ldlt!(F, A; kwargs...)
end

function decompose!!(F::Diagonal{E}, A::Diagonal{E}; shift=zero(E)) where {E}
    copyto!(F.diag, A.diag)
    if shift != 0
        _shift_diag!(F, convert(E, shift))
    end
    return F
end

"""
    decompose!!(F::Cholesky{E,<:SMatrix}, A::Hermitian{E,<:SMatrix}; kwargs...) where {E}
    decompose!!(F::Diagonal{E,<:SMatrix}, A::Diagonal{E,<:SMatrix}; kwargs...) where {E}

Return `A + shift * I`

Ignores `F` and forwards the call to `decompose(A; kwargs...)`. This provides a generic
factorization updating interface that also covers immutable factorization types.
"""
function decompose!!(
    F::Cholesky{E,<:SMatrix}, A::Hermitian{E,<:SMatrix}; kwargs...
) where {E}
    return decompose(A; kwargs...)
end

function decompose!!(
    F::Diagonal{E,<:SVector}, A::Diagonal{E,<:SVector}; kwargs...
) where {E}
    return decompose(A; kwargs...)
end

# Decomposition helpers
# ---------------------
_check_shift(shift) = iszero(imag(shift)) || throw(DomainError("shift must be real"))

function _shift_diag!(A, shift)
    @inbounds for i in diagind(A)
        A[i] += shift
    end
    return nothing
end

## Divison and inversion
## =====================
"""
    ldiv!!(A, B)

Left-divide `A` into `B` in-place such that `B` is overwritten by `A \\ B`, and return B
"""
ldiv!!(A, B) = ldiv!(A, B)

"""
    ldiv!!(A, B::AnySArray)
    ldiv!!(A::SuiteSparse.CHOLMOD.Factor, B)

Return `A \\ B`

This provides a generic left division interface that also covers cases where the most
efficient or only option is out-of-place division.
"""
ldiv!!(A, B::AnySArray) = A \ B
ldiv!!(A::SuiteSparse.CHOLMOD.Factor, B) = A \ B

"""
    invert(F)
    invert(F::Cholesky)
    invert(F::SuiteSparse.CHOLMOD.Factor)

Compute the inverse of a matrix from a factorization

The factorization object `F` should equal `decompose(A)` where `A` is the matrix to invert.

This function falls back to `inv` on types for which it is not specialized.
"""
invert(F) = inv(F)
invert(F::Cholesky) = Hermitian(inv(F), Symbol(F.uplo))

function invert(F::SuiteSparse.CHOLMOD.Factor)
    id = spdiagm(ones(eltype(F), size(F, 1)))
    return Hermitian(Matrix(F \ id))
end

"""
    invert!!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    invert!!(Ai::Diagonal{E}, F::Diagonal{E})

Compute the inverse of a matrix from a factorization, filling an existing array

The factorization object `F` should equal `decompose(A)` where `A` is the matrix to invert.
The appropriate type of `Ai` is the return type of `invert(decompose(A))`. Thus, appropriate
usage may look as follows:

```julia
F = decompose(A)
Ai = invert(F)
# Do stuff with Ai, which holds the inverse of A
F = decompose!!(F, A; shift=1)`
Ai = invert!!(Ai, F)
# Do more stuff with Ai, which now holds the inverse of A + I
```
"""
function invert!!(Ai::Hermitian{E}, F::Cholesky{E}) where {E}
    (Ai.uplo == F.uplo) || throw(ArgumentError("must have Ai.uplo == F.uplo"))
    copyto!(Ai.data, F.factors)
    LAPACK.potri!(Ai.uplo, Ai.data)
    return Ai
end

function invert!!(Ai::Diagonal{E}, F::Diagonal{E}) where {E}
    Ai.diag .= inv.(F.diag)
    return Ai
end

"""
    invert!!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}

Compute the inverse of a matrix from a factorization, using an existing array as workspace

!!! note
    `Ai` will not contain the inverse at the end; it is used as a working array only. The
    inverse is the function's return value.
"""
function invert!!(Ai::Hermitian{E}, F::SuiteSparse.CHOLMOD.Factor{E}) where {E}
    copyto!(Ai.data, I)
    return Hermitian(ldiv!!(F, Ai.data), Symbol(Ai.uplo))
end

"""
    invert!!(::Hermitian{E,<:SMatrix}, F::Cholesky{E,<:SMatrix}) where {E}
    invert!!(::Diagonal{E,<:SVector}, F::Cholesky{E,<:SVector}) where {E}

Compute the inverse of `F`

The call is forwarded to `invert(F)`. This provides a inverse updating interface that also
covers immutable inverse types.
"""
invert!!(::Hermitian{E,<:SMatrix}, F::Cholesky{E,<:SMatrix}) where {E} = invert(F)
invert!!(::Diagonal{E,<:SVector}, F::Diagonal{E,<:SVector}) where {E} = invert(F)

## Smallest eigenvalue
## ==================
"""
    eigmin!!(F, A)

Find the smallest eigenvalue of `A`, possibly via a factorization into `F`

The factorization object `F` may be updated in-place or remain unused, depending on the type
of `A`. The appropriate type of `F` is the return type of `decompose(A)`.
"""
function eigmin!!(F, A::Hermitian)
    Ai = _invmap!(F, A)
    schur, history = partialschur(Ai; nev=1, which=LM(), tol=_eigmin_reltol(Ai))
    (history.nconverged >= 1) || error("smallest eigenvalue not found: did not converge")
    return inv(real(first(schur.eigenvalues)))
end

eigmin!!(_, A::Diagonal{<:Real}) = minimum(A.diag)
eigmin!!(_, A::Diagonal{<:Real,<:SArray}) = minimum(A.diag)  # Ambiguity resolution
eigmin!!(_, A::AnySMatrix) = eigmin(A)

_eigmin_reltol(::Type{T}) where {T} = (eps(real(T))^(3 // 4)) * (oneunit(real(T))^(1 // 4))
_eigmin_reltol(x) = _eigmin_reltol(eltype(x))

function _invmap!(F::Cholesky{E}, A::Hermitian{E,<:StridedMatrix}) where {E}
    decompose!!(F, A)
    return LinearMap{E}(
        (y, x) -> ldiv!(y, F, x),
        size(A, 2);
        ismutating=true,
        ishermitian=true,
        isposdef=true,
    )
end

function _invmap!(F::SuiteSparse.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC})
    decompose!!(F, A)
    return LinearMap{eltype(F)}(
        x -> F \ x, size(A, 2); ismutating=false, ishermitian=true, isposdef=true
    )
end

end
