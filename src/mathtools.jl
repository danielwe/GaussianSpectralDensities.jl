module MathTools

# TODO: unit tests?

using ArnoldiMethod: ArnoldiMethod, partialschur
using LinearAlgebra:
    Bidiagonal,
    Cholesky,
    Diagonal,
    Hermitian,
    I,
    LAPACK,
    LinearAlgebra,
    PosDefException,
    SymTridiagonal,
    Tridiagonal,
    UniformScaling,
    cholesky,
    diagind,
    dot,
    eigmin,
    ldiv!,
    ldlt,
    ldlt!,
    mul!,
    tr
using LinearMaps: LinearMap
using MappedArrays: mappedarray
using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm
using StaticArrays: SArray, SMatrix, SVector, StaticMatrix

using ..ArrayTools: AnySArray, AnySMatrix, DiagonalOrUniform, map_diagonal, withsize

public decompose,
    decompose!!,
    eigmin!!,
    frobenius_normalized,
    herk,
    herk!!,
    inunitdisc,
    inverse,
    inverse!!,
    isapproxlinear,
    ldiv,
    ldiv!!,
    minus,
    monotonic,
    muladd!!,
    muladd_diagonal!!,
    phase,
    predictlinear,
    rdiv,
    samesign,
    tr_normalized

## Scalar predicates
## =================
inunitdisc(a::Number) = (abs2(a) <= 1)

phase(a::Number) = (a == 0) ? one(a) : a / abs(a)  # like sign except when a == 0

samesign(x::Number, y::Number) = (x * y > 0)  # WARN: is this good enough? (Inf/NaN etc.)

## Linear interpolation/extrapolation
## ==================================
function predictlinear(x, (x1, y1), (x2, y2))
    if x1 == x2
        # fall back to piecewise constant predictor; perhaps non-standard, definitely useful
        if x < x1
            return y1
        elseif x == x1
            return (y1 + y2) / 2
        else
            return y2
        end
    end
    return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)
end

function isapproxlinear((x, y), (x1, y1), (x2, y2); kwargs...)
    ret = isapprox((y2 - y) * (x1 - x), (y1 - y) * (x2 - x); kwargs...)
    ret || @error "" xs=(x, x1, x2) ys=(y, y1, y2)
    return ret
end

## Iterable monotonicity
## =====================
function monotonic(x)
    isempty(x) && return true
    x1, tail... = x
    return monotonic_inner(x1, tail)
end

# monotonic helper
function monotonic_inner(x1, tail)
    isempty(tail) && return true
    x2, tail2... = tail
    return (x1 < x2) && monotonic_inner(x2, tail2)
end

## Mixed-type matrix division
## ==========================
ldiv(L, A) = L \ A
ldiv(L::AnySMatrix, A::UniformScaling) = L \ SMatrix{size(L, 1),size(L, 1)}(A)  # should file issue
ldiv(L::Diagonal{<:Any,<:SArray}, A::UniformScaling) = L \ withsize(A, L)
ldiv(L::SparseMatrixCSC, A::UniformScaling) = L \ withsize(A, L)

rdiv(A, R) = A / R
rdiv(A::UniformScaling, R::AnySMatrix) = SMatrix{size(R, 2),size(R, 2)}(A) / R  # should file issue
rdiv(A::UniformScaling, R::Diagonal{<:Any,<:SArray}) = withsize(A, R') / R
rdiv(A::UniformScaling, R::SparseMatrixCSC) = withsize(A, R') / R

## Uplo helper
## ===========
@static if :uplo in names(LinearAlgebra)
    using LinearAlgebra: uplo
else
    uplo(A) = LinearAlgebra.sym_uplo(A.uplo)
end

## Array unary minus
## =================
"""
minus(A)

Return a non-allocating representation of `-A`

Uses `-` for static arrays,  `MappedArrays` otherwise. Preserves structured matrix wrappers.
"""
minus(A) = mappedarray(-, A)
minus(A::SArray) = -A
minus(A::Diagonal) = Diagonal(minus(parent(A)))
minus(A::Bidiagonal) = Bidiagonal(minus(A.dv), minus(A.ev), uplo(A))
minus(A::SymTridiagonal) = SymTridiagonal(minus(A.dv), minus(A.ev))
minus(A::Tridiagonal) = Tridiagonal(minus(A.dl), minus(A.d), minus(A.du))

## Bangbang muladd
## ===============
"""
    muladd!!(C, λ::Number, A, B)

Overwrite ``C`` with ``λA + B`` and return ``C``
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
    muladd_diagonal!!(C, λ::Number, A, B)

Overwrite the diagonal of ``C`` with `λ * diag(A) + diag(B)`, and return ``C``
"""
function muladd_diagonal!!(C, λ::Number, A, B)
    view_diagonal(C) .= muladd.(λ, view_diagonal(A), view_diagonal(B))
    return C
end

"""
    muladd_diagonal!!(C::AnySMatrix, λ::Number, A::Union{Diagonal,UniformScaling}, B)

Return ``λA + B``

This is intended as an out-of-place equivalent of the in-place ``_muladd_diag!!``
implementation. This only makes sense when ``C`` and ``B`` only differ on the diagonal,
which is the case of interest in this package.
"""
muladd_diagonal!!(::AnySMatrix, λ::Number, A::DiagonalOrUniform, B) = muladd(λ, A, B)

# muladd helpers
"""
    view_diagonal(A)

Return a view of the diagonal of the matrix ``A``
"""
view_diagonal(A) = view(A, diagind(A))
view_diagonal(A::Diagonal) = parent(A)
view_diagonal(A::Bidiagonal) = A.dv
view_diagonal(A::SymTridiagonal) = A.dv
view_diagonal(A::Tridiagonal) = A.d
view_diagonal(A::UniformScaling) = A.λ

## Traces and trace inner products
## ===============================
"""
    tr_normalized(A)

Compute the normalized trace `tr(A) / size(A, 1)`
"""
tr_normalized(A) = tr(A) / size(A, 1)

"""
    frobenius_normalized(A, B)

Compute the normalized Frobenius inner product `tr(A'B) / size(A, 1)`

The normalization is with respect to the dimension of the codomain of the matrices, i.e.,
the dimension contracted over (not the one traced over) when the inner product is written as
above. Of course, this distinction only matters if the matrices are regular, so it's purely
academic in the context of this package unless we add support for rectangular L and R in
[`MatrixEnsemble`](@ref).
"""
frobenius_normalized(A, B) = dot(A, B) / size(A, 1)

"""
    frobenius_normalized(S::Diagonal, A, B)
    frobenius_normalized(S::UniformScaling, A, B)

Compute the normalized trace `tr(S * A'B) / size(A, 1)`

This is a Hermitian form represented by the superoperator ``M ↦ MS``, and forms a weighted
Frobenius inner product when ``S`` is positive definite.
"""
function frobenius_normalized(S::Diagonal, A, B)
    @boundscheck begin
        if (axes(A) != axes(B)) || (axes(parent(S), 1) != axes(A, 2))
            throw(ArgumentError("matrix axes don't match"))
        end
    end
    out = zero(promote_type(eltype(S), eltype(A), eltype(B)))
    @inbounds for j in axes(A, 2)
        out += parent(S)[j] * dot(view(A, :, j), view(B, :, j))
    end
    return out / size(A, 1)
end

frobenius_normalized(S::UniformScaling, A, B) = S.λ * frobenius_normalized(A, B)

## Generic herk
## ============
"""
    herk(A)

Compute `A * A'` and return as an appropriate type
"""
herk(A) = Hermitian(A * A')
herk(A::Diagonal) = map_diagonal(abs2, A)

"""
    herk!!(H, A)

Compute `A * A'`, overwriting and returning `H` if possible
"""
herk!!(H, A) = herk!!(H, A, true, false)

"""
    herk!!(H::Hermitian, A, α, β)
    herk!!(H::Diagonal{<:Real}, A::Diagonal, α, β)

Overwrite `H` with `α * A * A' + β * H`, return `H`
"""
function herk!!(H::Hermitian, A, α, β)
    mul!(parent(H), A, A', α, β)
    return H
end

function herk!!(H::Diagonal{<:Real}, A::Diagonal, α, β)
    Hdiag = parent(H)
    Hdiag .= α .* abs2.(parent(A)) .+ β .* Hdiag
    return H
end

"""
    herk!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    herk!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)

Return `α * A * A' + β * H`
"""
function herk!!(H::Hermitian{<:Any,<:SMatrix}, A, α, β)
    return Hermitian(α * A * A' + β * parent(H), uplo(H))
end

function herk!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β)
    return Diagonal(α * map(abs2, parent(A)) + β * parent(H))
end

## Decomposition
## =============
"""
    decompose(H::Hermitian{T}; shift=zero(T)) where {T}

Compute the Cholesky factorization of `H + shift * I` with `A` Hermitian and `shift` real
"""
function decompose(H::Hermitian{T}; shift=zero(T)) where {T}
    Hdata = parent(H)
    target = if (Hdata isa StaticMatrix) && (uplo(H) === :L)
        # https://github.com/JuliaArrays/StaticArrays.jl/issues/1075
        Hermitian(Hdata', :U)
    else
        H
    end
    iszero(shift) && cholesky(target)
    checkreal(shift)
    return cholesky(Hermitian(parent(target) + T(shift) * I, uplo(target)))
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
function decompose(A::Diagonal{T}; shift=zero(T)) where {T}
    return A + T(shift) * I
end

"""
    decompose!!(F::Cholesky{T}, A::Hermitian{T}; shift=zero(T)) where {T}
    decompose!!(
        F::SparseArrays.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0
    )
    decompose!!(F::Diagonal{T}, A::Diagonal{T}; shift=zero(T)) where {T}

Factorize `A + shift * I` into an existing factorization instance `F` and return `F`

The factorization `F` is updated in-place and returned. The appropriate type of `F` is the
return type of `decompose(A)`; that is, appropriate usage may look as follows:

```julia
F = decompose(A)
# Do stuff with F, which holds a factorization of A
F = decompose!(F, A; shift=1)
# Do more stuff with F, which now holds a factorization of A + I
```

Note: Whenever an error is thrown, for example if `F::Cholesky` but `A + shift * I` is not
positive definite, the factorization object `F` may be left in an inconsistent and
unrecoverable state. For a safe but allocating alternative, use `decompose` instead.
"""
function decompose!!(F::Cholesky{T}, A::Hermitian{T}; shift=zero(T)) where {T}
    copyto!(F.factors, A)
    if !iszero(shift)
        checkreal(shift)
        shift_diagonal(F.factors, T(shift))
    end
    _, info = LAPACK.potrf!(F.uplo, F.factors)
    (info == 0) || throw(PosDefException(info))
    return F
end

function decompose!!(
    F::SparseArrays.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...
)
    return ldlt!(F, A; kwargs...)
end

function decompose!!(F::Diagonal{T}, A::Diagonal{T}; shift=zero(T)) where {T}
    copyto!(parent(F), parent(A))
    if shift != 0
        shift_diagonal(F, T(shift))
    end
    return F
end

"""
    decompose!!(F::Cholesky{T,<:SMatrix}, A::Hermitian{T,<:SMatrix}; kwargs...) where {T}
    decompose!!(F::Diagonal{T,<:SMatrix}, A::Diagonal{T,<:SMatrix}; kwargs...) where {T}

Return `A + shift * I`

Ignores `F` and forwards the call to `decompose(A; kwargs...)`. This is to provide a generic
factorization updating interface that also supports immutable factorization types.
"""
function decompose!!(
    ::Cholesky{T,<:SMatrix}, A::Hermitian{T,<:SMatrix}; kwargs...
) where {T}
    return decompose(A; kwargs...)
end

function decompose!!(::Diagonal{T,<:SVector}, A::Diagonal{T,<:SVector}; kwargs...) where {T}
    return decompose(A; kwargs...)
end

# decomposition helpers
checkreal(x::Number) = iszero(imag(x)) || throw(DomainError("shift must be real"))

function shift_diagonal(A, shift)
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
    ldiv!!(A::SparseArrays.CHOLMOD.Factor, B)

Return `A \\ B`

This provides a generic left division interface that also covers cases where the most
efficient or only option is out-of-place division.
"""
ldiv!!(A, B::AnySArray) = A \ B
ldiv!!(A::SparseArrays.CHOLMOD.Factor, B) = A \ B

"""
    ldiv!!(A::SparseArrays.CHOLMOD.Factor, B::AnySArray))

Return `A \\ Matrix(B)`

This method mainly exists to avoid method ambiguity, but it does the right thing.
(SparseArrays.CHOLMOD left division doesn't support structured wrappers like `Diagonal` and
`Hermitian` anyway, so no point in trying to preserve them.)
"""
ldiv!!(A::SparseArrays.CHOLMOD.Factor, B::AnySArray) = A \ Matrix(B)

"""
    inverse(F)
    inverse(F::Cholesky)
    inverse(F::SparseArrays.CHOLMOD.Factor)

Compute the inverse of a matrix from a factorization

The factorization object `F` should equal `decompose(A)` where `A` is the matrix to invert.

This function falls back to `inv` on types for which it is not specialized.
"""
inverse(F) = inv(F)
inverse(F::Cholesky) = Hermitian(inv(F), uplo(F))

function inverse(F::SparseArrays.CHOLMOD.Factor)
    id = spdiagm(ones(eltype(F), size(F, 1)))
    return Hermitian(Matrix(F \ id))
end

"""
    inverse!!(Ai::Hermitian{T}, F::Cholesky{T}) where {T}
    inverse!!(Ai::Diagonal{T}, F::Diagonal{T})

Compute the inverse of a matrix from a factorization, filling an existing array

The factorization object `F` should equal `decompose(A)` where `A` is the matrix to invert.
The appropriate type of `Ai` is the return type of `inverse(decompose(A))`. Thus, appropriate
usage may look as follows:

```julia
F = decompose(A)
Ai = inverse(F)
# Do stuff with Ai, which holds the inverse of A
F = decompose!!(F, A; shift=1)
Ai = inverse!!(Ai, F)
# Do more stuff with Ai, which now holds the inverse of A + I
```
"""
function inverse!!(Ai::Hermitian{T}, F::Cholesky{T}) where {T}
    (uplo(Ai) === uplo(F)) || throw(ArgumentError("must have uplo(Ai) == uplo(F)"))
    Aidata = parent(Ai)
    copyto!(Aidata, F.factors)
    LAPACK.potri!(Ai.uplo, Aidata)
    return Ai
end

function inverse!!(Ai::Diagonal{T}, F::Diagonal{T}) where {T}
    parent(Ai) .= inv.(parent(F))
    return Ai
end

"""
    inverse!!(Ai::Hermitian{T}, F::SparseArrays.CHOLMOD.Factor{T}) where {T}

Compute the inverse of a matrix from a factorization, using an existing array as workspace

!!! note

    `Ai` will not contain the inverse at the end; it is used as a working array only. The
    inverse is the function's return value.
"""
function inverse!!(Ai::Hermitian{T}, F::SparseArrays.CHOLMOD.Factor{T}) where {T}
    Aidata = parent(Ai)
    copyto!(Aidata, I)
    return Hermitian(ldiv!!(F, Aidata), uplo(Ai))
end

"""
    inverse!!(::Hermitian{T,<:SMatrix}, F::Cholesky{T,<:SMatrix}) where {T}
    inverse!!(::Diagonal{T,<:SVector}, F::Cholesky{T,<:SVector}) where {T}

Compute the inverse of `F`

The call is forwarded to `inverse(F)`. This is to provide an inverse updating interface that
also supports immutable inverse types.
"""
inverse!!(::Hermitian{T,<:SMatrix}, F::Cholesky{T,<:SMatrix}) where {T} = inverse(F)
inverse!!(::Diagonal{T,<:SVector}, F::Diagonal{T,<:SVector}) where {T} = inverse(F)

## Smallest eigenvalue
## ==================
"""
    eigmin!!(F, A)

Find the smallest eigenvalue of `A`, possibly via a factorization into `F`

The factorization object `F` may be updated in-place or remain unused, depending on the type
of `A`. The appropriate type of `F` is the return type of `decompose(A)`.
"""
function eigmin!!(F, A::Hermitian)
    Ai = inverse_as_map(F, A)
    which = ArnoldiMethod.LM()
    schur, history = partialschur(Ai; which, nev=1, tol=eigmin_reltol(Ai))
    (history.nconverged >= 1) || error("smallest eigenvalue not found: did not converge")
    return inv(real(first(schur.eigenvalues)))
end

eigmin!!(_, A::Diagonal{<:Real}) = minimum(parent(A))
eigmin!!(_, A::Diagonal{<:Real,<:SArray}) = minimum(parent(A))  # Ambiguity resolution
eigmin!!(_, A::AnySMatrix) = eigmin(A)

# eigmin helpers
eigmin_reltol(::Type{T}) where {T} = (eps(real(T))^(3 // 4)) * (oneunit(real(T))^(1 // 4))
eigmin_reltol(x) = eigmin_reltol(eltype(x))

function inverse_as_map(F::Cholesky{T}, A::Hermitian{T,<:StridedMatrix}) where {T}
    decompose!!(F, A)
    return LinearMap{T}(
        (y, x) -> ldiv!(y, F, x),
        size(A, 2);
        ismutating=true,
        ishermitian=true,
        isposdef=true,
    )
end

function inverse_as_map(
    F::SparseArrays.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}
)
    decompose!!(F, A)
    return LinearMap{eltype(F)}(
        x -> F \ x, size(A, 2); ismutating=false, ishermitian=true, isposdef=true
    )
end

end  # module MathTools
