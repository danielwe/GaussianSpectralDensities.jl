module MathTools

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
    cholesky!,
    diagind,
    dot,
    eigmin,
    eigvals,
    isposdef,
    issuccess,
    ldiv!,
    mul!,
    tr
using LinearMaps: LinearMap
using MappedArrays: mappedarray
using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm
using StaticArrays: MMatrix, SArray, SMatrix, SVector, StaticMatrix

using ..ArrayTools: AnySArray, AnySMatrix, DiagonalOrUniform, map_diagonal, withsize
using ..Defaults: Defaults

public ShiftedGramMatrix,
    decompose_pd,
    decompose_pd!!,
    frobenius_normalized,
    gramgap,
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

samesign(x::Number, y::Number) = (x * y > 0)  # XXX: probably not robust (Inf/NaN etc.)

## Linear interpolation/extrapolation
## ==================================
function predictlinear(x, (x1, y1), (x2, y2))
    if x1 == x2
        # fall back to piecewise constant predictor; perhaps non-standard, definitely useful
        if x < x1
            return y1
        elseif x2 < x
            return y2
        else
            return (y1 + y2) / 2
        end
    end
    return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)
end

function isapproxlinear((x, y), (x1, y1), (x2, y2))
    a, b = promote((y2 - y) * (x1 - x), (y1 - y) * (x2 - x))
    return Defaults.approxequal(a, b)
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

This is intended as an out-of-place equivalent of the in-place ``muladd_diagonal!!``
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

herk!!(H::Diagonal{<:Real,<:SVector}, A::Diagonal, α, β) = α * map_diagonal(abs2, A) + β * H

## Decomposition
## =============
"""
    decompose_pd(A::Hermitian{T}; shift=zero(T), check=true) where {T}

Compute the Cholesky factorization of `A + shift * I` with `A` Hermitian and `shift` real

If `check` is `true`, a `PosDefException` is thrown if `A + shift * I` is not positive
definite; if `false`, a incomplete factorization is returned, and the caller is responsible
for checking its status using `LinearAlgebra.issuccess`.
"""
function decompose_pd(A::Hermitian{T}; shift=zero(T), check=true) where {T}
    iszero(shift) && return cholesky_pd(A, check)
    checkrealshift(shift)
    return cholesky_pd(Hermitian(parent(A) + T(shift) * I, uplo(A)), check)
end

"""
    decompose_pd(A::Hermitian{<:Any,<:SparseMatrixCSC}; shift=0.0, check=true)

Get the Cholesky factorization of `A + shift * I` with Hermitian `A` and real `shift`

If `check` is `true`, a `PosDefException` is thrown if `A + shift * I` is not positive
definite; if `false`, a incomplete factorization is returned, and the caller is responsible
for checking its status using `LinearAlgebra.issuccess` or `LinearAlgebra.isposdef`.
"""
decompose_pd(A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...) = cholesky(A; kwargs...)

"""
    decompose_pd(A::Diagonal{T}; shift=zero(T), check=true) where {T}

Return `A + shift * I`

If `check` is `true`, a `PosDefException` is thrown if `A + shift * I` is not positive
definite; if `false`, no check of definiteness is performed.
"""
function decompose_pd(A::Diagonal{T}; shift=zero(T), check=true) where {T}
    F = A + T(shift) * I
    check && checkposdef(F)
    return F
end

"""
    decompose_pd!!(F::Cholesky{T}, A::Hermitian{T}; shift=zero(T), check=true) where {T}
    decompose_pd!!(
        F::SparseArrays.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC};
        shift=0.0, check=true,
    )
    decompose_pd!!(F::Diagonal{T}, A::Diagonal{T}; shift=zero(T), check=true) where {T}

Factorize `A + shift * I`, reusing memory from an existing factorization instance `F`

The memory underlying the factorization `F` is updated in-place. Whether the returned object
is the same exact instance as `F` or a new wrapper of the same type depends on the types of
the arguments. Hence, the user should always use the return value and discard their binding
to `F`.

The appropriate type of `F` is the return type of `decompose_pd(A)`; thus, appropriate usage
may look as follows:

```julia
F = decompose_pd(A)
# Do stuff with F, which holds a factorization of A
F = decompose_pd!!(F, A; shift=1)
# Do more stuff with F, which now holds a factorization of A + I that may reuse memory from
# the previous F
```

If `check` is `true`, a `PosDefException` is thrown if `A + shift * I` is not positive
definite; if `false`, a possibly incomplete factorization is returned, and the caller is
responsible for checking its status using `LinearAlgebra.isposdef` (or
`LinearAlgebra.issuccess`, except in the `Diagonal` case).

!!! note

    If the decomposition failed, the argument `F` may be left in a corrupt and unrecoverable
    state. For a safe but allocating alternative, use `decompose_pd` instead.
"""
function decompose_pd!!(
    F::Cholesky{T}, A::Hermitian{T}; shift=zero(T), check=true
) where {T}
    copyto!(F.factors, A)
    if !iszero(shift)
        checkrealshift(shift)
        shift_diagonal(F.factors, T(shift))
    end
    _, info = LAPACK.potrf!(F.uplo, F.factors)
    check && ((info == 0) || throw(PosDefException(info)))
    # Can't modify F.info, so construct new wrapper. If info == F.info this is === F anyway.
    return typeof(F)(F.factors, F.uplo, info)
end

function decompose_pd!!(
    F::SparseArrays.CHOLMOD.Factor, A::Hermitian{<:Any,<:SparseMatrixCSC}; kwargs...
)
    return cholesky!(F, A; kwargs...)
end

function decompose_pd!!(F::Diagonal{T}, A::Diagonal{T}; shift=zero(T), check=true) where {T}
    copyto!(parent(F), parent(A))
    if shift != 0
        shift_diagonal(F, T(shift))
    end
    check && checkposdef(F)
    return F
end

"""
    decompose_pd!!(F::Cholesky{T,<:SMatrix}, A::Hermitian{T,<:SMatrix}; kwargs...) where {T}
    decompose_pd!!(F::Diagonal{T,<:SMatrix}, A::Diagonal{T,<:SMatrix}; kwargs...) where {T}

Return an appropriate decomposition of `A + shift * I`, assuming positive definiteness

Ignores `F` and forwards the call to `decompose_pd(A; kwargs...)`. This is to provide a
generic factorization updating interface that also supports immutable factorization types.
"""
function decompose_pd!!(
    ::Cholesky{T,<:SMatrix}, A::Hermitian{T,<:SMatrix}; kwargs...
) where {T}
    return decompose_pd(A; kwargs...)
end

function decompose_pd!!(
    ::Diagonal{T,<:SVector}, A::Diagonal{T,<:SVector}; kwargs...
) where {T}
    return decompose_pd(A; kwargs...)
end

# decomposition helpers
cholesky_pd(A::Hermitian, check) = cholesky(A; check)

function cholesky_pd(A::Hermitian{<:Any,<:StaticMatrix}, check)
    if uplo(A) === :L
        # StaticArrays's cholesky does not handle uplo
        # issue: https://github.com/JuliaArrays/StaticArrays.jl/issues/1075
        # PR: https://github.com/JuliaArrays/StaticArrays.jl/pull/1369
        A = Hermitian(parent(A)', :U)
    end
    C = cholesky(A; check=false)
    # StaticArrays's cholesky fails to reject nonpositive pivots if the eltype is complex
    # and zero pivots if it's real, so we must validate the factorization ourselves before
    # returning. Since pivots for complex eltypes are calculated in complex arithmetic, even
    # real and positive ones may have a roundoff-level imaginary part, so `isreal` is out.
    # The case to reject is always the square root of a negative number, i.e., approximately
    # imaginary, so we don't need carefully chosen tolerances, we just check that the real
    # part is larger than the absolute imaginary part---which conveniently doubles as a
    # positivity check.
    # PR: https://github.com/JuliaArrays/StaticArrays.jl/pull/1368
    info = C.info
    if iszero(info)
        pivots = view(C.factors, diagind(C.factors))
        info = something(findfirst(d -> !(real(d) > abs(imag(d))), pivots), 0)
    end
    (check && !iszero(info)) && throw(PosDefException(info))
    # Remove imaginary noise on the diagonal
    factors = C.factors
    if eltype(factors) <: Complex
        factors_mutable = MMatrix(factors)
        dind = diagind(factors_mutable)
        factors_mutable[dind] .= real.(view(factors_mutable, dind))
        factors = typeof(factors)(factors_mutable)
    end
    return typeof(C)(factors, C.uplo, info)
end

checkrealshift(shift::Number) = isreal(shift) || throw(DomainError("shift must be real"))

function checkposdef(D::Diagonal)
    for (i, di) in pairs(D.diag)
        if !(isreal(di) && (real(di) > 0))
            throw(PosDefException(i))
        end
    end
    return nothing
end

function shift_diagonal(A, shift)
    @inbounds for i in diagind(A)
        A[i] += shift
    end
    return nothing
end

## Shifted Gram matrix
## ===================
"""
    ShiftedGramMatrix(A; shift=0, check=true, normalize=false)
    ShiftedGramMatrix(
        G::ShiftedGramMatrix;
        new_factor=nothing,
        new_shift=nothing,
        overwrite=false,
        check=true,
        normalize=false,
    )

Factorize and represent the shifted Gram matrix `A * A' + shift * I`

The smallest eigenvalue of the unshifted Gram matrix, `eigmin(A * A')`, is computed eagerly
on construction and stored in the field `gramgap`.

If `normalize` is `true`, `A * A'` is replaced with `(A * A' + A' * A) / 2`.

If `check` is `true`, a `PosDefException` is thrown if `A * A' + shift * I` is not positive
definite; if `false`, the caller is responsible for checking the status using
`LinearAlgebra.issuccess`. Failure to factorize the unshifted Gram matrix for computing
`gramgap` never throws, but implies `gramgap == 0`.

The second form derives a new shifted Gram matrix from `G`, replacing the factor `A` by
`new_factor` and/or the shift by `new_shift`. If only the shift is replaced, `gramgap` is
copied from `G` and not recomputed. If `overwrite` is `true`, memory from `G` may be reused,
leaving `G` in an invalid state.
"""
struct ShiftedGramMatrix{
    F<:Real,
    TG<:Union{Diagonal{F},Hermitian{Complex{F}}},
    # Don't use Complex{F}; factorizations can promote
    TF<:Union{Diagonal{F},Cholesky{<:Complex},SparseArrays.CHOLMOD.Factor{<:Complex}},
}
    _fact::TF
    gram::TG
    shift::F
    gramgap::F

    function ShiftedGramMatrix(A; shift=zero(real(eltype(A))), check=true, normalize=false)
        gram = herk(A)
        if (!(A isa Diagonal)) && normalize
            gram = herk!!(gram, A', 1 // 2, 1 // 2)
        end
        _fact, gramgap = _factorize_with_gramgap!!(nothing, gram, shift, check)
        return _new(_fact, gram, shift, gramgap)
    end

    function ShiftedGramMatrix(
        G::ShiftedGramMatrix;
        new_factor=nothing,
        new_shift=nothing,
        overwrite=false,
        check=true,
        normalize=false,
    )
        shift = isnothing(new_shift) ? G.shift : new_shift
        old_fact = overwrite ? G._fact : nothing
        if isnothing(new_factor)
            gram, gramgap = G.gram, G.gramgap
            _fact = _factorize!!(old_fact, gram, shift, check)
        else
            old_gram = overwrite ? G.gram : nothing
            gram = _gram!!(old_gram, new_factor, normalize)
            _fact, gramgap = _factorize_with_gramgap!!(old_fact, gram, shift, check)
        end
        return _new(_fact, gram, shift, gramgap)
    end

    # Inner constructor helpers
    function _new(_fact, gram, shift, gramgap)
        F = real(eltype(gram))
        return new{F,typeof(gram),typeof(_fact)}(_fact, gram, shift, gramgap)
    end

    function _gram!!(old_gram, A, normalize)
        gram = isnothing(old_gram) ? herk(A) : herk!!(old_gram, A)
        ((!(A isa Diagonal)) && normalize) && return herk!!(gram, A', 1 // 2, 1 // 2)
        return gram
    end

    function _factorize!!(old_fact, gram, shift, check)
        isnothing(old_fact) && return decompose_pd(gram; shift, check)
        return decompose_pd!!(old_fact, gram; shift, check)
    end

    function _factorize_with_gramgap!!(old_fact, gram, shift, check)
        # If iszero(shift), the bare factorization is the only one and must respect check
        check_bare = check && iszero(shift)
        fact = _factorize!!(old_fact, gram, zero(shift), check_bare)
        gramgap = _eigmin(fact, gram)
        if !iszero(shift)
            fact = _factorize!!(fact, gram, shift, check)
        end
        return fact, gramgap
    end

    _eigmin(_, gram::Diagonal) = minimum(parent(gram))
    _eigmin(_, gram::Hermitian{<:Any,<:SMatrix}) = eigmin_static(gram)

    function _eigmin(fact, _)
        # Gram matrices are positive semidefinite by construction. If decomposition failed, the
        # matrix is numerically singular, so the smallest eigenvalue is zero.
        issuccess(fact) || return zero(real(eltype(fact)))
        return eigmin_arnoldi(fact)
    end
end

function LinearAlgebra.issuccess(G::ShiftedGramMatrix)
    (G._fact isa Diagonal) && return isposdef(G._fact)
    return issuccess(G._fact)
end

gramgap(G::ShiftedGramMatrix) = G.gramgap

## Divison and inversion
## =====================
"""
    ldiv!!(G::ShiftedGramMatrix, B)

Left-divide the shifted Gram matrix `G` into `B`, possibly in-place in `B` if mutable

!!! note

    Even if `B` is mutable, it may not always be reused as output, depending on types and
    method support. Only the return value is guaranteed to hold the final result.

!!! note

    The status of the decomposition cached in `G` is not verified. If `G` was instantiated
    with `check=false`, `LinearAlgebra.issuccess(G)` may be `false`, and this function may
    produce garbage.
"""
ldiv!!(G::ShiftedGramMatrix, B) = _ldiv!!(G._fact, B)
_ldiv!!(A, B) = ldiv!(A, B)
_ldiv!!(A, B::AnySMatrix) = A \ B
_ldiv!!(A::SparseArrays.CHOLMOD.Factor, B) = A \ B

# This method is rarely useful and mainly exists to avoid method ambiguity, but it does the
# right thing; SparseArrays.CHOLMOD left division doesn't support structured wrappers like
# `Diagonal` and `Hermitian` anyway, so no point in trying to preserve them; Matrix is
# always the correct conversion type.
_ldiv!!(A::SparseArrays.CHOLMOD.Factor, B::AnySMatrix) = A \ Matrix(B)

## Need out-of-place division too
Base.:\(G::ShiftedGramMatrix, B) = G._fact \ B

"""
    inverse(G::ShiftedGramMatrix)

Compute the inverse of a shifted Gram matrix

!!! note

    The status of the decomposition cached in `G` is not verified. If `G` was instantiated
    with `check=false`, `LinearAlgebra.issuccess(G)` may be `false`, and this function may
    produce garbage.
"""
inverse(G::ShiftedGramMatrix) = _inverse(G._fact)
_inverse(F) = inv(F)
_inverse(F::Cholesky) = Hermitian(inv(F), uplo(F))

function _inverse(F::SparseArrays.CHOLMOD.Factor)
    id = spdiagm(ones(eltype(F), size(F, 1)))
    return Hermitian(Matrix(F \ id))
end

"""
    inverse!!(Ai, G::ShiftedGramMatrix)

Compute the inverse of a shifted Gram matrix `G`, possibly reusing `Ai` if mutable

The appropriate type of `Ai` is the return type of `inverse(G)`. Thus, appropriate usage may
look as follows:

```julia
G = ShiftedGramMatrix(A)
Ai = inverse(G)
# Do stuff with Ai, which holds the inverse of G
G = ShiftedGramMatrix(G; new_shift=1)
Ai = inverse!!(Ai, G)
# Do more stuff with Ai, which now holds the inverse of A + I
```

!!! note

    Even if `Ai` is mutable, it may be used only as workspace rather than as output. Only
    the return value is guaranteed to hold the final result.

!!! note

    The status of the decomposition cached in `G` is not verified. If `G` was instantiated
    with `check=false`, `LinearAlgebra.issuccess(G)` may be `false`, and this function may
    produce garbage.
"""
inverse!!(Ai, G::ShiftedGramMatrix) = _inverse!!(Ai, G._fact)

function _inverse!!(Ai::Hermitian{T}, F::Cholesky{T}) where {T}
    (uplo(Ai) === uplo(F)) || throw(ArgumentError("must have uplo(Ai) == uplo(F)"))
    Aidata = parent(Ai)
    copyto!(Aidata, F.factors)
    LAPACK.potri!(Ai.uplo, Aidata)
    return Ai
end

function _inverse!!(Ai::Diagonal{T}, F::Diagonal{T}) where {T}
    parent(Ai) .= inv.(parent(F))
    return Ai
end

# NOTE: With CHOLMOD, `Ai` is only used as workspace and will never contain the result. The
# inverse is the function's return value.
function _inverse!!(Ai::Hermitian{T}, F::SparseArrays.CHOLMOD.Factor{T}) where {T}
    Aidata = parent(Ai)
    copyto!(Aidata, I)
    return Hermitian(_ldiv!!(F, Aidata), uplo(Ai))
end

_inverse!!(::Hermitian{T,<:SMatrix}, F::Cholesky{T,<:SMatrix}) where {T} = _inverse(F)
_inverse!!(::Diagonal{T,<:SVector}, F::Diagonal{T,<:SVector}) where {T} = _inverse(F)

## Smallest eigenvalue
## ==================
# StaticArrays provides explicit eigvals method for sizes 1, 2, 3, however, they don't
# support the eigvals(H, n:m) form, so LinearAlgebra.eigmin won't dispatch to them, hence
# these helpers. For size 4 and up, any approach will make an Array copy and use LAPACK, so
# in that case it's better to just use LinearAlgebra.eigmin directly.
eigmin_static(H::Hermitian{T,<:SMatrix{1,1,T}}) where {T} = first(eigvals(H))
# On closer inspection, the 2x2 and 3x3 methods have sloppy numerics. Just let LAPACK do it.
# eigmin_static(H::Hermitian{T,<:SMatrix{2,2,T}}) where {T} = first(eigvals(H))
# eigmin_static(H::Hermitian{T,<:SMatrix{3,3,T}}) where {T} = first(eigvals(H))
eigmin_static(H::Hermitian{T,<:SMatrix{<:Any,<:Any,T}}) where {T} = eigmin(H)

function eigmin_arnoldi(F::Union{Cholesky,SparseArrays.CHOLMOD.Factor})
    Ai = inverse_as_map(F)
    which = ArnoldiMethod.LM()
    schur, history = partialschur(Ai; which, nev=1, tol=Defaults.rtol_strict(eltype(Ai)))
    (history.nconverged >= 1) || error("smallest eigenvalue not found: did not converge")
    return inv(real(first(schur.eigenvalues)))
end

function inverse_as_map(F::Cholesky{T}) where {T}
    return LinearMap{T}(
        (y, x) -> ldiv!(y, F, x),
        size(F, 2);
        ismutating=true,
        ishermitian=true,
        isposdef=true,
    )
end

function inverse_as_map(F::SparseArrays.CHOLMOD.Factor)
    return LinearMap{eltype(F)}(
        x -> F \ x, size(F, 2); ismutating=false, ishermitian=true, isposdef=true
    )
end

end  # module MathTools

@testitem "MathTools" begin
    using GaussianSpectralDensities.MathTools:
        ShiftedGramMatrix,
        decompose_pd,
        decompose_pd!!,
        frobenius_normalized,
        gramgap,
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

    using LinearAlgebra:
        Bidiagonal,
        Cholesky,
        Diagonal,
        Hermitian,
        I,
        PosDefException,
        SymTridiagonal,
        Tridiagonal,
        eigmin,
        issuccess,
        tr
    using SparseArrays: sparse
    using StaticArrays: @SMatrix, @SVector, SMatrix, SVector

    @testset "scalar predicates" begin
        @test inunitdisc(0.5)
        @test inunitdisc(1.0)
        @test !inunitdisc(1.1)
        @test inunitdisc(0.5 + 0.5im)

        @test phase(2.0) == 1.0
        @test phase(-3.0) == -1.0
        @test phase(0.0) == 1.0
        @test phase(2im) ≈ im

        @test samesign(2, 3)
        @test samesign(-2, -3)
        @test !samesign(2, -3)
    end

    @testset "predictlinear / isapproxlinear" begin
        @test predictlinear(1.5, (1.0, 10.0), (2.0, 20.0)) ≈ 15.0
        @test predictlinear(0.0, (1.0, 10.0), (2.0, 20.0)) ≈ 0.0  # extrapolation

        # degenerate case x1 == x2
        @test predictlinear(0.5, (1.0, 10.0), (1.0, 20.0)) == 10.0
        @test predictlinear(1.0, (1.0, 10.0), (1.0, 20.0)) == 15.0
        @test predictlinear(1.5, (1.0, 10.0), (1.0, 20.0)) == 20.0

        @test isapproxlinear((1.5, 15.0), (1.0, 10.0), (2.0, 20.0))
    end

    @testset "monotonic" begin
        @test monotonic([1, 2, 3])
        @test monotonic([1])
        @test monotonic(Int[])
        @test !monotonic([1, 3, 2])
        @test !monotonic([1, 1, 2])
    end

    @testset "ldiv / rdiv" begin
        A = [1.0 2.0; 3.0 4.0]
        @test ldiv(A, I) ≈ inv(A)
        @test rdiv(I, A) ≈ inv(A)

        S = @SMatrix [1.0 0.0; 0.0 2.0]
        @test ldiv(S, I) ≈ inv(S)
        @test rdiv(I, S) ≈ inv(S)

        # Diagonal with SArray storage
        Ds = Diagonal(@SVector [2.0, 4.0])
        @test ldiv(Ds, I) ≈ inv(Ds)
        @test rdiv(I, Ds) ≈ inv(Ds)

        # SparseMatrixCSC
        Sp = sparse([1.0 0.0; 0.0 2.0])
        Sp_inv = inv(Matrix(Sp))
        @test ldiv(Sp, I) ≈ Sp_inv
        @test rdiv(I, Sp) ≈ Sp_inv
    end

    @testset "minus" begin
        A = [1, 2, 3]
        @test collect(minus(A)) == [-1, -2, -3]

        D = Diagonal([1, 2])
        @test collect(minus(D)) == [-1 0; 0 -2]

        S = @SMatrix [1 2; 3 4]
        @test minus(S) == -S

        # Structured matrices
        Bi = Bidiagonal([1, 2, 3], [4, 5], :U)
        @test collect(minus(Bi)) == collect(-Bi)

        ST = SymTridiagonal([1, 2, 3], [4, 5])
        @test collect(minus(ST)) == collect(-ST)

        Tri = Tridiagonal([1, 2], [3, 4, 5], [6, 7])
        @test collect(minus(Tri)) == collect(-Tri)
    end

    @testset "muladd!!" begin
        C = zeros(3)
        A = [1.0, 2.0, 3.0]
        B = [10.0, 20.0, 30.0]
        @test muladd!!(C, 2.0, A, B) === C
        @test C == [12.0, 24.0, 36.0]

        Cs = @SVector zeros(3)
        @test muladd!!(Cs, 2.0, A, B) == [12.0, 24.0, 36.0]
    end

    @testset "muladd_diagonal!!" begin
        # In-place on mutable matrix
        C = [10.0 1.0; 2.0 20.0]
        A = Diagonal([1.0, 2.0])
        B = [5.0 1.0; 2.0 6.0]
        result = muladd_diagonal!!(C, 2.0, A, B)
        @test result === C
        @test C[1, 1] == 2.0 * 1.0 + 5.0
        @test C[2, 2] == 2.0 * 2.0 + 6.0

        # Out-of-place with static array
        Cs = @SMatrix [10.0 1.0; 2.0 20.0]
        @test muladd_diagonal!!(Cs, 2.0, A, B) == 2.0 * A + B

        # Structured matrix types for view_diagonal coverage
        Bi = Bidiagonal([1.0, 2.0, 3.0], [0.1, 0.2], :U)
        Cbi = copy(Bi)
        muladd_diagonal!!(Cbi, 2.0, Bi, Bi)
        @test Cbi.dv == 3.0 .* Bi.dv

        ST = SymTridiagonal([1.0, 2.0, 3.0], [0.1, 0.2])
        Cst = copy(ST)
        muladd_diagonal!!(Cst, 2.0, ST, ST)
        @test Cst.dv == 3.0 .* ST.dv

        Tri = Tridiagonal([0.1, 0.2], [1.0, 2.0, 3.0], [0.3, 0.4])
        Ctri = copy(Tri)
        muladd_diagonal!!(Ctri, 2.0, Tri, Tri)
        @test Ctri.d == 3.0 .* Tri.d

        # UniformScaling in A or B
        C2 = [10.0 1.0; 2.0 20.0]
        muladd_diagonal!!(C2, 2.0, I, C2)
        @test C2[1, 1] == 2.0 * 1.0 + 10.0
        @test C2[2, 2] == 2.0 * 1.0 + 20.0
    end

    @testset "tr_normalized / frobenius_normalized" begin
        A = [1.0 0.0; 0.0 3.0]
        @test tr_normalized(A) == 2.0

        B = [1.0 2.0; 3.0 4.0]
        @test frobenius_normalized(A, B) == tr(A' * B) / 2

        S = Diagonal([1.0, 2.0])
        @test frobenius_normalized(S, A, B) ≈ tr(S * A' * B) / 2

        # UniformScaling variant
        @test frobenius_normalized(2I, A, B) ≈ 2 * frobenius_normalized(A, B)

        # Bounds check error
        @test_throws ArgumentError frobenius_normalized(Diagonal([1.0]), A, B)
    end

    @testset "herk / herk!!" begin
        A = [1.0 2.0; 3.0 4.0]
        H = herk(A)
        @test H ≈ A * A'
        @test H isa Hermitian

        D = Diagonal([2.0, 3.0])
        @test herk(D) == Diagonal([4.0, 9.0])

        H2 = similar(H)
        @test herk!!(Hermitian(H2), A) ≈ A * A'

        # herk!! with α, β on mutable Diagonal
        Hd = Diagonal([1.0, 1.0])
        Ad = Diagonal([2.0, 3.0])
        result = herk!!(Hd, Ad, 2.0, 3.0)
        @test result === Hd
        @test Hd == Diagonal([2.0 * 4.0 + 3.0 * 1.0, 2.0 * 9.0 + 3.0 * 1.0])

        # herk!! with static arrays (out-of-place)
        Hs = Hermitian(@SMatrix [1.0 0.0; 0.0 1.0])
        As = @SMatrix [1.0 2.0; 3.0 4.0]
        @test herk!!(Hs, As, 2.0, 3.0) ≈ 2.0 * As * As' + 3.0 * parent(Hs)

        Hds = Diagonal(@SVector [1.0, 1.0])
        Ads = Diagonal(@SVector [2.0, 3.0])
        @test herk!!(Hds, Ads, 2.0, 3.0) == Diagonal([2.0 * 4.0 + 3.0, 2.0 * 9.0 + 3.0])
    end

    @testset "decompose_pd" begin
        A = [4.0 2.0; 2.0 5.0]
        Abad = [1.0 2.0; 2.0 1.0]
        b = [1.0, 2.0]
        for (name, H, Hbad) in (
            ("Array", Hermitian(A), Hermitian(Abad)),
            ("SMatrix", Hermitian(SMatrix{2,2}(A)), Hermitian(SMatrix{2,2}(Abad))),
            ("SparseMatrixCSC", Hermitian(sparse(A)), Hermitian(sparse(Abad))),
        )
            @testset "$name" begin
                @test decompose_pd(H) \ b ≈ A \ b
                @test decompose_pd(H; shift=1.0) \ b ≈ (A + I) \ b
                @test_throws PosDefException decompose_pd(Hbad)
                @test !issuccess(decompose_pd(Hbad; check=false))
                @test issuccess(decompose_pd(Hbad; shift=4.0))
            end
        end
        @test decompose_pd(Hermitian(A)) isa Cholesky

        # Complex indefinite and singular matrices (StaticArrays needs extra pivot checks)
        for (name, Abad) in (
                ("complex indefinite", ComplexF64[1 2im; -2im 1]),
                ("real singular", [1.0 1.0; 1.0 1.0]),
                ("complex singular", ComplexF64[1 1im; -1im 1]),
            ),
            (arrayname, toarray) in
            (("Array", identity), ("SMatrix", SMatrix{2,2}), ("SparseMatrixCSC", sparse))

            @testset "$name $arrayname" begin
                Hbad = Hermitian(toarray(Abad))
                @test_throws PosDefException decompose_pd(Hbad)
                @test !issuccess(decompose_pd(Hbad; check=false))
                @test issuccess(decompose_pd(Hbad; shift=4.0))
            end
        end

        # Static Hermitian with :L uplo (triggers workaround branch)
        HsL = Hermitian(@SMatrix([4.0 0.0; 2.0 5.0]), :L)
        @test decompose_pd(HsL) \ b ≈ A \ b
        @test decompose_pd(HsL; shift=1.0) \ b ≈ (A + I) \ b

        # Non-real shift
        @test_throws DomainError decompose_pd(Hermitian(A); shift=1im)
        @test_throws DomainError decompose_pd(Hermitian(SMatrix{2,2}(A)); shift=1im)

        for (name, D, Dbad) in (
            ("Vector", Diagonal([2.0, 3.0]), Diagonal([1.0, -1.0])),
            ("SVector", Diagonal(@SVector [2.0, 3.0]), Diagonal(@SVector [1.0, -1.0])),
        )
            @testset "Diagonal{$name}" begin
                @test decompose_pd(D) == D
                @test decompose_pd(D; shift=1.0) == D + I
                @test_throws PosDefException decompose_pd(Dbad)
                @test decompose_pd(Dbad; check=false) == Dbad
                @test decompose_pd(Dbad; shift=2.0) == Dbad + 2I
            end
        end
    end

    @testset "decompose_pd!!" begin
        A = [4.0 2.0; 2.0 5.0]
        A2 = [5.0 1.0; 1.0 6.0]
        Abad = [1.0 2.0; 2.0 1.0]
        b = [1.0, 2.0]
        for (name, H, H2, Hbad) in (
            ("Array", Hermitian(A), Hermitian(A2), Hermitian(Abad)),
            (
                "SMatrix",
                Hermitian(SMatrix{2,2}(A)),
                Hermitian(SMatrix{2,2}(A2)),
                Hermitian(SMatrix{2,2}(Abad)),
            ),
            (
                "SparseMatrixCSC",
                Hermitian(sparse(A)),
                Hermitian(sparse(A2)),
                Hermitian(sparse(Abad)),
            ),
        )
            @testset "$name" begin
                F = decompose_pd(H)
                F = decompose_pd!!(F, H2)
                @test F \ b ≈ A2 \ b
                F = decompose_pd!!(F, H; shift=1.0)
                @test F \ b ≈ (A + I) \ b
                @test_throws PosDefException decompose_pd!!(decompose_pd(H), Hbad)
                @test !issuccess(decompose_pd!!(decompose_pd(H), Hbad; check=false))
            end
        end

        # Mutable factorizations are updated in-place
        F = decompose_pd(Hermitian(A))
        @test decompose_pd!!(F, Hermitian(A2)) === F
        Fsp = decompose_pd(Hermitian(sparse(A)))
        @test decompose_pd!!(Fsp, Hermitian(sparse(A2))) === Fsp

        D = Diagonal([2.0, 3.0])
        D2 = Diagonal([4.0, 5.0])
        Dbad = Diagonal([1.0, -1.0])
        Fd = decompose_pd(D)
        @test decompose_pd!!(Fd, D2) === Fd
        @test Fd == D2
        @test decompose_pd!!(Fd, D; shift=1.0) == D + I
        @test_throws PosDefException decompose_pd!!(Fd, Dbad)
        @test decompose_pd!!(Fd, Dbad; check=false) == Dbad

        Ds = Diagonal(@SVector [2.0, 3.0])
        Ds2 = Diagonal(@SVector [4.0, 5.0])
        Dsbad = Diagonal(@SVector [1.0, -1.0])
        Fds = decompose_pd(Ds)
        @test decompose_pd!!(Fds, Ds2) == Ds2
        @test decompose_pd!!(Fds, Ds; shift=1.0) == Ds + I
        @test_throws PosDefException decompose_pd!!(Fds, Dsbad)
        @test decompose_pd!!(Fds, Dsbad; check=false) == Dsbad
    end

    @testset "ShiftedGramMatrix" begin
        for F in (Float32, Float64)
            Qdata = Complex{F}[2 0; 1 1]
            Q2data = Complex{F}[1 1; 0 2]
            qdiag = Complex{F}[2, 1]
            q2diag = Complex{F}[1, 2]
            B = Complex{F}[1 2; 3 4]
            cases = (
                ("Array", Qdata, Q2data),
                ("SMatrix", SMatrix{2,2}(Qdata), SMatrix{2,2}(Q2data)),
                ("SparseMatrixCSC", sparse(Qdata), sparse(Q2data)),
                ("Diagonal", Diagonal(qdiag), Diagonal(q2diag)),
                (
                    "Diagonal{SVector}",
                    Diagonal(SVector{2}(qdiag)),
                    Diagonal(SVector{2}(q2diag)),
                ),
            )
            for (name, Q, Q2) in cases
                @testset "F = $F, $name" begin
                    Qd, Q2d = Matrix(Q), Matrix(Q2)
                    gram, gram2 = Qd * Qd', Q2d * Q2d'
                    gramn = (Q isa Diagonal) ? gram : (gram + Qd' * Qd) / 2
                    gram2n = (Q isa Diagonal) ? gram2 : (gram2 + Q2d' * Q2d) / 2
                    gap = eigmin(Hermitian(gram))

                    G = ShiftedGramMatrix(Q)
                    @test G isa ShiftedGramMatrix{F}
                    @test G.gram ≈ gram
                    @test G.shift === zero(F)
                    @test gramgap(G) ≈ gap
                    @test issuccess(G)

                    Gs = ShiftedGramMatrix(Q; shift=1)
                    @test Gs.shift === one(F)
                    @test gramgap(Gs) ≈ gap
                    @test issuccess(Gs)
                    @test inverse(Gs) ≈ inv(gram + I)
                    @test inverse!!(inverse(G), Gs) ≈ inv(gram + I)
                    @test ldiv!!(Gs, copy(B)) ≈ (gram + I) \ B
                    @test ldiv!!(Gs, SMatrix{2,2}(B)) ≈ (gram + I) \ B
                    @test Gs \ B ≈ (gram + I) \ B

                    Gn = ShiftedGramMatrix(Q; normalize=true)
                    @test Gn.gram ≈ gramn
                    @test gramgap(Gn) ≈ eigmin(Hermitian(gramn))

                    # Shifting below the bare gap fails, but only the shifted factorization
                    negshift = -(gap + 1)
                    @test_throws PosDefException ShiftedGramMatrix(Q; shift=negshift)
                    Gneg = ShiftedGramMatrix(Q; shift=negshift, check=false)
                    @test !issuccess(Gneg)
                    @test gramgap(Gneg) ≈ gap
                    @test_throws PosDefException ShiftedGramMatrix(G; new_shift=negshift)

                    # Updating without overwrite leaves the original intact
                    Gc = ShiftedGramMatrix(Gs; new_factor=Q2, new_shift=2)
                    @test typeof(Gc) == typeof(Gs)
                    @test Gc.gram ≈ gram2
                    @test Gc.shift === F(2)
                    @test gramgap(Gc) ≈ eigmin(Hermitian(gram2))
                    @test inverse(Gc) ≈ inv(gram2 + 2I)
                    @test inverse(Gs) ≈ inv(gram + I)
                    Gc = ShiftedGramMatrix(Gs; new_factor=Q2, normalize=true)
                    @test Gc.gram ≈ gram2n
                    @test Gc.shift === one(F)
                    @test gramgap(Gc) ≈ eigmin(Hermitian(gram2n))
                    @test inverse(Gc) ≈ inv(gram2n + I)
                    Gc = ShiftedGramMatrix(Gs; new_shift=3)
                    @test Gc.gram ≈ gram
                    @test gramgap(Gc) === gramgap(Gs)
                    @test inverse(Gc) ≈ inv(gram + 3I)
                    @test inverse(Gs) ≈ inv(gram + I)

                    # Updating with overwrite
                    Go = ShiftedGramMatrix(Q; shift=1)
                    Go = ShiftedGramMatrix(Go; new_factor=Q2, new_shift=2, overwrite=true)
                    @test typeof(Go) == typeof(Gs)
                    @test Go.gram ≈ gram2
                    @test gramgap(Go) ≈ eigmin(Hermitian(gram2))
                    @test inverse(Go) ≈ inv(gram2 + 2I)
                    Go = ShiftedGramMatrix(Go; new_factor=Q, overwrite=true, normalize=true)
                    @test Go.gram ≈ gramn
                    @test gramgap(Go) ≈ eigmin(Hermitian(gramn))
                    @test inverse(Go) ≈ inv(gramn + 2I)
                    gap_old = gramgap(Go)
                    Go = ShiftedGramMatrix(Go; new_shift=0, overwrite=true)
                    @test Go.shift === zero(F)
                    @test gramgap(Go) === gap_old
                    @test inverse(Go) ≈ inv(gramn)
                end
            end

            singular_cases = (
                ("Array", Complex{F}[1 1; 0 0]),
                ("SMatrix", SMatrix{2,2}(Complex{F}[1 1; 0 0])),
                ("SparseMatrixCSC", sparse(Complex{F}[1 1; 0 0])),
                ("Diagonal", Diagonal(Complex{F}[1, 0])),
                ("Diagonal{SVector}", Diagonal(SVector{2}(Complex{F}[1, 0]))),
            )
            for (name, Q) in singular_cases
                @testset "F = $F, singular $name" begin
                    @test_throws PosDefException ShiftedGramMatrix(Q)
                    G = ShiftedGramMatrix(Q; check=false)
                    @test !issuccess(G)
                    @test gramgap(G) === zero(F)

                    # Failure of the bare factorization doesn't throw when shifted
                    Gs = ShiftedGramMatrix(Q; shift=1)
                    @test issuccess(Gs)
                    @test gramgap(Gs) === zero(F)

                    G = ShiftedGramMatrix(G; new_shift=1, overwrite=true)
                    @test issuccess(G)
                    @test gramgap(G) === zero(F)
                    G = ShiftedGramMatrix(G; new_factor=Q, overwrite=true)
                    @test issuccess(G)
                    @test G.shift === one(F)
                    @test gramgap(G) === zero(F)
                    @test_throws PosDefException ShiftedGramMatrix(
                        G; new_factor=Q, new_shift=0
                    )
                end
            end

            # 1x1 static matrices have their own eigmin path
            G = ShiftedGramMatrix(SMatrix{1,1}(Complex{F}[2]); shift=1)
            @test gramgap(G) == 4
        end
    end
end
