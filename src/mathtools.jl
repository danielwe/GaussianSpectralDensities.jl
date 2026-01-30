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
using ..Defaults: Defaults

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

samesign(x::Number, y::Number) = (x * y > 0)  # XXX: probably not robust (Inf/NaN etc.)

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
    schur, history = partialschur(Ai; which, nev=1, tol=Defaults.tol(eltype(Ai)))
    (history.nconverged >= 1) || error("smallest eigenvalue not found: did not converge")
    return inv(real(first(schur.eigenvalues)))
end

eigmin!!(_, A::Diagonal{<:Real}) = minimum(parent(A))
eigmin!!(_, A::Diagonal{<:Real,<:SArray}) = minimum(parent(A))  # Ambiguity resolution
eigmin!!(_, A::AnySMatrix) = eigmin(A)

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

@testitem "MathTools" begin
    using GaussianSpectralDensities.MathTools:
        decompose,
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

    using LinearAlgebra:
        Bidiagonal,
        Cholesky,
        Diagonal,
        Hermitian,
        I,
        SymTridiagonal,
        Tridiagonal,
        cholesky,
        tr
    using SparseArrays: sparse
    using StaticArrays: @SMatrix, @SVector, SMatrix

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

    @testset "decompose / inverse" begin
        A = [4.0 2.0; 2.0 5.0]
        H = Hermitian(A)
        F = decompose(H)
        @test F isa Cholesky
        @test inverse(F) ≈ inv(A)

        D = Diagonal([2.0, 3.0])
        @test decompose(D) == D
        @test inverse(decompose(D)) == Diagonal([0.5, 1 / 3])

        # Sparse decompose/inverse
        Asp = sparse(A)
        Hsp = Hermitian(Asp)
        Fsp = decompose(Hsp)
        @test inverse(Fsp) ≈ inv(A)

        # Static arrays
        As = @SMatrix [4.0 2.0; 2.0 5.0]
        Hs = Hermitian(As)
        Fs = decompose(Hs)
        @test inverse(Fs) ≈ inv(A)

        Ds = Diagonal(@SVector [2.0, 3.0])
        @test decompose(Ds) == Ds
        @test inverse(decompose(Ds)) == Diagonal([0.5, 1 / 3])

        # Static Hermitian with :L uplo (triggers workaround branch)
        HsL = Hermitian(@SMatrix([4.0 0.0; 2.0 5.0]), :L)
        FsL = decompose(HsL)
        @test inverse(FsL) ≈ inv([4.0 2.0; 2.0 5.0])
    end

    @testset "decompose!! / inverse!!" begin
        A = [4.0 2.0; 2.0 5.0]
        H = Hermitian(A)
        F = decompose(H)
        Ai = inverse(F)

        A2 = [5.0 1.0; 1.0 6.0]
        H2 = Hermitian(A2)
        F2 = decompose!!(F, H2)
        @test F2 === F
        Ai2 = inverse!!(Ai, F2)
        @test Ai2 ≈ inv(A2)

        # With shift
        F3 = decompose!!(F, H; shift=1.0)
        @test F3 === F
        @test inverse(F3) ≈ inv(A + I)

        # Diagonal decompose!!/inverse!!
        D = Diagonal([2.0, 3.0])
        Fd = decompose(D)
        Di = inverse(Fd)
        D2 = Diagonal([4.0, 5.0])
        Fd2 = decompose!!(Fd, D2)
        @test Fd2 === Fd
        Di2 = inverse!!(Di, Fd2)
        @test Di2 ≈ inv(D2)

        # Diagonal with shift
        Fd3 = decompose!!(Fd, D; shift=1.0)
        @test Fd3 == D + I

        # Sparse decompose!!/inverse!!
        Asp = sparse(A)
        Hsp = Hermitian(Asp)
        Fsp = decompose(Hsp)
        Aisp = inverse(Fsp)
        Asp2 = sparse(A2)
        Hsp2 = Hermitian(Asp2)
        Fsp2 = decompose!!(Fsp, Hsp2)
        @test Fsp2 === Fsp
        Aisp2 = inverse!!(Aisp, Fsp2)
        @test Aisp2 ≈ inv(A2)

        # Static array decompose!!/inverse!! (out-of-place)
        As = @SMatrix [4.0 2.0; 2.0 5.0]
        Hs = Hermitian(As)
        Fs = decompose(Hs)
        Ais = inverse(Fs)
        As2 = @SMatrix [5.0 1.0; 1.0 6.0]
        Hs2 = Hermitian(As2)
        Fs2 = decompose!!(Fs, Hs2)
        @test inverse(Fs2) ≈ inv(A2)
        Ais2 = inverse!!(Ais, Fs2)
        @test Ais2 ≈ inv(A2)

        Ds = Diagonal(@SVector [2.0, 3.0])
        Fds = decompose(Ds)
        Dis = inverse(Fds)
        Ds2 = Diagonal(@SVector [4.0, 5.0])
        Fds2 = decompose!!(Fds, Ds2)
        @test Fds2 == Ds2
        Dis2 = inverse!!(Dis, Fds2)
        @test Dis2 ≈ inv(D2)
    end

    @testset "ldiv!!" begin
        F = cholesky([4.0 2.0; 2.0 5.0])
        B = [1.0 0.0; 0.0 1.0]
        @test ldiv!!(F, copy(B)) ≈ F \ B

        Bs = @SMatrix [1.0 0.0; 0.0 1.0]
        @test ldiv!!(F, Bs) ≈ F \ B

        # Sparse CHOLMOD Factor
        A = [4.0 2.0; 2.0 5.0]
        Fsp = decompose(Hermitian(sparse(A)))
        @test ldiv!!(Fsp, copy(B)) ≈ A \ B
        @test ldiv!!(Fsp, Bs) ≈ A \ B
    end

    @testset "eigmin!!" begin
        A = Hermitian([4.0 1.0; 1.0 3.0])
        F = decompose(A)
        λmin = eigmin!!(F, A)
        @test λmin ≈ (7 - sqrt(5)) / 2 atol = 1e-6

        D = Diagonal([2.0, 5.0, 3.0])
        @test eigmin!!(nothing, D) == 2.0

        # Static Diagonal (ambiguity resolution method)
        Ds = Diagonal(@SVector [2.0, 5.0, 3.0])
        @test eigmin!!(nothing, Ds) == 2.0

        # Static Hermitian matrix
        As = Hermitian(@SMatrix [4.0 1.0; 1.0 3.0])
        @test eigmin!!(nothing, As) ≈ (7 - sqrt(5)) / 2 atol = 1e-6

        # Sparse Hermitian
        Asp = Hermitian(sparse([4.0 1.0; 1.0 3.0]))
        Fsp = decompose(Asp)
        @test eigmin!!(Fsp, Asp) ≈ (7 - sqrt(5)) / 2 atol = 1e-6
    end
end
