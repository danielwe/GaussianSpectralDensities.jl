# TODO: consider generalizations where M, L, R can be LinearMaps etc.
abstract type AbstractMatrixEnsemble{
    F<:AbstractFloat,
    TM<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
    TL<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
    TR<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
} end

"""

    MatrixEnsemble([N::Integer], M, L, R, T)
    MatrixEnsemble{F}([N::Integer], M, L, R, T)

Ensemble of square matrices on the form ``M + LJR`` where ``J`` has Girko ``T``-distribution

Providing the size `N` is not necessary as long as it can be inferred from one of `M`, `L`,
`R`, or `T`, but is required if, for example, all of them are of `UniformScaling` type.

The Girko ``T``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``T``. The ordinary Girko ensemble is recovered by
choosing `T = sqrt(τ) * I`.

All constructors support an optional type parameter `F` to enforce the floating point type.
"""
struct MatrixEnsemble{
    F,
    TM,
    TL,
    TR,
    TT<:DiagonalOrUniform{F},
    TLiRi<:AbstractMatrixOrUniform{Complex{F}},
    TLiMRi<:AbstractMatrix{Complex{F}},
} <: AbstractMatrixEnsemble{F,TM,TL,TR}
    N::Int
    M::TM
    L::TL
    R::TR
    T::TT
    LiRi::TLiRi
    LiMRi::TLiMRi

    function MatrixEnsemble{F}(N, M, L, R, T) where {F}
        for (A, name) in ((M, 'M'), (L, 'L'), (R, 'R'), (T, 'T'))
            if applicable(size, A) && (size(A) != (N, N))
                throw(DimensionMismatch("size($name) must be (N, N) == ($N, $N)"))
            end
        end
        if !((T isa DiagonalOrUniform{<:Real}) && _ispossemidef(T))
            msg = "T must be Diagonal or UniformScaling and positive semidefinite"
            throw(ArgumentError(msg))
        end
        M, L, R = _floatconvert.(F, (M, L, R))
        T = _elconvert(F, T)  # Always real
        LiMRi = _elconvert(Complex{F}, (L \ M) / R)  # Always complex---though, necessary?
        LiRi =_elconvert(Complex{F}, (L \ I) / R)
        if LiMRi isa UniformScaling
            LiMRi = LiMRi(N)
        end
        return new{F,map(typeof, (M, L, R, T, LiRi, LiMRi))...}(N, M, L, R, T, LiRi, LiMRi)
    end
end

function MatrixEnsemble{F}(M, L, R, T) where {F}
    N = _apply_earliest(size, (M, L, R, T), 1)
    if N === nothing
        msg = (
            "ensemble size N cannot be inferred from M, L, R, or T, so must be supplied "
            * "manually using the signature MatrixEnsemble(N, M, L, R, T)"
        )
        throw(ArgumentError(msg))
    end
    return MatrixEnsemble{F}(N, M, L, R, T)
end

MatrixEnsemble(M, L, R, T) = MatrixEnsemble{_floattype(M, L, R, T)}(M, L, R, T)
MatrixEnsemble(N, M, L, R, T) = MatrixEnsemble{_floattype(M, L, R, T)}(N, M, L, R, T)

_ispossemidef(a::Number) = ((imag(a) == 0) && (real(a) >= 0))
_ispossemidef(A::Diagonal) = all(_ispossemidef(a) for a in A.diag)
_ispossemidef(A::UniformScaling) = _ispossemidef(A.λ)

function _floatconvert(::Type{F}, A) where {F<:Real}
    (eltype(A) <: Complex) && return _elconvert(Complex{F}, A)
    return _elconvert(F, A)
end

function _elconvert(::Type{E}, A) where {E}
    (E === eltype(A)) && return A
    return convert.(E, A)
end

_elconvert(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

function _apply_earliest(f, As, args...)
    A = first(As)
    applicable(f, A, args...) && return f(A, args...)
    return _apply_earliest(f, Base.tail(As), args...)
end

_apply_earliest(f, ::Tuple{}, args...) = nothing

function _floattype(M, L, R, T)
    Mu, Lu, Ru, Tu = (oneunit ∘ eltype).((M, L, R, T))
    return real(typeof((Mu + (Lu * (Tu * Ru))) / (Lu * Ru)))
end
