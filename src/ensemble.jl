# TODO: consider generalizations where M, L, R can be LinearMaps etc.
abstract type AbstractMatrixEnsemble{
    F<:AbstractFloat,
    TM<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
    TL<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
    TR<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
} end

"""

    MatrixEnsemble(M, L, R, T)
    MatrixEnsemble{F}(M, L, R, T) where {F<:AbstractFloat}

Ensemble of square matrices on the form ``M + LJR`` where ``J`` has Girko ``T``-distribution

The Girko ``T``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``T``. The ordinary Girko ensemble is recovered by
choosing `T = sqrt(τ) * I`.

All of ``M, L, R`` must be square and of the same size, or `UniformScaling`. In addition,
``L`` and ``R`` must be nonsingular.

All constructors support an optional type parameter `F` to enforce the floating point type.
"""
struct MatrixEnsemble{
    F,
    TM,
    TL,
    TR,
    TT<:DiagonalOrUniform{F},
    TLiRi<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
    TLiMRi<:AbstractMatrixOrUniform{<:RealOrComplex{F}},
} <: AbstractMatrixEnsemble{F,TM,TL,TR}
    M::TM
    L::TL
    R::TR
    T::TT
    LiRi::TLiRi
    LiMRi::TLiMRi

    function MatrixEnsemble{F}(M, L, R, T) where {F}
        sizes = NamedTuple(
            name => size(A)
            for (name, A) in ((:M, M), (:L, L), (:R, R), (:T, T)) if applicable(size, A)
        )
        if length(sizes) > 0
            function throwsizeerror()
                msg = (
                    "M, L, R, and T must all be square matrices of the same size or "
                    * "UniformScaling. Got: "
                    * join(("size($name) == $s" for (name, s) in pairs(sizes)), "; ")
                )
                throw(DimensionMismatch(msg))
            end
            s = first(sizes)
            N = first(s)
            (s == (N, N)) || throwsizeerror()
            for nexts in Base.tail(sizes)
                (nexts == s) || throwsizeerror()
            end
        end
        if !((T isa DiagonalOrUniform{<:Real}) && _inunitinterval(T))
            msg = "T must be Diagonal or UniformScaling with elements in [0, 1]"
            throw(ArgumentError(msg))
        end
        M, L, R = _floatconvert.(F, (M, L, R))
        T = _elconvert(F, T)  # Always real
        LiRi =_floatconvert(F, _rdiv(_ldiv(L, I), R))
        LiMRi = _floatconvert(F, _rdiv(_ldiv(L, M), R))
        return new{F,map(typeof, (M, L, R, T, LiRi, LiMRi))...}(M, L, R, T, LiRi, LiMRi)
    end
end

MatrixEnsemble(M, L, R, T) = MatrixEnsemble{_floattype(M, L, R, T)}(M, L, R, T)

_inunitinterval(a::Number) = ((imag(a) == 0) && (0 <= real(a) <= 1))
_inunitinterval(v::AbstractVector) = all(_inunitinterval(a) for a in v)
_inunitinterval(A::Diagonal) = _inunitinterval(A.diag)
_inunitinterval(A::UniformScaling) = _inunitinterval(A.λ)

function _floattype(M, L, R, T)
    Mu, Lu, Ru, Tu = (oneunit ∘ eltype).((M, L, R, T))
    return real(typeof((Mu + (Lu * (Tu * Ru))) / (Lu * Ru)))
end

function _elconvert(::Type{E}, A) where {E}
    (E === eltype(A)) && return A
    return convert.(E, A)
end

# Work around lack of structure-preserving broadcast with SArray storage
function _elconvert(::Type{E}, A::Diagonal) where {E}
    (E === eltype(A)) && return A
    return Diagonal(convert.(E, A.diag))
end

_elconvert(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

function _floatconvert(::Type{F}, A) where {F<:Real}
    (eltype(A) <: Complex) && return _elconvert(Complex{F}, A)
    return _elconvert(F, A)
end

_ldiv(L, A) = L \ A
_ldiv(L, A::UniformScaling) = inv(L) * A  # Work around StaticMatrix \ UniformScaling issue

_rdiv(A, R) = A / R
_rdiv(A::UniformScaling, R) = A * inv(R)  # Should file an issue
