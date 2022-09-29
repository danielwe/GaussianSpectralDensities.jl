const _AbstractMatrixOrUniform{F} = Union{<:AbstractMatrix{F},UniformScaling{F}}
const _RealOrComplex{F<:Real} = Union{F,Complex{F}}

# TODO: consider generalizations where M, L, R can be LinearMaps etc.
"""
    MatrixEnsemble(M, L, R, T)
    MatrixEnsemble{F}(M, L, R, T) where {F<:AbstractFloat}

Ensemble of square matrices on the form ``M + LJR`` where ``J`` has Girko ``T``-distribution

The Girko ``T``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``T``. The ordinary Girko ensemble is recovered by
choosing `T = sqrt(τ) * I`.

All of ``M, L, R`` must be square and of the same size, or `UniformScaling`. In addition,
``L`` and ``R`` must be nonsingular.
"""
struct MatrixEnsemble{
    F<:AbstractFloat,
    TM<:_AbstractMatrixOrUniform{<:_RealOrComplex{F}},
    TL<:_AbstractMatrixOrUniform{<:_RealOrComplex{F}},
    TR<:_AbstractMatrixOrUniform{<:_RealOrComplex{F}},
    TT<:DiagonalOrUniform{F},
    TC<:AbstractMatrix{<:_RealOrComplex{F}},
}
    M::TM
    L::TL
    R::TR
    T::TT
    LiRi::TC
    LiMRi::TC

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
        if !((T isa DiagonalOrUniform) && _inunitdisc(T))
            msg = "T must be Diagonal or UniformScaling with elements in the unit disc"
            throw(ArgumentError(msg))
        end
        M, L, R = _floatconvert.(F, (M, L, _phase(T) * R))
        T = _floatconvert(F, _abs(T))
        LiMRi = _sized(_floatconvert(F, _rdiv(_ldiv(L, M), R)), T)
        LiRi = _sized(_floatconvert(F, _ldiv(R * L, I)), LiMRi)
        return new{F,typeof.((M, L, R, T, LiMRi))...}(M, L, R, T, LiRi, LiMRi)
    end
end

MatrixEnsemble(M, L, R, T) = MatrixEnsemble{_floattype(M, L, R, T)}(M, L, R, T)

## Traits, comparisons, conversions
## ================================
Base.size(e::MatrixEnsemble) = size(e.LiMRi)

function Base.:(==)(a::E, b::E) where {E<:MatrixEnsemble}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, e::Q) where {F,P<:MatrixEnsemble{F},Q<:MatrixEnsemble}
    (e isa P) && return e
    return MatrixEnsemble{F}(e.M, e.L, e.R, e.T)
end

## Helpers
## =======
_inunitdisc(a::Number) = (abs2(a) <= 1)
_inunitdisc(v::AbstractVector) = all(_inunitdisc, v)
_inunitdisc(A::Diagonal) = _inunitdisc(A.diag)
_inunitdisc(A::UniformScaling) = _inunitdisc(A.λ)

_phase(a::Number) = (a == 0) ? one(a) : a / abs(a)  # like sign except when a == 0
_phase(A::UniformScaling) = UniformScaling(_phase(A.λ))

function _phase(A::Diagonal)
    if all(a -> ((imag(a) == 0) && (real(a) >= 0)), A.diag)
        return UniformScaling(one(eltype(A)))
    end
    return Diagonal(_phase.(A.diag))
end

_abs(A::Diagonal) = Diagonal(abs.(A.diag))
_abs(A::UniformScaling) = UniformScaling(abs(A.λ))

function _floattype(M, L, R, T)
    Mu, Lu, Ru, Tu = (oneunit ∘ eltype).((M, L, R, T))
    return real(typeof((Mu + (Lu * (Tu * Ru))) / (Lu * Ru)))
end

function _floatconvert(::Type{F}, A) where {F<:Real}
    (eltype(A) <: Complex) && return _elconvert(Complex{F}, A)
    return _elconvert(F, A)
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

_ldiv(L, A) = L \ A
_ldiv(L::AnySMatrix, A::UniformScaling) = L \ SMatrix{size(L, 1),size(L, 1)}(A)  # should file issue
_ldiv(L::Diagonal{<:Any,<:SArray}, A::UniformScaling) = L \ _sized(A, L)
_ldiv(L::SparseMatrixCSC, A::UniformScaling) = L \ _sized(A, L)

_rdiv(A, R) = A / R
_rdiv(A::UniformScaling, R::AnySMatrix) = SMatrix{size(R, 2),size(R, 2)}(A) / R  # should file issue
_rdiv(A::UniformScaling, R::Diagonal{<:Any,<:SArray}) = _sized(A, R') / R
_rdiv(A::UniformScaling, R::SparseMatrixCSC) = _sized(A, R') / R

_sized(A::UniformScaling, T::UniformScaling) = Diagonal(SVector{1}(A.λ))
_sized(A::UniformScaling, T::AnySMatrix) = Diagonal(@SVector fill(A.λ, size(T, 1)))
_sized(A::UniformScaling, T) = A(size(T, 1))
_sized(A, T) = A
