# TODO: consider generalizations where M, L, R can be LinearMaps etc.
const AbstractMatrixOrUniform{T} = Union{<:AbstractMatrix{<:T},UniformScaling{<:T}}
const DiagonalOrUniform{T} = Union{<:Diagonal{<:T},UniformScaling{<:T}}
const RealOrComplex{T<:Real} = Union{T,Complex{T}}

abstract type AbstractMatrixEnsemble{
    F<:AbstractFloat,
    TM<:AbstractMatrixOrUniform{RealOrComplex{F}},
    TL<:AbstractMatrixOrUniform{RealOrComplex{F}},
    TR<:AbstractMatrixOrUniform{RealOrComplex{F}},
} end

"""

    MatrixEnsemble([N::Integer], M, L, R, T)

Ensemble of square matrices on the form ``M + LJR`` where ``J`` has Girko ``T``-distribution

Providing the size `N` is not necessary as long as it can be inferred from one of `M`, `L`,
`R`, or `T`, but is required if, for example, all of them are of `UniformScaling` type.

The Girko ``T``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``T``. The ordinary Girko ensemble is recovered by
choosing `T = sqrt(τ) * I`.
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

    function MatrixEnsemble(N, M, L, R, T)
        for (A, name) in ((M, 'M'), (L, 'L'), (R, 'R'), (T, 'T'))
            if applicable(size, A) && (size(A) != (N, N))
                throw(DimensionMismatch("size($name) must be (N, N) == ($N, $N)"))
            end
        end
        if !((T isa DiagonalOrUniform{<:Real}) && _ispossemidef(T))
            msg = "T must be Diagonal or UniformScaling and positive semidefinite"
            throw(ArgumentError(msg))
        end
        # Include eltypes of R and T early by pre-promoting L; avoids surprises if, e.g.,
        # M == L == I isa UniformScaling{Bool}, but eltype(R) === Float32, in which case the
        # desired behavior is to promote to Float32 but L \ M would give Float64
        L = _elpromote(promote_type(eltype(R), eltype(T)), L)
        LiMRi = _complex(_float((L \ M) / R))
        if LiMRi isa UniformScaling
            LiMRi = LiMRi(N)
        end
        CF = eltype(LiMRi)
        LiRi =_elconvert(CF, (L \ I) / R)  # Always complex
        F = real(CF)
        M, L, R = map(A -> _elpromote(F, A), (M, L, R))
        T = _elconvert(F, T)  # Always real
        return new{F,map(typeof, (M, L, R, T, LiRi, LiMRi))...}(N, M, L, R, T, LiRi, LiMRi)
    end
end

function MatrixEnsemble(M, L, R, T)
    N = _apply_earliest(size, (M, L, R, T), 1)
    if N === nothing
        msg = (
            "ensemble size N cannot be inferred from M, L, R, or T, so must be supplied "
            * "manually using the signature MatrixEnsemble(N, M, L, R, T)"
        )
        throw(ArgumentError(msg))
    end
    return MatrixEnsemble(N, M, L, R, T)
end

_ispossemidef(a::Number) = ((imag(a) == 0) && (real(a) >= 0))
_ispossemidef(A::Diagonal) = all(_ispossemidef(a) for a in A.diag)
_ispossemidef(A::UniformScaling) = _ispossemidef(A.λ)

_complex(A) = _elconvert(complex(eltype(A)), A)

_float(A) = _elconvert(float(eltype(A)), A)

_elpromote(T, A) = _elconvert(promote_type(T, eltype(A)), A)

function _elconvert(T, A)
    (T === eltype(A)) && return A
    return convert.(T, A)
end

_elconvert(T, A::UniformScaling) = UniformScaling(convert(T, A.λ))

function _apply_earliest(f, As, args...; kwargs...)
    A = first(As)
    applicable(f, A, args...; kwargs...) && return f(A, args...; kwargs...)
    return _apply_earliest(f, Base.tail(As), args...; kwargs...)
end

_apply_earliest(f, ::Tuple{}, args...; kwargs...) = nothing
