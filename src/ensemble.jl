const AMOrI{R} = Union{AbstractMatrix{R}, UniformScaling{R}}
const RealOrComplex{R<:Real} = Union{R, Complex{R}}

abstract type AbstractMatrixEnsemble{
    R<:Real,
    U<:AMOrI{<:RealOrComplex{R}},
    V<:AMOrI{<:RealOrComplex{R}},
    W<:AMOrI{<:RealOrComplex{R}},
} end

struct MatrixEnsemble{
    R, U, V, W, X<:AMOrI{Complex{R}}, Y<:AbstractMatrix{Complex{R}},
} <: AbstractMatrixEnsemble{R, U, V, W}
    N::Int
    M::U
    L::V
    R::W
    L⁻¹R⁻¹::X
    L⁻¹MR⁻¹::Y
    T::Vector{R}

    function MatrixEnsemble(M::U, L::V, R::W, T::AbstractVector) where {U, V, W}
        N = length(T)
        for (A, c) in ((M, 'M'), (L, 'L'), (R, 'R'))
            if !(A isa UniformScaling) && (size(A) != (N, N))
                throw(DimensionMismatch("the size of $c does not match the length of T"))
            end
        end
        Lf, Rf = _factorize(L), _factorize(_complex(R))
        L⁻¹R⁻¹ = A⁻¹BC⁻¹(Lf, I, Rf)
        L⁻¹MR⁻¹ = A⁻¹BC⁻¹(Lf, M, Rf)
        if L⁻¹MR⁻¹ isa UniformScaling
            L⁻¹MR⁻¹ = L⁻¹MR⁻¹(N)
        end
        return new{real(eltype(M)), U, V, W, typeof(L⁻¹R⁻¹), typeof(L⁻¹MR⁻¹)}(
            N, M, L, R, L⁻¹R⁻¹, L⁻¹MR⁻¹, T
        )
    end
end

function MatrixEnsemble(M, L, R, τ::Real)
    return MatrixEnsemble(M, L, R, fill(sqrt(τ), size(M, 1)))
end

_complex(A::UniformScaling) = UniformScaling(complex(A.λ))
_complex(A) = complex(A)

function _factorize(A)
    if applicable(factorize, A)
        return factorize(A)
    end
    return A
end

function A⁻¹BC⁻¹(A, B, C)
    tmp = B / C
    if applicable(ldiv!, A, tmp)
        ldiv!(A, tmp)
        return tmp
    end
    return A \ tmp
end

function A⁻¹BC⁻¹(A::UniformScaling, B, C::UniformScaling)
    return B / (A * C)
end

function Base.eltype(ensemble::AbstractMatrixEnsemble{R, U, V, W}) where {R, U, V, W}
    if any(T -> eltype(T) <: Complex, (U, V, W))
        return Complex{R}
    end
    return R
end
