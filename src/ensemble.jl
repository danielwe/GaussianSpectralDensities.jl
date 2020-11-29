const MatrixOrU{R} = Union{AbstractMatrix{R}, UniformScaling{R}}
const RealOrComplex{R} = Union{R, Complex{R}}

abstract type AbstractMatrixEnsemble{R<:Real} end

struct MatrixEnsemble{
    R,
    U<:MatrixOrU{<:RealOrComplex{R}},
    V<:MatrixOrU{<:RealOrComplex{R}},
    W<:MatrixOrU{<:RealOrComplex{R}},
    X<:MatrixOrU{Complex{R}},
    Y<:MatrixOrU{Complex{R}},
} <: AbstractMatrixEnsemble{R}
    M::U
    L::V
    R::W
    L⁻¹R⁻¹::X
    L⁻¹MR⁻¹::Y
    T::Vector{R}

    function MatrixEnsemble(M::U, L::V, R::W, T) where {U, V, W}
        N = size(M, 1)
        if size(M, 2) != N
            throw(DimensionMismatch("M is not square"))
        end
        for (A, label) in ((L, "L"), (R, "R"))
            if !(typeof(A) <: UniformScaling) && (size(A) != (N, N))
                throw(DimensionMismatch("the size of $label does not match M"))
            end
        end
        if length(T) != N
            throw(DimensionMismatch("the length of T does not match M"))
        end
        L⁻¹R⁻¹ = _complex((L \ I) / R)
        L⁻¹MR⁻¹ = _complex((L \ M) / R)
        return new{real(eltype(M)), U, V, W, typeof(L⁻¹R⁻¹), typeof(L⁻¹MR⁻¹)}(
            M, L, R, L⁻¹R⁻¹, L⁻¹MR⁻¹, T
        )
    end
end

function MatrixEnsemble(M, L, R, τ::Real)
    return MatrixEnsemble(M, L, R, fill(sqrt(τ), size(M, 1)))
end

_complex(x::UniformScaling) = UniformScaling(complex(x.λ))
_complex(x) = complex(x)
