module Internals

using Adapt, LinearAlgebra, StaticArrays, SparseArrays

export RealOrComplex
export AbstractMatrixOrUniform, DiagonalOrUniform, AnySArray, AnySMatrix
export isapprox, mapdiag, alldiag, inunitdisc, phase, converteltype, sizedmatrix

@static if VERSION >= v"1.11.0-DEV.469"
    # `public ldiv rdiv` without getting parse errors in 1.10
    eval(Expr(:public, :ldiv, :rdiv))
end

## Number type unions
## ==================
const RealOrComplex{T<:Real} = Union{T,Complex{T}}

## Array/operator type unions
## ==========================
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

## Exported helpers
## ================
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

# Work around lack of structure-preserving broadcast with SArray storage
converteltype(::Type{E}, A::Diagonal) where {E} = Diagonal(converteltype(E, A.diag))
converteltype(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

sizedmatrix(A::UniformScaling, ::UniformScaling) = Diagonal(SVector{1}(A.λ))
sizedmatrix(A::UniformScaling, T::AnySMatrix) = Diagonal(@SVector fill(A.λ, size(T, 1)))
sizedmatrix(A::UniformScaling, T::AbstractMatrix) = A(size(T, 1))
sizedmatrix(A::AbstractMatrix, ::AbstractMatrixOrUniform) = A

## Public unexported helpers
## =========================
ldiv(L, A) = L \ A
ldiv(L::AnySMatrix, A::UniformScaling) = L \ SMatrix{size(L, 1),size(L, 1)}(A)  # should file issue
ldiv(L::Diagonal{<:Any,<:SArray}, A::UniformScaling) = L \ sizedmatrix(A, L)
ldiv(L::SparseMatrixCSC, A::UniformScaling) = L \ sizedmatrix(A, L)

rdiv(A, R) = A / R
rdiv(A::UniformScaling, R::AnySMatrix) = SMatrix{size(R, 2),size(R, 2)}(A) / R  # should file issue
rdiv(A::UniformScaling, R::Diagonal{<:Any,<:SArray}) = sizedmatrix(A, R') / R
rdiv(A::UniformScaling, R::SparseMatrixCSC) = sizedmatrix(A, R') / R

end
