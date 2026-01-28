module ArrayTools

# TODO: unit tests?

using Adapt: WrappedArray
using LinearAlgebra: Diagonal, Hermitian, UniformScaling
using StaticArrays: @SVector, SArray, SVector

public AbstractMatrixOrUniform,
    AnySArray,
    AnySMatrix,
    DiagonalOrUniform,
    RealOrComplex,
    adapteltype,
    adaptrealtype,
    all_diagonal,
    copy!!,
    map_diagonal,
    withsize

## Number type aliases
## ===================
const RealOrComplex{T<:Real} = Union{T,Complex{T}}

## Array/operator type aliases
## ===========================
const AbstractMatrixOrUniform{T} = Union{AbstractMatrix{T},UniformScaling{T}}
const DiagonalOrUniform{T} = Union{Diagonal{T},UniformScaling{T}}

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

## Array copy helper
## =================
"""
    copy!!(dest, src)

Overwrite `dest` with the contents of `src` and return `dest`
"""
copy!!(dest, src) = copyto!(dest, src)

"""
    copy!!(::AnySArray, src)

Return a copy of `src`
"""
copy!!(::AnySArray, src) = deepcopy(src)

## Element type adaptation
## =======================
function adapteltype(::Type{E}, A) where {E}
    (E === eltype(A)) && return A
    return map(Ai -> convert(E, Ai), A)
end

adapteltype(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

# Work around lack of structure-preserving map with SArray storage
adapteltype(::Type{E}, A::Diagonal) where {E} = Diagonal(adapteltype(E, parent(A)))

## Float type adaptation
## =====================
function adaptrealtype(::Type{F}, A) where {F}
    (F <: Real) || throw(ArgumentError("adaptrealtype(F, A) requires F<:Real"))
    # Fallback assumes eltype(A) exists
    E = (eltype(A) <: Complex) ? Complex{F} : F
    return adapteltype(E, A)
end

## Diagonal mapping and reduction
## ==============================
map_diagonal(f, A::Diagonal) = Diagonal(map(f, parent(A)))
map_diagonal(f, A::UniformScaling) = UniformScaling(f(A.λ))

all_diagonal(f, A::Diagonal) = all(f, parent(A))
all_diagonal(f, A::UniformScaling) = f(A.λ) ? true : false  # use ? : to trigger appropriate error when f returns non-Bool

## Instantiate sized representation of A with size and type derived from B
## =======================================================================
# TODO: try to eliminate the need for this function by supporting unsized UniformScaling in
# MatrixEnsemble and all the way through the Workspace calculations
withsize(A::UniformScaling, ::UniformScaling) = Diagonal(SVector{1}(A.λ))
withsize(A::UniformScaling, B::AnySMatrix) = Diagonal(@SVector fill(A.λ, size(B, 1)))
withsize(A::UniformScaling, B::AbstractMatrix) = A(size(B, 1))
withsize(A::AbstractMatrix, ::AbstractMatrixOrUniform) = A

end  # ArrayTools
