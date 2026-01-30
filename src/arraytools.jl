module ArrayTools

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

module TestArrayTools
    using Test: @test, @test_throws, @testset
    using GaussianSpectralDensities.ArrayTools:
        AbstractMatrixOrUniform,
        AnySMatrix,
        DiagonalOrUniform,
        RealOrComplex,
        adapteltype,
        adaptrealtype,
        all_diagonal,
        copy!!,
        map_diagonal,
        withsize

    using LinearAlgebra: Diagonal, Hermitian, I, UniformScaling
    using StaticArrays: @SMatrix, @SVector, SMatrix, SVector

    @testset "type aliases" begin
        @test Int <: RealOrComplex{Int}
        @test Complex{Float64} <: RealOrComplex{Float64}
        @test Matrix{Float64} <: AbstractMatrixOrUniform{Float64}
        @test UniformScaling{Int} <: AbstractMatrixOrUniform{Int}
        @test Diagonal{Float64} <: DiagonalOrUniform{Float64}
        @test SMatrix{2,2,Float64} <: AnySMatrix{Float64}
        @test Hermitian{Float64,SMatrix{2,2,Float64,4}} <: AnySMatrix{Float64}
    end

    @testset "copy!!" begin
        dest = zeros(3)
        src = [1.0, 2.0, 3.0]
        @test copy!!(dest, src) === dest
        @test dest == src

        sdest = @SVector zeros(3)
        @test copy!!(sdest, src) == src
        @test copy!!(sdest, src) !== sdest  # returns a copy, not original
    end

    @testset "adapteltype" begin
        A = [1, 2, 3]
        @test adapteltype(Int, A) === A  # no conversion needed
        @test adapteltype(Float64, A) == [1.0, 2.0, 3.0]
        @test eltype(adapteltype(Float64, A)) === Float64

        @test adapteltype(Float32, 3I) === UniformScaling(3.0f0)
        @test adapteltype(Float64, Diagonal([1, 2])) == Diagonal([1.0, 2.0])
    end

    @testset "adaptrealtype" begin
        A = [1.0, 2.0]
        @test adaptrealtype(Float32, A) == Float32[1, 2]

        B = Complex{Float64}[1 + 2im, 3 + 4im]
        @test eltype(adaptrealtype(Float32, B)) === Complex{Float32}

        @test_throws ArgumentError adaptrealtype(Complex{Float64}, A)
    end

    @testset "map_diagonal / all_diagonal" begin
        D = Diagonal([1, 2, 3])
        @test map_diagonal(x -> 2x, D) == Diagonal([2, 4, 6])
        @test map_diagonal(x -> 2x, 5I) === UniformScaling(10)

        @test all_diagonal(>(0), D)
        @test !all_diagonal(>(2), D)
        @test all_diagonal(==(1), 1I)
        @test !all_diagonal(==(2), 1I)
    end

    @testset "withsize" begin
        B_static = @SMatrix zeros(3, 3)
        B_dense = zeros(3, 3)

        @test withsize(2I, B_static) == Diagonal(@SVector fill(2, 3))
        @test withsize(2I, B_dense) == Diagonal(fill(2, 3))
        @test withsize(2I, I) == Diagonal(SVector{1}(2))
        @test withsize(B_dense, I) === B_dense
    end
end
