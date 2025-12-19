# TODO: consider generalizations where M, L, R can be LinearMaps etc.
"""
    MatrixEnsemble(M, L, R, T)
    MatrixEnsemble{F}(M, L, R, T) where {F<:AbstractFloat}
    MatrixEnsemble{F}(e::MatrixEnsemble) where {F<:AbstractFloat}

Ensemble of square matrices on the form ``M + LJR`` where ``J`` is Girko ``T``-distributed

The Girko ``T``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``T``. The ordinary Girko ensemble is recovered by
choosing `T = sqrt(τ) * I`.

All of ``M, L, R`` must be square and of the same size, or `UniformScaling`, and
``L`` and ``R`` must be nonsingular.
"""
struct MatrixEnsemble{
    F<:AbstractFloat,
    RC<:Union{F,Complex{F}},
    TM<:AbstractMatrixOrUniform{RC},
    TL<:AbstractMatrixOrUniform{RC},
    TR<:AbstractMatrixOrUniform{RC},
    TT<:DiagonalOrUniform{F},
    TC<:AbstractMatrix{RC},
}
    M::TM
    L::TL
    R::TR
    T::TT
    LiRi::TC
    LiMRi::TC

    function MatrixEnsemble{F}(
        M::AbstractMatrixOrUniform,
        L::AbstractMatrixOrUniform,
        R::AbstractMatrixOrUniform,
        T::DiagonalOrUniform,
    ) where {F}
        sizes = NamedTuple(
            name => size(A)
            for (name, A) in ((:M, M), (:L, L), (:R, R), (:T, T)) if applicable(size, A)
        )
        if length(sizes) > 0
            N = first(first(sizes))
            for s in sizes
                (s == (N, N)) || throw(DimensionMismatch(
                    "M, L, R, and T must all be square matrices of the same size or UniformScaling. "
                    * "Got " * join(("size($name) == $s" for (name, s) in pairs(sizes)), "; ")
                ))
            end
        end
        alldiag(inunitdisc, T) || throw(ArgumentError("all elements of T must lie in the unit disc"))
        RC = any(eltype(A) <: Complex for A in (M, L, R, T)) ? Complex{F} : F
        Mrc, Lrc, Rrc = converteltype.(RC, (M, L, mapdiag(phase, T) * R))
        Tf = converteltype(F, mapdiag(abs, T))
        LiMRi = sizedmatrix(converteltype(RC, Internals.rdiv(Internals.ldiv(Lrc, Mrc), Rrc)), Tf)
        LiRi = sizedmatrix(converteltype(RC, Internals.ldiv(Rrc * Lrc, I)), LiMRi)
        return new{F,RC,typeof.((Mrc, Lrc, Rrc, Tf, LiMRi))...}(Mrc, Lrc, Rrc, Tf, LiRi, LiMRi)
    end
end

function MatrixEnsemble(M, L, R, T)
    Mu, Lu, Ru, Tu = (oneunit ∘ eltype).((M, L, R, T))
    RC = typeof((Mu + (Lu * (Tu * Ru))) / (Lu * Ru))
    return MatrixEnsemble{real(RC)}(M, L, R, T)
end

MatrixEnsemble{F}(e::MatrixEnsemble) where {F} = MatrixEnsemble{F}(e.M, e.L, e.R, e.T)

## Traits, comparisons, conversions
## ================================
Base.size(e::MatrixEnsemble) = size(e.LiMRi)

function Base.:(==)(a::E, b::E) where {E<:MatrixEnsemble}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, e::Q) where {F,P<:MatrixEnsemble{F},Q<:MatrixEnsemble}
    (e isa P) && return e
    return MatrixEnsemble{F}(e)
end

## Tests
## =====
@testitem "MatrixEnsemble with UniformScaling" begin
    using LinearAlgebra: I

    @testset "UniformScaling" begin
        ensemble = MatrixEnsemble(I, I, I, I)

        # Traits
        @test ensemble isa MatrixEnsemble{Float64}
        @test size(ensemble) == (1, 1)

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(I, I, I, I)

            # Floattype enforcing constructor
            @test ensembleF isa MatrixEnsemble{F}

            # Floattype conversions
            @test convert(MatrixEnsemble{F}, ensemble) === ensembleF
            @test convert(MatrixEnsemble{Float64}, ensembleF) === ensemble
            @test MatrixEnsemble{F}(ensemble) === ensembleF
            @test MatrixEnsemble{Float64}(ensembleF) === ensemble
        end
    end
end

@testitem "MatrixEnsemble with various array types" begin
    using LinearAlgebra: I, Diagonal
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    function isapprox(a::E, b::E) where {E<:MatrixEnsemble}
        return all(p -> getproperty(a, p) ≈ getproperty(b, p), propertynames(a))
    end

    A64data = ComplexF64[1.0 -1.0; 1.0 1.0]
    D64data = ComplexF64[0.75, 0.25*cispi(1 // 3)]
    N = length(D64data)

    @testset "$F" for F in [Float32, Float64]
        AFdata = convert.(Complex{F}, A64data)
        DFdata = convert.(Complex{F}, D64data)

        @testset "$label" for (label, A, D, AF, DF) in [
            ("Array", A64data, Diagonal(D64data), AFdata, Diagonal(DFdata)),
            ("SArray", SMatrix{N,N}(A64data), Diagonal(SVector{N}(D64data)), SMatrix{N,N}(AFdata), Diagonal(SVector{N}(DFdata))),
            ("SparseMatrixCSC", sparse(A64data), Diagonal(D64data), sparse(AFdata), Diagonal(DFdata)),
        ]
            ensemblesF = [
                MatrixEnsemble(AF, I, I, I),
                MatrixEnsemble(I, AF, I, I),
                MatrixEnsemble(I, I, AF, I),
                MatrixEnsemble(I, I, I, DF),
            ]
            ensembles64 = [
                MatrixEnsemble(A, I, I, I),
                MatrixEnsemble(I, A, I, I),
                MatrixEnsemble(I, I, A, I),
                MatrixEnsemble(I, I, I, D),
            ]

            # Size
            @test all(e -> size(e) == (N, N), ensemblesF)
            @test all(e -> size(e) == (N, N), ensembles64)

            # Floattype inferring constructor
            @test all(e -> e isa MatrixEnsemble{F}, ensemblesF)
            @test all(e -> e isa MatrixEnsemble{Float64}, ensembles64)

            # Floattype enforcing constructor
            @test isapprox(MatrixEnsemble{F}(A, I, I, I), ensemblesF[1])
            @test isapprox(MatrixEnsemble{F}(I, A, I, I), ensemblesF[2])
            @test isapprox(MatrixEnsemble{F}(I, I, A, I), ensemblesF[3])
            @test isapprox(MatrixEnsemble{F}(I, I, I, D), ensemblesF[4])
            @test isapprox(MatrixEnsemble{Float64}(AF, I, I, I), ensembles64[1])
            @test isapprox(MatrixEnsemble{Float64}(I, AF, I, I), ensembles64[2])
            @test isapprox(MatrixEnsemble{Float64}(I, I, AF, I), ensembles64[3])
            @test isapprox(MatrixEnsemble{Float64}(I, I, I, DF), ensembles64[4])

            # Floattype conversions
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[1]), ensemblesF[1])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[2]), ensemblesF[2])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[3]), ensemblesF[3])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[4]), ensemblesF[4])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[1]), ensembles64[1])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[2]), ensembles64[2])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[3]), ensembles64[3])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[4]), ensembles64[4])
            @test isapprox(MatrixEnsemble{F}(ensembles64[1]), ensemblesF[1])
            @test isapprox(MatrixEnsemble{F}(ensembles64[2]), ensemblesF[2])
            @test isapprox(MatrixEnsemble{F}(ensembles64[3]), ensemblesF[3])
            @test isapprox(MatrixEnsemble{F}(ensembles64[4]), ensemblesF[4])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[1]), ensembles64[1])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[2]), ensembles64[2])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[3]), ensembles64[3])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[4]), ensembles64[4])
        end
    end
end

@testitem "MatrixEnsemble errors" begin
    using LinearAlgebra: I, Diagonal

    Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
    Ddata = Diagonal(ComplexF64[0.75, 0.25*cispi(1 // 3)])
    N = length(Ddata.diag)

    @test_throws MethodError MatrixEnsemble(I, I, I, Adata)  # T not diagonal
    @test_throws ArgumentError MatrixEnsemble(I, I, I, 2im * I)  # T not in unit disc
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I(N + 1), I, I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I(N + 1), I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I, I(N + 1))
    @test_throws DimensionMismatch MatrixEnsemble(hcat(Adata, Ddata), I, I, I)
end
