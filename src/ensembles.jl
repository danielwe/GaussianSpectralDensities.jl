module Ensembles

using LinearAlgebra: I
using TestItems: @testitem

using ..MathHelpers: MathHelpers, AbstractMatrixOrUniform, DiagonalOrUniform, alldiag,
    inunitdisc, mapdiag, phase, converteltype, sizedmatrix

public MatrixEnsemble

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
    L_inv_R_inv::TC
    L_inv_M_R_inv::TC

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
                if s != (N, N)
                    size_strs = ("size($name) == $s" for (name, s) in pairs(sizes))
                    throw(DimensionMismatch(
                        "M, L, R, and T must all be same-size square matrices or "
                        * "Uniformscaling, got " * join(size_strs, "; ")
                    ))
                end
            end
        end
        if !alldiag(inunitdisc, T)
            throw(ArgumentError("all elements of T must lie in the unit disc"))
        end
        RC = any(eltype(A) <: Complex for A in (M, L, R, T)) ? Complex{F} : F
        M_RC, L_RC, R_RC = converteltype.(RC, (M, L, mapdiag(phase, T) * R))
        T_F = converteltype(F, mapdiag(abs, T))
        L_inv_M_R_inv = sizedmatrix(
            converteltype(RC, MathHelpers.rdiv(MathHelpers.ldiv(L_RC, M_RC), R_RC)),
            T_F,
        )
        L_inv_R_inv = sizedmatrix(
            converteltype(RC, MathHelpers.ldiv(R_RC * L_RC, I)),
            L_inv_M_R_inv,
        )
        return new{F,RC,typeof.((M_RC, L_RC, R_RC, T_F, L_inv_M_R_inv))...}(
            M_RC, L_RC, R_RC, T_F, L_inv_R_inv, L_inv_M_R_inv
        )
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
Base.size(e::MatrixEnsemble) = size(e.L_inv_M_R_inv)

function Base.:(==)(a::E, b::E) where {E<:MatrixEnsemble}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, e::MatrixEnsemble) where {F,P<:MatrixEnsemble{F}}
    return (e isa P) ? e : MatrixEnsemble{F}(e)
end

## Tests
## =====
@testitem "MatrixEnsemble with UniformScaling" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(I, I, I, I)

    # Traits
    @test ensemble isa MatrixEnsemble{Float64}
    @test size(ensemble) == (1, 1)

    # Fields
    @test ensemble.M === 1.0I
    @test ensemble.L === 1.0I
    @test ensemble.R === 1.0I
    @test ensemble.T === 1.0I
    @test isequal(ensemble.L_inv_R_inv, I(1))
    @test isequal(ensemble.L_inv_M_R_inv, I(1))

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

@testitem "MatrixEnsemble with various array types" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I, Diagonal
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    function isapprox(a::E, b::E) where {E<:MatrixEnsemble}
        return all(p -> getproperty(a, p) ≈ getproperty(b, p), propertynames(a))
    end

    A64data = ComplexF64[1.0 -1.0; 1.0 1.0]
    D64data = ComplexF64[0.75, 0.25cispi(1 // 3)]

    # These are just used for comparison, so we don't fuss about eltypes and array types
    Ainv = [0.5 0.5; -0.5 0.5]
    Dabs = Diagonal([0.75, 0.25])
    Dphase = Diagonal([1.0, cispi(1 // 3)])

    @testset "$F" for F in [Float32, Float64]
        AFdata = map(Complex{F}, A64data)
        DFdata = map(Complex{F}, D64data)

        @testset "$label" for (label, A, D, AF, DF) in [
            ("Array", A64data, Diagonal(D64data), AFdata, Diagonal(DFdata)),
            (
                "SArray",
                SMatrix{2,2}(A64data),
                Diagonal(SVector{2}(D64data)),
                SMatrix{2,2}(AFdata),
                Diagonal(SVector{2}(DFdata)),
            ),
            (
                "SparseMatrixCSC",
                sparse(A64data),
                Diagonal(D64data),
                sparse(AFdata),
                Diagonal(DFdata),
            ),
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
            @test all(e -> size(e) == (2, 2), ensemblesF)
            @test all(e -> size(e) == (2, 2), ensembles64)

            # Fields
            @test ensemblesF[1].M == AF
            @test ensemblesF[1].L == I
            @test ensemblesF[1].R == I
            @test ensemblesF[1].T == I
            @test ensemblesF[1].L_inv_R_inv == I(2)
            @test ensemblesF[1].L_inv_M_R_inv == AF
            @test ensemblesF[2].M == I
            @test ensemblesF[2].L == AF
            @test ensemblesF[2].R == I
            @test ensemblesF[2].T == I
            @test ensemblesF[2].L_inv_R_inv == Ainv
            @test ensemblesF[2].L_inv_M_R_inv == Ainv
            @test ensemblesF[3].M == I
            @test ensemblesF[3].L == I
            @test ensemblesF[3].R == AF
            @test ensemblesF[3].T == I
            @test ensemblesF[3].L_inv_R_inv == Ainv
            @test ensemblesF[3].L_inv_M_R_inv == Ainv
            # Phase is sensitive to floating point round errors, so use ≈
            @test ensemblesF[4].M == I
            @test ensemblesF[4].L == I
            @test ensemblesF[4].R ≈ Dphase
            @test ensemblesF[4].T == Dabs
            @test ensemblesF[4].L_inv_R_inv ≈ map(conj, Dphase)
            @test ensemblesF[4].L_inv_M_R_inv ≈ map(conj, Dphase)

            # Floattype inferring constructor
            @test all(e -> e isa MatrixEnsemble{F}, ensemblesF)
            @test all(e -> e isa MatrixEnsemble{Float64}, ensembles64)

            # Floattype enforcing constructor
            # These tests use == since any truncation is the same on both sides
            @test MatrixEnsemble{F}(A, I, I, I) == ensemblesF[1]
            @test MatrixEnsemble{F}(I, A, I, I) == ensemblesF[2]
            @test MatrixEnsemble{F}(I, I, A, I) == ensemblesF[3]
            @test MatrixEnsemble{F}(I, I, I, D) == ensemblesF[4]
            # These tests use isapprox since they RHS uses original values while the LHS may
            # have roundtripped through lower precision
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
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I, Diagonal

    Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
    Ddata = Diagonal(ComplexF64[0.75, 0.25*cispi(1 // 3)])
    N = length(Ddata.diag)

    @test_throws MethodError MatrixEnsemble(I, I, I, Adata)  # T not diagonal
    @test_throws ArgumentError MatrixEnsemble(I, I, I, 2im * I)  # T not in unit disc
    @test_throws "unit disc" MatrixEnsemble(I, I, I, 2im * I)  # T not in unit disc
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I(N + 1), I, I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I(N + 1), I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I, I(N + 1))
    @test_throws DimensionMismatch MatrixEnsemble(hcat(Adata, Ddata), I, I, I)
end

end  # module Ensembles
