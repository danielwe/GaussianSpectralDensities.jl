module Ensembles

using AutoHashEquals: @auto_hash_equals
using LinearAlgebra: I

using ..ArrayTools:
    AbstractMatrixOrUniform,
    ArrayTools,
    DiagonalOrUniform,
    adapteltype,
    all_diagonal,
    map_diagonal,
    withsize
using ..MathTools: inunitdisc, ldiv as generic_ldiv, phase, rdiv as generic_rdiv

public MatrixEnsemble

# TODO: consider generalizations where M, L, R can be LinearMaps etc.
"""
    MatrixEnsemble(M, L, R, S)
    MatrixEnsemble{F}(M, L, R, S) where {F<:AbstractFloat}
    MatrixEnsemble{F}(e::MatrixEnsemble) where {F<:AbstractFloat}

Ensemble of square matrices on the form ``M + LJR`` where ``J`` is Girko ``S``-distributed

The Girko ``S``-ensemble is a generalized Girko ensemble where the scalar symmetry parameter
``τ`` is replaced by a diagonal matrix ``S``. The ordinary Girko ensemble is recovered by
choosing `S = sqrt(τ) * I`.

All of ``M, L, R`` must be square and of the same size, or `UniformScaling`, and ``L`` and
``R`` must be nonsingular.
"""
@auto_hash_equals typearg=true struct MatrixEnsemble{
    F<:AbstractFloat,
    T<:Union{F,Complex{F}},
    TM<:AbstractMatrixOrUniform{T},
    TL<:AbstractMatrixOrUniform{T},
    TR<:AbstractMatrixOrUniform{T},
    TS<:DiagonalOrUniform{F},
    TG0i<:AbstractMatrix{T},
}
    M::TM
    L::TL
    R::TR
    S::TS
    Li_Ri::TG0i
    Li_M_Ri::TG0i

    function common(::Type{F}, M, L, R, S, Li_Ri, Li_M_Ri) where {F<:AbstractFloat}
        maybecomplex = (M, L, R, Li_Ri, Li_M_Ri)
        # HACK: using `local` here to silence an incorrect JETLS boxed capture warning.
        # Remove when aviatesk/JETLS.jl#508 is closed.
        local T = any(eltype(A) <: Complex for A in maybecomplex) ? Complex{F} : F
        M_T, L_T, R_T, Li_Ri_T, Li_M_Ri_T = adapteltype.(T, maybecomplex)
        S_F = adapteltype(F, S)
        return new{F,T,typeof.((M_T, L_T, R_T, S_F, Li_M_Ri_T))...}(
            M_T, L_T, R_T, S_F, Li_Ri_T, Li_M_Ri_T
        )
    end

    function MatrixEnsemble{F}(
        M::AbstractMatrixOrUniform,
        L::AbstractMatrixOrUniform,
        R::AbstractMatrixOrUniform,
        S::DiagonalOrUniform,
    ) where {F}
        sizes = NamedTuple(
            n => size(A) for (n, A) in pairs((; M, L, R, S)) if applicable(size, A)
        )
        if length(sizes) > 0
            N = first(first(sizes))
            for size in sizes
                if size != (N, N)
                    size_str = join(("size($n) == $s" for (n, s) in pairs(sizes)), "; ")
                    msg =
                        "M, L, R, and S must all be same-size square matrices or " *
                        "Uniformscaling, got $size_str"
                    throw(DimensionMismatch(msg))
                end
            end
        end
        if !all_diagonal(inunitdisc, S)
            throw(ArgumentError("all elements of S must lie in the unit disc"))
        end
        Sp, Rp = map_diagonal(abs, S), map_diagonal(phase, S) * R
        Li_M_Ri = withsize(generic_rdiv(generic_ldiv(L, M), Rp), Sp)
        Li_Ri = withsize(generic_ldiv(Rp * L, I), Li_M_Ri)
        return common(F, M, L, Rp, Sp, Li_Ri, Li_M_Ri)
    end

    function MatrixEnsemble{F}(e::MatrixEnsemble) where {F}
        return common(F, e.M, e.L, e.R, e.S, e.Li_Ri, e.Li_M_Ri)
    end
end

function MatrixEnsemble(M, L, R, S)
    Mu, Lu, Ru, Su = (oneunit ∘ eltype).((M, L, R, S))
    T = typeof((Mu + (Lu * (Su * Ru))) / (Lu * Ru))
    return MatrixEnsemble{real(T)}(M, L, R, S)
end

## Traits, conversions
## ===================
Base.size(e::MatrixEnsemble) = size(e.Li_M_Ri)

function Base.convert(::Type{P}, e::MatrixEnsemble) where {F,P<:MatrixEnsemble{F}}
    return (e isa P) ? e : MatrixEnsemble{F}(e)
end

function ArrayTools.adaptrealtype(::Type{F}, e::MatrixEnsemble) where {F}
    return convert(MatrixEnsemble{F}, e)
end

end  # module Ensembles

module TestMatrixEnsembleUniformScaling
    using Test: @test, @testset
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(I, I, I, I / 2)

    # Traits
    @test ensemble isa MatrixEnsemble{Float64}
    @test size(ensemble) == (1, 1)

    # Fields
    @test ensemble.M === 1.0I
    @test ensemble.L === 1.0I
    @test ensemble.R === 1.0I
    @test ensemble.S === 0.5I
    @test isequal(ensemble.Li_Ri, I(1))
    @test isequal(ensemble.Li_M_Ri, I(1))

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)

        # Floattype enforcing constructor
        @test ensembleF isa MatrixEnsemble{F}

        # Floattype conversions
        @test MatrixEnsemble{F}(ensemble) === ensembleF
        @test MatrixEnsemble{Float64}(ensembleF) === ensemble
        @test convert(MatrixEnsemble{F}, ensemble) === ensembleF
        @test convert(MatrixEnsemble{Float64}, ensembleF) === ensemble
        @test adaptrealtype(F, ensemble) === ensembleF
        @test adaptrealtype(Float64, ensembleF) === ensemble
    end
end

module TestMatrixEnsembletypes
    using Test: @test, @testset
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I
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
                MatrixEnsemble(AF, I, I, F(0.5) * I),
                MatrixEnsemble(I, AF, I, F(0.5) * I),
                MatrixEnsemble(I, I, AF, F(0.5) * I),
                MatrixEnsemble(I, I, I, DF),
            ]
            ensembles64 = [
                MatrixEnsemble(A, I, I, 0.5I),
                MatrixEnsemble(I, A, I, 0.5I),
                MatrixEnsemble(I, I, A, 0.5I),
                MatrixEnsemble(I, I, I, D),
            ]

            # Size
            @test all(e -> size(e) == (2, 2), ensemblesF)
            @test all(e -> size(e) == (2, 2), ensembles64)

            # Fields
            @test ensemblesF[1].M == AF
            @test ensemblesF[1].L == I
            @test ensemblesF[1].R == I
            @test ensemblesF[1].S == I / 2
            @test ensemblesF[1].Li_Ri == I(2)
            @test ensemblesF[1].Li_M_Ri == AF
            @test ensemblesF[2].M == I
            @test ensemblesF[2].L == AF
            @test ensemblesF[2].R == I
            @test ensemblesF[2].S == I / 2
            @test ensemblesF[2].Li_Ri == Ainv
            @test ensemblesF[2].Li_M_Ri == Ainv
            @test ensemblesF[3].M == I
            @test ensemblesF[3].L == I
            @test ensemblesF[3].R == AF
            @test ensemblesF[3].S == I / 2
            @test ensemblesF[3].Li_Ri == Ainv
            @test ensemblesF[3].Li_M_Ri == Ainv
            # Phase is sensitive to floating point round errors, so use ≈
            @test ensemblesF[4].M == I
            @test ensemblesF[4].L == I
            @test ensemblesF[4].R ≈ Dphase
            @test ensemblesF[4].S == Dabs
            @test ensemblesF[4].Li_Ri ≈ map(conj, Dphase)
            @test ensemblesF[4].Li_M_Ri ≈ map(conj, Dphase)

            # Floattype inferring constructor
            @test all(e -> e isa MatrixEnsemble{F}, ensemblesF)
            @test all(e -> e isa MatrixEnsemble{Float64}, ensembles64)

            # Floattype enforcing constructor
            # These tests use == since any truncation is the same on both sides
            @test MatrixEnsemble{F}(A, I, I, I / 2) == ensemblesF[1]
            @test MatrixEnsemble{F}(I, A, I, I / 2) == ensemblesF[2]
            @test MatrixEnsemble{F}(I, I, A, I / 2) == ensemblesF[3]
            @test MatrixEnsemble{F}(I, I, I, D) == ensemblesF[4]
            # These tests use isapprox since they RHS uses original values while the LHS may
            # have roundtripped through lower precision
            @test isapprox(MatrixEnsemble{Float64}(AF, I, I, I / 2), ensembles64[1])
            @test isapprox(MatrixEnsemble{Float64}(I, AF, I, I / 2), ensembles64[2])
            @test isapprox(MatrixEnsemble{Float64}(I, I, AF, I / 2), ensembles64[3])
            @test isapprox(MatrixEnsemble{Float64}(I, I, I, DF), ensembles64[4])

            # Floattype conversions
            @test isapprox(MatrixEnsemble{F}(ensembles64[1]), ensemblesF[1])
            @test isapprox(MatrixEnsemble{F}(ensembles64[2]), ensemblesF[2])
            @test isapprox(MatrixEnsemble{F}(ensembles64[3]), ensemblesF[3])
            @test isapprox(MatrixEnsemble{F}(ensembles64[4]), ensemblesF[4])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[1]), ensembles64[1])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[2]), ensembles64[2])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[3]), ensembles64[3])
            @test isapprox(MatrixEnsemble{Float64}(ensemblesF[4]), ensembles64[4])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[1]), ensemblesF[1])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[2]), ensemblesF[2])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[3]), ensemblesF[3])
            @test isapprox(convert(MatrixEnsemble{F}, ensembles64[4]), ensemblesF[4])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[1]), ensembles64[1])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[2]), ensembles64[2])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[3]), ensembles64[3])
            @test isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[4]), ensembles64[4])
            @test isapprox(adaptrealtype(F, ensembles64[1]), ensemblesF[1])
            @test isapprox(adaptrealtype(F, ensembles64[2]), ensemblesF[2])
            @test isapprox(adaptrealtype(F, ensembles64[3]), ensemblesF[3])
            @test isapprox(adaptrealtype(F, ensembles64[4]), ensemblesF[4])
            @test isapprox(adaptrealtype(Float64, ensemblesF[1]), ensembles64[1])
            @test isapprox(adaptrealtype(Float64, ensemblesF[2]), ensembles64[2])
            @test isapprox(adaptrealtype(Float64, ensemblesF[3]), ensembles64[3])
            @test isapprox(adaptrealtype(Float64, ensemblesF[4]), ensembles64[4])
        end
    end
end

module TestMatrixEnsembleerrors
    using Test: @test_throws
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble

    using LinearAlgebra: Diagonal, I

    Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
    Ddata = Diagonal(ComplexF64[0.75, 0.25cispi(1//3)])
    N = length(Ddata.diag)

    @test_throws MethodError MatrixEnsemble(I, I, I, Adata)  # S not diagonal
    @test_throws ArgumentError MatrixEnsemble(I, I, I, 2im * I)  # S not in unit disc
    @test_throws "unit disc" MatrixEnsemble(I, I, I, 2im * I)  # S not in unit disc
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I(N + 1), I, I / 2)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I(N + 1), I / 2)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I, I(N + 1) / 2)
    @test_throws DimensionMismatch MatrixEnsemble(hcat(Adata, Ddata), I, I, I / 2)
end
