const _AbstractMatrixOrUniform{F} = Union{AbstractMatrix{F},UniformScaling{F}}
const _RealOrComplex{F<:Real} = Union{F,Complex{F}}

# TODO: consider generalizations where M, L, R can be LinearMaps etc.
"""
    MatrixEnsemble(M, L, R, T)
    MatrixEnsemble{F}(M, L, R, T) where {F<:AbstractFloat}

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
    TM<:_AbstractMatrixOrUniform{RC},
    TL<:_AbstractMatrixOrUniform{RC},
    TR<:_AbstractMatrixOrUniform{RC},
    TT<:DiagonalOrUniform{F},
    TC<:AbstractMatrix{RC},
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
            for next_s in Base.tail(sizes)
                (next_s == s) || throwsizeerror()
            end
        end
        if !((T isa DiagonalOrUniform) && _inunitdisc(T))
            msg = "T must be Diagonal or UniformScaling with elements in the unit disc"
            throw(ArgumentError(msg))
        end
        RC = (_eltype(M, L, R, T) <: Complex) ? Complex{F} : F
        Mrc, Lrc, Rrc = _elconvert.(RC, (M, L, _phase(T) * R))
        Tf = _elconvert(F, _abs(T))
        LiMRi = _sized(_elconvert(RC, _rdiv(_ldiv(Lrc, Mrc), Rrc)), Tf)
        LiRi = _sized(_elconvert(RC, _ldiv(Rrc * Lrc, I)), LiMRi)
        return new{F,RC,typeof.((Mrc, Lrc, Rrc, Tf, LiMRi))...}(Mrc, Lrc, Rrc, Tf, LiRi, LiMRi)
    end
end

MatrixEnsemble(M, L, R, T) = MatrixEnsemble{real(_eltype(M, L, R, T))}(M, L, R, T)
MatrixEnsemble{F}(e::MatrixEnsemble) where {F} = MatrixEnsemble{F}(e.M, e.L, e.R, e.T)

## Traits, comparisons, conversions
## ================================
Base.size(e::MatrixEnsemble) = size(e.LiMRi)

function Base.:(==)(a::E, b::E) where {E<:MatrixEnsemble}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function _isapprox(a::E, b::E) where {E<:MatrixEnsemble}
    return all(p -> getproperty(a, p) ≈ getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, e::Q) where {F,P<:MatrixEnsemble{F},Q<:MatrixEnsemble}
    (e isa P) && return e
    return MatrixEnsemble{F}(e)
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

function _eltype(M, L, R, T)
    Mu, Lu, Ru, Tu = (oneunit ∘ eltype).((M, L, R, T))
    return typeof((Mu + (Lu * (Tu * Ru))) / (Lu * Ru))
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

_sized(A::UniformScaling, ::UniformScaling) = Diagonal(SVector{1}(A.λ))
_sized(A::UniformScaling, T::AnySMatrix) = Diagonal(@SVector fill(A.λ, size(T, 1)))
_sized(A::UniformScaling, T) = A(size(T, 1))
_sized(A, _) = A

## Tests
## =====
@testitem "MatrixEnsemble with UniformScaling" begin
    using LinearAlgebra

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
    using GaussianSpectralDensities: _elconvert, _isapprox
    using LinearAlgebra
    using SparseArrays
    using StaticArrays

    Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
    Ddata = Diagonal(ComplexF64[0.75, 0.25*cispi(1 // 3)])
    N = length(Ddata.diag)

    @testset "$label" for (label, (A, D)) in [
        ("Array", (Adata, Ddata)),
        ("SArray", (SMatrix{N,N}(Adata), Diagonal(SVector{N}(Ddata.diag)))),
        ("SparseMatrixCSC", (sparse(Adata), Ddata)),
    ]
        ensembles = [
            MatrixEnsemble(A, I, I, I),
            MatrixEnsemble(I, A, I, I),
            MatrixEnsemble(I, I, A, I),
            MatrixEnsemble(I, I, I, D),
        ]

        # Traits
        @test all(e -> e isa MatrixEnsemble{Float64}, ensembles)
        @test all(e -> size(e) == (N, N), ensembles)

        @testset "$F" for F in [Float32, Float64]
            AF, DF = _elconvert.(Complex{F}, (A, D))
            ensemblesF = [
                MatrixEnsemble(AF, I, I, I),
                MatrixEnsemble(I, AF, I, I),
                MatrixEnsemble(I, I, AF, I),
                MatrixEnsemble(I, I, I, DF),
            ]

            # Floattype inferring constructor
            @test all(e -> e isa MatrixEnsemble{F}, ensemblesF)

            # Floattype enforcing constructor
            @test _isapprox(MatrixEnsemble{F}(A, I, I, I), ensemblesF[1])
            @test _isapprox(MatrixEnsemble{F}(I, A, I, I), ensemblesF[2])
            @test _isapprox(MatrixEnsemble{F}(I, I, A, I), ensemblesF[3])
            @test _isapprox(MatrixEnsemble{F}(I, I, I, D), ensemblesF[4])
            @test _isapprox(MatrixEnsemble{Float64}(AF, I, I, I), ensembles[1])
            @test _isapprox(MatrixEnsemble{Float64}(I, AF, I, I), ensembles[2])
            @test _isapprox(MatrixEnsemble{Float64}(I, I, AF, I), ensembles[3])
            @test _isapprox(MatrixEnsemble{Float64}(I, I, I, DF), ensembles[4])

            # Floattype conversions
            @test _isapprox(convert(MatrixEnsemble{F}, ensembles[1]), ensemblesF[1])
            @test _isapprox(convert(MatrixEnsemble{F}, ensembles[2]), ensemblesF[2])
            @test _isapprox(convert(MatrixEnsemble{F}, ensembles[3]), ensemblesF[3])
            @test _isapprox(convert(MatrixEnsemble{F}, ensembles[4]), ensemblesF[4])
            @test _isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[1]), ensembles[1])
            @test _isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[2]), ensembles[2])
            @test _isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[3]), ensembles[3])
            @test _isapprox(convert(MatrixEnsemble{Float64}, ensemblesF[4]), ensembles[4])
            @test _isapprox(MatrixEnsemble{F}(ensembles[1]), ensemblesF[1])
            @test _isapprox(MatrixEnsemble{F}(ensembles[2]), ensemblesF[2])
            @test _isapprox(MatrixEnsemble{F}(ensembles[3]), ensemblesF[3])
            @test _isapprox(MatrixEnsemble{F}(ensembles[4]), ensemblesF[4])
            @test _isapprox(MatrixEnsemble{Float64}(ensemblesF[1]), ensembles[1])
            @test _isapprox(MatrixEnsemble{Float64}(ensemblesF[2]), ensembles[2])
            @test _isapprox(MatrixEnsemble{Float64}(ensemblesF[3]), ensembles[3])
            @test _isapprox(MatrixEnsemble{Float64}(ensemblesF[4]), ensembles[4])
        end
    end
end

@testitem "MatrixEnsemble errors" begin
    using LinearAlgebra

    Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
    Ddata = Diagonal(ComplexF64[0.75, 0.25*cispi(1 // 3)])
    N = length(Ddata.diag)

    @test_throws ArgumentError MatrixEnsemble(I, I, I, Adata)  # T not diagonal
    @test_throws ArgumentError MatrixEnsemble(I, I, I, 2im * I)  # T not in unit disc
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I(N + 1), I, I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I(N + 1), I)
    @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I, I(N + 1))
    @test_throws DimensionMismatch MatrixEnsemble(hcat(Adata, Ddata), I, I, I)
end
