using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt, γ², δ²
using GaussianSpectralDensities: psdmul!!, decompose, decompose!!, ldiv!!, invert, invert!!
using GaussianSpectralDensities: eigmin!!
using GaussianSpectralDensities: Workspace, G0i!, Q!, QQh!, P!
using GaussianSpectralDensities: newton, finite_difference_jacobian, NewtonException

using LinearAlgebra
using StaticArrays
using SparseArrays
using SuiteSparse

## Floating point and array type helpers
## =====================================
FLOATTYPES = [Float32, Float64]

function floattypes(args...)
    return [(F, floatconvert.(F, args)) for F in FLOATTYPES]
end

# Reuse some utility functions from ensemble.jl
function floatconvert(::Type{F}, A) where {F<:Real}
    (eltype(A) <: Complex) && return _elconvert(Complex{F}, A)
    return _elconvert(F, A)
end

function _elconvert(::Type{E}, A) where {E}
    (E === eltype(A)) && return A
    return convert.(E, A)
end

_elconvert(::Type{E}, A::UniformScaling) where {E} = UniformScaling(convert(E, A.λ))

# Work around lack of structure-preserving broadcast with SArray storage
function _elconvert(::Type{E}, A::Diagonal) where {E}
    (E === eltype(A)) && return A
    return Diagonal(convert.(E, A.diag))
end

function arraytypes(arrays...)
    return [
        ("Array", arrays),
        ("SArray", _sarray.(arrays)),
        ("SparseMatrixCSC", _sparse.(arrays)),
    ]
end

_sarray(A::AbstractMatrix) = SMatrix{size(A)...}(A)
_sarray(D::Diagonal) = Diagonal(SVector{length(D.diag)}(D.diag))

_sparse(A) = sparse(A)
_sparse(D::Diagonal) = Diagonal(sparse(D.diag))

## Test guidelines
## ===============
# For each type and file, test the interface that downstream code is meant to consume:
# - Construction and conversion signatures and the resulting type relations that you support
#   dispatching on
#   * E.g., you may support `x isa A` and `x isa A{B}` but consider `x isa A{B,C}` internal
# - Types and values returned from functions that are meant for downstream use
# - Explicitly handled error conditions in functions that are meant for downstream use
# - Don't test types and values of struct fields, even if accessing them directly is
#   supposed to be OK. It's too much---you'd be reimplementing the constructor in the test.
#   Use type parameters in the implementation as much as possible to enforce consistency. If
#   something is actually incorrect it should break a test of downstream functionality.
# Let's try to enforce the following convention: Names that should _not_ be used in other
# files start with an underscore (we use files as the main unit of informal
# "encapsulation"). Conversely, names not prefixed with an underscore should have
# a dedicated test (except struct fields).
#
# Run all tests across all relevant floating point and array types using the above helpers.

@testset "GaussianSpectralDensities.jl" begin
    @testset "MatrixEnsemble" begin
        @testset "UniformScaling" begin
            ensemble = MatrixEnsemble(I, I, I, I)

            # Traits
            @test ensemble isa MatrixEnsemble{Float64}
            @test size(ensemble) == (0, 0)

            @testset "$F" for F in FLOATTYPES
                ensembleF = MatrixEnsemble{F}(I, I, I, I)

                # Floattype enforcing constructor
                @test ensembleF isa MatrixEnsemble{F}

                # Floattype conversions
                @test convert(MatrixEnsemble{F}, ensemble) === ensembleF
                @test convert(MatrixEnsemble{Float64}, ensembleF) === ensemble
                @test adapt(F, ensemble) === ensembleF
                @test adapt(Float64, ensembleF) === ensemble
            end
        end

        Adata, Ddata = complex.([1.0 -1.0; 1.0 1.0]), Diagonal([0.75, 0.25])
        N = length(Ddata.diag)
        @testset "$label" for (label, (A, D)) in arraytypes(Adata, Ddata)
            ensembles = [
                MatrixEnsemble(A, I, I, I),
                MatrixEnsemble(I, A, I, I),
                MatrixEnsemble(I, I, A, I),
                MatrixEnsemble(I, I, I, D),
            ]

            # Traits
            @test all(e -> e isa MatrixEnsemble{Float64}, ensembles)
            @test all(e -> size(e) == (N, N), ensembles)

            @testset "$F" for (F, (AF, DF)) in floattypes(A, D)
                ensemblesF = [
                    MatrixEnsemble(AF, I, I, I),
                    MatrixEnsemble(I, AF, I, I),
                    MatrixEnsemble(I, I, AF, I),
                    MatrixEnsemble(I, I, I, DF),
                ]

                # Floattype inferring constructor
                @test all(e -> e isa MatrixEnsemble{F}, ensemblesF)

                # Floattype enforcing constructor
                @test MatrixEnsemble{F}(A, I, I, I) == ensemblesF[1]
                @test MatrixEnsemble{F}(I, A, I, I) == ensemblesF[2]
                @test MatrixEnsemble{F}(I, I, A, I) == ensemblesF[3]
                @test MatrixEnsemble{F}(I, I, I, D) == ensemblesF[4]
                @test MatrixEnsemble{Float64}(AF, I, I, I) == ensembles[1]
                @test MatrixEnsemble{Float64}(I, AF, I, I) == ensembles[2]
                @test MatrixEnsemble{Float64}(I, I, AF, I) == ensembles[3]
                @test MatrixEnsemble{Float64}(I, I, I, DF) == ensembles[4]

                # Floattype conversions
                @test all(
                    ((e, eF),) -> convert(MatrixEnsemble{F}, e) == eF,
                    zip(ensembles, ensemblesF),
                )
                @test all(
                    ((e, eF),) -> convert(MatrixEnsemble{Float64}, eF) == e,
                    zip(ensembles, ensemblesF)
                )
                @test all(((e, eF),) -> adapt(F, e) == eF, zip(ensembles, ensemblesF))
                @test all(((e, eF),) -> adapt(Float64, eF) == e, zip(ensembles, ensemblesF))
            end
        end

        # Sanity checks
        @test_throws ArgumentError MatrixEnsemble(I, I, I, real(Adata))  # T not diagonal
        @test_throws ArgumentError MatrixEnsemble(I, I, I, complex(Ddata))  # T not real
        @test_throws ArgumentError MatrixEnsemble(I, I, I, 2I)  # T not in unit interval
        @test_throws ArgumentError MatrixEnsemble(I, I, I, -I)  # T not in unit interval
        @test_throws DimensionMismatch MatrixEnsemble(Adata, I(N + 1), I, I)
        @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I(N + 1), I)
        @test_throws DimensionMismatch MatrixEnsemble(Adata, I, I, I(N + 1))
        @test_throws DimensionMismatch MatrixEnsemble(hcat(Adata, Ddata), I, I, I)
    end
    @testset "Regularizer" begin
        γ²value, δ²value = 0.0, 0.5
        s²min = δ²value - γ²value
        γr, δr = Regularizer(:γ, γ²value), Regularizer(:δ, δ²value)

        # Traits
        @test γr isa Regularizer{Float64}
        @test δr isa Regularizer{Float64}

        @testset "$F" for (F, (γ²valueF, δ²valueF, s²minF)) in floattypes(
            γ²value, δ²value, s²min
        )
            γrF, δrF = Regularizer(:γ, γ²valueF), Regularizer(:δ, δ²valueF)

            # Floattype inferring constructor
            @test γrF isa Regularizer{F}
            @test δrF isa Regularizer{F}

            # Floattype enforcing constructor
            @test Regularizer{F}(:γ, γ²value) === γrF
            @test Regularizer{F}(:δ, δ²value) === δrF
            @test Regularizer{Float64}(:γ, γ²valueF) === γr
            @test Regularizer{Float64}(:δ, δ²valueF) === δr

            # Floattype conversions
            @test convert(Regularizer{F}, γr) === γrF
            @test convert(Regularizer{F}, δr) === δrF
            @test convert(Regularizer{Float64}, γrF) === γr
            @test convert(Regularizer{Float64}, δrF) === δr
            @test adapt(F, γr) === γrF
            @test adapt(F, δr) === δrF
            @test adapt(Float64, γrF) === γr
            @test adapt(Float64, δrF) === δr

            # Interface
            @test γ²(γrF, s²minF) === γ²valueF
            @test δ²(γrF, s²minF) === δ²valueF
            @test γ²(δrF, s²minF) === γ²valueF
            @test δ²(δrF, s²minF) === δ²valueF
        end
    end
    @testset "objectives" begin
        kvalue, rkind, rvalue = 1.0, :δ, 0.5
        reg = Regularizer(rkind, rvalue)
        ko = TraceObjective(kvalue)
        ro = RegularizerObjective(reg)

        # Traits
        @test ko isa TraceObjective{Float64}
        @test ro isa RegularizerObjective{Float64}

        # Defaults
        @test TraceObjective() === TraceObjective(DEFAULTS.kobjective)
        @test RegularizerObjective() === RegularizerObjective(DEFAULTS.robjective)

        @testset "$F" for (F, (kvalueF, rvalueF)) in floattypes(kvalue, rvalue)
            regF = Regularizer(rkind, rvalueF)
            koF = TraceObjective(kvalueF)
            roF = RegularizerObjective(regF)

            # Floattype inferring constructor
            @test koF isa TraceObjective{F}
            @test roF isa RegularizerObjective{F}
            # - All the signatures
            @test RegularizerObjective(rkind, rvalueF) === roF

            # Floattype enforcing constructor
            @test TraceObjective{F}(kvalue) === koF
            @test RegularizerObjective{F}(reg) === roF
            @test TraceObjective{Float64}(kvalueF) === ko
            @test RegularizerObjective{Float64}(regF) === ro

            # Floattype conversions
            @test convert(TraceObjective{F}, ko) === koF
            @test convert(RegularizerObjective{F}, ro) === roF
            @test convert(TraceObjective{Float64}, koF) === ko
            @test convert(RegularizerObjective{Float64}, roF) === ro
            @test adapt(F, ko) === koF
            @test adapt(F, ro) === roF
            @test adapt(Float64, koF) === ko
            @test adapt(Float64, roF) === ro
        end
    end
    @testset "ResolventProblem" begin
        ro = RegularizerObjective()
        @testset "$label" for (label, z) in [
            ("z scalar", 1.0),
            ("z matrix", [0.0 1.0]),
            ("z matrix with missing", [0.0 1.0; missing missing]),
        ]
            @testset "$F" for F in FLOATTYPES
                ensembleF = MatrixEnsemble{F}(I, I, I, I)

                # Floattype inferring constructor
                @test ResolventProblem(ensembleF, z) isa ResolventProblem{F}
                @test ResolventProblem(ensembleF, z, ro) isa ResolventProblem{F}
                p = ResolventProblem(ensembleF, z; normalizeQ=true)
                @test p isa ResolventProblem{F}
                p = ResolventProblem(ensembleF, z, ro; normalizeQ=true)
                @test p isa ResolventProblem{F}
            end
        end
    end
    @testset "algorithms" begin
        @testset "DirectSolve" begin
            q0, δ0 = complex(eps()), 0.0
            dsolve = DirectSolve(q0, δ0)

            # Traits
            @test dsolve isa DirectSolve{Float64}
            @test length(dsolve) == 0

            # Defaults
            @test DirectSolve() === DirectSolve(DEFAULTS.q0, DEFAULTS.δ0)

            @testset "$F" for (F, (q0F, δ0F)) in floattypes(q0, δ0)
                dsolveF = DirectSolve(q0F, δ0F)

                # Floattype inferring constructor
                @test dsolveF isa DirectSolve{F}

                # Floattype enforcing constructor
                @test DirectSolve{F}(q0, δ0) isa DirectSolve{F}
                @test DirectSolve{Float64}(q0F, δ0F) isa DirectSolve{Float64}

                # Floattype conversions
                @test convert(DirectSolve{F}, dsolve) === dsolveF
                @test convert(DirectSolve{Float64}, dsolveF) === dsolve
                @test adapt(F, dsolve) === dsolveF
                @test adapt(Float64, dsolveF) === dsolve
            end
        end
        @testset "RegularizerSchedule" begin
            kind, values, γ²min = :δ, 2.0.^(2:-1:-2), 0.0
            rschedule = Regularizer.(kind, values)
            schedule = RegularizerSchedule(rschedule, γ²min)

            # Traits
            @test schedule isa RegularizerSchedule{Float64}
            @test length(schedule) == length(values)

            # Defaults
            @test RegularizerSchedule() === RegularizerSchedule(
                DEFAULTS.rschedule, DEFAULTS.γ²min
            )

            @testset "$F" for (F, (valuesF, γ²minF)) in floattypes(values, γ²min)
                rscheduleF = Regularizer.(kind, valuesF)
                scheduleF = RegularizerSchedule(rscheduleF, γ²minF)

                # Floattype inferring constructor
                @test scheduleF isa RegularizerSchedule{F}
                # - All the signatures
                @test RegularizerSchedule(kind, valuesF, γ²minF) == scheduleF

                # Floattype enforcing constructor
                @test RegularizerSchedule{F}(rschedule, γ²min) == scheduleF
                @test RegularizerSchedule{Float64}(rscheduleF, γ²minF) == schedule

                # Floattype conversions
                @test convert(RegularizerSchedule{F}, schedule) == scheduleF
                @test convert(RegularizerSchedule{Float64}, scheduleF) == schedule
                @test adapt(F, schedule) == scheduleF
                @test adapt(Float64, scheduleF) == schedule
            end
        end
        @testset "SymmetrySchedule" begin
            ialg, steps, γ²min = RegularizerSchedule(), 3, 0.0
            μschedule = range(0.0, 1.0, steps)
            schedule = SymmetrySchedule(ialg, μschedule, γ²min)

            # Traits
            @test schedule isa SymmetrySchedule{Float64,<:RegularizerSchedule{Float64}}
            @test length(schedule) == length(ialg) + length(μschedule)

            @testset "$F" for (F, (γ²minF,)) in floattypes(γ²min)
                ialgF = adapt(F, ialg)
                μscheduleF = adapt(F, μschedule)
                scheduleF = SymmetrySchedule(ialgF, μscheduleF, γ²minF)

                # Floattype inferring constructor
                @test scheduleF isa SymmetrySchedule{F,<:RegularizerSchedule{F}}
                # - All the signatures
                @test SymmetrySchedule(ialgF, steps, γ²minF) == scheduleF
                # - Defaults
                @test SymmetrySchedule(ialgF) == SymmetrySchedule{F}(
                    ialgF, DEFAULTS.μschedule, DEFAULTS.γ²min
                )

                # Capture any roundtrip truncation loss---that's not what we want to test
                ialg64 = adapt(Float64, ialgF)
                schedule64 = SymmetrySchedule(ialg64, μschedule, γ²min)

                # Floattype enforcing constructor
                @test SymmetrySchedule{F}(ialg, μschedule, γ²min) == scheduleF
                @test SymmetrySchedule{Float64}(ialgF, μscheduleF, γ²minF) == schedule64

                # Floattype conversions
                @test convert(SymmetrySchedule{F}, schedule) == scheduleF
                @test convert(SymmetrySchedule{Float64}, scheduleF) == schedule64
                @test adapt(F, schedule) == scheduleF
                @test adapt(Float64, scheduleF) == schedule64
            end

            # Sanity checks
            @test_throws ArgumentError SymmetrySchedule(ialg, [0.2, 0.6, 1.0], γ²min)  # Doesn't start at 0
            @test_throws ArgumentError SymmetrySchedule(ialg, [0.0, 0.4, 0.8], γ²min)  # Doesn't end at 1
            @test_throws ArgumentError SymmetrySchedule(ialg, [0.0, 0.67, 0.33, 1.0], γ²min)  # Not monotonic
        end
    end
    @testset "linalg" begin
        _isequal(C1, C2) = isequal(C1, C2)

        function _isequal(C1::C, C2::C) where {C<:Cholesky}
            return (
                isequal(C1.info, C2.info)
                && isequal(C1.uplo, C2.uplo)
                && isequal(C1.L, C2.L)
            )
        end

        function _isequal(C1::C, C2::C) where {C<:SuiteSparse.CHOLMOD.Factor}
            return (
                isequal(issuccess(C1), issuccess(C2))
                && isequal(C1.p, C2.p)
                && isequal(sparse(C1.LD), sparse(C2.LD))
            )
        end

        function linalg_tests(
            A, Ai_true, Aeigmin_true, shift, Asi_true, B, A_B_true, As_B_true, X
        )
            F = decompose(A)
            Ai = invert(F)
            @test Ai ≈ Ai_true
            A_B = ldiv!!(F, copy(B))
            @test A_B ≈ A_B_true
            F = decompose!!(F, A; shift=shift)
            @test _isequal(F, decompose(A; shift=shift))
            Ai = invert!!(Ai, F)
            @test Ai ≈ Asi_true
            A_B = ldiv!!(F, copy(B))
            @test A_B ≈ As_B_true

            @test eigmin!!(F, A) ≈ Aeigmin_true

            H = psdmul!!(deepcopy(A), X)
            @test H ≈ X * X'
            α, β = convert.(real(eltype(X)), (0.25, 1.33))
            H = psdmul!!(H, X', α, β)
            @test H ≈ α * X' * X + β * X * X'
        end

        commondata = (
            shift=0.5,
            U=2.0I,
            Ui=0.5I,
            Usi=0.4I,
            D=Diagonal([1.5, 2.0]),
            Di=Diagonal([2, 1.5] / 3),
            Dsi=Diagonal([0.5, 0.4]),
        )
        @testset "$field" for field in ["Real", "Complex"]
            @testset "$F" for (
                F,
                (
                    shift,
                    U,
                    Ui,
                    Usi,
                    D,
                    Di,
                    Dsi,
                    B,
                    U_B,
                    Us_B,
                    D_B,
                    Ds_B,
                    Adata,
                    Aeigmin,
                    Aidata,
                    AU_B,
                    AL_B,
                    Asidata,
                    AsU_B,
                    AsL_B,
                )
            ) in floattypes(
                commondata...,
                (
                    if field === "Real"
                        (
                            B=[1 -1; 1 1],
                            U_B=[0.5 -0.5; 0.5 0.5],
                            Us_B=[0.4 -0.4; 0.4 0.4],
                            D_B=[2 -2; 1.5 1.5] / 3,
                            Ds_B=[0.5 -0.5; 0.4 0.4],
                            Adata=[1.5 -1; 1 2],  # deliberate asymmetry, masked in two different ways
                            Aeigmin=0.719223593595585,
                            Aidata=[1 0.5; -0.5 0.75],
                            AU_B=[1.5 -0.5; 1.25 0.25],
                            AL_B=[0.5 -1.5; 0.25 1.25],
                            Asidata=[5 2; -2 4] / 8,
                            AsU_B=[7 -3; 6 2] / 8,
                            AsL_B=[3 -7; 2 6] / 8,
                        )
                    else
                        (
                            B=[(1 + im) -1; 1 (1 - im)],
                            U_B=[(0.5 + 0.5im) -0.5; 0.5 (0.5 - 0.5im)],
                            Us_B=[(0.4 + 0.4im) -0.4; 0.4 (0.4 - 0.4im)],
                            D_B=[(2 + 2im) -2; 1.5 (1.5 - 1.5im)] / 3,
                            Ds_B=[(0.5 + 0.5im) -0.5; 0.4 (0.4 - 0.4im)],
                            Adata=[1.5 (-1 + im); (1 + im) 2],  # deliberate ahermiticity
                            Aeigmin=0.3138593383654923,
                            Aidata=[2 (1 - im); (-1 - im) 1.5],
                            AU_B=[(3 + im) (-2 - 2im); (1.5 + 2im) (0.5 - 2.5im)],
                            AL_B=[(1 + 3im) (-2 + 2im); (1.5 - 2im) (2.5 - 0.5im)],
                            Asidata=[5 (2 - 2im); (-2 - 2im) 4] / 6,
                            AsU_B=[(7 + 3im) (-5 - 4im); (4 + 4im) (2 - 6im)] / 6,
                            AsL_B=[(3 + 7im) (-5 + 4im); (4 - 4im) (6 - 2im)] / 6,
                        )
                    end
                )...,
            )
                @testset "UniformScaling" begin
                    linalg_tests(U, Ui, U.λ, shift, Usi, B, U_B, Us_B, first(B) * I)
                end
                @testset "Diagonal" begin
                    X = Diagonal(B)
                    @testset "Vector" begin
                        linalg_tests(D, Di, minimum(D.diag), shift, Dsi, B, D_B, Ds_B, X)
                    end
                    @testset "SVector" begin
                        linalg_tests(
                            _sarray.((D, Di))...,
                            minimum(D.diag),
                            shift,
                            _sarray.((Dsi, B, D_B, Ds_B, X))...,
                        )
                    end
                end
                # Can't use arraytypes here because which ones to convert and which to leave
                # as plain Arrays vary between array types
                @testset "uplo=:$uplo" for (uplo, A_B, As_B) in (
                    (:U, AU_B, AsU_B),
                    (:L, AL_B, AsL_B),
                )
                    Ai = Hermitian(Aidata, uplo)
                    Asi = Hermitian(Asidata, uplo)
                    X = B
                    @testset "Array" begin
                        A = Hermitian(Adata, uplo)
                        @test_throws DomainError decompose(A; shift=im)
                        linalg_tests(A, Ai, Aeigmin, shift, Asi, B, A_B, As_B, X)
                    end
                    @testset "SArray" begin
                        A = Hermitian(_sarray(Adata), uplo)
                        Ai = Hermitian(_sarray(Aidata), uplo)
                        Asi = Hermitian(_sarray(Asidata), uplo)
                        @test_throws DomainError decompose(A; shift=im)
                        linalg_tests(
                            A, Ai, Aeigmin, shift, Asi, _sarray.((B, A_B, As_B, X))...,
                        )
                    end
                    @testset "SparseMatrixCSC" begin
                        A = Hermitian(_sparse(Adata), uplo)
                        @test_throws TypeError decompose(A; shift=im)
                        linalg_tests(A, Ai, Aeigmin, shift, Asi, B, A_B, As_B, _sparse(X))
                    end
                end
            end
        end
    end
    @testset "workspace" begin
        @testset "Workspace" begin
            function workspace_tests(F, ens)
                _inv(A) = inv(A)
                _inv(A::SparseMatrixCSC) = inv(Matrix(A))
                _hermitian(A) = Hermitian(A)
                _hermitian(A::SparseMatrixCSC) = Hermitian(Matrix(A))
                _hermitian(A::Union{<:Diagonal,<:UniformScaling}) = real(A)
                _eigmin(A) = eigmin(A)
                _eigmin(A::UniformScaling) = A.λ

                ws = Workspace(ens)
                @test ws isa Workspace{F}

                z, q = convert.(Complex{F}, (-(im + 2) / 4, (1 - im) / 2))
                G0i = _inv(ens.L) * (z * I - ens.M) * _inv(ens.R)
                Q = G0i - q * ens.T
                @test G0i!(ws, ens, z) ≈ G0i
                @test Q!(ws, ens, z, q) ≈ Q

                QQh = Q * Q'
                QQhn = (QQh + Q' * Q) / 2
                @test QQh!(ws, ens, z, q) ≈ QQh
                @test QQh!(ws, ens, z, q, false) ≈ QQh
                @test QQh!(ws, ens, z, q, true) ≈ QQh

                s²min, s²minn = _eigmin(_hermitian(QQh)), _eigmin(_hermitian(QQhn))
                γ², δ² = Regularizer{F}(:γ, 0.25), Regularizer{F}(:δ, 0.5)
                Pγ = _inv(QQh + γ².value * I)
                Pδ = _inv(QQh + (δ².value - s²min) * I)
                Pγn = _inv(QQhn + γ².value * I)
                Pδn = _inv(QQhn + (δ².value - s²min) * I)
                @test P!(ws, ens, z, q, γ²) ≈ Pγ
                @test P!(ws, ens, z, q, δ²) ≈ Pδ
                @test P!(ws, ens, z, q, γ², false) ≈ Pγ
                @test P!(ws, ens, z, q, δ², false) ≈ Pδ
                @test P!(ws, ens, z, q, γ², true) ≈ Pγn
                @test P!(ws, ens, z, q, δ², true) ≈ Pδn
            end

            Adata, Ddata = ComplexF64[1 -1; 1 1], Diagonal([0.75, 0.25])
            @testset "$label" for (label, (A, D)) in arraytypes(Adata, Ddata)
                @testset "$F" for F in FLOATTYPES
                    workspace_tests(F, MatrixEnsemble{F}(I, I, I, I))
                    workspace_tests(F, MatrixEnsemble{F}(A, I, I, I))
                    workspace_tests(F, MatrixEnsemble{F}(I, A, I, I))
                    workspace_tests(F, MatrixEnsemble{F}(I, I, A, I))
                    workspace_tests(F, MatrixEnsemble{F}(I, I, I, D))
                end
            end
        end
    end
    @testset "newton" begin
        @testset "scalar" begin
            f(x) = cos(x) - x^3
            df(x, _) = -sin(x) - 3x^2
            x0value = 0.5

            @testset "$F" for (F, (x0,)) in floattypes(x0value)
                # finite_difference_jacobian
                fx0 = f(x0)
                dfx0 = df(x0, fx0)
                rtol = eps(F)^convert(F, 0.4)
                @test finite_difference_jacobian(f, x0, fx0) ≈ dfx0 rtol=rtol
                @test finite_difference_jacobian(f, x0) ≈ dfx0 rtol=rtol

                # newton
                atol = sqrt(eps(F)) / 2
                @test norm(f(first(newton(f, x0)))) < atol
                @test norm(f(first(newton(f, df, x0)))) < atol

                @test_throws NewtonException newton(f, x0; maxiters=0)
                @test_throws NewtonException newton(f, df, x0; maxiters=0)
            end
        end
        @testset "SVector" begin
            g(x) = (x .* (x.^2 .- 1)) + (reverse(x) ./ 2)

            function jg(x::SVector{L}, _) where {L}
                jac = fill!(similar(x, Size(L, L)), 0)
                for i in 1:L
                    jac[i, i] = 3x[i]^2 - 1
                    jac[L - i + 1, i] = 0.5
                end
                return SMatrix(jac)
            end

            x0value = SVector(0.3, -0.7)

            @testset "$F" for (F, (x0,)) in floattypes(x0value)
                # finite_difference_jacobian
                gx0 = g(x0)
                jgx0 = jg(x0, gx0)
                rtol = eps(F)^convert(F, 0.4)
                @test finite_difference_jacobian(g, x0, gx0) ≈ jgx0 rtol=rtol
                @test finite_difference_jacobian(g, x0) ≈ jgx0 rtol=rtol

                # newton
                atol = sqrt(eps(F)) / 2
                normalizednorm(y) = norm(y) / sqrt(length(y))
                @test normalizednorm(g(first(newton(g, x0)))) < atol
                @test normalizednorm(g(first(newton(g, jg, x0)))) < atol

                @test_throws NewtonException newton(g, x0; maxiters=0)
                @test_throws NewtonException newton(g, jg, x0; maxiters=0)
            end
        end
    end
end
