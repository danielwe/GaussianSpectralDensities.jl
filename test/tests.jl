using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt, γ², δ²
using GaussianSpectralDensities: Regularizer, LocalResolventEquation
using GaussianSpectralDensities: LocalQProblem, LocalQRegularizerProblem
using GaussianSpectralDensities: Workspace, qrhs!, γ²!, δ²!, k!, E!
using GaussianSpectralDensities: LocalResolventSolution, eqsolve!
using GaussianSpectralDensities: psdmul, psdmul!!, decompose, decompose!!, ldiv!!
using GaussianSpectralDensities: invert, invert!!, eigmin!!
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
_sparse(D::Diagonal) = D

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
            @test size(ensemble) == (1, 1)

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

        function _isapprox(a::E, b::E) where {E<:MatrixEnsemble}
            return all(p -> getproperty(a, p) ≈ getproperty(b, p), propertynames(a))
        end

        Adata = ComplexF64[1.0 -1.0; 1.0 1.0]
        Ddata = Diagonal(ComplexF64[0.75, 0.25 * cis(pi / 3)])
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
                @test _isapprox(
                    convert(MatrixEnsemble{Float64}, ensemblesF[1]), ensembles[1]
                )
                @test _isapprox(
                    convert(MatrixEnsemble{Float64}, ensemblesF[2]), ensembles[2]
                )
                @test _isapprox(
                    convert(MatrixEnsemble{Float64}, ensemblesF[3]), ensembles[3]
                )
                @test _isapprox(
                    convert(MatrixEnsemble{Float64}, ensemblesF[4]), ensembles[4]
                )
                @test _isapprox(adapt(F, ensembles[1]), ensemblesF[1])
                @test _isapprox(adapt(F, ensembles[2]), ensemblesF[2])
                @test _isapprox(adapt(F, ensembles[3]), ensemblesF[3])
                @test _isapprox(adapt(F, ensembles[4]), ensemblesF[4])
                @test _isapprox(adapt(Float64, ensemblesF[1]), ensembles[1])
                @test _isapprox(adapt(Float64, ensemblesF[2]), ensembles[2])
                @test _isapprox(adapt(Float64, ensemblesF[3]), ensembles[3])
                @test _isapprox(adapt(Float64, ensemblesF[4]), ensembles[4])
            end
        end

        # Sanity checks
        @test_throws ArgumentError MatrixEnsemble(I, I, I, Adata)  # T not diagonal
        @test_throws ArgumentError MatrixEnsemble(I, I, I, 2im * I)  # T not in unit disc
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
            @test γ²(γrF, () -> s²minF) === γ²valueF
            @test δ²(γrF, s²minF) === δ²valueF
            @test δ²(γrF, () -> s²minF) === δ²valueF
            @test γ²(δrF, s²minF) === γ²valueF
            @test γ²(δrF, () -> s²minF) === γ²valueF
            @test δ²(δrF, s²minF) === δ²valueF
            @test δ²(δrF, () -> s²minF) === δ²valueF
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
    @testset "problems/solutions" begin
        ensemble, z, μ, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, 1.0, false
        k, rkind, rvalue, q0, E = 1.0, :δ, 0.5, complex(eps()), 0.5 - im
        regularizer = Regularizer(rkind, rvalue)

        eq = LocalResolventEquation(ensemble, z, μ, normalizeQ)
        qproblem = LocalQProblem(eq, regularizer, q0)
        qrproblem = LocalQRegularizerProblem(eq, k, q0, regularizer)
        solution = LocalResolventSolution(qproblem, q0, regularizer, rvalue, rvalue, k, E)

        # Traits
        @test eq isa LocalResolventEquation{Float64}
        @test qproblem isa LocalQProblem{Float64}
        @test qrproblem isa LocalQRegularizerProblem{Float64}
        @test solution isa LocalResolventSolution{Float64}

        @testset "$F" for (
            F, (zF, μF, kF, rvalueF, q0F, EF)
        ) in floattypes(z, μ, k, rvalue, q0, E)
            ensembleF = MatrixEnsemble{F}(I, I, I, I)
            regularizerF = Regularizer(rkind, rvalueF)

            eqF = LocalResolventEquation(ensembleF, zF, μF, normalizeQ)
            qproblemF = LocalQProblem(eqF, regularizerF, q0F)
            qrproblemF = LocalQRegularizerProblem(eqF, kF, q0F, regularizerF)
            solutionF = LocalResolventSolution(
                qproblemF, q0F, regularizerF, rvalueF, rvalueF, kF, EF
            )

            # Floattype inferring constructor
            @test eqF isa LocalResolventEquation{F}
            @test qproblemF isa LocalQProblem{F}
            @test qrproblemF isa LocalQRegularizerProblem{F}
            @test solutionF isa LocalResolventSolution{F}

            # Floattype enforcing constructor
            @test LocalResolventEquation{F}(ensemble, z, μ, normalizeQ) === eqF
            @test LocalQProblem{F}(eq, regularizer, q0) === qproblemF
            @test LocalQRegularizerProblem{F}(eq, k, q0, regularizer) === qrproblemF
            @test LocalResolventSolution{F}(
                qproblem, q0, regularizer, rvalue, rvalue, k, E
            ) === solutionF
            @test LocalResolventEquation{Float64}(ensembleF, zF, μF, normalizeQ) === eq
            @test LocalQProblem{Float64}(eqF, regularizerF, q0F) === qproblem
            @test LocalQRegularizerProblem{Float64}(
                eqF, kF, q0F, regularizerF
            ) === qrproblem
            @test LocalResolventSolution{Float64}(
                qproblemF, q0F, regularizerF, rvalueF, rvalueF, kF, EF
            ) === solution

            # Floattype conversions
            @test convert(LocalResolventEquation{F}, eq) === eqF
            @test convert(LocalQProblem{F}, qproblem) === qproblemF
            @test convert(LocalQRegularizerProblem{F}, qrproblem) === qrproblemF
            @test convert(LocalResolventSolution{F}, solution) === solutionF
            @test convert(LocalResolventEquation{Float64}, eqF) === eq
            @test convert(LocalQProblem{Float64}, qproblemF) === qproblem
            @test convert(LocalQRegularizerProblem{Float64}, qrproblemF) === qrproblem
            @test convert(LocalResolventSolution{Float64}, solutionF) === solution
            @test adapt(F, eq) === eqF
            @test adapt(F, qproblem) === qproblemF
            @test adapt(F, qrproblem) === qrproblemF
            @test adapt(F, solution) === solutionF
            @test adapt(Float64, eqF) === eq
            @test adapt(Float64, qproblemF) === qproblem
            @test adapt(Float64, qrproblemF) === qrproblem
            @test adapt(Float64, solutionF) === solution
        end

        # Sanity checks: μ ∈ [0, 1]
        @test_throws ArgumentError LocalResolventEquation(ensemble, z, -0.1, normalizeQ)
        @test_throws ArgumentError LocalResolventEquation(ensemble, z, 1.1, normalizeQ)
        @test_throws ArgumentError LocalResolventEquation(ensemble, z, 0.5im, normalizeQ)
    end
    #=
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
    =#
    @testset "Workspace" begin
        function workspace_tests(F, M, L, R, T)
            _inv(A) = inv(A)
            _inv(A::SparseMatrixCSC) = inv(Matrix(A))
            _hermitian(A) = Hermitian(A)
            _hermitian(A::SparseMatrixCSC) = Hermitian(Matrix(A))
            _hermitian(A::Diagonal) = real(A)
            _trn(A) = tr(A) / size(A, 1)

            M, L, R, T = floatconvert.(F, (M, L, R, T))
            ens = MatrixEnsemble(M, L, R, T)
            ws = Workspace(ens)
            @test ws isa Workspace{F}

            z, q = convert.(Complex{F}, (-(im + 2) / 4, (1 - im) / 2))
            γ², δ² = Regularizer{F}(:γ, 0.25), Regularizer{F}(:δ, 0.5)
            μ = convert(F, 2 // 3)
            μens = MatrixEnsemble(M, L, R, μ * T)

            eq = LocalResolventEquation(ens, z, μ, false)
            eqn = LocalResolventEquation(ens, z, μ, true)
            μeq = LocalResolventEquation(μens, z, 1, false)
            μeqn = LocalResolventEquation(μens, z, 1, true)

            G0i = _inv(L) * (z * I - M) * _inv(R)
            if G0i isa UniformScaling
                G0i = SMatrix{size(ens)...}(G0i)
            end
            Q = G0i - q * μ * T
            QQh = Q * Q'
            QQhn = (QQh + Q' * Q) / 2
            s²min, s²minn = eigmin(_hermitian(QQh)), eigmin(_hermitian(QQhn))
            γ²valueδ = δ².value - s²min
            δ²valueγ = γ².value + s²min
            γ²valueδn = δ².value - s²minn
            δ²valueγn = γ².value + s²minn
            Pγ = _inv(QQh + γ².value * I)
            Pδ = _inv(QQh + γ²valueδ * I)
            Pγn = _inv(QQhn + γ².value * I)
            Pδn = _inv(QQhn + γ²valueδn * I)
            qrhsγ, qrhsδ = μ * _trn(T * Q' * Pγ), μ * _trn(T * Q' * Pδ)
            qrhsγn, qrhsδn = μ * _trn(T * Q' * Pγn), μ * _trn(T * Q' * Pδn)
            @test qrhs!(ws, eq, q, γ²) ≈ qrhsγ
            @test qrhs!(ws, eq, q, δ²) ≈ qrhsδ
            @test qrhs!(ws, eqn, q, γ²) ≈ qrhsγn
            @test qrhs!(ws, eqn, q, δ²) ≈ qrhsδn
            @test qrhs!(ws, μeq, q, γ²) ≈ qrhsγ
            @test qrhs!(ws, μeq, q, δ²) ≈ qrhsδ
            @test qrhs!(ws, μeqn, q, γ²) ≈ qrhsγn
            @test qrhs!(ws, μeqn, q, δ²) ≈ qrhsδn

            @test γ²!(ws, eq, q, γ²) ≈ γ².value
            @test γ²!(ws, eq, q, δ²) ≈ γ²valueδ
            @test γ²!(ws, eqn, q, γ²) ≈ γ².value
            @test γ²!(ws, eqn, q, δ²) ≈ γ²valueδn
            @test δ²!(ws, eq, q, γ²) ≈ δ²valueγ
            @test δ²!(ws, eq, q, δ²) ≈ δ².value
            @test δ²!(ws, eqn, q, γ²) ≈ δ²valueγn
            @test δ²!(ws, eqn, q, δ²) ≈ δ².value
            @test γ²!(ws, μeq, q, γ²) ≈ γ².value
            @test γ²!(ws, μeq, q, δ²) ≈ γ²valueδ
            @test γ²!(ws, μeqn, q, γ²) ≈ γ².value
            @test γ²!(ws, μeqn, q, δ²) ≈ γ²valueδn
            @test δ²!(ws, μeq, q, γ²) ≈ δ²valueγ
            @test δ²!(ws, μeq, q, δ²) ≈ δ².value
            @test δ²!(ws, μeqn, q, γ²) ≈ δ²valueγn
            @test δ²!(ws, μeqn, q, δ²) ≈ δ².value

            LiRi = _inv(R * L)
            Eγ, Eδ = _trn(LiRi * Q' * Pγ), _trn(LiRi * Q' * Pδ)
            Eγn, Eδn = _trn(LiRi * Q' * Pγn), _trn(LiRi * Q' * Pδn)
            @test E!(ws, eq, q, γ²) ≈ Eγ
            @test E!(ws, eq, q, δ²) ≈ Eδ
            @test E!(ws, eqn, q, γ²) ≈ Eγn
            @test E!(ws, eqn, q, δ²) ≈ Eδn
            @test E!(ws, μeq, q, γ²) ≈ Eγ
            @test E!(ws, μeq, q, δ²) ≈ Eδ
            @test E!(ws, μeqn, q, γ²) ≈ Eγn
            @test E!(ws, μeqn, q, δ²) ≈ Eδn
        end

        Adata, Ddata = ComplexF64[1 -1; 1 1], Diagonal([0.75, 0.25 * cis(pi / 3)])
        @testset "$F" for F in FLOATTYPES
            workspace_tests(F, I, I, I, I)
            @testset "$label" for (label, (A, D)) in arraytypes(Adata, Ddata)
                workspace_tests(F, A, I, I, I)
                workspace_tests(F, I, A, I, I)
                workspace_tests(F, I, I, A, I)
                workspace_tests(F, I, I, I, D)
            end
        end
    end
    @testset "eqsolve" begin
        function eqsolve_tests!(F, ws, problem)
            tol = eps(F)^convert(F, 3 // 4)
            sol = eqsolve!(ws, problem)
            @test sol.problem === problem
            if problem isa LocalQProblem
                @test sol.regularizer === problem.regularizer
            elseif problem isa LocalQRegularizerProblem
                @test sol.k === problem.k
            end
            @test sol.q ≈ qrhs!(ws, problem.equation, sol.q, sol.regularizer) rtol=tol atol=tol
            @test sol.γ² ≈ γ²!(ws, problem.equation, sol.q, sol.regularizer) rtol=tol atol=tol
            @test sol.δ² ≈ δ²!(ws, problem.equation, sol.q, sol.regularizer) rtol=tol atol=tol
            @test sol.k ≈ k!(ws, problem.equation, sol.q, sol.regularizer) rtol=tol atol=tol
            @test sol.E ≈ E!(ws, problem.equation, sol.q, sol.regularizer) rtol=tol atol=tol
        end

        Adata, Ddata = ComplexF64[1 0.1im; 0 -1], Diagonal([0.75, 0.25 * cis(pi / 3)])
        z_, μ_, γ²value_, δ²value_ = complex(0.8), 0.5, 100.0, 100.0
        k_, q0_ = 0.01, DEFAULTS.q0
        @testset "$label" for (label, (A, D)) in arraytypes(Adata, Ddata)
            @testset "$F" for (
                F, (z, μ, γ²value, δ²value, k, q0)
            ) in floattypes(z_, μ_, γ²value_, δ²value_, k_, q0_)
                γ², δ² = Regularizer(:γ, γ²value), Regularizer(:δ, δ²value)
                @testset "$args" for (args, ensemble) in (
                    ("(I, I, I, I)", MatrixEnsemble{F}(I, I, I, I)),
                    ("(A, I, I, I)", MatrixEnsemble{F}(A, I, I, I)),
                    ("(I, A, I, I)", MatrixEnsemble{F}(I, A, I, I)),
                    ("(I, I, A, I)", MatrixEnsemble{F}(I, I, A, I)),
                    ("(I, I, I, D)", MatrixEnsemble{F}(I, I, I, D)),
                )
                    ws = Workspace(ensemble)
                    eq = LocalResolventEquation(ensemble, z, μ, false)
                    eqn = LocalResolventEquation(ensemble, z, μ, true)

                    eqsolve_tests!(F, ws, LocalQProblem(eq, γ², q0))
                    eqsolve_tests!(F, ws, LocalQProblem(eq, δ², q0))
                    eqsolve_tests!(F, ws, LocalQProblem(eqn, γ², q0))
                    eqsolve_tests!(F, ws, LocalQProblem(eqn, δ², q0))
                    eqsolve_tests!(F, ws, LocalQRegularizerProblem(eq, k, q0, γ²))
                    eqsolve_tests!(F, ws, LocalQRegularizerProblem(eq, k, q0, δ²))
                    eqsolve_tests!(F, ws, LocalQRegularizerProblem(eqn, k, q0, γ²))
                    eqsolve_tests!(F, ws, LocalQRegularizerProblem(eqn, k, q0, δ²))
                end
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

            H = psdmul(X)
            @test H ≈ X * X'
            H = psdmul!!(H, X)
            @test H ≈ X * X'
            α, β = convert.(real(eltype(X)), (0.25, 1.33))
            H = psdmul!!(H, X', α, β)
            @test H ≈ α * X' * X + β * X * X'
        end

        commondata = (
            shift=0.5,
            D=Diagonal([1.5, 2.0]),
            Di=Diagonal([2, 1.5] / 3),
            Dsi=Diagonal([0.5, 0.4]),
        )
        @testset "$field" for field in ["Real", "Complex"]
            @testset "$F" for (
                F,
                (
                    shift,
                    D,
                    Di,
                    Dsi,
                    B,
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
                            A, Ai, Aeigmin, shift, Asi, _sarray.((B, A_B, As_B, X))...
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
    @testset "newton" begin
        @testset "scalar" begin
            f(x) = cos(x) - x^3
            df(x, _) = -sin(x) - 3x^2
            x0value = 0.5

            @testset "$F" for (F, (x0,)) in floattypes(x0value)
                # finite_difference_jacobian
                rtol = eps(F)^convert(F, 2 // 5)
                fx0 = f(x0)
                dfx0 = df(x0, fx0)
                @test finite_difference_jacobian(f, x0, fx0) ≈ dfx0 rtol=rtol
                @test finite_difference_jacobian(f, x0) ≈ dfx0 rtol=rtol

                # newton
                atol = eps(F)^convert(F, 3 // 4)
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
                rtol = eps(F)^convert(F, 2 // 5)
                gx0 = g(x0)
                jgx0 = jg(x0, gx0)
                @test finite_difference_jacobian(g, x0, gx0) ≈ jgx0 rtol=rtol
                @test finite_difference_jacobian(g, x0) ≈ jgx0 rtol=rtol

                # newton
                atol = eps(F)^convert(F, 3 // 4)
                normalizednorm(y) = norm(y) / sqrt(length(y))
                @test normalizednorm(g(first(newton(g, x0)))) < atol
                @test normalizednorm(g(first(newton(g, jg, x0)))) < atol

                @test_throws NewtonException newton(g, x0; maxiters=0)
                @test_throws NewtonException newton(g, jg, x0; maxiters=0)
            end
        end
    end
end
