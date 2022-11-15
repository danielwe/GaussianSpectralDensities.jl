using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt, γ², δ²
using GaussianSpectralDensities: Regularizer, LocalResolventEquation
using GaussianSpectralDensities: LocalRegularizerProblem, LocalTraceProblem
using GaussianSpectralDensities: Workspace, q_μrhs!, γ²!, δ²!, k!, E!
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
        γreg, δreg = Regularizer(:γ, γ²value), Regularizer(:δ, δ²value)

        # Traits
        @test γreg isa Regularizer{Float64}
        @test δreg isa Regularizer{Float64}

        @testset "$F" for (F, (γ²valueF, δ²valueF, s²minF)) in floattypes(
            γ²value, δ²value, s²min
        )
            γregF, δregF = Regularizer(:γ, γ²valueF), Regularizer(:δ, δ²valueF)

            # Floattype inferring constructor
            @test γregF isa Regularizer{F}
            @test δregF isa Regularizer{F}

            # Floattype enforcing constructor
            @test Regularizer{F}(:γ, γ²value) === γregF
            @test Regularizer{F}(:δ, δ²value) === δregF
            @test Regularizer{Float64}(:γ, γ²valueF) === γreg
            @test Regularizer{Float64}(:δ, δ²valueF) === δreg

            # Floattype conversions
            @test convert(Regularizer{F}, γreg) === γregF
            @test convert(Regularizer{F}, δreg) === δregF
            @test convert(Regularizer{Float64}, γregF) === γreg
            @test convert(Regularizer{Float64}, δregF) === δreg
            @test adapt(F, γreg) === γregF
            @test adapt(F, δreg) === δregF
            @test adapt(Float64, γregF) === γreg
            @test adapt(Float64, δregF) === δreg

            # Interface
            @test γ²(γregF, s²minF) === γ²valueF
            @test γ²(γregF, () -> s²minF) === γ²valueF
            @test δ²(γregF, s²minF) === δ²valueF
            @test δ²(γregF, () -> s²minF) === δ²valueF
            @test γ²(δregF, s²minF) === γ²valueF
            @test γ²(δregF, () -> s²minF) === γ²valueF
            @test δ²(δregF, s²minF) === δ²valueF
            @test δ²(δregF, () -> s²minF) === δ²valueF
        end
    end
    @testset "objectives" begin
        kvalue, regkind, regvalue = 1.0, :δ, 0.5
        reg = Regularizer(regkind, regvalue)
        ko = TraceObjective(kvalue)
        rego = RegularizerObjective(reg)

        # Traits
        @test ko isa TraceObjective{Float64}
        @test rego isa RegularizerObjective{Float64}

        # Defaults
        @test TraceObjective() === TraceObjective(DEFAULTS.kobjective)
        @test RegularizerObjective() === RegularizerObjective(DEFAULTS.regobjective)

        @testset "$F" for (F, (kvalueF, regvalueF)) in floattypes(kvalue, regvalue)
            regF = Regularizer(regkind, regvalueF)
            koF = TraceObjective(kvalueF)
            regoF = RegularizerObjective(regF)

            # Floattype inferring constructor
            @test koF isa TraceObjective{F}
            @test regoF isa RegularizerObjective{F}
            # - All the signatures
            @test RegularizerObjective(regkind, regvalueF) === regoF

            # Floattype enforcing constructor
            @test TraceObjective{F}(kvalue) === koF
            @test RegularizerObjective{F}(reg) === regoF
            @test TraceObjective{Float64}(kvalueF) === ko
            @test RegularizerObjective{Float64}(regF) === rego

            # Floattype conversions
            @test convert(TraceObjective{F}, ko) === koF
            @test convert(RegularizerObjective{F}, rego) === regoF
            @test convert(TraceObjective{Float64}, koF) === ko
            @test convert(RegularizerObjective{Float64}, regoF) === rego
            @test adapt(F, ko) === koF
            @test adapt(F, rego) === regoF
            @test adapt(Float64, koF) === ko
            @test adapt(Float64, regoF) === rego
        end
    end
    @testset "problems/solutions" begin
        ensemble, z, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, false
        k, regkind, regvalue, q, μ, E = 1.0, :δ, 0.5, complex(eps()), 1.0, 0.5 - im
        reg = Regularizer(regkind, regvalue)

        eq = LocalResolventEquation(ensemble, z, normalizeQ)
        regproblem = LocalRegularizerProblem(eq, reg, q, μ)
        kproblem = LocalTraceProblem(eq, k, q, reg, μ)
        solution = LocalResolventSolution(regproblem, q, reg, regvalue, regvalue, k, E)

        # Traits
        @test eq isa LocalResolventEquation{Float64}
        @test regproblem isa LocalRegularizerProblem{Float64}
        @test kproblem isa LocalTraceProblem{Float64}
        @test solution isa LocalResolventSolution{Float64}

        # Defaults
        @test LocalResolventEquation(
            ensemble, z
        ) === LocalResolventEquation(ensemble, z, false)
        @test LocalRegularizerProblem(eq, reg, q) === LocalRegularizerProblem(eq, reg, q, 1)
        @test LocalTraceProblem(eq, k, q, reg) === LocalTraceProblem(eq, k, q, reg, 1)

        @testset "$F" for (
            F, (zF, kF, regvalueF, qF, μF, EF)
        ) in floattypes(z, k, regvalue, q, μ, E)
            ensembleF = MatrixEnsemble{F}(I, I, I, I)
            regF = Regularizer(regkind, regvalueF)

            eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)
            regproblemF = LocalRegularizerProblem(eqF, regF, qF, μF)
            kproblemF = LocalTraceProblem(eqF, kF, qF, regF, μF)
            solutionF = LocalResolventSolution(
                regproblemF, qF, regF, regvalueF, regvalueF, kF, EF
            )

            # Floattype inferring constructor
            @test eqF isa LocalResolventEquation{F}
            @test regproblemF isa LocalRegularizerProblem{F}
            @test kproblemF isa LocalTraceProblem{F}
            @test solutionF isa LocalResolventSolution{F}

            # Floattype enforcing constructor
            @test LocalResolventEquation{F}(ensemble, z, normalizeQ) === eqF
            @test LocalRegularizerProblem{F}(eq, reg, q, μ) === regproblemF
            @test LocalTraceProblem{F}(eq, k, q, reg, μ) === kproblemF
            @test LocalResolventSolution{F}(
                regproblem, q, reg, regvalue, regvalue, k, E
            ) === solutionF
            @test LocalResolventEquation{Float64}(ensembleF, zF, normalizeQ) === eq
            @test LocalRegularizerProblem{Float64}(eqF, regF, qF, μF) === regproblem
            @test LocalTraceProblem{Float64}(eqF, kF, qF, regF, μF) === kproblem
            @test LocalResolventSolution{Float64}(
                regproblemF, qF, regF, regvalueF, regvalueF, kF, EF
            ) === solution

            # Floattype conversions
            @test convert(LocalResolventEquation{F}, eq) === eqF
            @test convert(LocalRegularizerProblem{F}, regproblem) === regproblemF
            @test convert(LocalTraceProblem{F}, kproblem) === kproblemF
            @test convert(LocalResolventSolution{F}, solution) === solutionF
            @test convert(LocalResolventEquation{Float64}, eqF) === eq
            @test convert(LocalRegularizerProblem{Float64}, regproblemF) === regproblem
            @test convert(LocalTraceProblem{Float64}, kproblemF) === kproblem
            @test convert(LocalResolventSolution{Float64}, solutionF) === solution
            @test adapt(F, eq) === eqF
            @test adapt(F, regproblem) === regproblemF
            @test adapt(F, kproblem) === kproblemF
            @test adapt(F, solution) === solutionF
            @test adapt(Float64, eqF) === eq
            @test adapt(Float64, regproblemF) === regproblem
            @test adapt(Float64, kproblemF) === kproblem
            @test adapt(Float64, solutionF) === solution
        end

        # Sanity checks: μ ∈ [0, 1]
        @test_throws ArgumentError LocalRegularizerProblem(eq, reg, q, -0.1)
        @test_throws ArgumentError LocalTraceProblem(eq, k, q, reg, -0.1)
        @test_throws ArgumentError LocalRegularizerProblem(eq, reg, q, 1.1)
        @test_throws ArgumentError LocalTraceProblem(eq, k, q, reg, 1.1)
        @test_throws ArgumentError LocalRegularizerProblem(eq, reg, q, 0.5im)
        @test_throws ArgumentError LocalTraceProblem(eq, k, q, reg, 0.5im)
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
            ensemble = MatrixEnsemble(M, L, R, T)
            ws = Workspace(ensemble)
            @test ws isa Workspace{F}

            z, q = convert.(Complex{F}, (-(im + 2) / 4, (1 - im) / 2))
            γ², δ² = Regularizer{F}(:γ, 0.25), Regularizer{F}(:δ, 0.5)
            μ = convert(F, 2 // 3)
            μq = μ * q
            μensemble = MatrixEnsemble(M, L, R, μ * T)

            eq = LocalResolventEquation(ensemble, z, false)
            eqn = LocalResolventEquation(ensemble, z, true)
            μeq = LocalResolventEquation(μensemble, z, false)
            μeqn = LocalResolventEquation(μensemble, z, true)

            G0i = _inv(L) * (z * I - M) * _inv(R)
            if G0i isa UniformScaling
                G0i = SMatrix{size(ensemble)...}(G0i)
            end
            Q = G0i - μq * T
            QQh = Q * Q'
            QQhn = (QQh + Q' * Q) / 2
            s²min, s²minn = eigmin(_hermitian(QQh)), eigmin(_hermitian(QQhn))

            γ²valueδ, δ²valueγ= δ².value - s²min, γ².value + s²min
            γ²valueδn, δ²valueγn = δ².value - s²minn, γ².value + s²minn
            @test γ²!(ws, eq, μq, γ²) ≈ γ².value
            @test γ²!(ws, eq, μq, δ²) ≈ γ²valueδ
            @test γ²!(ws, eqn, μq, γ²) ≈ γ².value
            @test γ²!(ws, eqn, μq, δ²) ≈ γ²valueδn
            @test δ²!(ws, eq, μq, γ²) ≈ δ²valueγ
            @test δ²!(ws, eq, μq, δ²) ≈ δ².value
            @test δ²!(ws, eqn, μq, γ²) ≈ δ²valueγn
            @test δ²!(ws, eqn, μq, δ²) ≈ δ².value
            @test γ²!(ws, μeq, q, γ²) ≈ γ².value
            @test γ²!(ws, μeq, q, δ²) ≈ γ²valueδ
            @test γ²!(ws, μeqn, q, γ²) ≈ γ².value
            @test γ²!(ws, μeqn, q, δ²) ≈ γ²valueδn
            @test δ²!(ws, μeq, q, γ²) ≈ δ²valueγ
            @test δ²!(ws, μeq, q, δ²) ≈ δ².value
            @test δ²!(ws, μeqn, q, γ²) ≈ δ²valueγn
            @test δ²!(ws, μeqn, q, δ²) ≈ δ².value

            Pγ, Pδ = _inv(QQh + γ².value * I), _inv(QQh + γ²valueδ * I)
            Pγn, Pδn = _inv(QQhn + γ².value * I), _inv(QQhn + γ²valueδn * I)

            q_μrhsγ, q_μrhsδ = _trn(T * Q' * Pγ), _trn(T * Q' * Pδ)
            q_μrhsγn, q_μrhsδn = _trn(T * Q' * Pγn), _trn(T * Q' * Pδn)
            @test q_μrhs!(ws, eq, μq, γ²) ≈ q_μrhsγ
            @test q_μrhs!(ws, eq, μq, δ²) ≈ q_μrhsδ
            @test q_μrhs!(ws, eqn, μq, γ²) ≈ q_μrhsγn
            @test q_μrhs!(ws, eqn, μq, δ²) ≈ q_μrhsδn
            @test q_μrhs!(ws, μeq, q, γ²) ≈ μ * q_μrhsγ
            @test q_μrhs!(ws, μeq, q, δ²) ≈ μ * q_μrhsδ
            @test q_μrhs!(ws, μeqn, q, γ²) ≈ μ * q_μrhsγn
            @test q_μrhs!(ws, μeqn, q, δ²) ≈ μ * q_μrhsδn

            kγ, kδ = _trn(Pγ), _trn(Pδ)
            kγn, kδn = _trn(Pγn), _trn(Pδn)
            @test k!(ws, eq, μq, γ²) ≈ kγ
            @test k!(ws, eq, μq, δ²) ≈ kδ
            @test k!(ws, eqn, μq, γ²) ≈ kγn
            @test k!(ws, eqn, μq, δ²) ≈ kδn
            @test k!(ws, μeq, q, γ²) ≈ kγ
            @test k!(ws, μeq, q, δ²) ≈ kδ
            @test k!(ws, μeqn, q, γ²) ≈ kγn
            @test k!(ws, μeqn, q, δ²) ≈ kδn

            LiRi = _inv(R * L)

            Eγ, Eδ = _trn(LiRi * Q' * Pγ), _trn(LiRi * Q' * Pδ)
            Eγn, Eδn = _trn(LiRi * Q' * Pγn), _trn(LiRi * Q' * Pδn)
            @test E!(ws, eq, μq, γ²) ≈ Eγ
            @test E!(ws, eq, μq, δ²) ≈ Eδ
            @test E!(ws, eqn, μq, γ²) ≈ Eγn
            @test E!(ws, eqn, μq, δ²) ≈ Eδn
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
            if problem isa LocalRegularizerProblem
                @test sol.reg === problem.reg
            elseif problem isa LocalTraceProblem
                @test sol.k === problem.k
            end
            μq = problem.μ * sol.q
            @test sol.q ≈ problem.μ * q_μrhs!(ws, problem.eq, μq, sol.reg) rtol=tol atol=tol
            @test sol.γ² ≈ γ²!(ws, problem.eq, μq, sol.reg) rtol=tol atol=tol
            @test sol.δ² ≈ δ²!(ws, problem.eq, μq, sol.reg) rtol=tol atol=tol
            @test sol.k ≈ k!(ws, problem.eq, μq, sol.reg) rtol=tol atol=tol
            @test sol.E ≈ E!(ws, problem.eq, μq, sol.reg) rtol=tol atol=tol
        end

        Adata, Ddata = ComplexF64[1 0.1im; 0 -1], Diagonal([0.75, 0.25 * cis(pi / 3)])
        z_, μ_, γ²value_, δ²value_ = complex(0.8), 0.5, 100.0, 100.0
        k_, q_ = 0.01, DEFAULTS.q0
        @testset "$label" for (label, (A, D)) in arraytypes(Adata, Ddata)
            @testset "$F" for (
                F, (z, μ, γ²value, δ²value, k, q)
            ) in floattypes(z_, μ_, γ²value_, δ²value_, k_, q_)
                γ², δ² = Regularizer(:γ, γ²value), Regularizer(:δ, δ²value)
                @testset "$args" for (args, ensemble) in (
                    ("(I, I, I, I)", MatrixEnsemble{F}(I, I, I, I)),
                    ("(A, I, I, I)", MatrixEnsemble{F}(A, I, I, I)),
                    ("(I, A, I, I)", MatrixEnsemble{F}(I, A, I, I)),
                    ("(I, I, A, I)", MatrixEnsemble{F}(I, I, A, I)),
                    ("(I, I, I, D)", MatrixEnsemble{F}(I, I, I, D)),
                )
                    ws = Workspace(ensemble)
                    eq = LocalResolventEquation(ensemble, z, false)
                    eqn = LocalResolventEquation(ensemble, z, true)

                    eqsolve_tests!(F, ws, LocalRegularizerProblem(eq, γ², q, μ))
                    eqsolve_tests!(F, ws, LocalRegularizerProblem(eq, δ², q, μ))
                    eqsolve_tests!(F, ws, LocalRegularizerProblem(eqn, γ², q, μ))
                    eqsolve_tests!(F, ws, LocalRegularizerProblem(eqn, δ², q, μ))
                    eqsolve_tests!(F, ws, LocalTraceProblem(eq, k, q, γ², μ))
                    eqsolve_tests!(F, ws, LocalTraceProblem(eq, k, q, δ², μ))
                    eqsolve_tests!(F, ws, LocalTraceProblem(eqn, k, q, γ², μ))
                    eqsolve_tests!(F, ws, LocalTraceProblem(eqn, k, q, δ², μ))
                end
            end
        end
    end
    @testset "algorithms" begin
        @testset "DirectSolve" begin
            dsolve = DirectSolve()

            # Traits
            @test dsolve isa DirectSolve
            @test length(dsolve) == 0

            @testset "$F" for F in FLOATTYPES
                @test adapt(F, dsolve) === dsolve
            end
        end
        @testset "RegularizerSchedule" begin
            regkind, regvalues, γ²min = :δ, 2.0.^(2:-1:-2), 0.0
            regschedule = Regularizer.(regkind, regvalues)
            schedule = RegularizerSchedule(regschedule, γ²min)

            # Traits
            @test schedule isa RegularizerSchedule{Float64}
            @test length(schedule) == length(regvalues)

            # Defaults
            @test RegularizerSchedule() === RegularizerSchedule(
                DEFAULTS.regschedule, DEFAULTS.γ²min
            )

            @testset "$F" for (F, (regvaluesF, γ²minF)) in floattypes(regvalues, γ²min)
                regscheduleF = Regularizer.(regkind, regvaluesF)
                scheduleF = RegularizerSchedule(regscheduleF, γ²minF)

                # Floattype inferring constructor
                @test scheduleF isa RegularizerSchedule{F}
                # - All the signatures
                @test RegularizerSchedule(regkind, regvaluesF, γ²minF) == scheduleF

                # Floattype enforcing constructor
                @test RegularizerSchedule{F}(regschedule, γ²min) == scheduleF
                @test RegularizerSchedule{Float64}(regscheduleF, γ²minF) == schedule

                # Floattype conversions
                @test convert(RegularizerSchedule{F}, schedule) == scheduleF
                @test convert(RegularizerSchedule{Float64}, scheduleF) == schedule
                @test adapt(F, schedule) == scheduleF
                @test adapt(Float64, scheduleF) == schedule
            end
        end
        @testset "SymmetrySchedule" begin
            steps, γ²min, ialg = 3, 0.0, RegularizerSchedule()
            μschedule = range(0.0, 1.0, steps)
            schedule = SymmetrySchedule(μschedule, γ²min, ialg)

            # Traits
            @test schedule isa SymmetrySchedule{Float64,<:RegularizerSchedule{Float64}}
            @test length(schedule) == length(ialg) + length(μschedule)

            # All the signatures
            @test SymmetrySchedule(steps, γ²min, ialg) === schedule

            # Defaults
            @test SymmetrySchedule() === SymmetrySchedule(DEFAULTS.μschedule, DEFAULTS.γ²min)

            @testset "$F" for (F, (γ²minF,)) in floattypes(γ²min)
                μscheduleF, ialgF = adapt.(F, (μschedule, ialg))
                scheduleF = SymmetrySchedule(μscheduleF, γ²minF, ialgF)

                # Floattype inferring constructor
                @test scheduleF isa SymmetrySchedule{F,<:RegularizerSchedule{F}}
                # - Defaults
                @test SymmetrySchedule(μscheduleF, γ²min) === SymmetrySchedule{F}(
                    μschedule, γ²min, DirectSolve()
                )
                @test SymmetrySchedule(steps, γ²minF) === SymmetrySchedule{F}(
                    μschedule, γ²min, DirectSolve()
                )

                # Capture any roundtrip truncation loss---that's not what we want to test
                ialg64 = adapt(Float64, ialgF)
                schedule64 = SymmetrySchedule(μschedule, γ²min, ialg64)

                # Floattype enforcing constructor
                @test SymmetrySchedule{F}(μschedule, γ²min, ialg) == scheduleF
                @test SymmetrySchedule{Float64}(μscheduleF, γ²minF, ialgF) == schedule64

                # Floattype conversions
                @test convert(SymmetrySchedule{F}, schedule) == scheduleF
                @test convert(SymmetrySchedule{Float64}, scheduleF) == schedule64
                @test adapt(F, schedule) == scheduleF
                @test adapt(Float64, scheduleF) == schedule64
            end

            # Sanity checks
            @test_throws ArgumentError SymmetrySchedule([0.2, 0.6, 1.0], γ²min, ialg)  # Doesn't start at 0
            @test_throws ArgumentError SymmetrySchedule(range(0.2, 1.0, 3), γ²min, ialg)
            @test_throws ArgumentError SymmetrySchedule([0.0, 0.4, 0.8], γ²min, ialg)  # Doesn't end at 1
            @test_throws ArgumentError SymmetrySchedule(range(0.0, 0.8, 3), γ²min, ialg)
            @test_throws ArgumentError SymmetrySchedule([0.0, 0.67, 0.33, 1.0], γ²min, ialg)  # Not monotonic
            @test_throws ArgumentError SymmetrySchedule(range(1.0, 0.0, 3), γ²min, ialg)
        end
    end
    @testset "solve" begin

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
