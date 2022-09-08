using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt
using GaussianSpectralDensities: _psdmul!!, _factorize, _factorize!!, _ldiv!!, _inv, _inv!!
using GaussianSpectralDensities: _eigmin!!
using GaussianSpectralDensities: Workspace, G0i!, Q!, P!, _ldiv, _rdiv
using GaussianSpectralDensities: newton, finite_difference_jacobian, NewtonException

using LinearAlgebra
using StaticArrays
using SparseArrays
using SuiteSparse

sarray(A::AbstractMatrix) = SMatrix{size(A)...}(A)
sarray(D::Diagonal) = Diagonal(SVector{length(D.diag)}(D.diag))

_sparse(A) = sparse(A)
_sparse(D::Diagonal) = Diagonal(sparse(D.diag))

@testset "GaussianSpectralDensities.jl" begin
    @testset "ensembles" begin
        @testset "MatrixEnsemble" begin
            A32, A64 = ComplexF32[1 -1; 1 1], ComplexF64[1 -1; 1 1]
            D32, D64 = Diagonal.((Float32[0.75, 0.25], Float64[0.75, 0.25]))
            N = length(D64.diag)

            @testset "$label" for (label, M, arrays...) in (
                ("MatrixEnsemble{Float32}", MatrixEnsemble{Float32}, A32, D32, A64, D64),
                ("MatrixEnsemble{Float64}", MatrixEnsemble{Float64}, A64, D64, A32, D32),
            )
                @testset "$label" for (label, A, D, cA, cD) in (
                    ("Array", arrays...),
                    ("SArray", sarray.(arrays)...),
                    ("SparseMatrixCSC", _sparse.(arrays)...),
                )
                    # Floattype enforcing constructor
                    @test M(I, I, I, I) isa M
                    @test M(cA, I, I, I) isa M
                    @test M(I, cA, I, I) isa M
                    @test M(I, I, cA, I) isa M
                    @test M(I, I, I, cD) isa M

                    # Floattype inferring constructor
                    @test MatrixEnsemble(A, I, I, I) isa M
                    @test MatrixEnsemble(I, A, I, I) isa M
                    @test MatrixEnsemble(I, I, A, I) isa M
                    @test MatrixEnsemble(I, I, I, D) isa M
                end
            end

            # Sanity checks
            @test_throws ArgumentError MatrixEnsemble(I, I, I, real(A64))  # T not diagonal
            @test_throws ArgumentError MatrixEnsemble(I, I, I, complex(D64))  # T not real
            @test_throws ArgumentError MatrixEnsemble(I, I, I, 2I)  # T not in unit interval
            @test_throws ArgumentError MatrixEnsemble(I, I, I, -I)  # T not in unit interval
            @test_throws DimensionMismatch MatrixEnsemble(A64, I(N + 1), I, I)
            @test_throws DimensionMismatch MatrixEnsemble(A64, I, I(N + 1), I)
            @test_throws DimensionMismatch MatrixEnsemble(A64, I, I, I(N + 1))
            @test_throws DimensionMismatch MatrixEnsemble(hcat(A64, D64), I, I, I)
        end
    end
    @testset "regularizers" begin
        γ32, γ64 = Regularizer.(:γ, (0.0f0, 0.0))
        δ32, δ64 = Regularizer.(:δ, (0.5f0, 0.5))

        @test γ32.value === 0.0f0
        @test δ32.value === 0.5f0
        @test γ32.kind == :γ
        @test δ32.kind == :δ
        @test :γ == γ32.kind 
        @test :δ == δ32.kind

        @test γ64.value === 0.0
        @test δ64.value === 0.5
        @test γ64.kind == :γ
        @test δ64.kind == :δ
        @test :γ == γ64.kind 
        @test :δ == δ64.kind

        @test convert(Regularizer{Float32}, γ64) === γ32
        @test convert(Regularizer{Float32}, δ64) === δ32

        @test convert(Regularizer{Float64}, γ64) === γ64
        @test convert(Regularizer{Float64}, δ64) === δ64

        @test adapt(Float32, γ64) === γ32
        @test adapt(Float32, δ64) === δ32

        @test adapt(Float64, γ64) === γ64
        @test adapt(Float64, δ64) === δ64
    end
    @testset "objectives" begin
        @test TraceObjective() === TraceObjective(DEFAULTS.kobjective)
        @test TraceObjective(1.0) === TraceObjective(value=1.0)
        @test RegularizerObjective() === RegularizerObjective(DEFAULTS.robjective)
        r = Regularizer(:γ, 0.0)
        @test RegularizerObjective(kind=:γ, value=0.0) === RegularizerObjective(value=r)
        @test RegularizerObjective(r) === RegularizerObjective(value=r)

        k32, k64 = map(TraceObjective, (1.0f0, 1.0))
        γ32, γ64 = map(r -> RegularizerObjective(kind=:γ, value=r), (0.0f0, 0.0))
        δ32, δ64 = map(r -> RegularizerObjective(kind=:δ, value=r), (0.5f0, 0.5))

        @test k32.value === 1.0f0
        @test γ32.value === Regularizer(:γ, 0.0f0)
        @test δ32.value === Regularizer(:δ, 0.5f0)

        @test k64.value === 1.0
        @test γ64.value === Regularizer(:γ, 0.0)
        @test δ64.value === Regularizer(:δ, 0.5)

        @test convert(TraceObjective{Float32}, k64) === k32
        @test convert(RegularizerObjective{Float32}, γ64) === γ32
        @test convert(RegularizerObjective{Float32}, δ64) === δ32

        @test convert(TraceObjective{Float64}, k64) === k64
        @test convert(RegularizerObjective{Float64}, γ64) === γ64
        @test convert(RegularizerObjective{Float64}, δ64) === δ64

        @test adapt(Float32, k64) === k32
        @test adapt(Float32, γ64) === γ32
        @test adapt(Float32, δ64) === δ32

        @test adapt(Float64, k64) === k64
        @test adapt(Float64, γ64) === γ64
        @test adapt(Float64, δ64) === δ64
    end
    @testset "problems" begin
        ensemble = MatrixEnsemble(0.5f0I(2), I, I, I)

        # z deliberately not complex and/or Float32
        z = 1.0
        prob = ResolventProblem(ensemble, z)
        @test prob.z == z
        @test prob.z isa ComplexF32
        @test prob.objective === adapt(Float32, TraceObjective())
        @test !prob.normalizeQ 

        # eltype(zm) deliberately not complex and/or Float32
        @testset "$label" for (label, zm) in (
            ("without missing", [0.0 1.0]), ("with missing", [0.0 1.0; missing missing])
        )
            probm = ResolventProblem(ensemble, zm)
            @test isequal(probm.z, zm)  # isequal to handle missing correctly for tests
            @test eltype(probm.z) === Union{Missing,ComplexF32}
        end

        # regularizer deliberately not Float32
        regularizer = adapt(Float64, RegularizerObjective())
        probreg = ResolventProblem(ensemble, z, regularizer)
        @test probreg.objective === adapt(Float32, regularizer)

        # hitting all the signatures
        @test ResolventProblem(ensemble, z; normalizeQ=true).normalizeQ
        @test ResolventProblem(ensemble, z, regularizer; normalizeQ=true).normalizeQ
    end
    @testset "algorithms" begin
        @testset "DirectSolve" begin
            q0 = complex(eps())
            δ0 = 0.0
            @testset "$label" for (label, D) in (
                ("DirectSolve", DirectSolve),
                ("DirectSolve{Float32}", DirectSolve{Float32}),
                ("DirectSolve{Float64}", DirectSolve{Float64}),
            )
                @test D() === D(DEFAULTS.q0, DEFAULTS.δ0)
                @test D(q0, δ0) === D(q0=q0, δ0=δ0)
                @test length(D()) == 0
            end

            ds32, ds64 = DirectSolve{Float32}(), DirectSolve{Float64}()
            @test convert(DirectSolve{Float32}, ds64) === ds32
            @test convert(DirectSolve{Float64}, ds64) === ds64
            @test adapt(Float32, ds64) === ds32
            @test adapt(Float64, ds64) === ds64
        end
        @testset "RegularizerSchedule" begin
            k, v = :δ, 2.0.^(2:-1:-2)
            schedule = Regularizer.(k, v)
            γ²min = 0.0
            @testset "$label" for (label, R) in (
                ("RegularizerSchedule", RegularizerSchedule),
                ("RegularizerSchedule{Float32}", RegularizerSchedule{Float32}),
                ("RegularizerSchedule{Float64}", RegularizerSchedule{Float64}),
            )
                @test R() == R(DEFAULTS.rschedule, DEFAULTS.γ²min)
                @test R(schedule, γ²min) == R(schedule=schedule, γ²min=γ²min)
                @test R(schedule, γ²min) == R(k, v, γ²min)
                @test R(schedule, γ²min) == R(kind=k, schedule=v, γ²min=γ²min)
                @test length(R(schedule=schedule)) == length(schedule)
            end

            rs32, rs64 = RegularizerSchedule{Float32}(), RegularizerSchedule{Float64}()
            @test convert(RegularizerSchedule{Float32}, rs64) == rs32
            @test convert(RegularizerSchedule{Float64}, rs64) === rs64
            @test adapt(Float32, rs64) == rs32
            @test adapt(Float64, rs64) === rs64
        end
        @testset "SymmetrySchedule" begin
            steps = 3
            schedule = range(0, 1, steps)
            vschedule = [0.0, 0.5, 1.0]
            γ²min = 0.0
            ialg = RegularizerSchedule()
            @testset "$label" for (label, S) in (
                ("SymmetrySchedule", SymmetrySchedule),
                ("SymmetrySchedule{Float32}", SymmetrySchedule{Float32}),
                ("SymmetrySchedule{Float64}", SymmetrySchedule{Float64}),
            )
                @test S() == S(DEFAULTS.μschedule, DEFAULTS.γ²min, DirectSolve())
                ss = S(schedule=schedule, γ²min=γ²min, initialalg=ialg)
                @test ss == S(steps, γ²min, ialg)
                @test ss == S(schedule, γ²min, ialg)
                @test ss == S(schedule=steps, γ²min=γ²min, initialalg=ialg)
                @test length(S(vschedule, γ²min, ialg)) == length(vschedule) + length(ialg)
                @test_throws ArgumentError S(schedule=range(0.0, 0.8, 3))
                @test_throws ArgumentError S(schedule=range(0.2, 1.0, 3))
                @test_throws ArgumentError S(schedule=[0.0, 0.75, 0.25, 1.0])
            end

            ss32, ss64 = SymmetrySchedule{Float32}(), SymmetrySchedule{Float64}()
            @test convert(SymmetrySchedule{Float32}, ss64) == ss32
            @test convert(SymmetrySchedule{Float64}, ss64) === ss64
            @test adapt(Float32, ss64) == ss32
            @test adapt(Float64, ss64) === ss64
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
            F = _factorize(A)
            Ai = _inv(F)
            @test Ai ≈ Ai_true
            A_B = _ldiv!!(F, copy(B))
            @test A_B ≈ A_B_true
            F = _factorize!!(F, A; shift=shift)
            @test _isequal(F, _factorize(A; shift=shift))
            Ai = _inv!!(Ai, F)
            @test Ai ≈ Asi_true
            A_B = _ldiv!!(F, copy(B))
            @test A_B ≈ As_B_true

            @test _eigmin!!(F, A) ≈ Aeigmin_true

            H = _psdmul!!(deepcopy(A), X)
            @test H ≈ X * X'
            α, β = convert.(real(eltype(X)), (0.25, 1.33))
            H = _psdmul!!(H, X', α, β)
            @test H ≈ α * X' * X + β * X * X'
        end

        @testset "$E" for E in (Float32, Float64, ComplexF32, ComplexF64)
            _shift = 0.5
            D = Diagonal(real(E)[1.5, 2.0])
            Di = Diagonal(real(E)[2, 1.5] / 3)
            Dsi = Diagonal(real(E)[0.5, 0.4])
            U = real(E)(2) * I
            Ui = real(E)(0.5) * I
            Usi = real(E)(0.4) * I
            if E <: Real
                Adata = E[1.5 -1; 1 2]  # deliberate asymmetry, masked in two different ways
                Aidata = E[1 0.5; -0.5 0.75]
                Aeigmin = E(0.719223593595585)
                shift = E(_shift)
                Asidata = E[5 2; -2 4] / 8
                B = E[1 -1; 1 1]
                A_B = Dict(:U => E[1.5 -0.5; 1.25 0.25], :L => E[0.5 -1.5; 0.25 1.25])
                As_B = Dict(:U => E[7 -3; 6 2] / 8, :L => E[3 -7; 2 6] / 8)
                D_B = E[2 -2; 1.5 1.5] / 3
                Ds_B = E[0.5 -0.5; 0.4 0.4]
                U_B = E[0.5 -0.5; 0.5 0.5]
                Us_B = E[0.4 -0.4; 0.4 0.4]
            else
                Adata = E[1.5 (-1 + im); (1 + im) 2]  # deliberate ahermiticity
                Aidata = E[2 (1 - im); (-1 - im) 1.5]
                Aeigmin = real(E)(0.3138593383654923)
                shift = real(E)(_shift)
                Asidata = E[5 (2 - 2im); (-2 - 2im) 4] / 6
                B = E[(1 + im) -1; 1 (1 - im)]
                A_B = Dict(
                    :U => E[(3 + im) (-2 - 2im); (1.5 + 2im) (0.5 - 2.5im)],
                    :L => E[(1 + 3im) (-2 + 2im); (1.5 - 2im) (2.5 - 0.5im)],
                )
                As_B = Dict(
                    :U => E[(7 + 3im) (-5 - 4im); (4 + 4im) (2 - 6im)] / 6,
                    :L => E[(3 + 7im) (-5 + 4im); (4 - 4im) (6 - 2im)] / 6,
                )
                D_B = E[(2 + 2im) -2; 1.5 (1.5 - 1.5im)] / 3
                Ds_B = E[(0.5 + 0.5im) -0.5; 0.4 (0.4 - 0.4im)]
                U_B = E[(0.5 + 0.5im) -0.5; 0.5 (0.5 - 0.5im)]
                Us_B = E[(0.4 + 0.4im) -0.4; 0.4 (0.4 - 0.4im)]
            end
            X = B
            D_X = Diagonal(B)
            U_X = UniformScaling(first(B))
            @testset "uplo=:$uplo" for uplo in (:U, :L)
                Ai = Hermitian(Aidata, uplo)
                Asi = Hermitian(Asidata, uplo)
                @testset "Matrix" begin
                    A = Hermitian(Adata, uplo)
                    @test_throws DomainError _factorize(A; shift=im)
                    linalg_tests(A, Ai, Aeigmin, shift, Asi, B, A_B[uplo], As_B[uplo], X)
                end
                @testset "SparseMatrixCSC" begin
                    A = Hermitian(sparse(Adata), uplo)
                    @test_throws TypeError _factorize(A; shift=im)
                    linalg_tests(
                        A, Ai, Aeigmin, shift, Asi, B, A_B[uplo], As_B[uplo], sparse(X)
                    )
                end
                @testset "SMatrix" begin
                    A = Hermitian(sarray(Adata), uplo)
                    Ai = Hermitian(sarray(Aidata), uplo)
                    Asi = Hermitian(sarray(Asidata), uplo)
                    @test_throws DomainError _factorize(A; shift=im)
                    linalg_tests(
                        A,
                        Ai,
                        Aeigmin,
                        shift,
                        Asi,
                        sarray(B),
                        sarray.((A_B[uplo], As_B[uplo], X))...,
                    )
                end
            end
            @testset "diagonal" begin
                @testset "Vector" begin
                    linalg_tests(D, Di, minimum(D.diag), shift, Dsi, B, D_B, Ds_B, D_X)
                end
                @testset "SVector" begin
                    linalg_tests(
                        sarray.((D, Di))...,
                        minimum(D.diag),
                        shift,
                        sarray.((Dsi, B, D_B, Ds_B, D_X))...,
                    )
                end
            end
            @testset "uniform" begin
                linalg_tests(U, Ui, U.λ, shift, Usi, B, U_B, Us_B, U_X)
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

                z = convert(Complex{F}, -(im + 2) / 4)
                q = convert(Complex{F}, (1 - im) / 2)
                G0i = _inv(ens.L) * (z * I - ens.M) * _inv(ens.R)
                Q = G0i - q * ens.T
                @test G0i!(ws, ens, z) ≈ G0i
                @test Q!(ws, ens, z, q) ≈ Q

                # Deliberately not testing QQh!, it's an implementation detail

                QQh = Q * Q'
                QQhn = (QQh + Q' * Q) / 2
                s²min, s²minn = _eigmin(_hermitian(QQh)), _eigmin(_hermitian(QQhn))
                γ², δ²= Regularizer{F}(:γ, 0.25), Regularizer{F}(:δ, 0.5)
                Pγ, Pγn = _inv(QQh + γ².value * I), _inv(QQhn + γ².value * I)
                Pδ = _inv(QQh + (δ².value - s²min) * I)
                Pδn = _inv(QQhn + (δ².value - s²min) * I)
                @test P!(ws, ens, z, q, γ²) ≈ Pγ
                @test P!(ws, ens, z, q, δ²) ≈ Pδ
                @test P!(ws, ens, z, q, γ², false) ≈ Pγ
                @test P!(ws, ens, z, q, δ², false) ≈ Pδ
                @test P!(ws, ens, z, q, γ², true) ≈ Pγn
                @test P!(ws, ens, z, q, δ², true) ≈ Pδn
            end

            A_ = ComplexF64[1 -1; 1 1]
            D_ = Diagonal(Float64[0.75, 0.25])
            @testset "$label" for (label, (A, D)) in (
                ("Array", (A_, D_)),
                ("SArray", sarray.((A_, D_))),
                ("SparseMatrixCSC", _sparse.((A_, D_))),
            )
                @testset "$label" for (label, F) in (("Float32", Float32), ("Float64", Float64))
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
            x0 = 0.5

            # finite_difference_jacobian
            fx0 = f(x0)
            dfx0 = df(x0, fx0)
            rtol = eps()^0.4
            @test finite_difference_jacobian(f, x0, fx0) ≈ dfx0 rtol=rtol
            @test finite_difference_jacobian(f, x0) ≈ dfx0 rtol=rtol

            # newton
            atol = sqrt(eps()) / 2
            @test norm(f(first(newton(f, x0)))) < atol
            @test norm(f(first(newton(f, df, x0)))) < atol

            @test_throws NewtonException newton(f, x0; maxiters=0)
            @test_throws NewtonException newton(f, df, x0; maxiters=0)
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

            x0 = SVector(0.3, -0.7)

            # finite_difference_jacobian
            gx0 = g(x0)
            jgx0 = jg(x0, gx0)
            rtol = eps()^0.4
            @test finite_difference_jacobian(g, x0, gx0) ≈ jgx0 rtol=rtol
            @test finite_difference_jacobian(g, x0) ≈ jgx0 rtol=rtol

            # newton
            atol = sqrt(eps()) / 2
            normalizednorm(y) = norm(y) / sqrt(length(y))
            @test normalizednorm(g(first(newton(g, x0)))) < atol
            @test normalizednorm(g(first(newton(g, jg, x0)))) < atol

            @test_throws NewtonException newton(g, x0; maxiters=0)
            @test_throws NewtonException newton(g, jg, x0; maxiters=0)
        end
    end
end
