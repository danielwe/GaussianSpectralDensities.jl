using GaussianSpectralDensities
using GaussianSpectralDensities: DEFAULTS, adapt
using GaussianSpectralDensities: newton, finite_difference_jacobian, NewtonException

using LinearAlgebra
using StaticArrays

@testset "GaussianSpectralDensities.jl" begin
    @testset "ensembles" begin
        @testset "MatrixEnsemble" begin
            N = 2
            @test MatrixEnsemble(N, I, I, I, I).N == 2

            D32 = 0.5f0I(N)
            D64 = 0.5I(N)
            @testset "$args" for (args, e32, e64) in (
                ("M=D, L=I, R=I, T=I", map(D -> MatrixEnsemble(D, I, I, I), (D32, D64))...),
                ("M=I, L=D, R=I, T=I", map(D -> MatrixEnsemble(I, D, I, I), (D32, D64))...),
                ("M=I, L=I, R=D, T=I", map(D -> MatrixEnsemble(I, I, D, I), (D32, D64))...),
                ("M=I, L=I, R=I, T=D", map(D -> MatrixEnsemble(I, I, I, D), (D32, D64))...),
            )
                @test e32.N == N
                @test e64.N == N
                @test e32 isa MatrixEnsemble{Float32}
                @test e64 isa MatrixEnsemble{Float64}
            end

            @test_throws ArgumentError MatrixEnsemble(I, I, I, I)
            @test_throws DimensionMismatch MatrixEnsemble(N + 1, D64, I, I, I)
            @test_throws DimensionMismatch MatrixEnsemble(D64, I, I, 0.5I(N + 1))
            @test_throws ArgumentError MatrixEnsemble(I, I, I, Matrix(D64))
            @test_throws ArgumentError MatrixEnsemble(I, I, I, -D64)
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
            @testset "$desc" for (desc, D) in (
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
            @testset "$desc" for (desc, R) in (
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
            @testset "$desc" for (desc, S) in (
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
