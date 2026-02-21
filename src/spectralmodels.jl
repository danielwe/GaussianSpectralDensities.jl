# module SpectralModels — implementation pending
# end  # module SpectralModels

@testitem "Spectral{Problem,Solver} from MatrixEnsemble" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, SpreadAnnealing
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.SpectralModels: SpectralProblem, SpectralSolver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)

    @testset "default args/kwargs" begin
        prob = SpectralProblem(ensemble)
        @test prob isa SpectralProblem{Float64}
        @test prob.equation isa ResolventEquation{Float64}
        @test prob.constraint isa PTraceConstraint{Float64}
        @test prob.shiftoffset == Float64(1 // 12)
        solver = SpectralSolver(prob)
        @test solver.problem === prob
        @test solver.algorithm isa SpreadAnnealing
        @test solver.nsamples == 64
        @test solver.hcoeff == 1
        @test solver.max_retries == 8
    end

    @testset "custom args/kwargs" begin
        constraint = PTraceConstraint(1.0; shift0=1.0, shape0=complex(0.1))
        prob = SpectralProblem(ensemble, constraint; shiftmin=-0.25, shiftoffset=1 // 6)
        @test prob.equation.ensemble === ensemble
        @test prob.equation.shiftmin == -0.25
        @test prob.constraint === constraint
        @test prob.shiftoffset == Float64(1 // 6)
        solver = SpectralSolver(prob, DirectSolve(); nsamples=32, hcoeff=2, max_retries=5)
        @test solver.problem === prob
        @test solver.algorithm isa DirectSolve
        @test solver.nsamples == 32
        @test solver.hcoeff == 2
        @test solver.max_retries == 5
    end

    @testset "Float32" begin
        ensemble32 = MatrixEnsemble{Float32}(0.0I, I, I, 0.5I)
        prob = SpectralProblem(ensemble32)
        @test prob isa SpectralProblem{Float32}
        @test prob.equation isa ResolventEquation{Float32}
    end
end

@testitem "SpectralProblem from ResolventEquation" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.SpectralModels: SpectralProblem

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble; shiftmin=-0.25)

    @testset "default args/kwargs" begin
        prob = SpectralProblem(eq)
        @test prob isa SpectralProblem{Float64}
        @test prob.equation === eq
        @test prob.constraint isa PTraceConstraint{Float64}
        @test prob.shiftoffset == Float64(1 // 12)
    end

    @testset "custom args/kwargs" begin
        constraint = PTraceConstraint(1.0; shift0=1.0, shape0=complex(0.1))
        prob = SpectralProblem(eq, constraint; shiftoffset=1 // 6)
        @test prob.equation === eq
        @test prob.constraint === constraint
        @test prob.shiftoffset == Float64(1 // 6)
    end
end

@testitem "SpectralProblem type-enforcing constructor" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.SpectralModels: SpectralProblem

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    shiftoffset = 1 // 6
    prob = SpectralProblem{Float32}(eq, constraint, shiftoffset)
    @test prob isa SpectralProblem{Float32}
    @test prob.equation isa ResolventEquation{Float32}
    @test prob.constraint isa PTraceConstraint{Float32}
end

@testitem "SpectralSolver validation" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels: SpectralProblem, SpectralSolver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)

    @test_throws ArgumentError SpectralSolver(prob; nsamples=0)
    @test_throws ArgumentError SpectralSolver(prob; nsamples=-1)
    @test_throws ArgumentError SpectralSolver(prob; hcoeff=0)
    @test_throws ArgumentError SpectralSolver(prob; hcoeff=-1 // 2)
    @test_throws ArgumentError SpectralSolver(prob; max_retries=0)
    @test_throws ArgumentError SpectralSolver(prob; max_retries=-1)
end

@testitem "CommonSolve.init produces SpectralSolver" begin
    using CommonSolve: init
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels: SpectralProblem, SpectralSolver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)

    @test init(prob) isa SpectralSolver{Float64}
end

@testitem "CommonSolve.solve! produces SpectralModel" begin
    using CommonSolve: solve!
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels:
        SpectralModel, SpectralProblem, SpectralSolver
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)
    solver = SpectralSolver(prob; nsamples=16, hcoeff=2, max_retries=5)

    # TODO: Identify deterministic parameter values/configuration that force mesh failure so
    # `solve!` retry exhaustion can be tested reliably.
    model = solve!(solver)
    @test model isa SpectralModel{Float64}
    @test model.problem === prob
    @test model.surrogate isa SurrogateModel{Float64}
end

@testitem "CommonSolve.solve convenience" begin
    using CommonSolve: solve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels: SpectralModel, SpectralProblem
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)

    model = solve(prob; nsamples=16)
    @test model isa SpectralModel{Float64}
    @test model.problem === prob
    @test model.surrogate isa SurrogateModel{Float64}
end

@testitem "SpectralModel fields" begin
    using CommonSolve: solve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels: SpectralProblem
    using GaussianSpectralDensities.Surrogates: zvals

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)
    nsamples = 16

    model = solve(prob; nsamples)

    # The refined surrogate should have more points than the unrefined nsamples
    @test length(zvals(model.surrogate)) > nsamples
end
