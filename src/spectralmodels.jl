module SpectralModels

using AutoHashEquals: @auto_hash_equals
using CommonSolve: CommonSolve

using ..Algorithms: AbstractAlgorithm, SpreadAnnealing
using ..ArrayTools: adaptrealtype
using ..Ensembles: MatrixEnsemble
using ..Meshing: mesh
using ..Problems: AbstractConstraint, PTraceConstraint, ResolventEquation
using ..Surrogates: SurrogateModel, zvals

public SpectralModel, SpectralProblem

@auto_hash_equals typearg=true struct SpectralProblem{
    F<:AbstractFloat,E<:ResolventEquation{F},C<:AbstractConstraint{F}
}
    equation::E
    constraint::C
    shiftoffset::F

    function SpectralProblem{F}(equation, constraint, shiftoffset) where {F}
        eqF = adaptrealtype(F, equation)
        constraintF = adaptrealtype(F, constraint)
        return new{F,typeof(eqF),typeof(constraintF)}(eqF, constraintF, shiftoffset)
    end
end

function SpectralProblem(
    equation::ResolventEquation{F},
    constraint=PTraceConstraint{F}(one(F); shape0=zero(Complex{F}), shift0=one(F));
    shiftoffset::Real=F(1 // 12),
) where {F}
    return SpectralProblem{F}(equation, constraint, shiftoffset)
end

function SpectralProblem(
    ensemble::MatrixEnsemble{F},
    constraint::AbstractConstraint...;
    shiftmin::Real=(-one(F)),
    kwargs...,
) where {F}
    return SpectralProblem(ResolventEquation(ensemble; shiftmin), constraint...; kwargs...)
end

## SpectralSolver
## ===============
"""
    SpectralSolver(problem::SpectralProblem; algorithm, nsamples=64, hcoeff=1, max_retries=8)

CommonSolve-compatible solver state. Created by `CommonSolve.init` and consumed by
`CommonSolve.solve!` to produce a `SpectralModel`.

  - `nsamples`: number of random eigenvalue samples for initial surrogate construction.
  - `hcoeff`: mesh element size relative to surrogate point spacing (akin to a tolerance).
  - `max_retries`: number of meshing attempts before giving up.
"""
struct SpectralSolver{F,P<:SpectralProblem{F},A<:AbstractAlgorithm}
    problem::P
    algorithm::A
    nsamples::Int
    hcoeff::F
    max_retries::Int

    function SpectralSolver(
        problem::SpectralProblem{F},
        algorithm=adaptrealtype(F, SpreadAnnealing());
        nsamples::Integer=64,
        hcoeff::Real=one(F),
        max_retries::Integer=8,
    ) where {F}
        algorithmF = adaptrealtype(F, algorithm)
        (nsamples > 0) || throw(ArgumentError("nsamples must be positive"))
        (hcoeff > 0) || throw(ArgumentError("hcoeff must be positive"))
        (max_retries > 0) || throw(ArgumentError("max_retries must be positive"))
        return new{F,typeof(problem),typeof(algorithmF)}(
            problem, algorithmF, nsamples, hcoeff, max_retries
        )
    end
end

## SpectralModel
## ==============
@auto_hash_equals typearg=true struct SpectralModel{
    F<:AbstractFloat,P<:SpectralProblem{F},S<:SurrogateModel{F}
}
    problem::P
    surrogate::S
end

## CommonSolve interface
## =====================
function CommonSolve.init(
    problem::SpectralProblem, algorithm::AbstractAlgorithm...; kwargs...
)
    return SpectralSolver(problem, algorithm...; kwargs...)
end

function CommonSolve.solve!(solver::SpectralSolver)
    (; problem, algorithm, nsamples, hcoeff, max_retries) = solver
    (; equation, constraint, shiftoffset) = problem

    for _ in 1:max_retries
        model = SurrogateModel(equation, constraint, algorithm, nsamples)
        refinement = try
            mesh(model; hcoeff, shiftoffset)
        catch
            continue
        end
        refined_zs = zvals(refinement)
        refined_model = SurrogateModel(equation, constraint, algorithm, refined_zs)
        return SpectralModel(problem, refined_model)
    end
    return error("Failed to build spectral model after $max_retries attempts")
end

end  # module SpectralModels

@testitem "Spectral{Problem,Solver} from MatrixEnsemble" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation, SpreadAnnealing
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
    using GaussianSpectralDensities: MatrixEnsemble, PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities: MatrixEnsemble, PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities: MatrixEnsemble
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
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.SpectralModels: SpectralProblem, SpectralSolver

    using CommonSolve: init
    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)

    @test init(prob) isa SpectralSolver{Float64}
end

@testitem "CommonSolve.solve! produces SpectralModel" begin
    using GaussianSpectralDensities: MatrixEnsemble
    using GaussianSpectralDensities.Surrogates: SurrogateModel
    using GaussianSpectralDensities.SpectralModels:
        SpectralModel, SpectralProblem, SpectralSolver

    using CommonSolve: solve!
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
    using GaussianSpectralDensities: MatrixEnsemble, solve
    using GaussianSpectralDensities.Surrogates: SurrogateModel
    using GaussianSpectralDensities.SpectralModels: SpectralModel, SpectralProblem

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)

    model = solve(prob; nsamples=16)
    @test model isa SpectralModel{Float64}
    @test model.problem === prob
    @test model.surrogate isa SurrogateModel{Float64}
end

@testitem "SpectralModel fields" begin
    using GaussianSpectralDensities: MatrixEnsemble, solve, zvals
    using GaussianSpectralDensities.SpectralModels: SpectralProblem

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    prob = SpectralProblem(ensemble)
    nsamples = 16

    model = solve(prob; nsamples)

    # The refined surrogate should have more points than the unrefined nsamples
    @test length(zvals(model.surrogate)) > nsamples
end
