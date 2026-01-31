module Solvers

using AutoHashEquals: @auto_hash_equals
using CommonSolve: CommonSolve
using StructArrays: StructArray

using ..Algorithms: AbstractAlgorithm, AbstractAlgorithmState, Algorithms, statetype
using ..Problems: AbstractConstraint, PointProblem
using ..Solutions: PointSolution, ResolventParams
using ..Workspaces: Workspace

using ..ArrayTools: adaptrealtype

public Solution, Solver

@auto_hash_equals typearg=true mutable struct Solver{
    F,
    E,
    W<:Workspace{F},
    P<:PointProblem{F,E},
    A<:AbstractAlgorithm,
    S<:AbstractAlgorithmState,
    T<:StructArray{ResolventParams{F}},
}
    const ws::W
    const problem::P
    const algorithm::A
    state::Union{Nothing,S}
    current::Union{Nothing,PointSolution{F,E}}
    const trace::T

    function Solver{F}(problem::PointProblem, algorithm::AbstractAlgorithm) where {F}
        problemF = adaptrealtype(F, problem)
        algorithmF = adaptrealtype(F, algorithm)
        wsF = Workspace(problemF.equation)
        traceF = StructArray(ResolventParams{F}[])
        E, W, P, A, T = typeof.((problemF.equation, wsF, problemF, algorithmF, traceF))
        return new{F,E,W,P,A,statetype(A),T}(
            wsF, problemF, algorithmF, nothing, nothing, traceF
        )
    end
end

Solver(problem::PointProblem{F}, algorithm) where {F} = Solver{F}(problem, algorithm)

@auto_hash_equals typearg=true struct Solution{
    F,
    S<:PointSolution{F},
    C<:AbstractConstraint{F},
    A<:AbstractAlgorithm,
    T<:StructArray{ResolventParams{F}},
}
    solution::S
    constraint::C
    algorithm::A
    trace::T

    function Solution(solver::Solver{F}) where {F}
        if !Base.isdone(solver)
            throw(ArgumentError("cannot create Solution from unfinished Solver"))
        end
        (; problem, algorithm, trace) = solver
        solution = solver.current
        (; constraint) = problem
        S, C, A, T = typeof.((solution, constraint, algorithm, trace))
        return new{F,S,C,A,T}(solution, constraint, algorithm, trace)
    end
end

# Iterator interface
# ------------------
Base.IteratorSize(::Type{<:Solver}) = Base.SizeUnknown()
Base.eltype(::Type{S}) where {S<:Solver} = S

function Base.isdone(solver::Solver, (::Nothing)=nothing)
    return Algorithms.isdone(solver.problem, solver.algorithm, solver.state)
end

function Base.iterate(solver::Solver, (::Nothing)=nothing)
    result = Algorithms.step!(solver.ws, solver.problem, solver.algorithm, solver.state)
    isnothing(result) && return nothing
    stepsolution, stepstate = result
    solver.state = stepstate
    solver.current = stepsolution
    push!(solver.trace, stepsolution.params)
    return solver, nothing
end

# CommonSolve interface
# ---------------------
function CommonSolve.init(problem::PointProblem, algorithm::AbstractAlgorithm)
    return Solver(problem, algorithm)
end

function CommonSolve.step!(solver::Solver)
    result = iterate(solver)
    isnothing(result) && throw(ArgumentError("cannot step! a finished Solver"))
    # CommonSolve.step! generally returns a retcode; in our case just success/failure
    return !isnothing(solver.current) && solver.current.success
end

function CommonSolve.solve!(solver::Solver)
    for _ in solver
        # Just stepping through
    end
    return Solution(solver)
end

end  # module Solvers

@testitem "Solver construction" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solvers: Solver
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    z64 = 0.5 + 0.5im
    regc64 = RegularizerConstraint(; shift=0.25, q0=complex(0.1))
    ptracec64 = PTraceConstraint(1.0; gap0=1.0, q0=complex(0.1))

    alg_direct = DirectSolve()
    alg_schedule = RegularizerSchedule([2.0, 1.0, 0.5, 0.25], :gap)

    @testset "$F" for F in [Float32, Float64]
        @testset "$(nameof(typeof(constraint)))" for constraint in [regc64, ptracec64]
            problem = PointProblem(eq64, constraint, z64)
            @testset "$(nameof(typeof(alg)))" for alg in [alg_direct, alg_schedule]
                # Type-inferring constructor
                solver = Solver(problem, alg)
                @test solver isa Solver{Float64}

                # Type-enforcing constructor
                solverF = Solver{F}(problem, alg)
                @test solverF isa Solver{F}

                # Internals are converted to correct float type
                @test solverF.problem isa PointProblem{F}
                @test solverF.ws isa Workspace{F}

                # Initial state
                @test isnothing(solverF.state)
                @test isnothing(solverF.current)
                @test isempty(solverF.trace)
                @test !Base.isdone(solverF)
            end
        end
    end
end

@testitem "Solver iteration interface" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution, solves
    using GaussianSpectralDensities.Solvers: Solver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    constraint = PTraceConstraint(1.0; gap0=1.0, q0=complex(0.1))
    problem = PointProblem(eq, constraint, z)

    @testset "DirectSolve" begin
        solver = Solver(problem, DirectSolve())

        # Iterator traits
        @test Base.IteratorSize(typeof(solver)) === Base.SizeUnknown()
        @test eltype(solver) === typeof(solver)

        # Before iteration
        @test !Base.isdone(solver)
        @test isempty(solver.trace)

        # First iteration
        result = iterate(solver)
        @test !isnothing(result)
        returned_solver, _ = result
        @test returned_solver === solver
        @test solver.current isa PointSolution{Float64}
        @test length(solver.trace) == 1
        @test first(solver.trace) == solver.current.params

        # After DirectSolve, should be done
        @test Base.isdone(solver)
        @test isnothing(iterate(solver))

        # Solution should match target
        @test solver.current.success
        @test solves(solver.current, solver.problem)
    end

    @testset "RegularizerSchedule" begin
        schedule_vals = [4.0, 2.0, 1.0, 0.5, 0.25]
        alg = RegularizerSchedule(schedule_vals, :gap)
        constraint_sched = PTraceConstraint(1.0; gap0=schedule_vals[1], q0=complex(0.1))
        problem_sched = PointProblem(eq, constraint_sched, z)
        solver = Solver(problem_sched, alg)

        # Step through via for-loop
        step_count = 0
        for s in solver
            step_count += 1
            @test s === solver
        end

        # RegularizerSchedule terminates early once ptrace crosses target,
        # so we don't necessarily exhaust the full schedule
        @test step_count > 0
        @test step_count <= length(schedule_vals) + 1
        @test length(solver.trace) == step_count
        @test Base.isdone(solver)

        # Final solution should match target
        @test solver.current.success
        @test solves(solver.current, solver.problem)

        # Trace contains resolvent params
        @test last(solver.trace) == solver.current.params
    end
end

@testitem "CommonSolve interface" begin
    using CommonSolve: init, solve!, step!

    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solvers: Solution, Solver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    constraint = PTraceConstraint(1.0; gap0=1.0, q0=complex(0.1))
    problem = PointProblem(eq, constraint, z)

    @testset "init" begin
        solver = init(problem, DirectSolve())
        @test solver isa Solver{Float64}
        @test solver.problem === problem
    end

    @testset "step!" begin
        solver = init(problem, DirectSolve())

        # First step succeeds
        success = step!(solver)
        @test success isa Bool
        @test length(solver.trace) == 1

        # DirectSolve is done after one step
        @test Base.isdone(solver)

        # Additional step! calls error but don't change the solver
        @test_throws ArgumentError step!(solver)
        @test_throws "finished" step!(solver)
        @test length(solver.trace) == 1
    end

    @testset "solve!" begin
        schedule_vals = [4.0, 2.0, 1.0, 0.5]
        constraint_sched = PTraceConstraint(1.0; gap0=schedule_vals[1], q0=complex(0.1))
        problem_sched = PointProblem(eq, constraint_sched, z)
        solver = init(problem_sched, RegularizerSchedule(schedule_vals, :gap))

        solution = solve!(solver)
        @test solution isa Solution{Float64}
        @test solution.solution === solver.current
        @test solution.constraint === solver.problem.constraint
        @test solution.algorithm === solver.algorithm
        @test solution.trace === solver.trace
        @test Base.isdone(solver)
    end
end

@testitem "Solution construction" begin
    using CommonSolve: init, solve!

    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Solvers: Solution

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    constraint = PTraceConstraint(1.0; gap0=1.0, q0=complex(0.1))
    problem = PointProblem(eq, constraint, z)

    @testset "Solution from completed solver" begin
        solver = init(problem, DirectSolve())
        solve!(solver)

        solution = Solution(solver)
        @test solution isa Solution{Float64}
        @test solution.solution === solver.current
        @test solution.constraint === solver.problem.constraint
        @test solution.algorithm === solver.algorithm
        @test solution.trace === solver.trace
    end

    @testset "Solution from incomplete solver errors" begin
        solver = init(problem, DirectSolve())
        # Solver not yet stepped through
        @test !Base.isdone(solver)
        @test_throws ArgumentError Solution(solver)
    end

    @testset "Solution from failed solver" begin
        # z outside support with shiftmin=0 will fail to find valid solution
        eq_impossible = ResolventEquation(ensemble; shiftmin=0.0)
        z_impossible = 2.0 + 2.0im
        problem_impossible = PointProblem(eq_impossible, constraint, z_impossible)
        solver = init(problem_impossible, DirectSolve())
        for _ in solver
        end
        @test Base.isdone(solver)

        # Solution can still be constructed; it just has success=false
        solution = Solution(solver)
        @test solution isa Solution{Float64}
        @test !solution.solution.success
    end

    @testset "Solution properties" begin
        schedule_vals = [4.0, 2.0, 1.0, 0.5]
        constraint_sched = PTraceConstraint(1.0; gap0=schedule_vals[1], q0=complex(0.1))
        problem_sched = PointProblem(eq, constraint_sched, z)
        alg = RegularizerSchedule(schedule_vals, :gap)
        solver = init(problem_sched, alg)
        solution = solve!(solver)

        # Final solution matches target
        @test solution.solution.success
        @test solves(solution.solution, problem_sched)

        # Trace captures history (may terminate early when ptrace crosses target)
        @test length(solution.trace) > 0
        @test length(solution.trace) <= length(alg) + 1
        @test last(solution.trace) == solution.solution.params
    end
end

@testitem "Solver early termination on failure" begin
    using CommonSolve: init, solve!

    using GaussianSpectralDensities.Algorithms: RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation

    using LinearAlgebra: I

    # z outside Ginibre support with shiftmin=0 => negative shift needed => fails
    ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    z = 1.0 + 1.0im
    constraint = PTraceConstraint(1.0; gap0=4.0, q0=complex(0.1))
    problem = PointProblem(eq, constraint, z)

    alg = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125], :gap)
    solver = init(problem, alg)
    solution = solve!(solver)

    # Should terminate early (not exhaust full schedule + final step)
    @test length(solver.trace) < length(alg) + 1
    @test Base.isdone(solver)

    # Last solution failed
    @test !solution.solution.success
    @test solution.solution.params.shift < eq.shiftmin

    # Trace is still what we expect
    @test solution.trace === solver.trace
    @test last(solution.trace) == solution.solution.params
end
