module Solvers

using CommonSolve: CommonSolve
using StructArrays: StructArray

using ..Algorithms: AbstractAlgorithm, AbstractAlgorithmState, Algorithms, statetype
using ..Problems: PointProblem
using ..Solutions: PointSolution
using ..Workspaces: Workspace

using ..ArrayTools: adaptrealtype

mutable struct Solver{
    F,
    W<:Workspace{F},
    P<:PointProblem{F},
    A<:AbstractAlgorithm,
    T<:StructArray{PointSolution{F}},
    S<:AbstractAlgorithmState,
}
    const ws::W
    const pp::P
    const algorithm::A
    const trace::T
    state::Union{Nothing,S}

    function Solver{F}(pp::PointProblem, algorithm::AbstractAlgorithm) where {F}
        ppF = adaptrealtype(F, pp)
        algorithmF = adaptrealtype(F, algorithm)
        wsF = Workspace(ppF.problem.eq)
        traceF = StructArray(PointSolution{F}[])
        W, P, A, T = typeof.((wsF, ppF, algorithmF, traceF))
        return new{F,W,P,A,T,statetype(A)}(wsF, ppF, algorithmF, traceF, nothing)
    end
end

function Solver(pp::PointProblem{F}, algorithm) where {F}
    return Solver{F}(pp, algorithm)
end

struct Solution{F,P<:PointProblem{F},A<:AbstractAlgorithm,T<:StructArray{PointSolution{F}}}
    pp::P
    algorithm::A
    solution::PointSolution{F}
    trace::T

    function Solution(solver::Solver{F}) where {F}
        if !Base.isdone(solver)
            throw(ArgumentError("cannot create Solution from incomplete Solver"))
        end
        (; pp, algorithm, trace) = solver
        solution = trace[end]
        return new{F,typeof(pp),typeof(algorithm),typeof(trace)}(
            pp, algorithm, solution, trace
        )
    end
end

# Iterator interface
# ------------------
Base.IteratorSize(::Type{<:Solver}) = Base.SizeUnknown()
Base.eltype(::S) where {S<:Solver} = S

function Base.isdone(solver::Solver, (::Nothing)=nothing)
    isnothing(solver.state) && return false
    return Algorithms.isdone(solver.pp, solver.algorithm, solver.state)
end

function Base.iterate(solver::Solver, (::Nothing)=nothing)
    state = solver.state
    result = if isnothing(state)
        Algorithms.step!(solver.ws, solver.pp, solver.algorithm)
    else
        Algorithms.step!(solver.ws, solver.pp, solver.algorithm, state)
    end
    isnothing(result) && return nothing
    stepsolution, stepstate = result
    push!(solver.trace, stepsolution)
    solver.state = stepstate
    return solver, nothing
end

# CommonSolve interface
# ---------------------
function CommonSolve.init(pp::PointProblem, algorithm::AbstractAlgorithm)
    return Solver(pp, algorithm)
end

function CommonSolve.step!(solver::Solver)
    iterate(solver)
    # CommonSolve.step! returns generally returns a retcode; we just do success vs. failure
    (length(solver.trace) > 0) || return false
    return solver.trace.success[end]
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
        PTraceProblem, PointProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solvers: Solver
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    z64 = 0.5 + 0.5im
    reg64 = Regularizer(0.25, :shift)
    ptrace64 = 1.0
    init64 = (; reg=Regularizer(1.0, :gap), q=complex(0.1))

    regproblem64 = RegularizerProblem(eq64, reg64)
    ptraceproblem64 = PTraceProblem(eq64, ptrace64)

    alg_direct = DirectSolve()
    alg_schedule = RegularizerSchedule([2.0, 1.0, 0.5, 0.25], :gap)

    @testset "$F" for F in [Float32, Float64]
        @testset "$(nameof(typeof(problem)))" for problem in [regproblem64, ptraceproblem64]
            pp = PointProblem(problem, z64, init64)
            @testset "$(nameof(typeof(alg)))" for alg in [alg_direct, alg_schedule]
                # Type-inferring constructor
                solver = Solver(pp, alg)
                @test solver isa Solver{Float64}

                # Type-enforcing constructor
                solverF = Solver{F}(pp, alg)
                @test solverF isa Solver{F}

                # Internals are converted to correct float type
                @test solverF.pp isa PointProblem{F}
                @test solverF.ws isa Workspace{F}

                # Initial state
                @test isnothing(solverF.state)
                @test isempty(solverF.trace)
                @test !Base.isdone(solverF)
            end
        end
    end
end

@testitem "Solver iteration interface" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution
    using GaussianSpectralDensities.Solvers: Solver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    ptrace = 1.0
    init = (; reg=Regularizer(1.0, :gap), q=complex(0.1))
    problem = PTraceProblem(eq, ptrace)
    pp = PointProblem(problem, z, init)

    @testset "DirectSolve" begin
        solver = Solver(pp, DirectSolve())

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
        @test length(solver.trace) == 1
        @test solver.trace[1] isa PointSolution{Float64}

        # After DirectSolve, should be done
        @test Base.isdone(solver)
        @test isnothing(iterate(solver))
    end

    @testset "RegularizerSchedule" begin
        schedule_values = [4.0, 2.0, 1.0, 0.5, 0.25]
        alg = RegularizerSchedule(schedule_values, :gap)
        init_sched = (; reg=Regularizer(schedule_values[1], :gap), q=complex(0.1))
        problem_sched = PTraceProblem(eq, ptrace)
        pp_sched = PointProblem(problem_sched, z, init_sched)
        solver = Solver(pp_sched, alg)

        # Step through via for-loop
        step_count = 0
        for s in solver
            step_count += 1
            @test s === solver
        end

        # RegularizerSchedule terminates early once ptrace crosses target,
        # so we don't necessarily exhaust the full schedule
        @test step_count > 0
        @test step_count <= length(schedule_values) + 1
        @test length(solver.trace) == step_count
        @test Base.isdone(solver)

        # Final solution should match target
        @test solver.trace.success[end]
        @test solver.trace.ptrace[end] ≈ ptrace

        # Trace contains PointSolutions
        @testset "step $i" for i in eachindex(solver.trace)
            @test solver.trace[i] isa PointSolution{Float64}
        end
    end
end

@testitem "CommonSolve interface" begin
    using CommonSolve: init, solve!, step!

    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution
    using GaussianSpectralDensities.Solvers: Solution, Solver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    ptrace = 1.0
    initvals = (; reg=Regularizer(1.0, :gap), q=complex(0.1))
    problem = PTraceProblem(eq, ptrace)
    pp = PointProblem(problem, z, initvals)

    @testset "init" begin
        solver = init(pp, DirectSolve())
        @test solver isa Solver{Float64}
        @test solver.pp === pp
    end

    @testset "step!" begin
        solver = init(pp, DirectSolve())

        # First step succeeds
        success = step!(solver)
        @test success isa Bool
        @test length(solver.trace) == 1

        # DirectSolve is done after one step
        @test Base.isdone(solver)

        # Additional step! calls still work (no-op due to isdone check in iterate)
        step!(solver)
        @test length(solver.trace) == 1
    end

    @testset "solve!" begin
        schedule_values = [4.0, 2.0, 1.0, 0.5]
        initvals_sched = (; reg=Regularizer(schedule_values[1], :gap), q=complex(0.1))
        problem_sched = PTraceProblem(eq, ptrace)
        pp_sched = PointProblem(problem_sched, z, initvals_sched)
        solver = init(pp_sched, RegularizerSchedule(schedule_values, :gap))

        solution = solve!(solver)
        @test solution isa Solution{Float64}
        @test solution.pp === solver.pp
        @test solution.algorithm === solver.algorithm
        @test solution.solution isa PointSolution{Float64}
        @test solution.trace === solver.trace
        @test Base.isdone(solver)
    end
end

@testitem "Solution construction" begin
    using CommonSolve: init, solve!

    using GaussianSpectralDensities.Algorithms: DirectSolve, RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution
    using GaussianSpectralDensities.Solvers: Solution, Solver

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    z = 0.5 + 0.5im
    ptrace = 1.0
    initvals = (; reg=Regularizer(1.0, :gap), q=complex(0.1))
    problem = PTraceProblem(eq, ptrace)
    pp = PointProblem(problem, z, initvals)

    @testset "Solution from completed solver" begin
        solver = init(pp, DirectSolve())
        solve!(solver)

        solution = Solution(solver)
        @test solution isa Solution{Float64}
        @test solution.pp === solver.pp
        @test solution.algorithm === solver.algorithm
        @test solution.solution === solver.trace[end]
        @test solution.trace === solver.trace
    end

    @testset "Solution from incomplete solver errors" begin
        solver = init(pp, DirectSolve())
        # Solver not yet stepped through
        @test !Base.isdone(solver)
        @test_throws ArgumentError Solution(solver)
    end

    @testset "Solution from failed solver" begin
        # z outside support with shiftmin=0 will fail to find valid solution
        eq_impossible = ResolventEquation(ensemble; shiftmin=0.0)
        z_impossible = 2.0 + 2.0im
        problem_impossible = PTraceProblem(eq_impossible, ptrace)
        pp_impossible = PointProblem(problem_impossible, z_impossible, initvals)
        solver = init(pp_impossible, DirectSolve())
        for _ in solver
        end
        @test Base.isdone(solver)

        # Solution can still be constructed; it just has success=false
        solution = Solution(solver)
        @test solution isa Solution{Float64}
        @test !solution.solution.success
    end

    @testset "Solution properties" begin
        schedule_values = [4.0, 2.0, 1.0, 0.5]
        initvals_sched = (; reg=Regularizer(schedule_values[1], :gap), q=complex(0.1))
        problem_sched = PTraceProblem(eq, ptrace)
        pp_sched = PointProblem(problem_sched, z, initvals_sched)
        alg = RegularizerSchedule(schedule_values, :gap)
        solver = init(pp_sched, alg)
        solution = solve!(solver)

        # Final solution matches target
        @test solution.solution.ptrace ≈ ptrace
        @test solution.solution.success

        # Trace captures history (may terminate early when ptrace crosses target)
        @test length(solution.trace) > 0
        @test length(solution.trace) <= length(alg) + 1
    end
end

@testitem "Solver early termination on failure" begin
    using CommonSolve: init, solve!

    using GaussianSpectralDensities.Algorithms: RegularizerSchedule
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solvers: Solver

    using LinearAlgebra: I

    # z outside Ginibre support with shiftmin=0 => negative shift needed => fails
    ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    z = 1.0 + 1.0im
    ptrace = 1.0
    initvals = (; reg=Regularizer(4.0, :gap), q=complex(0.1))
    problem = PTraceProblem(eq, ptrace)
    pp = PointProblem(problem, z, initvals)

    alg = RegularizerSchedule([4.0, 2.0, 1.0, 0.5, 0.25, 0.125], :gap)
    solver = init(pp, alg)
    solution = solve!(solver)

    # Should terminate early (not exhaust full schedule + final step)
    @test length(solver.trace) < length(alg) + 1
    @test Base.isdone(solver)

    # Last solution failed
    @test !solution.solution.success
    @test solution.solution.shift < eq.shiftmin
end
