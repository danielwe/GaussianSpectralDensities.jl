const ConvergenceException = Union{NewtonException, SingularException, LAPACKException}

struct SpectralProblem{
    R,
    Z<:Union{MissingOr{Complex{R}}, Matrix{<:MissingOr{Complex{R}}}},
    O<:Objective{<:Any, R},
    M<:AbstractMatrixEnsemble{R},
}
    z::Z
    objective::O
    ensemble::M

    function SpectralProblem(z, objective, ensemble::AbstractMatrixEnsemble{R}) where {R}
        z = convert.(MissingOr{Complex{R}}, z)
        objective = adapt(R, objective)
        return new{R, typeof(z), typeof(objective), typeof(ensemble)}(
            z, objective, ensemble
        )
    end
end

struct SpectralSolver{R, W<:AbstractWorkspace{R}, O<:Objective{<:Any, R}, A<:Algorithm{R}}
    workspace::W
    objective::O
    algorithm::A

    function SpectralSolver(
        workspace::AbstractWorkspace{R}, objective, algorithm
    ) where {R}
        objective = adapt(R, objective)
        algorithm = adapt(R, algorithm)
        return new{R, typeof(workspace), typeof(objective), typeof(algorithm)}(
            workspace, objective, algorithm
        )
    end
end

function SpectralSolver(ensemble::AbstractMatrixEnsemble, objective, algorithm)
    return SpectralSolver(workspace(ensemble), objective, algorithm)
end

function SpectralSolver(problem, algorithm)
    return SpectralSolver(problem.ensemble, problem.objective, algorithm)
end

function solve(
    problem::SpectralProblem{R, <:AbstractMatrix}, algorithm; chunks_per_thread=1, kwargs...
) where {R}
    algorithm = adapt(R, algorithm)
    solvers = [SpectralSolver(problem, algorithm) for _ in 1:Threads.nthreads()]
    solution = GlobalSpectralSolution{R, Union{typeof(algorithm), StraightShot{R}}}(
        undef, size(problem.z)
    )
    nrows, ncols = size(problem.z)
    chunk_height = (nrows + 1) ÷ chunks_per_thread
    chunk_width = (ncols + 1) ÷ Threads.nthreads()
    slices = [
        (
            istart:min(istart + chunk_height - 1, nrows),
            jstart:min(jstart + chunk_width - 1, ncols),
        ) for jstart in 1:chunk_width:ncols for istart in 1:chunk_height:nrows
    ]
    @qthreads for slice in slices
        solve!(
            solvers[Threads.threadid()],
            (@view problem.z[slice...]),
            (@view solution[slice...]);
            kwargs...,
        )
    end
    return solution
end

function solve(problem, algorithm, args...; kwargs...)
    solver = SpectralSolver(problem, algorithm)
    return solve!(solver, problem.z, args...; kwargs...)
end

function solve!(solver, z::AbstractMatrix, solution::AbstractMatrix; kwargs...)
    m, n = size(z)
    imid, jmid = cld(m, 2), cld(n, 2)
    solution[imid, jmid] = solve_local_info!(solver, z[imid, jmid]; kwargs...)
    for i in (imid + 1) : m
        q_, δ_² = solution[i - 1, jmid].q, solution[i - 1, jmid].δ²
        solution[i, jmid] = solve_local_info!(solver, z[i, jmid], q_, δ_²; kwargs...)
    end
    for i in (imid - 1) : -1 : 1
        q_, δ_² = solution[i + 1, jmid].q, solution[i + 1, jmid].δ²
        solution[i, jmid] = solve_local_info!(solver, z[i, jmid], q_, δ_²; kwargs...)
    end
    for j in (jmid + 1) : n
        q_, δ_² = solution[imid, j - 1].q, solution[imid, j - 1].δ²
        solution[imid, j] = solve_local_info!(solver, z[imid, j], q_, δ_²; kwargs...)
        for i in (imid + 1) : m
            qa_, δa_² = solution[i - 1, j].q, solution[i - 1, j].δ²
            qb_, δb_² = solution[i, j - 1].q, solution[i, j - 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_local_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
        for i in (imid - 1) : -1 : 1
            qa_, δa_² = solution[i + 1, j].q, solution[i + 1, j].δ²
            qb_, δb_² = solution[i, j - 1].q, solution[i, j - 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_local_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
    end
    for j in (jmid - 1) : -1 : 1
        q_, δ_² = solution[imid, j + 1].q, solution[imid, j + 1].δ²
        solution[imid, j] = solve_local_info!(solver, z[imid, j], q_, δ_²; kwargs...)
        for i in (imid + 1) : m
            qa_, δa_² = solution[i - 1, j].q, solution[i - 1, j].δ²
            qb_, δb_² = solution[i, j + 1].q, solution[i, j + 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_local_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
        for i in (imid - 1) : -1 : 1
            qa_, δa_² = solution[i + 1, j].q, solution[i + 1, j].δ²
            qb_, δb_² = solution[i, j + 1].q, solution[i, j + 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_local_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
    end
    return nothing
end

function solve!(solver, z, q₀, δ₀²; kwargs...)
    eager_algorithm = StraightShot(q₀, δ₀²)
    eager_solver = SpectralSolver(solver.workspace, solver.objective, eager_algorithm)
    try
        return solve!(eager_solver, z; kwargs...)
    catch err
        if err isa ConvergenceException
            return solve!(solver, z; kwargs...)
        end
        rethrow()
    end
end

solve!(solver, z, ::Missing, ::Any; kwargs...) = solve!(solver, z; kwargs...)

function solve!(
    solver::SpectralSolver{R},
    z;
    verbosity=default(:verbosity),
    store_trace=default(:store_trace),
) where {R}
    z = convert(MissingOr{Complex{R}}, z)
    print_z(z, verbosity)
    workspace, objective, algorithm = solver.workspace, solver.objective, solver.algorithm
    if z === missing
        return SpectralSolution(BareSpectralSolution(objective), algorithm, empty_trace(R))
    end
    print_header(verbosity)
    n = length(algorithm)
    if store_trace && (n > 0)
        trace = SpectralTrace{R}(undef, n)
        solution, ti = solve!(workspace, objective, algorithm, z, trace; verbosity=verbosity)
        trace = trace[1:ti]
    else
        solution, _ = solve!(workspace, objective, algorithm, z; verbosity=verbosity)
        trace = empty_trace(R)
    end
    print_footer(solution, verbosity)
    return SpectralSolution(solution, algorithm, trace)
end

function solve!(workspace, objective, algorithm::Ramp, z, trace=nothing, ti=0; verbosity)
    q_ = q₀!(workspace, algorithm, z)
    γ_² = δ_² = δ₀²(algorithm)
    step_alg = initial_algorithm(algorithm, q_, δ_²)
    final_sol = nothing
    for i in 1:length(algorithm)
        step_obj = step_objective(objective, algorithm, i)
        step_sol, ti = solve!(
            workspace, step_obj, step_alg, z, trace, ti; verbosity=verbosity
        )
        if trace !== nothing
            ti += 1
            trace[ti] = step_sol
        end
        print_solution(step_sol, verbosity)
        if invalid_step(algorithm, step_sol, γ_²)
            final_sol = BareSpectralSolution(objective)
            break
        end
        if done(objective, algorithm, step_sol)
            final_sol = step_sol
            break
        end
        if final(objective, algorithm, step_sol)
            q_, δ_² = select(q_, step_sol.q, δ_², step_sol.δ²)
            step_alg = StraightShot(q_ , δ_²)
            break
        end
        if abort(algorithm, step_sol)
            final_sol = BareSpectralSolution(objective)
            break
        end
        q_, γ_², δ_² = step_sol.q, step_sol.γ², step_sol.δ²
        step_alg = StraightShot(q_, δ_²)
    end
    print_foot_divider(verbosity)
    if final_sol === nothing
        final_sol, _ = solve!(workspace, objective, step_alg, z; verbosity=verbosity)
    end
    return final_sol, ti
end

function solve!(
    workspace,
    objective::Fixk{QNormal},
    algorithm::StraightShot,
    z,
    ::Any=nothing,
    ti=0;
    verbosity,
) where {QNormal}
    q_, δ_² = q₀!(workspace, algorithm, z), δ₀²(algorithm)
    q, δ² = qsolve!(workspace, objective, z, q_, δ_²; verbosity=verbosity)
    return BareSpectralSolution(workspace, Fixδ²{QNormal}(δ², objective.μ), z, q), ti
end

function solve!(
    workspace, objective::Fixr², algorithm::StraightShot, z, ::Any=nothing, ti=0; verbosity
)
    local q
    q_ = q₀!(workspace, algorithm, z)
    try
        q = qsolve!(workspace, objective, z, q_; verbosity=verbosity)
    catch err
        if objective.value <= zero(objective.value) && err isa ConvergenceException
            return BareSpectralSolution(objective), ti
        end
        rethrow()
    end
    return BareSpectralSolution(workspace, objective, z, q), ti
end

function solve_local_info!(solver, z, args...; kwargs...)
    try
        return solve!(solver, z, args...; kwargs...)
    catch
        print_local_info(solver, z, args...)
        rethrow()
    end
end

function select(qa_, qb_, δa_², δb_²)
    if qb_ !== missing
        if qa_ !== missing
            return (qa_ + qb_) / 2, sqrt(δa_² * δb_²)
        end
        return qb_, δb_²
    end
    return qa_, δa_²
end
