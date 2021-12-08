const ConvergenceException = Union{NewtonException, SingularException, LAPACKException}

struct SpectralProblem{
    R,
    E<:AbstractMatrixEnsemble{R},
    Z<:Union{MissingOr{Complex{R}}, Matrix{<:MissingOr{Complex{R}}}},
}
    ensemble::E
    objective::Objective{<:Any, R}
    z::Z

    function SpectralProblem(
        ensemble::E, objective, z
    ) where {R, E<:AbstractMatrixEnsemble{R}}
        objective = adapt(R, objective)
        z = convert.(MissingOr{Complex{R}}, z)
        return new{R, E, typeof(z)}(ensemble, objective, z)
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

function set_ensemble!(solver::SpectralSolver, ensemble)
    return set_ensemble!(solver.workspace, ensemble)
end

function solve(
    problem::SpectralProblem{R, <:Any, <:AbstractMatrix},
    algorithm;
    chunks_per_thread=1,
    kwargs...
) where {R}
    algorithm = adapt(R, algorithm)
    nbthreads = max(1, Threads.nthreads() - 1)
    solvers = [SpectralSolver(problem, algorithm) for _ in 1:nbthreads]
    solution = GlobalSpectralSolution{R}(undef, size(problem.z))
    ranges = chunk_ranges(size(problem.z), chunks_per_thread * nbthreads)
    @qbthreads for range in ranges
        i = max(1, Threads.threadid() - 1)
        solve!(
            solvers[i], (@view problem.z[range...]), (@view solution[range...]); kwargs...
        )
    end
    return solution
end

function chunk_indices(size, n)
    nrows, ncols = size
    grows = ceil(Int, sqrt(n * nrows / ncols))
    gcols = cld(n, grows)
    chunk_height = cld(nrows, grows)
    chunk_width = fld(ncols, gcols)
    excess_height = nrows % chunk_height
    excess_width = ncols % chunk_width
    row_offset = excess_height > 0 ? (chunk_height - excess_height) ÷ 2 : 0
    col_offset = excess_width > 0 ? (chunk_width - excess_width) ÷ 2 : 0
    return [
        [max(istart, 1), min(istart + chunk_height - 1, nrows)]
        for istart in (1 - row_offset):chunk_height:nrows
    ], [
        [max(jstart, 1), min(jstart + chunk_width - 1, ncols)]
        for jstart in (1 - col_offset):chunk_width:ncols
    ]
end

function chunk_ranges(args...)
    return product_ranges(chunk_indices(args...))
end

function product_ranges(indices)
    return [(i[1]:i[2], j[1]:j[2]) for j in indices[2] for i in indices[1]]
end

function solve(problem, algorithm, args...; kwargs...)
    solver = SpectralSolver(problem, algorithm)
    return solve!(solver, problem.z, args...; kwargs...)
end

function solve!(solver, z::AbstractMatrix, solution::AbstractMatrix; kwargs...)
    m, n = size(z)
    imid, jmid = cld(m, 2), cld(n, 2)
    solution[imid, jmid] = solve_or_print_info!(solver, z[imid, jmid]; kwargs...)
    for i in (imid + 1) : m
        q_, δ_² = solution[i - 1, jmid].q, solution[i - 1, jmid].δ²
        solution[i, jmid] = solve_or_print_info!(solver, z[i, jmid], q_, δ_²; kwargs...)
    end
    for i in (imid - 1) : -1 : 1
        q_, δ_² = solution[i + 1, jmid].q, solution[i + 1, jmid].δ²
        solution[i, jmid] = solve_or_print_info!(solver, z[i, jmid], q_, δ_²; kwargs...)
    end
    for j in (jmid + 1) : n
        q_, δ_² = solution[imid, j - 1].q, solution[imid, j - 1].δ²
        solution[imid, j] = solve_or_print_info!(solver, z[imid, j], q_, δ_²; kwargs...)
        for i in (imid + 1) : m
            qa_, δa_² = solution[i - 1, j].q, solution[i - 1, j].δ²
            qb_, δb_² = solution[i, j - 1].q, solution[i, j - 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_or_print_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
        for i in (imid - 1) : -1 : 1
            qa_, δa_² = solution[i + 1, j].q, solution[i + 1, j].δ²
            qb_, δb_² = solution[i, j - 1].q, solution[i, j - 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_or_print_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
    end
    for j in (jmid - 1) : -1 : 1
        q_, δ_² = solution[imid, j + 1].q, solution[imid, j + 1].δ²
        solution[imid, j] = solve_or_print_info!(solver, z[imid, j], q_, δ_²; kwargs...)
        for i in (imid + 1) : m
            qa_, δa_² = solution[i - 1, j].q, solution[i - 1, j].δ²
            qb_, δb_² = solution[i, j + 1].q, solution[i, j + 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_or_print_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
        for i in (imid - 1) : -1 : 1
            qa_, δa_² = solution[i + 1, j].q, solution[i + 1, j].δ²
            qb_, δb_² = solution[i, j + 1].q, solution[i, j + 1].δ²
            q_, δ_² = select(qa_, qb_, δa_², δb_²)
            solution[i, j] = solve_or_print_info!(solver, z[i, j], q_, δ_²; kwargs...)
        end
    end
    return nothing
end

function solve!(solver, z, q₀, δ₀²; kwargs...)
    eager_algorithm = StraightShot(q₀, δ₀²)
    eager_solver = SpectralSolver(solver.workspace, solver.objective, eager_algorithm)
    eager_solution = solve!(eager_solver, z; kwargs...)
    if (eager_solution.q === missing) && (z !== missing)
        return solve!(solver, z; kwargs...)
    end
    return eager_solution
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
        solution, ti = solve!(
            workspace, objective, algorithm, z, trace; verbosity=verbosity
        )
        trace = trace[1:ti]
    else
        solution, _ = solve!(workspace, objective, algorithm, z; verbosity=verbosity)
        trace = empty_trace(R)
    end
    print_footer(solution, verbosity)
    return SpectralSolution(solution, algorithm, trace)
end

function solve!(workspace, objective, algorithm::Ramp, z, trace=nothing, ti=0; verbosity=0)
    final_sol = nothing
    q_ = q₀!(workspace, algorithm, z)
    δ_² = δ₀²(algorithm)
    ftype = typeof(δ_²)
    γ_² = ftype(NaN)
    s_ = one(ftype)
    step_alg = initial_algorithm(algorithm, q_, δ_²)
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
        if invalid(step_sol, algorithm, γ_²)  # Coming here was a mistake
            final_sol = BareSpectralSolution(objective)
            break
        end
        if done(step_sol, algorithm, objective)  # This is it!
            final_sol = step_sol
            break
        end
        if final(step_sol, algorithm, objective)  # We have visual
            q_, δ_² = select(
                q_, step_sol.q, δ_², step_sol.δ², s_, score(step_sol, objective),
            )
            step_alg = StraightShot(q_ , δ_²)
            break
        end
        if abort(step_sol, algorithm)  # No point in proceeding if this wasn't it
            final_sol = BareSpectralSolution(objective)
            break
        end
        q_, δ_², γ_² = step_sol.q, step_sol.δ², step_sol.γ²
        s_ = score(step_sol, objective)
        step_alg = StraightShot(q_, δ_²)
    end
    print_foot_divider(verbosity)
    if final_sol === nothing
        final_sol, _ = solve!(workspace, objective, step_alg, z; verbosity=verbosity)
        if invalid(final_sol, algorithm, γ_²)
            final_sol = BareSpectralSolution(objective)
        end
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
    verbosity=0,
) where {QNormal}
    q_, δ_² = q₀!(workspace, algorithm, z), δ₀²(algorithm)
    local q, δ²
    try
        q, δ² = qsolve!(workspace, objective, z, q_, δ_², verbosity)
    catch err
        if err isa ConvergenceException
            return BareSpectralSolution(objective), ti
        end
        rethrow()
    end
    sol = BareSpectralSolution(workspace, Fixδ²{QNormal}(δ², objective.μ), z, q)
    if invalid(sol, algorithm)
        sol = BareSpectralSolution(objective)
    end
    return sol, ti
end

function solve!(
    workspace,
    objective::Fixr²,
    algorithm::StraightShot,
    z,
    ::Any=nothing,
    ti=0;
    verbosity=0,
)
    q_ = q₀!(workspace, algorithm, z)
    local q
    try
        q = qsolve!(workspace, objective, z, q_, verbosity)
    catch err
        if err isa ConvergenceException
            return BareSpectralSolution(objective), ti
        end
        rethrow()
    end
    sol = BareSpectralSolution(workspace, objective, z, q)
    if invalid(sol, algorithm)
        sol = BareSpectralSolution(objective)
    end
    return sol, ti
end

function solve_or_print_info!(solver, z, args...; kwargs...)
    try
        return solve!(solver, z, args...; kwargs...)
    catch err
        print_info(stdout, solver, z, args...)
        flush(stdout)
        rethrow()
    end
end

function select(qa_, qb_, δa_², δb_², wa=one(δa_²), wb=one(δb_²))
    if qb_ !== missing
        if qa_ !== missing
            wtot = wa + wb
            wa /= wtot
            wb /= wtot
            return (qa_ * wa) + (qb_ * wb), (δa_²^wa) * (δb_²^wb)
        end
        return qb_, δb_²
    end
    return qa_, δa_²
end
