struct SpectralSolver{
    R,
    P<:SpectralProblem{R},
    A<:Algorithm{R},
    W<:AbstractWorkspace{R},
}
    problem::P
    algorithm::A
    workspaces::Vector{W}
    chunks_per_thread::Int

    function SpectralSolver(
        problem::SpectralProblem{R},
        algorithm::Algorithm,
        workspaces::Vector{W};
        chunks_per_thread=1,
    ) where {R, W<:AbstractWorkspace{R}}
        algorithm = _adapt(R, algorithm)
        for ws in workspaces
            set_ensemble!(ws, problem.ensemble)
        end
        return new{R, typeof(problem), typeof(algorithm), eltype(workspaces)}(
            problem, algorithm, workspaces, chunks_per_thread
        )
    end
end

function SpectralSolver(
    problem::SpectralProblem{R},
    algorithm::Algorithm;
    nthreads=max(1, Threads.nthreads() - 1),
    chunks_per_thread=1,
) where {R}
    workspaces = [workspace(problem.ensemble) for _ in 1:nthreads]
    return new{R, typeof(problem), typeof(algorithm), eltype(workspaces)}(
        problem, algorithm, workspaces, chunks_per_thread
    )
end

function SpectralSolver(ensemble::AbstractMatrixEnsemble, objective, algorithm; kwargs...)
    return SpectralSolver(workspace(ensemble), objective, algorithm; kwargs...)
end

function SpectralSolver(problem, algorithm; kwargs...)
    return SpectralSolver(problem.ensemble, problem.objective, algorithm; kwargs...)
end

function set_ensemble!(solver::SpectralSolver, ensemble)
    for ws in solver.workspaces
        set_ensemble!(ws, ensemble)
    end
end

function solve!(solver::SpectralSolver{R}) where {R}
    algorithm = _adapt(R, algorithm)
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
