#=
## solution score
## ==============

function score(solution, r::Regularizer)
    for reg in instances(Regularizers.T)
        (r.type === reg) && return score(getproperty(solution, reg), r.value)
    end
end
=#

## Attic---may be scavenged for further functionality
## ==================================================
#=
function set_PQ!(status::WorkspaceStatus, PQ)
    set_QPQ!(status, false)
    set_QᴴPQ!(status, false)
    status.PQ = PQ
    return nothing
end

function set_QᴴPQ!(status::WorkspaceStatus, QᴴPQ)
    status.QᴴPQ = QᴴPQ
    return nothing
end

function set_QPQ!(status::WorkspaceStatus, QPQ)
    if QPQ
        set_PL⁻¹R⁻¹!(status, false)
    end
    status.QPQ = QPQ
    return nothing
end

function set_QP!(status::WorkspaceStatus, QᴴPQ)
    status.QP = QP
    return nothing
end
=#

#=
function _PQ!(ws)
    @inbounds copyto!(ws.PQ, ws.Q)
    LAPACK.potrs!(ws.P.uplo, ws.P⁻¹f, ws.PQ)
    return nothing
end

function _QᴴPQ!(ws)
    mul!(ws.QᴴPQ.data, ws.Q', ws.PQ)
    return nothing
end

function _QPQ!(ws)
    mul!(ws.QPQ_PL⁻¹R⁻¹, ws.Q, ws.PQ)
    return nothing
end

function _QP!(ws)
    if ws.P.uplo == 'L'
        L = LowerTriangular(ws.P⁻¹f)
        U = L'
    else
        U = UpperTriangular(ws.P⁻¹f)
        L = U'
    end
    @inbounds copyto!(ws.QP, ws.Q)
    rdiv!(ws.QP, U)
    rdiv!(ws.QP, L)
    return nothing
end

function PQ!(problem, q, ws)
    P!(problem, q, ws)
    if !ws.status.PQ
        _PQ!(ws)
        set_PQ!(ws.status, true)
    end
    return ws.PQ
end

function QᴴPQ!(problem, q, ws)
    PQ!(problem, q, ws)
    if !ws.status.QᴴPQ
        _QᴴPQ!(ws)
        set_QᴴPQ!(ws.status, true)
    end
    return ws.QᴴPQ
end

function QPQ!(problem, q, ws)
    PQ!(problem, q, ws)
    if !ws.status.QPQ
        _QPQ!(ws)
        set_QPQ!(ws.status, true)
    end
    return ws.QPQ_PL⁻¹R⁻¹
end

function QP!(problem, q, ws)
    P!(problem, q, ws)
    if !ws.status.QP
        _QP!(ws)
        set_QP!(ws.status, true)
    end
    return ws.QP
end

function q∞!(problem, γ∞², ws)
    T = problem.ensemble.T
    q = _dot(G₀⁻¹_diag!(problem, ws), T)
    return q / (γ∞² * length(T))
end
=#

## Attic
## =====
# function qj!(
#     problem::LocalSpectralProblem{NormalizeQ, <:Any, <:Fixγ²}, q, ws
# ) where {NormalizeQ}
#     T = problem.ensemble.T
#     P = P!(problem, q, ws)
#     PQ = PQ!(problem, q, ws)
#     QᴴPQ = QᴴPQ!(problem, q, ws)
#     df_dq = trTAᴴTB(T, PQ, PQ')  # Tr[(TQᴴP)²]
#     df_dq⁺ = trTAᴴTB(T, QᴴPQ, P)  # Tr[TQᴴPQTP]
#     if NormalizeQ
#         QP = QP!(problem, q, ws)
#         QPQ = QPQ!(problem, q, ws)
#         df_dq += trTAᴴTB(T, QPQ, P)  # Tr[TQᴴPQᴴTP]
#         df_dq⁺ += trTAᴴTB(T, PQ, QP)  # Tr[TQᴴPTQP]
#         df_dq /= 2
#         df_dq⁺ /= 2
#     end
#     df_dq⁺ -= trT²A(T, P)  # Tr[T²P]
#     μ²N⁻¹ = problem.μ^2 / problem.ensemble.N
#     df_dq = muladd(df_dq, μ²N⁻¹, -one(q))
#     df_dq⁺ *= μ²N⁻¹
#     return df_dq, df_dq⁺
# end

#=
"""
    ResolventProblem(
        ensemble::MatrixEnsemble,
        z::Union{Number,AbstractMatrix},
        objective=TraceObjective();
        normalizeQ=false,
    )
    ResolventProblem{F}(ensemble, z, objective, normalizeQ) where {F}

Specification for the resolvent averaging problem at a single point in the complex plane

# Arguments

- `ensemble`: random matrix ensemble over which to average the resolvent
- `z`: coordinates in the complex plane at which to calculate the averaged resolvent. This
    can be either a single complex number or a matrix representing a rectangular grid in the
    complex plane. Use `missing` to exclude grid points from calculation.
- `objective`: objective to solve for. The default trace-based
    objective obtains the true averaged resolvent, but other objectives, including
    regularizer-based ones, are relevant for certain purposes.
- `normalizeQ`: whether to replace ``QQ^†`` with ``(QQ^† + Q^†Q) / 2`` in the
    self-consistent equations for the averaged resolvent, mimicking the case where
    ``QQ^† = Q^†Q``, i.e., where ``Q`` is normal. This is an attempt to get around an issue
    relating to order of limits that can arise for highly non-normal `Q` (see Ahmadian,
    Fumarola, and Miller (2015)). Performing this normalization is not mathematically sound,
    and in most cases gives answers that are subtly but unambiguously wrong; however, in
    certain experiments it seems to do the trick in terms of removing regions of spurious
    spectral support. The option is provided here to support further experimenting.
"""
struct ResolventProblem{
    # TODO: what's a better name than ResolventProblem?
    F,
    E<:MatrixEnsemble{F},
    Z<:Union{Complex{F},Matrix{Union{Missing,Complex{F}}}},
    O<:AbstractObjective{F},
}
    ensemble::E
    z::Z
    objective::O
    normalizeQ::Bool
    # TODO: would a list of points, i.e., unstructured meshes, be better than a rectangular
    # grid represented by a matrix? Big change, and it seems harder to approximate
    # derivatives and hence the spectral density, but finite element packages might have
    # built-in solutions for that

    function ResolventProblem{F}(ensemble, z, objective, normalizeQ) where {F}
        ensemble = adapt(F, ensemble)
        if z isa Number
            z = convert(Complex{F}, z)
        elseif z isa AbstractMatrix
            z = convert(Matrix{Union{Missing,Complex{F}}}, z)
        end
        objective = adapt(F, objective)
        return new{F,typeof(ensemble),typeof(z),typeof(objective)}(
            ensemble, z, objective, normalizeQ
        )
    end
end

function ResolventProblem(
    ensemble::MatrixEnsemble{F}, z, objective=TraceObjective(); normalizeQ=false
) where {F}
    return ResolventProblem{F}(ensemble, z, objective, normalizeQ)
end
=#

#= WIP automesh is not ready for this yet, and this would depend on LazySets.jl

ResolventProblem(objective, ensemble; kwargs...)
    return SpectralProblem(objective, automesh(ensemble), ensemble; kwargs...)
end
=#

# function _reinit(problem::LocalTraceProblem, currentsolution, μ=problem.μ)
#     return LocalTraceProblem(
#         problem.eq, problem.k, currentsolution.q, currentsolution.reg, μ
#     )
# end
#
# function _reinit(problem::LocalRegularizerProblem, currentsolution, μ=problem.μ)
#     return LocalRegularizerProblem(problem.eq, problem.reg, currentsolution.q, μ)
# end

# function solve!(solver, z, q₀, δ₀²; kwargs...)
#     eager_algorithm = DirectSolve(q₀, δ₀²)
#     eager_solver = SpectralSolver(solver.workspace, solver.objective, eager_algorithm)
#     eager_solution = solve!(eager_solver, z; kwargs...)
#     if (eager_solution.q === missing) && (z !== missing)
#         return solve!(solver, z; kwargs...)
#     end
#     return eager_solution
# end
#
# solve!(solver, z, ::Missing, ::Any; kwargs...) = solve!(solver, z; kwargs...)
#
# function solve!(
#     solver::SpectralSolver{R},
#     z;
#     verbosity=default(:verbosity),
#     store_trace=default(:store_trace),
# ) where {R}
#     z = convert(Union{Missing,Complex{R}}, z)
#     print_z(z, verbosity)
#     workspace, objective, algorithm = solver.workspace, solver.objective, solver.algorithm
#     if z === missing
#         return SpectralSolution(BareSpectralSolution(objective), algorithm, empty_trace(R))
#     end
#     print_header(verbosity)
#     n = length(algorithm)
#     if store_trace && (n > 0)
#         trace = SpectralTrace{R}(undef, n)
#         solution, ti = solve!(
#             workspace, objective, algorithm, z, trace; verbosity=verbosity
#         )
#         trace = trace[1:ti]
#     else
#         solution, _ = solve!(workspace, objective, algorithm, z; verbosity=verbosity)
#         trace = empty_trace(R)
#     end
#     print_footer(solution, verbosity)
#     return SpectralSolution(solution, algorithm, trace)
# end
#
# function solve!(workspace, objective, algorithm::Ramp, z, trace=nothing, ti=0; verbosity=0)
#     final_sol = nothing
#     q_ = q₀!(workspace, algorithm, z)
#     δ_² = δ₀²(algorithm)
#     ftype = typeof(δ_²)
#     γ_² = ftype(NaN)
#     s_ = one(ftype)
#     step_alg = initial_algorithm(algorithm, q_, δ_²)
#     for i in 1:length(algorithm)
#         step_obj = step_objective(objective, algorithm, i)
#         step_sol, ti = solve!(
#             workspace, step_obj, step_alg, z, trace, ti; verbosity=verbosity
#         )
#         if trace !== nothing
#             ti += 1
#             trace[ti] = step_sol
#         end
#         print_solution(step_sol, verbosity)
#         if invalid(step_sol, algorithm, γ_²)  # Coming here was a mistake
#             final_sol = BareSpectralSolution(objective)
#             break
#         end
#         if done(step_sol, algorithm, objective)  # This is it!
#             final_sol = step_sol
#             break
#         end
#         if final(step_sol, algorithm, objective)  # We have visual
#             q_, δ_² = select(  # Select whether to jump off of the previous step or this one
#                 q_, step_sol.q, δ_², step_sol.δ², s_, score(step_sol, objective),
#             )
#             step_alg = StraightShot(q_ , δ_²)
#             break
#         end
#         if abort(step_sol, algorithm)  # No point in proceeding if this wasn't it
#             final_sol = BareSpectralSolution(objective)
#             break
#         end
#         q_, δ_², γ_² = step_sol.q, step_sol.δ², step_sol.γ²
#         s_ = score(step_sol, objective)
#         step_alg = StraightShot(q_, δ_²)
#     end
#     print_foot_divider(verbosity)
#     if final_sol === nothing
#         final_sol, _ = solve!(workspace, objective, step_alg, z; verbosity=verbosity)
#         if invalid(final_sol, algorithm, γ_²)
#             final_sol = BareSpectralSolution(objective)
#         end
#     end
#     return final_sol, ti
# end
#
# function solve!(
#     workspace,
#     objective::Fixk{NormalizeQ},
#     algorithm::StraightShot,
#     z,
#     ::Any=nothing,
#     ti=0;
#     verbosity=0,
# ) where {NormalizeQ}
#     q_, δ_² = q₀!(workspace, algorithm, z), δ₀²(algorithm)
#     local q, δ²
#     try
#         q, δ² = qsolve!(workspace, objective, z, q_, δ_², verbosity)
#     catch err
#         if err isa ConvergenceException
#             return BareSpectralSolution(objective), ti
#         end
#         rethrow()
#     end
#     sol = BareSpectralSolution(workspace, Fixδ²{NormalizeQ}(δ², objective.μ), z, q)
#     if invalid(sol, algorithm)
#         sol = BareSpectralSolution(objective)
#     end
#     return sol, ti
# end
#
# function solve!(
#     workspace,
#     objective::Fixr²,
#     algorithm::StraightShot,
#     z,
#     ::Any=nothing,
#     ti=0;
#     verbosity=0,
# )
#     q_ = q₀!(workspace, algorithm, z)
#     local q
#     try
#         q = qsolve!(workspace, objective, z, q_, verbosity)
#     catch err
#         if err isa ConvergenceException
#             return BareSpectralSolution(objective), ti
#         end
#         rethrow()
#     end
#     sol = BareSpectralSolution(workspace, objective, z, q)
#     if invalid(sol, algorithm)
#         sol = BareSpectralSolution(objective)
#     end
#     return sol, ti
# end
#
# function solve_or_print_info!(solver, z, args...; kwargs...)
#     try
#         return solve!(solver, z, args...; kwargs...)
#     catch err
#         print_info(stdout, solver, z, args...)
#         flush(stdout)
#         rethrow()
#     end
# end
#
# function select(qa_, qb_, δa_², δb_², wa=one(δa_²), wb=one(δb_²))
#     if qb_ !== missing
#         if qa_ !== missing
#             wtot = wa + wb
#             wa /= wtot
#             wb /= wtot
#             return (qa_ * wa) + (qb_ * wb), (δa_²^wa) * (δb_²^wb)
#         end
#         return qb_, δb_²
#     end
#     return qa_, δa_²
# end
#         if qa_ !== missing
#             wtot = wa + wb
#             wa /= wtot
#             wb /= wtot
#             return (qa_ * wa) + (qb_ * wb), (δa_²^wa) * (δb_²^wb)
#         end
#         return qb_, δb_²
#     end
#     return qa_, δa_²
# end
