struct LocalSolution{
    F,
    P<:AbstractLocalResolventProblem{F},
    A<:Union{DirectSolve,AbstractSchedule{F}},
    IT<:StructArray{<:LocalResolventSolution{F}},
    T<:StructArray{<:LocalResolventSolution{F}},
}
    problem::P
    algorithm::A
    initialtrace::IT
    trace::T
    success::Bool
    q::Complex{F}
    reg::Regularizer{F}
    γ²::F
    δ²::F
    k::F
    E::Complex{F}
end

mutable struct LocalResolventSolver{
    F,
    A<:Union{DirectSolve,AbstractSchedule{F}},
    P<:AbstractLocalResolventProblem{F},
    W<:Workspace{F},
    IT<:StructArray{<:LocalResolventSolution{F}},
    T<:StructArray{<:LocalResolventSolution{F}},
    S<:LocalSolution{F},
}
    algorithm::A
    problem::P
    ws::W
    initialtrace::IT
    trace::T
    step::Int
    initial::Bool
    final::Bool
    # XXX: what does this refer to? Was uncommented
    # Warning: do not implement these methods if the underlying storage is not actually strided, as it may lead to incorrect results or segmentation faults.
    done::Bool
    solution::S

    function LocalResolventSolver{F}(problem, algorithm) where {F}
        problem = adapt(F, problem)
        algorithm = adapt(F, algorithm)
        ws = Workspace(problem.eq.ensemble)
        itrace = StructArray{_istepsoltype(problem, algorithm)}(
            undef, initiallength(algorithm)
        )
        trace = StructArray{_stepsoltype(problem, algorithm)}(undef, length(algorithm))
        soltype = _fullsoltype(problem, algorithm, itrace, trace)
        return new{F,typeof.((algorithm, problem, ws, itrace, trace))...,soltype}(
            algorithm, problem, ws, itrace, trace, 0, true, false, false  # solution uninitialized TODO: replace with blank solution
        )
    end
end

function LocalResolventSolver(
    problem::AbstractLocalResolventProblem{F}, algorithm
) where {F}
    return LocalResolventSolver{F}(problem, algorithm)
end

function _makesolution(solver, solution)
    return LocalSolution(
        solver.problem,
        solver.algorithm,
        solver.initialtrace,
        solver.trace,
        solution.success,
        solution.q,
        solution.reg,
        solution.γ²,
        solution.δ²,
        solution.k,
        solution.E,
    )
end

function _fullsoltype(::Type{P}, ::Type{A}, ::Type{IT}, ::Type{T}) where {
    F,
    P<:AbstractLocalResolventProblem{F},
    A<:AbstractAlgorithm{F},
    IT<:StructArray{<:LocalResolventSolution{F}},
    T<:StructArray{<:LocalResolventSolution{F}},
}
    return LocalSolution{F,P,A,IT,T}
end

_fullsoltype(args...) = _fullsoltype(typeof.(args)...)

function _soltype(::Type{P}) where {F,P<:AbstractLocalResolventProblem{F}}
    return LocalResolventSolution{F,P}
end

_soltype(problem) = _soltype(typeof(problem))

function _stepprobtype(
    ::Type{P}, ::Type{<:Union{DirectSolve,<:SymmetrySchedule{F}}}
) where {F,P<:AbstractLocalResolventProblem{F}}
    return P
end

function _stepprobtype(
    ::Type{<:AbstractLocalResolventProblem{F,E}}, ::Type{<:AbstractAlgorithm{F}}
) where {F,E}
    return LocalRegularizerProblem{F,E}
end

_stepprobtype(args...) = _stepprobtype(typeof.(args)...)

_stepsoltype(args...) = _soltype(_stepprobtype(args...))

function _istepprobtype(
    ::Type{P}, ::Type{<:SymmetrySchedule{F,IA}}
) where {F,P<:AbstractLocalResolventProblem{F},IA}
    return _stepprobtype(P, IA)
end

function _istepprobtype(
    ::Type{P}, ::Type{A}
) where {F,P<:AbstractLocalResolventProblem{F},A<:AbstractAlgorithm{F}}
    return _stepprobtype(P, A)
end

_istepprobtype(args...) = _istepprobtype(typeof.(args)...)

_istepsoltype(args...) = _soltype(_istepprobtype(args...))

function init(problem::AbstractLocalResolventProblem, algorithm::AbstractAlgorithm)
    return LocalResolventSolver(problem, algorithm)
end

function step!(solver::LocalResolventSolver)
    ## Step header

    solver.done && return nothing
    solver.initial && return _initialstep!(solver)  # Take care of all the hard things here?
    solver.final && return _finalstep!(solver)  # Take care of all the hard things here?

    current = solver.trace[solver.step]
    solver.step += 1
    stepproblem = _stepproblem(current, solver.algorithm, solver.step)

    ## Perform step

    solver.trace[solver.step] = eqsolve!(solver.ws, stepproblem)

    ## Step footer

    _stepfooter!(solver)
    return nothing
end

function _step!(ws, current, algorithm::RegularizerSchedule, step)
    stepproblem = _stepproblem(current, algorithm, step)
    return eqsolve!(ws, stepproblem)
end

function _step!(ws, current, algorithm::SymmetrySchedule, step)
    stepproblem = _stepproblem(current, algorithm, step)
    return eqsolve!(ws, stepproblem)
end

function _initialstep!(solver::LocalResolventSolver{<:Any,DirectSolve})
    solver.step += 1
    solver.solution = _makesolution(solver, eqsolve!(solver.ws, solver.problem))
    solver.done = true
    return nothing
end

function _initialstep!(solver::LocalResolventSolver{<:Any,<:RegularizerSchedule})
    solver.step += 1
    stepproblem = _stepproblem(solver.problem, solver.algorithm, solver.step)
    solver.trace[solver.step] = eqsolve!(solver.ws, stepproblem)
    solver.initial = false
    _stepfooter!(solver)
    return nothing
end

function _initialstep!(solver::LocalResolventSolver{<:Any,<:SymmetrySchedule})
    if solver.step == 0
        current = solver.problem
    else
        current = solver.initialtrace[solver.step]
    end
    solver.step += 1
    stepproblem = _stepproblem(solver.problem, solver.algorithm.initialalg, solver.step)
    solver.initialtrace[solver.step] = eqsolve!(solver.ws, stepproblem)
    _stepfooter!(solver, true)
    return nothing
end

function _finalstep!(solver)
    current = solver.trace[solver.step]
    solver.step += 1
    stepproblem = _reinit(solver.problem, current)
    solver.solution = _makesolution(solver, eqsolve!(solver.ws, stepproblem))
    solver.done = true
    return nothing
end

function _stepfooter!(solver, initial=false)
    if invalid(solver)  # failure condition observed (nans and stuff)
        solver.solution = failed_solution
        solver.done = true
    elseif done(solver)  # solution found
        solver.solution = _makesolution(solver)
        solver.done = true
    elseif final(solver)  # solution intersected, DirectSolve next
        solver.final = true
    elseif abort(solver)  # termination condition observed without having intersected the solution
        solver.solution = failed_solution
        solver.done = true
    end
end

function _stepproblem(problem::LocalRegularizerProblem, alg::SymmetrySchedule, step)
    return LocalRegularizerProblem(
        problem.eq, problem.reg, problem.q0, alg.schedule[step]
    )
end

function _stepproblem(problem::LocalTraceProblem, alg::SymmetrySchedule, step)
    return LocalTraceProblem(
        problem.eq, problem.k, problem.q0, problem.reg0, alg.schedule[step]
    )
end

function _stepproblem(
    current::LocalResolventSolution{<:Any,<:LocalRegularizerProblem},
    alg::SymmetrySchedule,
    step,
)
    return LocalRegularizerProblem(
        current.problem.eq, current.reg, current.q, alg.schedule[step]
    )
end

function _stepproblem(
    current::LocalResolventSolution{<:Any,<:LocalTraceProblem},
    alg::SymmetrySchedule,
    step,
)
    return LocalTraceProblem(
        current.problem.eq, current.k, current.q, current.reg, alg.schedule[step]
    )
end

function _reinit(problem::LocalTraceProblem, currentsolution, μ=problem.μ)
    return LocalTraceProblem(
        problem.eq, problem.k, currentsolution.q, currentsolution.reg, μ
    )
end

function _reinit(problem::LocalRegularizerProblem, currentsolution, μ=problem.μ)
    return LocalRegularizerProblem(problem.eq, problem.reg, currentsolution.q, μ)
end

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
