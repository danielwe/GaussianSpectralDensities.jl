function solve(ensemble::GaussianMatrixEnsemble, z, args...; kwargs)
    return create_solver(ensemble)(z, args...; kwargs...)
end

function create_solver(ensemble::GaussianMatrixEnsemble)
    let ws = Workspace(ensemble)
        function solver(z, args...; kwargs...)
            return solve!(ws, z, args...; kwargs...)
        end
        return solver
    end
end

function solve!(ws::Workspace, z, algorithm::Algorithm = DEFAULT_ALGORITHM; debug=false)
    print_z(z)
    if debug
        trace = fill(IntermediateSolution(), length(algorithm.rsqs))
    else
        trace = SolutionTrace()
    end
    if !isfinite(z)
        return Solution(trace)
    end
    if debug
        print_hline()
    end
    G0_inv_and_Q_off_diag!(ws, z)
    q0 = q_asymptote(ws, algorithm.rsqs[1])
    solution = solve!(trace, ws, z, q0, algorithm; debug=debug)
    if debug
        print_row(solution)
        print_hline()
    end
    return solution
end

function solve!(trace::SolutionTrace, ws::Workspace, z, q0, algorithm::FixK; debug=false)
    if debug
        print_header()
    end
    local q_, k_, γsq_
    solution = Solution(trace)
    γsq0 = algorithm.rsqs[1]
    for (i, δsq_) in enumerate(algorithm.rsqs)
        q_ = q!(ws, δsq_, q0, δ=true)
        γsq_ = ws.γsq[]
        k_ = k(ws)
        if debug
            trace[i] = IntermediateSolution(γsq_, δsq_, q_, k_)
            print_row(trace[i])
        end
        if γsq_ * δsq_ < -0.5
            break
        end
        if k_ >= 1.0
            γsq_, q_ = rsq_and_q!(ws, γsq0, q0, δ=false)
            δsq_ = δsq(ws)
            solution = Solution(γsq_, δsq_, q_, 1.0, trace)
            break
        end
        q0 = q_
        γsq0 = γsq_
    end
    if debug
        print_footer()
    end
    return solution
end

function solve!(trace::SolutionTrace, ws::Workspace, z, q0, algorithm::FixΓ; debug=false)
    if debug
        print_header()
    end
    local q_, k_, δsq_
    for (i, γsq_) in enumerate(algorithm.rsqs)
        q_ = q!(ws, γsq_, q0; δ=false)
        k_ = k(ws)
        if debug
            δsq_ = δsq(ws)
            trace[i] = IntermediateSolution(γsq_, δsq_, q_, k_)
            print_row(trace[i])
        end
        q0 = q_
    end
    try
        q_ = q!(ws, 0.0, q0; δ=false)
    catch err
        if isa(err, SingularException) || isa(err, LAPACKException)
            q_ = complex(NaN, NaN)
        else
            rethrow()
        end
    end
    if isfinite(q_)
        k_ = k(ws)
        δsq_ = δsq(ws)
    else
        k_ = Inf
        δsq_ = 0.0
    end
    if debug
        print_footer()
    end
    return Solution(0.0, δsq_, q_, k_, trace)
end

