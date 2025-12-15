const ConvergenceException = Union{NewtonException,SingularException,LAPACKException}

"""
    eqsolve!(
        ws::Workspace{F}, problem::AbstractLocalResolventProblem{F}, verbosity=0
    ) where {F}

Solve an averaged resolvent equation at a point in the complex plane

Returns a `LocalResolventSolution`
"""
function eqsolve!(
    ws::Workspace{F}, problem::LocalRegularizerProblem{F}, verbosity=0
) where {F}
    fq = let μ=problem.μ, ws=ws, eq=problem.eq, reg=problem.reg
        function fq(qv)
            q = complex(qv...)
            dq = μ * q_μrhs!(ws, eq, μ * q, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
    q0v = SVector(real(problem.q0), imag(problem.q0))
    local qv, niters, success
    try
        qv, niters = newton(fq, q0v)
        success = true
    catch e
        if e isa ConvergenceException
            qv, niters = problem.q0, -1
            success = false
        else
            rethrow()
        end
    end
    # print_niters(niters, verbosity)
    q, reg = complex(qv...), problem.reg
    k = k!(ws, problem.eq, problem.μ * q, reg)
    return _local_resolvent_solution!(ws, problem, success, q, reg, k)
end

function eqsolve!(ws::Workspace{F}, problem::LocalTraceProblem{F}, verbosity=0) where {F}
    fqreg = let kind=problem.reg0.kind, μ=problem.μ, ws=ws, eq=problem.eq, klhs=problem.k
        function fqreg(qregv)
            q, reg = complex(qregv[1], qregv[2]), Regularizer(kind, qregv[3])
            μq = μ * q
            dq = μ * q_μrhs!(ws, eq, μq, reg) - q
            dk = k!(ws, eq, μq, reg) - klhs
            return SVector(real(dq), imag(dq), dk)
        end
    end
    qreg0v = SVector(real(problem.q0), imag(problem.q0), problem.reg0.value)
    local qregv, niters, success
    try
        qregv, niters = newton(fqreg, qreg0v)
        success = true
    catch e
        if e isa ConvergenceException
            qregv, niters = qreg0v, -1
            success = false
        else
            rethrow()
        end
    end
    # print_niters(niters, verbosity)
    q, reg = complex(qregv[1], qregv[2]), Regularizer(problem.reg0.kind, qregv[3])
    return _local_resolvent_solution!(ws, problem, success, q, reg, problem.k)
end

function _local_resolvent_solution!(ws, problem, success, q, reg, k)
    μq = problem.μ * q
    γ² = γ²!(ws, problem.eq, μq, reg)
    δ² = δ²!(ws, problem.eq, μq, reg)
    E = E!(ws, problem.eq, μq, reg)
    return LocalResolventSolution(problem, success, q, reg, γ², δ², k, E)
end

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
