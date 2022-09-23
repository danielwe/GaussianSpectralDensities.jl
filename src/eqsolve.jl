"""
    eqsolve!(
        ws::Workspace{F}, problem::AbstractLocalResolventProblem{F}, verbosity=0
    ) where {F}

Solve an averaged resolvent equation at a point in the complex plane

Returns a `LocalResolventSolution`
"""
function eqsolve!(ws::Workspace{F}, problem::LocalQProblem{F}, verbosity=0) where {F}
    fq = let ws=ws, eq=problem.equation, regularizer=problem.regularizer
        function fq(qv)
            q = complex(qv...)
            dq = qrhs!(ws, eq, q, regularizer) - q
            return SVector(real(dq), imag(dq))
        end
    end
    q0v = SVector(real(problem.q0), imag(problem.q0))
    qv, niters = newton(fq, q0v)
    # print_niters(niters, verbosity)
    q, regularizer = complex(qv...), problem.regularizer
    k = k!(ws, problem.equation, q, regularizer)
    return _local_resolvent_solution!(ws, problem, q, regularizer, k)
end

function eqsolve!(
    ws::Workspace{F}, problem::LocalQRegularizerProblem{F}, verbosity=0
) where {F}
    fqr = let kind=problem.regularizer0.kind, ws=ws, eq=problem.equation, klhs=problem.k
        function fqr(qrv)
            q, regularizer = complex(qrv[1], qrv[2]), Regularizer(kind, qrv[3])
            dq = qrhs!(ws, eq, q, regularizer) - q
            dk = k!(ws, eq, q, regularizer) - klhs
            return SVector(real(dq), imag(dq), dk)
        end
    end
    qr0v = SVector(real(problem.q0), imag(problem.q0), problem.regularizer0.value)
    qrv, niters = newton(fqr, qr0v)
    # print_niters(niters, verbosity)
    q, regularizer = complex(qrv[1], qrv[2]), Regularizer(problem.regularizer0.kind, qrv[3])
    return _local_resolvent_solution!(ws, problem, q, regularizer, problem.k)
end

function _local_resolvent_solution!(ws, problem, q, regularizer, k)
    γ² = γ²!(ws, problem.equation, q, regularizer)
    δ² = δ²!(ws, problem.equation, q, regularizer)
    E = E!(ws, problem.equation, q, regularizer)
    return LocalResolventSolution(problem, q, regularizer, γ², δ², k, E)
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
