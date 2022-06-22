function qsolve!(problem::LocalSpectralProblem{<:Any, <:Any, <:Fixr²}, q₀, ws, verbosity=0)
    function f(qv)
        f = qf!(problem, complex(qv[1], qv[2]), ws)
        return @SVector [real(f), imag(f)]
    end
    q₀v = @SVector [real(q₀), imag(q₀)]
    if applicable(qj!, problem, q₀, ws)
        function j(qv, _)
            df_dq, df_dq⁺ = qj!(problem, complex(qv[1], qv[2]), ws)
            df_dqR = df_dq + df_dq⁺
            df_idqI = df_dq - df_dq⁺
            return (@SMatrix [real(df_dqR) -imag(df_idqI); imag(df_dqR) real(df_idqI)])
        end
        qv, niters = newton(f, j, q₀v)
    else
        qv, niters = newton(f, q₀v)
    end
    print_niters(niters, verbosity)
    return complex(qv[1], qv[2])
end

function qsolve!(
    problem::LocalSpectralProblem{<:Any, <:Any, <:Fixk}, q₀, r₀²::O, ws, verbosity=0
) where {O}
    function f(qr²v)
        qf, r²f = qr²f!(problem, complex(qr²v[1], qr²v[2]), O(qr²v[3]), ws)
        return @SVector [real(qf), imag(qf), r²f.value]
    end
    qr²v, niters = newton(f, @SVector [real(q₀), imag(q₀), r₀².value])
    print_niters(niters, verbosity)
    return complex(qδ²v[1], qδ²v[2]), O(qr²v[3])
end

function qf!(problem::LocalSpectralProblem{<:Any, <:Any, <:Fixr²}, q, ws)
    f = trTAᴴB(problem.ensemble.T, Q!(problem, q, ws), P!(problem, q, ws))  # Tr[TQᴴP]
    return muladd(f, problem.μ / problem.ensemble.N, -q)
end

function qj!(
    problem::LocalSpectralProblem{NormalizeQ, <:Any, <:Fixγ²}, q, ws
) where {NormalizeQ}
    T = problem.ensemble.T
    P = P!(problem, q, ws)
    PQ = PQ!(problem, q, ws)
    QᴴPQ = QᴴPQ!(problem, q, ws)
    df_dq = trTAᴴTB(T, PQ, PQ')  # Tr[(TQᴴP)²]
    df_dq⁺ = trTAᴴTB(T, QᴴPQ, P)  # Tr[TQᴴPQTP]
    if NormalizeQ
        QP = QP!(problem, q, ws)
        QPQ = QPQ!(problem, q, ws)
        df_dq += trTAᴴTB(T, QPQ, P)  # Tr[TQᴴPQᴴTP]
        df_dq⁺ += trTAᴴTB(T, PQ, QP)  # Tr[TQᴴPTQP]
        df_dq /= 2
        df_dq⁺ /= 2
    end
    df_dq⁺ -= trT²A(T, P)  # Tr[T²P]
    μ²N⁻¹ = problem.μ^2 / problem.ensemble.N
    df_dq = muladd(df_dq, μ²N⁻¹, -one(q))
    df_dq⁺ *= μ²N⁻¹
    return df_dq, df_dq⁺
end

function qr²f!(
    problem::LocalSpectralProblem{NormalizeQ, <:Any, <:Fixk}, q, r²::O, ws
) where {NormalizeQ, O<:Fixr²}
    r²problem = LocalSpectralProblem{NormalizeQ}(r², problem.ensemble, problem.z, problem.μ)
    qf = qf!(r²problem, q, ws)
    r²f = O(k!(r²problem, q, ws) - problem.objective.value)
    return qf, r²f
end
