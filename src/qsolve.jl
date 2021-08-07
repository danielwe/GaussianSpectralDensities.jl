function qsolve!(workspace, objective::Fixγ², z, q₀, verbosity=0)
    f(qv) = qf!(workspace, objective, z, qv)
    j(qv, _) = qj!(workspace, objective, z, qv)
    qv, niters = newton(f, j, (@SVector [real(q₀), imag(q₀)]))
    print_niters(niters, verbosity)
    return complex(qv[1], qv[2])
end

function qsolve!(workspace, objective::Fixδ², z, q₀, verbosity=0)
    f(qv) = qf!(workspace, objective, z, qv)
    qv, niters = newton(f, (@SVector [real(q₀), imag(q₀)]))
    print_niters(niters, verbosity)
    return complex(qv[1], qv[2])
end

function qsolve!(workspace, objective::Fixk, z, q₀, δ₀², verbosity=0)
    f(qδ²v) = qδ²f!(workspace, objective, z, qδ²v)
    qδ²v, niters = newton(f, (@SVector [real(q₀), imag(q₀), δ₀²]))
    print_niters(niters, verbosity)
    return complex(qδ²v[1], qδ²v[2]), qδ²v[3]
end

function qf!(workspace, objective, z, qv)
    f = _qf!(workspace, objective, z, complex(qv[1], qv[2]))
    return (@SVector [real(f), imag(f)])
end

function _qf!(workspace, objective, z, q)
    T = ensemble(workspace).T
    Q = Q!(workspace, objective, z, q)
    P = P!(workspace, objective, z, q)
    f = trTAᴴB(T, Q, P)  # Tr[TQᴴP]
    return muladd(f, objective.μ / length(T), -q)
end

function qj!(workspace, objective, z, qv)
    q = complex(qv[1], qv[2])
    df_dq, df_dq⁺ = _qj!(workspace, objective, z, q)
    df_dqR = df_dq + df_dq⁺
    df_idqI = df_dq - df_dq⁺
    return (@SMatrix [real(df_dqR) -imag(df_idqI); imag(df_dqR) real(df_idqI)])
end

function _qj!(workspace, objective::Fixγ²{Qnormal}, z, q) where {Qnormal}
    T = ensemble(workspace).T
    P = P!(workspace, objective, z, q)
    PQ = PQ!(workspace, objective, z, q)
    QᴴPQ = QᴴPQ!(workspace, objective, z, q)
    df_dq = trTAᴴTB(T, PQ, PQ')  # Tr[(TQᴴP)²]
    df_dq⁺ = trTAᴴTB(T, QᴴPQ, P)  # Tr[TQᴴPQTP]

    if Qnormal
        QP = QP!(workspace, objective, z, q)
        QPQ = QPQ!(workspace, objective, z, q)
        df_dq += trTAᴴTB(T, QPQ, P)  # Tr[TQᴴPQᴴTP]
        df_dq⁺ += trTAᴴTB(T, PQ, QP)  # Tr[TQᴴPTQP]
        df_dq /= 2
        df_dq⁺ /= 2
    end

    df_dq⁺ -= trT²A(T, P)  # Tr[T²P]
    μ²N⁻¹ = objective.μ^2 / length(T)
    df_dq = muladd(df_dq, μ²N⁻¹, -one(q))
    df_dq⁺ *= μ²N⁻¹
    return df_dq, df_dq⁺
end

function qδ²f!(workspace, objective, z, qδ²v)
    qf, δ²f = _qδ²f!(workspace, objective, z, complex(qδ²v[1], qδ²v[2]), qδ²v[3])
    return (@SVector [real(qf), imag(qf), δ²f])
end

function _qδ²f!(workspace, objective::Fixk{Qnormal}, z, q, δ²) where {Qnormal}
    δ²objective = Fixδ²{Qnormal}(δ², objective.μ)
    qf = _qf!(workspace, δ²objective, z, q)
    δ²f = k!(workspace, δ²objective, z, q) - objective.value
    return qf, δ²f
end
