function qsolve!(workspace, objective::Fixr², z, q₀; verbosity)
    f(qv) = qf!(workspace, objective, z, qv)
    qv, niters = newton(f, (@SVector [real(q₀), imag(q₀)]))
    print_niters(niters, verbosity)
    return complex(qv[1], qv[2])
end

function qsolve!(workspace, objective::Fixk, z, q₀, δ₀²; verbosity)
    f(qδ²v) = qδ²f!(workspace, objective, z, qδ²v)
    qδ²v, niters = newton(f, (@SVector [real(q₀), imag(q₀), δ₀²]))
    print_niters(niters, verbosity)
    return complex(qδ²v[1], qδ²v[2]), qδ²v[3]
end

function qf!(workspace, objective, z, qv)
    f = _qf!(workspace, objective, z, complex(qv[1], qv[2]))
    return (@SVector [real(f), imag(f)])
end

function _qf!(workspace, objective::Fixr², z, q)
    Q = Q!(workspace, objective, z, q)
    P = P!(workspace, objective, z, q)
    f = @inbounds trTAᴴB(workspace.ensemble.T, Q, P)  # Tr[TQᴴP]
    return muladd(f, objective.μ / length(workspace.ensemble.T), -q)
end

function qj!(workspace, objective, z, qv)
    q = complex(qv[1], qv[2])
    df_dq, df_dq⁺ = _qj!(workspace, objective, z, q)
    df_dqR = df_dq + df_dq⁺
    df_idqI = df_dq - df_dq⁺
    return (@SMatrix [real(df_dqR) -imag(df_idqI); imag(df_dqR) real(df_idqI)])
end

function _qj!(workspace, objective::Fixγ²{Qnormal}, z, q) where {Qnormal}
    P = P!(workspace, objective, z, q)
    QᴴP = QᴴP!(workspace, objective, z, q)
    QᴴPQ = QᴴPQ!(workspace, objective, z, q)
    ensemble = workspace.ensemble
    df_dq = @inbounds trTAᴴTB(ensemble.T, QᴴP', QᴴP)  # Tr[(TQᴴP)²]
    df_dq⁺ = @inbounds trTAᴴTB(ensemble.T, QᴴPQ, P)  # Tr[TQᴴPQTP] = Tr[T(QᴴPQ)ᴴTP]

    if Qnormal
        PQᴴ = PQᴴ!(workspace, objective, z, q)
        QᴴPQᴴ = QᴴPQᴴ!(workspace, objective, z, q)
        df_dq += @inbounds trTAᴴTB(ensemble.T, P, QᴴPQᴴ)  # Tr[TQᴴPQᴴTP] == Tr[TPᴴTQᴴPQᴴ]
        df_dq⁺ += @inbounds trTAᴴTB(ensemble.T, PQᴴ, QᴴP)  # Tr[TQᴴPTQP] == Tr[T(PQᴴ)ᴴTQᴴP]
        df_dq /= 2
        df_dq⁺ /= 2
    end

    N = length(ensemble.T)
    @inbounds for i in 1:N  # Tr[T²P]
        df_dq⁺ -= ensemble.T[i]^2 * real(P[i, i])
    end
    μ²N⁻¹ = objective.μ^2 / N
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

function trTAᴴB(T, A, B)
    out = zero(promote_type(eltype(A), eltype(B)))
    for j in 1:length(T), i in 1:length(T)
        out += T[j] * conj(A[i, j]) * B[i, j]
    end
    return out
end

function trTAᴴTB(T, A, B)
    out = zero(promote_type(eltype(A), eltype(B)))
    for j in 1:length(T), i in 1:length(T)
        out += T[i] * T[j] * conj(A[i, j]) * B[i, j]
    end
    return out
end

function trTAᴴTB(T, A::Hermitian, B::Hermitian)
    out = zero(promote_type(eltype(A), eltype(B)))
    for j in 1:length(T)
        for i in 1 : (j - 1)
            out += 2 * T[i] * T[j] * real(conj(A[i, j]) * B[i, j])
        end
        out += T[j]^2 * real(A[j, j]) * real(B[j, j])
    end
    return out
end
