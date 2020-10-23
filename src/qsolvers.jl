function residues!(Fv, Jv, ws::Workspace, rsq, qv, ::Val{δ}) where δ
    qc = complex(qv[1], qv[2])
    Q_and_P!(ws, qc, rsq, Val(δ))
    Fc = complex(0.0)
    if Jv === nothing
        @inbounds for j in 1:ws.ensemble.N, i in 1:ws.ensemble.N  # optimized Tr(TQhP)
            Fc += ws.ensemble.T[j] * conj(ws.Q[i, j]) * ws.P[i, j]
        end
    else
        QhP_and_QhPQ_I!(ws)
        dFc_dq = complex(0.0)
        dFc_dqconj = 0.0
        @inbounds for j in 1:ws.ensemble.N  # optimized(?) Tr((TQhP)^2) & Tr(T(QhPQ - I)TP)
            Fc += ws.ensemble.T[j] * ws.QhP[j, j]
            for i in 1:ws.ensemble.N
                # [j, i] does not respect memory order. Would copying help?
                τ = ws.ensemble.T[i] * ws.ensemble.T[j]
                dFc_dq += τ * ws.QhP[i, j] * ws.QhP[j, i]
                dFc_dqconj += τ * real(ws.QhPQ_I[i, j] * ws.P[j, i])
            end
        end
        dFc_dq = muladd(dFc_dq, ws.N_inv, -1.0)
        dFc_dqconj *= ws.N_inv
        Jv[1, 1] = real(dFc_dq) + dFc_dqconj
        Jv[2, 1] = imag(dFc_dq)
        Jv[1, 2] = -imag(dFc_dq)
        Jv[2, 2] = real(dFc_dq) - dFc_dqconj
    end
    Fc = muladd(Fc, ws.N_inv, -qc)
    Fv[1] = real(Fc)
    Fv[2] = imag(Fc)
    return nothing
end

function residues!(F, ws::Workspace, rsq_and_qv, ::Val{δ}) where δ
    residues!((@view F[2:3]), nothing, ws, rsq_and_qv[1], (@view rsq_and_qv[2:3]), Val(δ))
    F[1] = k(ws) - 1.0
    return nothing
end

function q!(ws::Workspace, rsq, q0; δ)
    qv = nlsolve(
        #only_fj!((Fv, Jv, qv) -> residues!(Fv, Jv, ws, rsq, qv, Val(δ))),
        #[real(q0); imag(q0)];
        #method=:newton,
        (Fv, qv) -> residues!(Fv, nothing, ws, rsq, qv, Val(δ)),
        [real(q0); imag(q0)];
        method=:anderson,
        m=1,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return complex(qv[1], qv[2])
end

function rsq_and_q!(ws::Workspace, rsq0, q0; δ)
    rsq_and_qv = nlsolve(
        (F, rsq_and_qv) -> residues!(F, ws, rsq_and_qv, Val(δ)),
        [rsq0; real(q0); imag(q0)];
        method=:anderson,
        m=1,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return rsq_and_qv[1], complex(rsq_and_qv[2], rsq_and_qv[3])
end

function q_asymptote(ws::Workspace, γsq)
    @assert ws.status.G0_inv_and_Q_off_diag
    q = complex(0.0)
    @inbounds for i in 1:ws.ensemble.N  # optimized Tr(TG0_inv')
        q += ws.ensemble.T[i] * conj(ws.G0_inv[i, i])
    end
    return q * ws.N_inv / γsq
end
