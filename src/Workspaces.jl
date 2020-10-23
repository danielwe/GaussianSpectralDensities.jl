mutable struct WorkspaceStatus
    G0_inv_and_Q_off_diag::Bool
    Q_and_P::Bool
    QhP_and_QhPQ_I::Bool
    γsq::Bool

    WorkspaceStatus() = new(false, false, false, false)
end

function reset!(status::WorkspaceStatus)
    status.G0_inv_and_Q_off_diag = false
    status.Q_and_P = false
    status.QhP_and_QhPQ_I = false
    status.γsq = false
    return nothing
end

struct Workspace{U,V}
    ensemble::MatrixEnsemble{U,V}
    N_inv::Float64
    G0_inv::U
    Q::U
    P::Hermitian{ComplexF64,Matrix{ComplexF64}}
    QhP::Matrix{ComplexF64}
    QhPQ_I::Hermitian{ComplexF64,Matrix{ComplexF64}}
    γsq::Ref{Float64}
    status::WorkspaceStatus

    function Workspace(ensemble::MatrixEnsemble{U,V}) where {U,V}
        Q = similar(ensemble.LiMRi)
        QhP = Matrix{ComplexF64}(undef, ensemble.N, ensemble.N)
        return new{U,V}(
            ensemble,
            1.0 / ensemble.N,
            similar(Q),
            Q,
            Hermitian(similar(QhP)),
            QhP,
            Hermitian(similar(QhP)),
            Ref(NaN),
            WorkspaceStatus(),
        )
    end
end

function G0_inv_and_Q_off_diag!(ws::Workspace, z)
    reset!(ws.status)
    @inbounds bare_resolvent_inv!(ws.ensemble, ws.G0_inv, z)
    @inbounds copy!(ws.Q, ws.G0_inv)
    ws.status.G0_inv_and_Q_off_diag = true
    return nothing
end

function Q_and_P!(ws::Workspace, q, rsq, ::Val{δ}) where δ
    @assert ws.status.G0_inv_and_Q_off_diag
    @inbounds for i in 1:ws.ensemble.N
        ws.Q[i, i] = muladd(-q, ws.ensemble.T[i], ws.G0_inv[i, i])
    end
    mul!(ws.P.data, ws.Q, ws.Q')
    if δ
        rsq -= eigmin(ws.P)
    end
    ws.γsq[] = rsq
    ws.status.γsq = true
    @inbounds for i in diagind(ws.P)
        ws.P[i] += rsq
    end
    LAPACK.potrf!('U', ws.P.data)
    LAPACK.potri!('U', ws.P.data)
    ws.status.Q_and_P = true
    ws.status.QhP_and_QhPQ_I = false
    return nothing
end

function QhP_and_QhPQ_I!(ws::Workspace)
    @assert ws.status.Q_and_P
    mul!(ws.QhP, ws.Q', ws.P)
    mul!(ws.QhPQ_I.data, ws.QhP, ws.Q)
    @inbounds for i in diagind(ws.QhPQ_I)
        ws.QhPQ_I[i] -= 1.0
    end
    ws.status.QhP_and_QhPQ_I = true
    return nothing
end

function k(ws::Workspace)
    @assert ws.status.Q_and_P
    return tr(ws.P) * ws.N_inv
end

function δsq(ws::Workspace)
    @assert ws.status.Q_and_P
    return 1.0 / eigmax(ws.P)
end
