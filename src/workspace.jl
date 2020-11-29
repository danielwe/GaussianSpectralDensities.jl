mutable struct WorkspaceStatus{R}
    z::MissingOr{Complex{R}}
    μq⁻::MissingOr{Complex{R}}
    r²::MissingOr{Tuple{DataType, R}}
    QᴴP::Bool
    QᴴPQ::Bool
    PQᴴ::Bool
    QᴴPQᴴ::Bool

    WorkspaceStatus{R}() where {R} = new{R}(
        missing, missing, missing, false, false, false, false
    )
end

function set_z!(status, z)
    set_μq⁻!(status, missing)
    status.z = z
    return nothing
end

function set_μq⁻!(status::WorkspaceStatus{R}, μq⁻) where {R}
    set_r²!(status, missing)
    status.μq⁻ = μq⁻
    return nothing
end

function set_r²!(status, r²)
    status.QᴴPQᴴ = false
    status.PQᴴ = false
    status.QᴴPQ = false
    status.QᴴP = false
    status.r² = r²
    return nothing
end

abstract type AbstractWorkspace{R<:Real} end

struct Workspace{R, U, V, W, X, Y} <: AbstractWorkspace{R}
    ensemble::MatrixEnsemble{R, U, V, W, X, Y}
    G₀⁻¹::Y
    Q::Y
    QQᴴ::Hermitian{Complex{R},Matrix{Complex{R}}}
    P::Hermitian{Complex{R},Matrix{Complex{R}}}
    QᴴP::Matrix{Complex{R}}
    QᴴPQ::Hermitian{Complex{R},Matrix{Complex{R}}}
    PQᴴ::Matrix{Complex{R}}
    QᴴPQᴴ::Matrix{Complex{R}}
    s²min::Ref{R}
    status::WorkspaceStatus{R}

    function Workspace(ensemble::MatrixEnsemble{R, U, V, W, X, Y}) where {R, U, V, W, X, Y}
        G₀⁻¹ = similar(ensemble.L⁻¹MR⁻¹)
        P = Hermitian(Matrix{Complex{R}}(undef, length(ensemble.T), length(ensemble.T)))
        return new{R, U, V, W, X, Y}(
            ensemble,
            G₀⁻¹,
            similar(G₀⁻¹),
            similar(P),
            P,
            similar(P.data),
            similar(P),
            similar(P.data),
            similar(P.data),
            Ref(convert(R, NaN)),
            WorkspaceStatus{R}(),
        )
    end
end

workspace(ensemble::MatrixEnsemble) = Workspace(ensemble)

function G₀⁻¹!(workspace, z)
    if workspace.status.z !== z
        ensemble = workspace.ensemble
        for j in 1:size(workspace.G₀⁻¹, 2), i in 1:size(workspace.G₀⁻¹, 1)
            workspace.G₀⁻¹[i, j] = workspace.Q[i, j] = muladd(
                z, ensemble.L⁻¹R⁻¹[i, j], -ensemble.L⁻¹MR⁻¹[i, j]
            )
        end
        set_z!(workspace.status, z)
    end
    return workspace.G₀⁻¹
end

function Q!(workspace, objective::Fixr²{QNormal}, z, q) where {QNormal}
    G₀⁻¹ = G₀⁻¹!(workspace, z)
    μq⁻ = -objective.μ * q
    if workspace.status.μq⁻ !== μq⁻
        @inbounds for i in 1:length(workspace.ensemble.T)
            workspace.Q[i, i] = muladd(μq⁻, workspace.ensemble.T[i], G₀⁻¹[i, i])
        end
        mul!(workspace.QQᴴ.data, workspace.Q, workspace.Q')
        if QNormal
            # This is an attempt to get around the order of limits issue by
            # replacing QQ' with (QQ' + Q'Q) / 2, cancelling any non-normality in
            # Q. This is not mathematically sound in general, but it does the trick
            # in certain experiments.
            half = one(eltype(Q)) / 2
            mul!(workspace.QQᴴ.data, workspace.Q', workspace.Q, half, half)
        end
        copy!(workspace.P.data, workspace.QQᴴ.data)
        workspace.s²min[] = eigvals!(workspace.P, 1:1)[1]
        set_μq⁻!(workspace.status, μq⁻)
    end
    return workspace.Q
end

function P!(workspace, objective, z, q)
    Q!(workspace, objective, z, q)
    r² = (typeof(objective), objective.value)
    if workspace.status.r² !== r²
        γ² = _γ²!(objective, workspace.s²min[])
        copy!(workspace.P.data, workspace.QQᴴ.data)
        @inbounds for i in 1:size(workspace.P, 1)
            workspace.P[i, i] += γ²
        end
        LAPACK.potrf!(workspace.P.uplo, workspace.P.data)
        LAPACK.potri!(workspace.P.uplo, workspace.P.data)
        set_r²!(workspace.status, r²)
    end
    return workspace.P
end

_γ²!(objective::Fixγ², ::Any) = objective.value
_γ²!(objective::Fixδ², s²min) = objective.value - s²min

function QᴴP!(workspace, objective, z, q)
    Q = Q!(workspace, objective, z, q)
    P = P!(workspace, objective, z, q)
    if !workspace.status.QᴴP
        mul!(workspace.QᴴP, Q', P)
        workspace.status.QᴴP = true
    end
    return workspace.QᴴP
end

function QᴴPQ!(workspace, objective, z, q)
    Q = Q!(workspace, objective, z, q)
    QᴴP = QᴴP!(workspace, objective, z, q)
    if !workspace.status.QᴴPQ
        mul!(workspace.QᴴPQ.data, QᴴP, Q)
        workspace.status.QᴴPQ = true
    end
    return workspace.QᴴPQ
end

function PQᴴ!(workspace, objective, z, q)
    Q = Q!(workspace, objective, z, q)
    P = P!(workspace, objective, z, q)
    if !workspace.status.PQᴴ
        mul!(workspace.PQᴴ, P, Q')
        workspace.status.PQᴴ = true
    end
    return workspace.PQᴴ
end

function QᴴPQᴴ!(workspace, objective, z, q)
    Q = Q!(workspace, objective, z, q)
    PQᴴ = PQᴴ!(workspace, objective, z, q)
    if !workspace.status.QᴴPQᴴ
        mul!(workspace.QᴴPQᴴ, Q', PQᴴ)
        workspace.status.QᴴPQᴴ = true
    end
    return workspace.QᴴPQᴴ
end

γ²!(::Any, objective::Fixγ², ::Any, ::Any) = objective.value

function γ²!(workspace, objective::Fixδ², z, q)
    Q!(workspace, objective, z, q)
    return objective.value - workspace.s²min[]
end

δ²!(::Any, objective::Fixδ², ::Any, ::Any) = objective.value

function δ²!(workspace, objective::Fixγ², z, q)
    Q!(workspace, objective, z, q)
    return objective.value + workspace.s²min[]
end

function k!(workspace, objective, z, q)
    P = P!(workspace, objective, z, q)
    return tr(P) / size(P, 1)
end

function q∞!(workspace, γ∞², z)
    q = zero(z)
    G₀⁻¹ = G₀⁻¹!(workspace, z)
    N = length(workspace.ensemble.T)
    @inbounds for i in 1:N  # Tr[TG₀⁻¹]'
        q += workspace.ensemble.T[i] * conj(G₀⁻¹[i, i])
    end
    return q / (γ∞² * N)
end
