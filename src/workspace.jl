mutable struct WorkspaceStatus{R, E<:AbstractMatrixEnsemble{R}}
    ensemble::E
    z::MissingOr{Complex{R}}
    μq⁻::MissingOr{Complex{R}}
    r²::MissingOr{Objective{<:Any, R}}
    PQ::Bool
    QᴴPQ::Bool
    QP::Bool
    QPQ::Bool
    PL⁻¹R⁻¹::Bool

    function WorkspaceStatus(ensemble::AbstractMatrixEnsemble{R}) where {R}
        return new{R, typeof(ensemble)}(
            ensemble, missing, missing, missing, false, false, false, false
        )
    end
end

function set_ensemble!(status::WorkspaceStatus{R, E}, ensemble::E) where {R, E}
    set_z!(status, missing)
    status.ensemble = ensemble
    return nothing
end

function set_z!(status::WorkspaceStatus, z)
    set_μq⁻!(status, missing)
    status.z = z
    return nothing
end

function set_μq⁻!(status::WorkspaceStatus, μq⁻)
    set_r²!(status, missing)
    status.μq⁻ = μq⁻
    return nothing
end

function set_r²!(status::WorkspaceStatus, r²)
    status.PL⁻¹R⁻¹ = false
    status.QPQ = false
    status.QP = false
    status.QᴴPQ = false
    status.PQ = false
    status.r² = r²
    return nothing
end

abstract type AbstractWorkspace{R<:Real} end

struct Workspace{R, U, V, W, X, Y} <: AbstractWorkspace{R}
    status::WorkspaceStatus{R, MatrixEnsemble{R, U, V, W, X, Y}}
    G₀⁻¹_diag::Vector{Complex{R}}
    Q::Y
    QQᴴ::Hermitian{Complex{R},Matrix{Complex{R}}}
    P⁻¹f::Matrix{Complex{R}}
    P::Hermitian{Complex{R},Matrix{Complex{R}}}
    PQ::Matrix{Complex{R}}
    QᴴPQ::Hermitian{Complex{R},Matrix{Complex{R}}}
    QP::Matrix{Complex{R}}
    QPQ::Matrix{Complex{R}}  # Also used for PL⁻¹R⁻¹
    s²min::Base.RefValue{R}

    function Workspace(ensemble::MatrixEnsemble{R, U, V, W, X, Y}) where {R, U, V, W, X, Y}
        status = WorkspaceStatus(ensemble)
        N = length(ensemble.T)
        G₀⁻¹_diag = Vector{Complex{R}}(undef, N)
        Q = similar(ensemble.L⁻¹MR⁻¹)
        QQᴴ = Hermitian(Matrix{Complex{R}}(undef, N, N))
        P⁻¹f = Matrix{Complex{R}}(undef, N, N)
        P = Hermitian(Matrix{Complex{R}}(undef, N, N))
        PQ = Matrix{Complex{R}}(undef, N, N)
        QᴴPQ = Hermitian(Matrix{Complex{R}}(undef, N, N))
        QP = Matrix{Complex{R}}(undef, N, N)
        QPQ = Matrix{Complex{R}}(undef, N, N)
        return new{R, U, V, W, X, Y}(
            status, G₀⁻¹_diag, Q, QQᴴ, P⁻¹f, P, PQ, QᴴPQ, QP, QPQ, Ref(R(NaN))
        )
    end
end

workspace(ensemble::MatrixEnsemble) = Workspace(ensemble)

ensemble(workspace::AbstractWorkspace) = workspace.status.ensemble

function update_ensemble!(workspace::AbstractWorkspace, ensemble)
    return set_ensemble!(workspace.status, ensemble)
end

function _G₀⁻¹_diag!(workspace, z)
    ens = ensemble(workspace)
    @inbounds zA⁻B!(workspace.Q, z, ens.L⁻¹R⁻¹, ens.L⁻¹MR⁻¹)
    @inbounds copy_diag!(workspace.G₀⁻¹_diag, workspace.Q)
    return nothing
end

function _Q!(workspace, ::Fixr²{QNormal}, μq⁻) where {QNormal}
    @inbounds set_diag_muladd!(workspace.Q, μq⁻, ensemble(workspace).T, workspace.G₀⁻¹_diag)
    mul!(workspace.QQᴴ.data, workspace.Q, workspace.Q')
    if QNormal
        # This is an attempt to get around the order of limits issue by
        # replacing QQ' with (QQ' + Q'Q) / 2, cancelling any non-normality in
        # Q. This is not mathematically sound in general, but it does the trick
        # in certain experiments.
        half = one(eltype(workspace.Q)) / 2
        mul!(workspace.QQᴴ.data, workspace.Q', workspace.Q, half, half)
    end
    @inbounds copyto!(workspace.P.data, workspace.QQᴴ.data)
    workspace.s²min[] = eigmin!(workspace.P)
    return nothing
end

function _P!(workspace, r²)
    γ² = _γ²!(r², workspace.s²min[])
    @inbounds copyto!(workspace.P⁻¹f, workspace.QQᴴ.data)
    @inbounds add_to_diag!(workspace.P⁻¹f, γ²)
    LAPACK.potrf!(workspace.P.uplo, workspace.P⁻¹f)
    @inbounds copyto!(workspace.P.data, workspace.P⁻¹f)
    LAPACK.potri!(workspace.P.uplo, workspace.P.data)
    return nothing
end

_γ²!(r²::Fixγ², ::Any) = r².value
_γ²!(r²::Fixδ², s²min) = r².value - s²min

function _PQ!(workspace)
    @inbounds copyto!(workspace.PQ, workspace.Q)
    LAPACK.potrs!(workspace.P.uplo, workspace.P⁻¹f, workspace.PQ)
    return nothing
end

function _QᴴPQ!(workspace)
    mul!(workspace.QᴴPQ.data, workspace.Q', workspace.PQ)
    return nothing
end

function _QP!(workspace)
    if workspace.P.uplo === 'L'
        L = LowerTriangular(workspace.P⁻¹f)
        U = L'
    else
        U = UpperTriangular(workspace.P⁻¹f)
        L = U'
    end
    @inbounds copyto!(workspace.QP, workspace.Q)
    rdiv!(rdiv!(workspace.QP, U), L)
    return nothing
end

function _QPQ!(workspace)
    mul!(workspace.QPQ, workspace.Q, workspace.PQ)
    return nothing
end

function _PL⁻¹R⁻¹!(workspace)
    ens = ensemble(workspace)
    if ens.L⁻¹R⁻¹ isa UniformScaling
        mul!(workspace.QPQ, workspace.P, ens.L⁻¹R⁻¹)
    else
        @inbounds copyto!(workspace.QPQ, ens.L⁻¹R⁻¹)
        LAPACK.potrs!(workspace.P.uplo, workspace.P⁻¹f, workspace.QPQ)
    end
    return nothing
end

function G₀⁻¹_diag!(workspace, z)
    if workspace.status.z !== z
        _G₀⁻¹_diag!(workspace, z)
        set_z!(workspace.status, z)
    end
    return workspace.G₀⁻¹_diag
end

function Q!(workspace, objective, z, q)
    G₀⁻¹_diag!(workspace, z)
    μq⁻ = -objective.μ * q
    if workspace.status.μq⁻ !== μq⁻
        _Q!(workspace, objective, μq⁻)
        set_μq⁻!(workspace.status, μq⁻)
    end
    return workspace.Q
end

function P!(workspace, objective::O, z, q) where {O}
    Q!(workspace, objective, z, q)
    r² = O(objective.value, one(objective.μ))  # Don't include μ in this status, see Q!
    if workspace.status.r² !== r²
        _P!(workspace, r²)
        set_r²!(workspace.status, r²)
    end
    return workspace.P
end

function PQ!(workspace, objective, z, q)
    Q!(workspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.PQ
        _PQ!(workspace)
        workspace.status.PQ = true
    end
    return workspace.PQ
end

function QᴴPQ!(workspace, objective, z, q)
    Q!(workspace, objective, z, q)
    PQ!(workspace, objective, z, q)
    if !workspace.status.QᴴPQ
        _QᴴPQ!(workspace)
        workspace.status.QᴴPQ = true
    end
    return workspace.QᴴPQ
end

function QP!(workspace, objective, z, q)
    Q!(workspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.QP
        _QP!(workspace)
        workspace.status.QP = true
    end
    return workspace.QP
end

function QPQ!(workspace, objective, z, q)
    Q!(workspace, objective, z, q)
    PQ!(workspace, objective, z, q)
    if !workspace.status.QPQ
        workspace.status.PL⁻¹R⁻¹ = false
        _QPQ!(workspace)
        workspace.status.QPQ = true
    end
    return workspace.QPQ
end

function PL⁻¹R⁻¹!(workspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.PL⁻¹R⁻¹
        workspace.status.QPQ = false
        _PL⁻¹R⁻¹!(workspace)
        workspace.status.PL⁻¹R⁻¹ = true
    end
    return workspace.QPQ
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
    return _tr(P) / size(P, 1)
end

function r!(workspace, objective, z, q)
    Q = Q!(workspace, objective, z, q)
    PL⁻¹R⁻¹ = PL⁻¹R⁻¹!(workspace, objective, z, q)
    return _dot(Q, PL⁻¹R⁻¹)
end

function q∞!(workspace, γ∞², z)
    T = ensemble(workspace).T
    q = _dot(G₀⁻¹_diag!(workspace, z), T)
    return q / (γ∞² * length(T))
end
