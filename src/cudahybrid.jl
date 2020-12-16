struct CuHybridMatrixEnsemble{R, U, V, W, X} <: AbstractMatrixEnsemble{R}
    ensemble::CuMatrixEnsemble{R, U, V, W, X}
end

function CuHybridMatrixEnsemble(args...)
    return CuHybridMatrixEnsemble(CuMatrixEnsemble(args...))
end

function Base.getproperty(ensemble::CuHybridMatrixEnsemble, s::Symbol)
    return getproperty(getfield(ensemble, :ensemble), s)
end

struct CuHybridWorkspace{R, U, V, W, X} <: AbstractWorkspace{R}
    cuworkspace::CuWorkspace{R, U, V, W, X}
    hostT::Vector{R}
    hostG₀⁻¹_diag::Vector{Complex{R}}
    hostQ::Matrix{Complex{R}}
    hostP::Hermitian{Complex{R},Matrix{Complex{R}}}
    hostPQ::Matrix{Complex{R}}
    hostQᴴPQ::Hermitian{Complex{R},Matrix{Complex{R}}}
    hostQP::Matrix{Complex{R}}
    hostQPQ::Matrix{Complex{R}}  # Also used for PL⁻¹R⁻¹

    function CuHybridWorkspace(
        ensemble::CuHybridMatrixEnsemble{R, U, V, W, X}
    ) where {R, U, V, W, X}
        cuworkspace = CuWorkspace(getfield(ensemble, :ensemble))
        N = length(ensemble.T)
        hostT = Vector(ensemble.T)
        hostG₀⁻¹_diag = Vector{Complex{R}}(undef, N)
        hostQ = Matrix{Complex{R}}(undef, N, N)
        hostP = Hermitian(Matrix{Complex{R}}(undef, N, N))
        hostPQ = Matrix{Complex{R}}(undef, N, N)
        hostQᴴPQ = Hermitian(Matrix{Complex{R}}(undef, N, N))
        hostQP = Matrix{Complex{R}}(undef, N, N)
        hostQPQ = Matrix{Complex{R}}(undef, N, N)
        for A in (hostG₀⁻¹_diag, hostQ, hostP.data, hostPQ, hostQᴴPQ.data, hostQP, hostQPQ)
            Mem.pin(A)
        end
        return new{R, U, V, W, X}(
            cuworkspace,
            hostT,
            hostG₀⁻¹_diag,
            hostQ,
            hostP,
            hostPQ,
            hostQᴴPQ,
            hostQP,
            hostQPQ,
        )
    end
end

function Base.getproperty(workspace::CuHybridWorkspace, s::Symbol)
    if startswith(string(s), "host")
        return getfield(workspace, s)
    end
    return getproperty(getfield(workspace, :cuworkspace), s)
end

workspace(ensemble::CuHybridMatrixEnsemble) = CuHybridWorkspace(ensemble)

T(workspace::CuHybridWorkspace) = workspace.hostT

function G₀⁻¹_diag!(workspace::CuHybridWorkspace, z)
    if workspace.status.z !== z
        _G₀⁻¹_diag!(workspace, z)
        @inbounds copyto!(workspace.hostG₀⁻¹_diag, workspace.G₀⁻¹_diag)
        synchronize(CuDefaultStream())
    end
    return workspace.hostG₀⁻¹_diag
end

function Q!(workspace::CuHybridWorkspace, objective, z, q)
    G₀⁻¹_diag!(workspace, z)
    μq⁻ = -objective.μ * q
    if workspace.status.μq⁻ !== μq⁻
        _Q!(workspace, objective, μq⁻)
        @inbounds copyto!(workspace.hostQ, workspace.Q)
        synchronize(CuDefaultStream())
    end
    return workspace.hostQ
end

function P!(workspace::CuHybridWorkspace, objective, z, q)
    Q!(workspace, objective, z, q)
    r² = (typeof(objective), objective.value)
    if workspace.status.r² !== r²
        _P!(workspace, objective, r²)
        @inbounds copyto!(workspace.hostP.data, workspace.P.data)
        synchronize(CuDefaultStream())
    end
    return workspace.hostP
end

function PQ!(workspace::CuHybridWorkspace, objective, z, q)
    Q!(workspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.PQ
        _PQ!(workspace)
        @inbounds copyto!(workspace.hostPQ, workspace.PQ)
        synchronize(CuDefaultStream())
    end
    return workspace.hostPQ
end

function QᴴPQ!(workspace::CuHybridWorkspace, objective, z, q)
    Q!(workspace, objective, z, q)
    PQ!(workspace, objective, z, q)
    if !workspace.status.QᴴPQ
        _QᴴPQ!(workspace)
        @inbounds copyto!(workspace.hostQᴴPQ.data, workspace.QᴴPQ.data)
        synchronize(CuDefaultStream())
    end
    return workspace.hostQᴴPQ
end

function QP!(workspace::CuHybridWorkspace, objective, z, q)
    Q!(workspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.QP
        _QP!(workspace)
        @inbounds copyto!(workspace.hostQP, workspace.QP)
        synchronize(CuDefaultStream())
    end
    return workspace.hostQP
end

function QPQ!(workspace::CuHybridWorkspace, objective, z, q)
    Q!(workspace, objective, z, q)
    PQ!(workspace, objective, z, q)
    if !workspace.status.QPQ
        _QPQ!(workspace)
        @inbounds copyto!(workspace.hostQPQ, workspace.QPQ)
        synchronize(CuDefaultStream())
    end
    return workspace.hostQPQ
end

function PL⁻¹R⁻¹!(workspace::CuHybridWorkspace, objective, z, q)
    P!(workspace, objective, z, q)
    if !workspace.status.PL⁻¹R⁻¹
        _PL⁻¹R⁻¹!(workspace)
        @inbounds copyto!(workspace.hostQPQ, workspace.QPQ)
        synchronize(CuDefaultStream())
    end
    return workspace.hostQPQ
end
