struct CuHybridMatrixEnsemble{R, U, V, W} <: AbstractMatrixEnsemble{R, U, V, W}
    ensemble::CuMatrixEnsemble{R, U, V, W}
end

function CuHybridMatrixEnsemble(args...)
    return CuHybridMatrixEnsemble(CuMatrixEnsemble(args...))
end

function Base.getproperty(ensemble::CuHybridMatrixEnsemble, s::Symbol)
    return getproperty(getfield(ensemble, :ensemble), s)
end

struct CuHybridWorkspace{R, U, V, W} <: AbstractWorkspace{R}
    cuworkspace::CuWorkspace{R, U, V, W}
    T::Vector{R}
    G₀⁻¹_diag::Vector{Complex{R}}
    Q::Matrix{Complex{R}}
    P::Hermitian{Complex{R},Matrix{Complex{R}}}
    PQ::Matrix{Complex{R}}
    QᴴPQ::Hermitian{Complex{R},Matrix{Complex{R}}}
    QP::Matrix{Complex{R}}
    QPQ::Matrix{Complex{R}}  # Also used for PL⁻¹R⁻¹

    function CuHybridWorkspace(
        ensemble::CuHybridMatrixEnsemble{R, U, V, W}
    ) where {R, U, V, W}
        cuworkspace = CuWorkspace(getfield(ensemble, :ensemble))
        N = length(ensemble.T)
        T = Vector(ensemble.T)
        G₀⁻¹_diag = Mem.pin(Vector{Complex{R}}(undef, N))
        Q = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        P = Mem.pin(Hermitian(Matrix{Complex{R}}(undef, N, N)))
        PQ = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        QᴴPQ = Mem.pin(Hermitian(Matrix{Complex{R}}(undef, N, N)))
        QP = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        QPQ = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        return new{R, U, V, W}(cuworkspace, T, G₀⁻¹_diag, Q, P, PQ, QᴴPQ, QP, QPQ)
    end
end

function Base.getproperty(workspace::CuHybridWorkspace, s::Symbol)
    if s in (:status, :s²min)
        return getproperty(workspace.cuworkspace, s)
    end
    return getfield(workspace, s)
end

workspace(ensemble::CuHybridMatrixEnsemble) = CuHybridWorkspace(ensemble)

T(workspace::CuHybridWorkspace) = workspace.T

function _G₀⁻¹_diag!(workspace::CuHybridWorkspace, z)
    _G₀⁻¹_diag!(workspace.cuworkspace, z)
    @inbounds copyto!(workspace.G₀⁻¹_diag, workspace.cuworkspace.G₀⁻¹_diag)
    synchronize(CuDefaultStream())
    return nothing
end

function _Q!(workspace::CuHybridWorkspace, objective::Fixr²{QNormal}, μq⁻) where {QNormal}
    _Q!(workspace.cuworkspace, objective, μq⁻)
    @inbounds copyto!(workspace.Q, workspace.cuworkspace.Q)
    synchronize(CuDefaultStream())
    return nothing
end

function _P!(workspace::CuHybridWorkspace, r²)
    _P!(workspace.cuworkspace, r²)
    @inbounds copyto!(workspace.P, workspace.cuworkspace.P)
    synchronize(CuDefaultStream())
    return nothing
end

function _PQ!(workspace::CuHybridWorkspace)
    _PQ!(workspace.cuworkspace)
    @inbounds copyto!(workspace.PQ, workspace.cuworkspace.PQ)
    synchronize(CuDefaultStream())
    return nothing
end

function _QᴴPQ!(workspace::CuHybridWorkspace)
    _QᴴPQ!(workspace.cuworkspace)
    @inbounds copyto!(workspace.QᴴPQ, workspace.cuworkspace.QᴴPQ)
    synchronize(CuDefaultStream())
    return nothing
end

function _QP!(workspace::CuHybridWorkspace)
    _QP!(workspace.cuworkspace)
    @inbounds copyto!(workspace.QP, workspace.cuworkspace.QP)
    synchronize(CuDefaultStream())
    return nothing
end

function _QPQ!(workspace::CuHybridWorkspace)
    _QPQ!(workspace.cuworkspace)
    @inbounds copyto!(workspace.QPQ, workspace.cuworkspace.QPQ)
    synchronize(CuDefaultStream())
    return nothing
end

function _PL⁻¹R⁻¹!(workspace::CuHybridWorkspace)
    _PL⁻¹R⁻¹!(workspace.cuworkspace)
    @inbounds copyto!(workspace.QPQ, workspace.cuworkspace.QPQ)
    synchronize(CuDefaultStream())
    return nothing
end
