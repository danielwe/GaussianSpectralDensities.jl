struct CuHybridMatrixEnsemble{R, U, V, W} <: AbstractCuMatrixEnsemble{R, U, V, W}
    cuensemble::CuMatrixEnsemble{R, U, V, W}
    T::Vector{R}
end

function CuHybridMatrixEnsemble(args...)
    cuensemble = CuMatrixEnsemble(args...)
    return CuHybridMatrixEnsemble(cuensemble, Vector(cuensemble.T))
end

function Base.getproperty(ensemble::CuHybridMatrixEnsemble, s::Symbol)
    if !hasfield(typeof(ensemble), s)
        return getproperty(ensemble.cuensemble, s)
    end
    return getfield(ensemble, s)
end

struct CuHybridWorkspace{R, U, V, W} <: AbstractWorkspace{R}
    cuworkspace::CuWorkspace{R, U, V, W}
    status::WorkspaceStatus{R, CuHybridMatrixEnsemble{R, U, V, W}}
    # Could use unified memory instead of explicit host mirrors
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
        cuworkspace = CuWorkspace(ensemble.cuensemble)
        status = WorkspaceStatus(ensemble)
        N = length(ensemble.T)
        G₀⁻¹_diag = Mem.pin(Vector{Complex{R}}(undef, N))
        Q = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        P = Hermitian(Mem.pin(Matrix{Complex{R}}(undef, N, N)))
        PQ = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        QᴴPQ = Hermitian(Mem.pin(Matrix{Complex{R}}(undef, N, N)))
        QP = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        QPQ = Mem.pin(Matrix{Complex{R}}(undef, N, N))
        synchronize()
        return new{R, U, V, W}(cuworkspace, status, G₀⁻¹_diag, Q, P, PQ, QᴴPQ, QP, QPQ)
    end
end

function Base.getproperty(workspace::CuHybridWorkspace, s::Symbol)
    if !hasfield(typeof(workspace), s)
        return getproperty(workspace.cuworkspace, s)
    end
    return getfield(workspace, s)
end

workspace(ensemble::CuHybridMatrixEnsemble) = CuHybridWorkspace(ensemble)

function set_ensemble!(workspace::CuHybridWorkspace, ensemble)
    set_ensemble!(workspace.cuworkspace, ensemble.cuensemble)
    return set_ensemble!(workspace.status, ensemble)
end

# Wrapper methods: can't use varargs, these signatures must be strictly more specific
#
# One arg
for (func, field) in (
    (:_PQ!, :PQ), (:_QᴴPQ!, :QᴴPQ), (:_QP!, :QP), (:_QPQ!, :QPQ), (:_PL⁻¹R⁻¹!, :QPQ)
)
    @eval begin
        function $func(workspace::CuHybridWorkspace)
            $func(workspace.cuworkspace)
            @inbounds copyto!(workspace.$field, workspace.cuworkspace.$field)
            synchronize()
            return nothing
        end
    end
end

# Two args
for (func, field) in ((:_G₀⁻¹_diag!, :G₀⁻¹_diag), (:_P!, :P))
    @eval begin
        function $func(workspace::CuHybridWorkspace, arg2)
            $func(workspace.cuworkspace, arg2)
            @inbounds copyto!(workspace.$field, workspace.cuworkspace.$field)
            synchronize()
            return nothing
        end
    end
end

# Three args with type constrained second arg
function _Q!(workspace::CuHybridWorkspace, objective::Fixr²{QNormal}, μq⁻) where {QNormal}
    _Q!(workspace.cuworkspace, objective, μq⁻)
    @inbounds copyto!(workspace.Q, workspace.cuworkspace.Q)
    synchronize()
    return nothing
end
