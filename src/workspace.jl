abstract type AbstractWorkspace{F<:AbstractFloat} end

"""

    Workspace(ensemble::MatrixEnsemble)

Workspace holding intermediate arrays used to compute the average resolvent from an ensemble
"""
mutable struct Workspace{
    F,
    TE<:MatrixEnsemble{F},
    TG0i_diag<:Union{Complex{F},AbstractVector{Complex{F}}},
    TQ<:AbstractMatrixOrUniform{Complex{F}},
    TQQh<:Union{<:DiagonalOrUniform{F},<:Hermitian{Complex{F}}},
    TPifact<:Union{<:DiagonalOrUniform{F},<:Factorization{Complex{F}}},
    TP<:Union{<:DiagonalOrUniform{F},<:Hermitian{Complex{F}}},
} <: AbstractWorkspace{F}
    ensemble::TE
    z::Complex{F}
    q::Complex{F}  # Depends on G0i_diag
    regularizer::Regularizer{F}  # Depends on Q
    G0i_diag::TG0i_diag  # Depends on z, used when updating Q with new q
    Q::TQ
    QQh::TQQh
    Pifact::TPifact
    P::TP
    s²min::F  # Depends on Q (smallest eigenvalue of QQh)
    G0i_diag_current::Bool
    Q_current::Bool  # Depends on G0i_diag
    P_current::Bool  # Depends on Q
    # PQ::TQ
    # QP::TQ
    # PLiRi_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}
    # PQ_current::Bool  # Depends on P
    # QP_current::Bool  # Depends on P
    # PLiRi_current::Bool  # Depends on P, voids QPQ
    # QPQ_current::Bool  # Depends on PQ, voids PLiRi
    # QhPQ_current::Bool  # Depends on PQ

    function Workspace(ensemble::E) where {F,E<:MatrixEnsemble{F}}
        z = zero(Complex{F})
        q = zero(Complex{F})
        regularizer = first(DEFAULTS.rschedule)
        G0i = muladd(z, ensemble.LiRi, -ensemble.LiMRi)
        Q = muladd(-q, _diagonaloruniform(ensemble.T), G0i)
        G0i_diag = _diag(Q)  # Matrix(Q) == Matrix(G0i) since q == 0 during construction
        QQh = (Q isa DiagonalOrUniform) ? real(Q * Q') : Hermitian(Q * Q')
        Pifact = _factorize(QQh)
        P = _inv(Pifact)
        s²min = _eigmin(QQh)
        return new{F,E,typeof(G0i_diag),typeof(Q),typeof(QQh),typeof(Pifact),typeof(P)}(
            ensemble,
            z,
            q,
            regularizer,
            G0i_diag,
            Q,
            QQh,
            Pifact,
            P,
            s²min,
            true,
            true,
            true,
            # PQ,
            # QhPQ,
            # QPQ_PLiRi,
            # QP,
            # true,
            # true,
            # true,
            # true,
            # false,
        )
    end
end

#=
mutable struct WorkspaceStatus{R,E<:AbstractMatrixEnsemble{R}}
    N::Int
    ensemble::E
    z::Union{Missing,Complex{R}}  # Also G₀⁻¹_diag::Bool
    nμq::Union{Missing,Complex{R}}  # Also Q::Bool depends G₀⁻¹_diag
    r²::Union{Missing,Fixr²{R}}  # Also P::Bool depends Q
    PQ::Bool  # depends P
    QᴴPQ::Bool  # depends PQ
    QPQ::Bool  # depends PQ, voids PL⁻¹R⁻¹
    QP::Bool  # depends P
    PL⁻¹R⁻¹::Bool  # depends P, voids QPQ

    function WorkspaceStatus(ensemble::AbstractMatrixEnsemble{R}) where {R}
        return new{R, typeof(ensemble)}(
            ensemble.N, ensemble, missing, missing, missing, false, false, false, false, false
        )
    end
end

function set_ensemble!(status::WorkspaceStatus{R, E}, ensemble::E) where {R, E}
    if !(ensemble.N == status.N)
        throw(ArgumentError("ensemble and workspace dimension do not match"))
    end
    set_z!(status, missing)
    status.ensemble = ensemble
    return nothing
end

function set_z!(status::WorkspaceStatus, z)
    set_nμq!(status, missing)
    status.z = z
    return nothing
end

function set_nμq!(status::WorkspaceStatus, nμq)
    set_r²!(status, missing)
    status.nμq = nμq
    return nothing
end

function set_r²!(status::WorkspaceStatus, r²)
    set_PL⁻¹R⁻¹!(status, false)
    set_QP!(status, false)
    set_PQ!(status, false)
    status.r² = r²
    return nothing
end

function set_PQ!(status::WorkspaceStatus, PQ)
    set_QPQ!(status, false)
    set_QᴴPQ!(status, false)
    status.PQ = PQ
    return nothing
end

function set_QᴴPQ!(status::WorkspaceStatus, QᴴPQ)
    status.QᴴPQ = QᴴPQ
    return nothing
end

function set_QPQ!(status::WorkspaceStatus, QPQ)
    if QPQ
        set_PL⁻¹R⁻¹!(status, false)
    end
    status.QPQ = QPQ
    return nothing
end

function set_QP!(status::WorkspaceStatus, QᴴPQ)
    status.QP = QP
    return nothing
end

function set_PL⁻¹R⁻¹!(status::WorkspaceStatus, PL⁻¹R⁻¹)
    PL⁻¹R⁻¹ && set_QPQ!(status, false)
    status.PL⁻¹R⁻¹ = PL⁻¹R⁻¹
    return nothing
end
=#

#=
workspace(ensemble::MatrixEnsemble) = Workspace(ensemble)

function _G₀⁻¹_diag!(problem, ws)
    ens = problem.ensemble
    @inbounds zA⁻B!(ws.Q, problem.z, ens.L⁻¹R⁻¹, ens.L⁻¹MR⁻¹)
    @inbounds copy_diag!(ws.G₀⁻¹_diag, ws.Q)
    return nothing
end

function _Q!(problem::LocalSpectralProblem{NormalizeQ}, nμq, ws) where {NormalizeQ}
    @inbounds muladd_diag!(ws.Q, nμq, problem.ensemble.T, ws.G₀⁻¹_diag)
    mul!(ws.QQᴴ.data, ws.Q, ws.Q')
    if NormalizeQ
        # This is an attempt to get around the order of limits issue by
        # replacing QQ' with (QQ' + Q'Q) / 2, cancelling any non-normality in
        # Q. This is not mathematically sound in general, but it does the trick
        # in certain experiments.
        half = one(eltype(ws.Q)) / 2
        mul!(ws.QQᴴ.data, ws.Q', ws.Q, half, half)
    end
    @inbounds copyto!(ws.P.data, ws.QQᴴ.data)
    ws.s²min[] = eigmin!(ws.P)
    return nothing
end

function _P!(problem, ws)
    copyto!(ws.Pichol, ws.QQᴴ)
    add_to_diag!(ws.Pichol, _γ²!(problem.objective, ws))
    _cholinv!(ws.P, ws.Pichol)
    return nothing
end
=#

#=
function _PQ!(ws)
    @inbounds copyto!(ws.PQ, ws.Q)
    LAPACK.potrs!(ws.P.uplo, ws.P⁻¹f, ws.PQ)
    return nothing
end

function _QᴴPQ!(ws)
    mul!(ws.QᴴPQ.data, ws.Q', ws.PQ)
    return nothing
end

function _QPQ!(ws)
    mul!(ws.QPQ_PL⁻¹R⁻¹, ws.Q, ws.PQ)
    return nothing
end

function _QP!(ws)
    if ws.P.uplo == 'L'
        L = LowerTriangular(ws.P⁻¹f)
        U = L'
    else
        U = UpperTriangular(ws.P⁻¹f)
        L = U'
    end
    @inbounds copyto!(ws.QP, ws.Q)
    rdiv!(ws.QP, U)
    rdiv!(ws.QP, L)
    return nothing
end

function _PL⁻¹R⁻¹!(problem, ws)
    ens = problem.ensemble
    if ens.L⁻¹R⁻¹ isa UniformScaling
        mul!(ws.QPQ_PL⁻¹R⁻¹, ws.P, ens.L⁻¹R⁻¹)
    else
        @inbounds copyto!(ws.QPQ_PL⁻¹R⁻¹, ens.L⁻¹R⁻¹)
        LAPACK.potrs!(ws.P.uplo, ws.P⁻¹f, ws.QPQ_PL⁻¹R⁻¹)
    end
    return nothing
end

function G₀⁻¹_diag!(problem, ws)
    if ws.status.ensemble !== problem.ensemble
        set_ensemble!(ws.status, problem.ensemble)
    end
    if ws.status.z !== problem.z
        _G₀⁻¹_diag!(problem.z, ws)
        set_z!(ws.status, problem.z)
    end
    return ws.G₀⁻¹_diag
end

function Q!(problem, q, ws)
    G₀⁻¹_diag!(problem, ws)
    nμq = -problem.μ * q
    if ws.status.nμq !== nμq
        _Q!(problem, nμq, ws)
        set_nμq!(ws.status, nμq)
    end
    return ws.Q
end

function P!(problem, q, ws)
    Q!(problem, q, ws)
    if ws.status.r² !== problem.objective
        _P!(problem, ws)
        set_r²!(ws.status, problem.objective)
    end
    return ws.P
end

function PQ!(problem, q, ws)
    P!(problem, q, ws)
    if !ws.status.PQ
        _PQ!(ws)
        set_PQ!(ws.status, true)
    end
    return ws.PQ
end

function QᴴPQ!(problem, q, ws)
    PQ!(problem, q, ws)
    if !ws.status.QᴴPQ
        _QᴴPQ!(ws)
        set_QᴴPQ!(ws.status, true)
    end
    return ws.QᴴPQ
end

function QPQ!(problem, q, ws)
    PQ!(problem, q, ws)
    if !ws.status.QPQ
        _QPQ!(ws)
        set_QPQ!(ws.status, true)
    end
    return ws.QPQ_PL⁻¹R⁻¹
end

function QP!(problem, q, ws)
    P!(problem, q, ws)
    if !ws.status.QP
        _QP!(ws)
        set_QP!(ws.status, true)
    end
    return ws.QP
end

function PL⁻¹R⁻¹!(problem, q, ws)
    P!(problem, q, ws)
    if !ws.status.PL⁻¹R⁻¹
        _PL⁻¹R⁻¹!(ws)
        set_PL⁻¹R⁻¹!(ws.status, true)
    end
    return ws.QPQ_PL⁻¹R⁻¹
end
=#

#=
_γ²!(r²::Fixγ², ::Any) = r².value
_γ²!(r²::Fixδ², ws) = r².value - ws.s²min[]
_δ²!(r²::Fixγ², ws) = r².value + ws.s²min[]
_δ²!(r²::Fixδ², ::Any) = r².value

function γ²!(problem, q, ws)
    r² = problem.objective
    if !(r² isa Fixγ²)
        Q!(problem, q, ws)
    end
    return _γ²(r², ws)
end

function δ²!(problem, q, ws)
    r² = problem.objective
    if !(r² isa Fixδ²)
        Q!(problem, q, ws)
    end
    return _δ²(r², ws)
end

function k!(problem, q, ws)
    P = P!(problem, q, ws)
    return _tr(P) / size(P, 1)
end

# TODO: Ahmadian calls this E(z) so why not use that? r is a bad name for this variable and
# conflicts with r² as a common name for γ² and δ²
function r!(problem, q, ws)
    return _dot(Q!(problem, q, ws), PL⁻¹R⁻¹!(problem, q, ws))
end

function q∞!(problem, γ∞², ws)
    T = problem.ensemble.T
    q = _dot(G₀⁻¹_diag!(problem, ws), T)
    return q / (γ∞² * length(T))
end
=#
