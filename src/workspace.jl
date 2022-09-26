abstract type AbstractWorkspace{F<:AbstractFloat} end

"""
    Workspace(ensemble::MatrixEnsemble)

Workspace holding intermediate arrays used to compute the average resolvent from an ensemble
"""
mutable struct Workspace{
    F,
    TE<:MatrixEnsemble{F},
    TG0i<:AbstractMatrix{Complex{F}},
    TQ<:AbstractMatrix{Complex{F}},
    TQQh<:Union{<:Diagonal{F},<:Hermitian{Complex{F}}},
    TPifact<:Union{<:Diagonal{F},<:Factorization{<:Complex}},  # factorizations can promote
    TP<:Union{<:Diagonal{F},<:Hermitian{<:Complex}},
    TPLiRi<:AbstractMatrix{<:Number},
} <: AbstractWorkspace{F}
    ensemble::TE
    z::Complex{F}
    μq::Complex{F}
    normalizeQ::Bool
    regularizer::Regularizer{F}
    G0i::TG0i
    Q::TQ
    QQh::TQQh
    Pifact::TPifact
    P::TP
    PLiRi::TPLiRi
    s²min::F
    state::UInt8
    # PQ::TQ
    # QP::TQ
    # PLiRi_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}

    function Workspace(ensemble::E) where {F,E<:MatrixEnsemble{F}}
        z = zero(Complex{F})
        μq = convert(Complex{F}, DEFAULTS.q0)
        regularizer = convert(Regularizer{F}, first(DEFAULTS.rschedule))
        G0i = muladd(z, ensemble.LiRi, _neg(ensemble.LiMRi))
        Q = muladd(-μq, ensemble.T, G0i)
        QQh = psdmul(Q)
        Pifact = decompose(QQh)
        s²min = eigmin!!(Pifact, QQh)
        Pifact = decompose!!(Pifact, QQh; shift=γ²(regularizer, s²min))
        P = invert(Pifact)
        PLiRi = Pifact \ ensemble.LiRi
        return new{F,E,typeof.((G0i,Q,QQh,Pifact,P,PLiRi))...}(
            ensemble,
            z,
            μq,
            false,
            regularizer,
            G0i,
            Q,
            QQh,
            Pifact,
            P,
            PLiRi,
            s²min,
            ALLVALID,
            # PQ,
            # QhPQ,
            # QPQ_PLiRi,
            # QP,
        )
    end
end

const G0iCURRENT = 0b00000001  # Also covers off-diagonal parts of Q
const QCURRENT = 0b00000010
const QQhCURRENT = 0b00000100  # Also covers s²min
const PifactCURRENT = 0b00001000
const PCURRENT = 0b00010000
const PLiRiCURRENT = 0b00100000

const G0iVALID = G0iCURRENT
const QVALID = QCURRENT | G0iVALID
const QQhVALID =  QQhCURRENT |QVALID
const PifactVALID =  PifactCURRENT |QQhVALID
const PVALID =  PCURRENT | PifactVALID
const PLiRiVALID =  PLiRiCURRENT | PifactVALID
const ALLVALID = PVALID | PLiRiVALID

_clearstate(ws) = (ws.state = 0b00000000)
_clearstateexcept(ws, flags) = (ws.state &= flags)
_addstate(ws, flags) = (ws.state |= flags)
_statecontains(ws, flags) = ((ws.state & flags) == flags)

## Matrix getters
## ==============
"""
    G0i!(ws, eq)

Calculate the bare inverse resolvent `G0i(z) = L \\ (z * I - M) / R`
"""
function G0i!(ws, eq)
    _setG0iparams!(ws, eq)
    _updateG0i!(ws)
    return ws.G0i
end

function _setG0iparams!(ws, eq)
    if ws.ensemble !== eq.ensemble
        if size(ws.ensemble) != size(eq.ensemble)
            throw(ArgumentError("new ensemble size does not match workspace"))
        end
        _clearstate(ws)
        ws.ensemble = eq.ensemble
    end
    if ws.z != eq.z
        _clearstate(ws)
        ws.z = eq.z
    end
    return nothing
end

function _updateG0i!(ws)
    _statecontains(ws, G0iVALID) && return nothing
    ws.G0i = _muladd!!(ws.G0i, ws.z, ws.ensemble.LiRi, _neg(ws.ensemble.LiMRi))
    if !(ws.Q isa Diagonal)
        ws.Q = _copy!!(ws.Q, ws.G0i)  # set off-diagonal elements of Q
    end
    _addstate(ws, G0iCURRENT)
    @assert _statecontains(ws, G0iVALID)
    return nothing
end

"""
    Q!(ws, eq, q)

Calculate `Q(z, q) = G0i(z) - q * T`
"""
function Q!(ws, eq, q)
    _setQparams!(ws, eq, q)
    _updateQ!(ws)
    return ws.Q
end

function _setQparams!(ws, eq, q)
    _setG0iparams!(ws, eq)
    μq = eq.μ * q
    (ws.μq == μq) && return nothing
    _clearstateexcept(ws, G0iVALID)
    ws.μq = μq
    return nothing
end

function _updateQ!(ws)
    _statecontains(ws, QVALID) && return nothing
    _updateG0i!(ws)
    ws.Q = _muladd_diag!!(ws.Q, -ws.μq, ws.ensemble.T, ws.G0i)
    _addstate(ws, QCURRENT)
    @assert _statecontains(ws, QVALID)
    return nothing
end

"""
    _QQh!(ws, eq, q)

Calculate `Q(z, q) * Q'(z, q)`

If `eq.normalizeQ` is `true`, this actually calculates `(Q * Q' + Q' * Q) / 2`. See the
`ResolventProblem` documentation for further explanation.
"""
function _QQh!(ws, eq, q)
    _setQQhparams!(ws, eq, q)
    _updateQQh!(ws)
    return ws.QQh
end

function _setQQhparams!(ws, eq, q)
    _setQparams!(ws, eq, q)
    (ws.normalizeQ == eq.normalizeQ) && return nothing
    _clearstateexcept(ws, QVALID)
    ws.normalizeQ = eq.normalizeQ
    return nothing
end

function _updateQQh!(ws)
    _statecontains(ws, QQhVALID) && return nothing
    _updateQ!(ws)
    ws.QQh = psdmul!!(ws.QQh, ws.Q)
    if (!(ws.Q isa Diagonal)) && ws.normalizeQ
        half = one(real(eltype(ws.QQh))) / 2
        ws.QQh = psdmul!!(ws.QQh, ws.Q', half, half)
    end
    ws.s²min = eigmin!!(ws.Pifact, ws.QQh)
    _addstate(ws, QQhCURRENT)
    @assert _statecontains(ws, QQhVALID)
    return nothing
end

"""
    _Pifact!(ws, eq, q, regularizer)

Factorize `P⁻¹ = Q * Q' + γ² * I`
"""
function _Pifact!(ws, eq, q, regularizer)
    _setPifactparams!(ws, eq, q, regularizer)
    _updatePifact!(ws)
    return ws.Pifact
end

function _setPifactparams!(ws, eq, q, regularizer)
    _setQQhparams!(ws, eq, q)
    (ws.regularizer == regularizer) && return nothing
    _clearstateexcept(ws, QQhVALID)
    ws.regularizer = regularizer
    return nothing
end

function _updatePifact!(ws)
    _statecontains(ws, PifactVALID) && return nothing
    _updateQQh!(ws)
    ws.Pifact = decompose!!(ws.Pifact, ws.QQh; shift=γ²(ws.regularizer, ws.s²min))
    _addstate(ws, PifactCURRENT)
    @assert _statecontains(ws, PifactVALID)
    return nothing
end

"""
    P!(ws, eq, q, regularizer)

Calculate `P(z, q, γ²) = inv(Q * Q' + γ² * I)`

The value of `γ²` is provided through `regularizer::Regularizer`, which can hold either the
value of `γ²` itself or `δ² = γ² + eigmin(Q * Q')`.
"""
function P!(ws, eq, q, regularizer)
    _setPparams!(ws, eq, q, regularizer)
    _updateP!(ws)
    return ws.P
end

_setPparams!(ws, eq, q, regularizer) = _setPifactparams!(ws, eq, q, regularizer)

function _updateP!(ws)
    _statecontains(ws, PVALID) && return nothing
    _updatePifact!(ws)
    ws.P = invert!!(ws.P, ws.Pifact)
    _addstate(ws, PCURRENT)
    @assert _statecontains(ws, PVALID)
    return nothing
end

"""
    _PLiRi!(ws, eq, q, regularizer)

Calculate `P / (RL) = (Q * Q' + γ² * I) * (RL)`

The arguments are the same as for `P!`.
"""
function _PLiRi!(ws, eq, q, regularizer)
    _setPLiRiparams!(ws, eq, q, regularizer)
    _updatePLiRi!(ws)
    return ws.PLiRi
end

_setPLiRiparams!(ws, eq, q, regularizer) = _setPifactparams!(ws, eq, q, regularizer)

function _updatePLiRi!(ws)
    _statecontains(ws, PLiRiVALID) && return nothing
    _updatePifact!(ws)
    ws.PLiRi = ldiv!!(ws.Pifact, _copy!!(ws.PLiRi, ws.ensemble.LiRi))
    _addstate(ws, PLiRiCURRENT)
    @assert _statecontains(ws, PLiRiVALID)
    return nothing
end

## Scalar getters
## ==============
"""
    qrhs!(ws, eq, q, regularizer)

Calculate the RHS of the self-consistency relation for ``q``: `μ * tr(T * Q'P) / size(P, 2)`

The arguments are the same as for `P!`
"""
function qrhs!(ws, eq, q, regularizer)
    return eq.μ * _hsn(eq.ensemble.T, Q!(ws, eq, q), P!(ws, eq, q, regularizer))
end

"""
    γ²!(ws, eq, q, regularizer)

Calculate the regularizer ``γ²``

The arguments are the same as for `P!`
"""
function γ²!(ws, eq, q, regularizer)
    _QQh!(ws, eq, q)
    return γ²(regularizer, ws.s²min)
end

"""
    δ²!(ws, eq, q, regularizer)

Calculate the regularizer ``δ²``

The arguments are the same as for `P!`
"""
function δ²!(ws, eq, q, regularizer)
    _QQh!(ws, eq, q)
    return δ²(regularizer, ws.s²min)
end

"""
    k!(ws, eq, q, regularizer)

Calculate the normalized trace of `P(z, q, γ²)`, i.e., `tr(P) / size(P, 2)`

The arguments are the same as for `P!`
"""
k!(ws, eq, q, regularizer) = _trn(P!(ws, eq, q, regularizer))

"""
    E!(ws, eq, q, regularizer)

Calculate the normalized resolvent trace, i.e., trace of `(RL)⁻¹ * Q'(z, q) * P(z, q, γ²)`

The arguments are the same as for `P!`
"""
E!(ws, eq, q, regularizer) = _hsn(Q!(ws, eq, q), _PLiRi!(ws, eq, q, regularizer))

## Array arithmetic helpers
## ========================
"""
    _copy!!(dest, src)

Overwrite `dest` with the contents of `src` and return `dest`
"""
_copy!!(dest, src) = copyto!(dest, src)

"""
    _copy!!(::AnySArray, src)

Return a copy of `src` 
"""
_copy!!(::AnySArray, src) = deepcopy(src)

"""
    _diagview(A)

Return a view of the diagonal of the matrix ``A``
"""
_diagview(A) = view(A, diagind(A))
_diagview(A::Diagonal) = A.diag
_diagview(A::Union{<:Bidiagonal,<:SymTridiagonal}) = A.dv
_diagview(A::Tridiagonal) = A.d
_diagview(A::UniformScaling) = A.λ

"""
   _neg(A)

Return a non-allocating representation of `-A`

Uses `-` for static arrays and `MappedArrays` otherwise, and preserves structured matrix
types.
"""
_neg(A) = mappedarray(-, A)
_neg(A::SArray) = -A
_neg(A::Diagonal) = Diagonal(_neg(A.diag))
_neg(A::Bidiagonal) = Bidiagonal(_neg(A.dv), _neg(A.ev), Symbol(A.uplo))
_neg(A::SymTridiagonal) = SymTridiagonal(_neg(A.dv), _neg(A.ev))
_neg(A::Tridiagonal) = Tridiagonal(_neg(A.dl), _neg(A.d), _neg(A.du))

"""
    _muladd!!(C, λ::Number, A, B)

Overwrite ``C`` with ``λA + B` and return ``C``
"""
function _muladd!!(C, λ::Number, A, B)
    C .= muladd.(λ, A, B)
    return C
end

"""
    _muladd!!(::AnySArray, λ, A, B)

Return ``λA + B``
"""
_muladd!!(::AnySArray, λ::Number, A, B) = muladd(λ, A, B)

"""
    _muladd_diag!!(C, λ::Number, A, B)

Overwrite the diagonal of ``C`` with `λ * diag(A) + diag(B)`, and return ``C``
"""
function _muladd_diag!!(C, λ::Number, A, B)
    _diagview(C) .= muladd.(λ, _diagview(A), _diagview(B))
    return C
end

"""
    _muladd_diag!!(C::AnySArray, λ::Number, A::Union{Diagonal,UniformScaling}, B)

Return ``λA + B``

This is intended as an out-of-place equivalent of the in-place ``_muladd_diag!!``
implementation. This only makes sense when ``C`` and ``B`` only differ on the diagonal,
which is the case of interest in this package.
"""
_muladd_diag!!(::AnySArray, λ::Number, A::DiagonalOrUniform, B) = muladd(λ, A, B)

## Traces and trace inner products
## ===============================
"""
    _trn(A)

Compute the normalized trace `tr(A) / size(A, 2)`
"""
_trn(A) = tr(A) / size(A, 2)

"""
    _hsn(A, B)

Compute the normalized Hilbert-Schmidt inner product `tr(A'B) / size(B, 2)`
"""
_hsn(A, B) = dot(A, B) / size(B, 2)

"""
    _hsn(T::Diagonal, A, B)
    _hsn(T::UniformScaling, A, B)

Compute the normalized trace `tr(T * A'B) / size(B, 2)`

For positive definite `T` this can be seen as a generalized Hilbert-Schmidt inner product
"""
function _hsn(T::Diagonal, A, B)
    @boundscheck begin
        if (axes(A) != axes(B)) || (axes(T.diag, 1) != axes(A, 2))
            throw(ArgumentError("matrix axes don't match"))
        end
    end
    out = zero(promote_type(eltype(T), eltype(A), eltype(B)))
    @inbounds for j in axes(A, 2)
        out += T.diag[j] * dot(view(A, :, j), view(B, :, j))
    end
    return out / length(T.diag)
end

_hsn(T::UniformScaling, A, B) = T.λ * _hsn(A, B)

## Attic---may be scavenged for further functionality
## ==================================================
#=
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

function q∞!(problem, γ∞², ws)
    T = problem.ensemble.T
    q = _dot(G₀⁻¹_diag!(problem, ws), T)
    return q / (γ∞² * length(T))
end
=#
