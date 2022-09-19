abstract type AbstractWorkspace{F<:AbstractFloat} end

"""
    Workspace(ensemble::MatrixEnsemble)

Workspace holding intermediate arrays used to compute the average resolvent from an ensemble
"""
mutable struct Workspace{
    F,
    TE<:MatrixEnsemble{F},
    TG0i<:AbstractMatrixOrUniform{Complex{F}},
    TQ<:AbstractMatrixOrUniform{Complex{F}},
    TQQh<:Union{<:DiagonalOrUniform{F},<:Hermitian{Complex{F}}},
    TPifact<:Union{<:DiagonalOrUniform{F},<:Factorization{<:Complex}},  # factorizations can promote
    TP<:Union{<:DiagonalOrUniform{F},<:Hermitian{<:Complex}},
} <: AbstractWorkspace{F}
    ensemble::TE
    z::Complex{F}
    q::Complex{F}
    normalizeQ::Bool
    regularizer::Regularizer{F}
    G0i::TG0i
    Q::TQ
    QQh::TQQh
    Pifact::TPifact
    P::TP
    s²min::F
    G0icurrent::Bool  # also covers off-diagonal elements of Q
    Qcurrent::Bool
    QQhcurrent::Bool  # also covers s²min
    Pifactcurrent::Bool
    Pcurrent::Bool
    # PQ::TQ
    # QP::TQ
    # PLiRi_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}
    # PQcurrent::Bool  # Depends on P
    # QPcurrent::Bool  # Depends on P
    # PLiRicurrent::Bool  # Depends on P, voids QPQ
    # QPQcurrent::Bool  # Depends on PQ, voids PLiRi
    # QhPQcurrent::Bool  # Depends on PQ

    function Workspace(ensemble::E) where {F,E<:MatrixEnsemble{F}}
        z = zero(Complex{F})
        q = convert(Complex{F}, DEFAULTS.q0)
        regularizer = convert(Regularizer{F}, first(DEFAULTS.rschedule))
        G0i = muladd(z, ensemble.LiRi, -ensemble.LiMRi)
        Q = muladd(-q, ensemble.T, G0i)
        QQh = (Q isa DiagonalOrUniform) ? real(Q * Q') : Hermitian(Q * Q')
        Pifact = decompose(QQh)
        s²min = eigmin!!(Pifact, QQh)
        Pifact = decompose!!(Pifact, QQh; shift=γ²(regularizer, s²min))
        P = invert(Pifact)
        return new{F,E,typeof.((G0i,Q,QQh,Pifact,P))...}(
            ensemble,
            z,
            q,
            false,
            regularizer,
            G0i,
            Q,
            QQh,
            Pifact,
            P,
            s²min,
            true,
            true,
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
            # false,
        )
    end
end

## Matrix getters
## ==============
"""
    G0i!(ws, ensemble, z)

Calculate the bare inverse resolvent `G0i(z) = L \\ (z * I - M) / R`
"""
function G0i!(ws, ensemble, z)
    _set_ensemble!(ws, ensemble)
    _set_z!(ws, z)
    _refresh_G0i!(ws)
    return ws.G0i
end

"""
    Q!(ws, ensemble, z, q)

Calculate `Q(z, q) = G0i(z) - q * T`
"""
function Q!(ws, ensemble, z, q)
    G0i!(ws, ensemble, z)
    _set_q!(ws, q)
    _refresh_Q!(ws)
    return ws.Q
end

"""
    QQh!(ws, ensemble, z, q, normalizeQ=false)

Calculate `Q(z, q) * Q'(z, q)`

If `normalizeQ` is `true`, this actually calculates `(Q * Q' + Q' * Q) / 2`. See the
`ResolventProblem` documentation for further explanation.
"""
function QQh!(ws, ensemble, z, q, normalizeQ=false)
    Q!(ws, ensemble, z, q)
    _set_normalizeQ!(ws, normalizeQ)
    _refresh_QQh!(ws)
    return ws.QQh
end

"""
    _Pifact!(ws, ensemble, z, q, regularizer, normalizeQ=false)

Factorize `P⁻¹ = Q * Q' + γ² * I`
"""
function _Pifact!(ws, ensemble, z, q, regularizer, normalizeQ=false)
    QQh!(ws, ensemble, z, q, normalizeQ)
    _set_regularizer!(ws, regularizer)
    _refresh_Pifact!(ws)
    return ws.Pifact
end

"""
    P!(ws, ensemble, z, q, regularizer, normalizeQ=false)

Calculate `P(z, q, γ²) = inv(Q * Q' + γ² * I)`

The value of `γ²` is provided through `regularizer::Regularizer`, which can hold either the
value of `γ²` itself or `δ² = γ² + eigmin(Q * Q')`.
"""
function P!(ws, ensemble, z, q, regularizer, normalizeQ=false)
    _Pifact!(ws, ensemble, z, q, regularizer, normalizeQ)
    _refresh_P!(ws)
    return ws.P
end

## Parameter setters
## =================
function _set_ensemble!(ws::Workspace{<:Any,E}, ensemble::E) where {E}
    (ws.ensemble == ensemble) && return nothing
    if size(ws.ensemble) != size(ensemble)
        throw(ArgumentError("new ensemble size does not match workspace"))
    end
    ws.G0icurrent = false
    ws.Qcurrent = false
    ws.QQhcurrent = false
    ws.Pifactcurrent = false
    ws.Pcurrent = false
    ws.ensemble = ensemble
    return nothing
end

function _set_z!(ws::Workspace, z)
    (ws.z == z) && return nothing
    ws.G0icurrent = false
    ws.Qcurrent = false
    ws.QQhcurrent = false
    ws.Pifactcurrent = false
    ws.Pcurrent = false
    ws.z = z
    return nothing
end

function _set_q!(ws::Workspace, q)
    (ws.q == q) && return nothing
    ws.Qcurrent = false
    ws.QQhcurrent = false
    ws.Pifactcurrent = false
    ws.Pcurrent = false
    ws.q = q
    return nothing
end

function _set_normalizeQ!(ws::Workspace, normalizeQ)
    (ws.normalizeQ == normalizeQ) && return nothing
    ws.QQhcurrent = false
    ws.Pifactcurrent = false
    ws.Pcurrent = false
    ws.normalizeQ = normalizeQ
    return nothing
end

function _set_regularizer!(ws::Workspace, regularizer)
    (ws.regularizer == regularizer) && return nothing
    ws.Pifactcurrent = false
    ws.Pcurrent = false
    ws.regularizer = regularizer
    return nothing
end

## Matrix refreshers
## =================
function _refresh_G0i!(ws)
    ws.G0icurrent && return nothing
    ws.G0i = _muladd!!(ws.G0i, ws.z, ws.ensemble.LiRi, _mapped(-, ws.ensemble.LiMRi))
    if !(ws.Q isa DiagonalOrUniform)
        ws.Q = _copy!!(ws.Q, ws.G0i)  # set off-diagonal elements of Q
    end
    ws.G0icurrent = true
    return nothing
end

function _refresh_Q!(ws)
    ws.Qcurrent && return nothing
    ws.Q = _muladd_diag!!(ws.Q, -ws.q, ws.ensemble.T, ws.G0i)
    ws.Qcurrent = true
    return nothing
end

function _refresh_QQh!(ws)
    ws.QQhcurrent && return nothing
    ws.QQh = psdmul!!(ws.QQh, ws.Q)
    if !(ws.Q isa DiagonalOrUniform) && ws.normalizeQ
        half = one(real(eltype(ws.QQh))) / 2
        ws.QQh = psdmul!!(ws.QQh, ws.Q', half, half)
    end
    ws.s²min = eigmin!!(ws.Pifact, ws.QQh)
    ws.QQhcurrent = true
    return nothing
end

function _refresh_Pifact!(ws)
    ws.Pifactcurrent && return nothing
    ws.Pifact = decompose!!(ws.Pifact, ws.QQh; shift=γ²(ws.regularizer, ws.s²min))
    ws.Pifactcurrent = true
    return nothing
end

function _refresh_P!(ws)
    ws.Pcurrent && return nothing
    ws.P = invert!!(ws.P, ws.Pifact)
    ws.Pcurrent = true
    return nothing
end

## Array arithmetic helpers
## ========================
"""
    _copy!!(dest, src)

Overwrite `dest` with the contents of `src` and return `dest`
"""
_copy!!(dest, src) = copyto!(dest, src)

"""
    _copy!!(::Immutable, src)

Return a copy of `src` 
"""
_copy!!(::Immutable, src) = deepcopy(src)

"""
    _mapped(f, A)
    _mapped(f, A::UniformScaling)

Wrapper for `mappedarray(f, A)` that also works when `A isa UniformScaling`
"""
_mapped(f, A) = mappedarray(f, A)
_mapped(f, A::UniformScaling) = UniformScaling(f(A.λ))

"""
    _diagview(A)

Return a view of the diagonal of the matrix ``A``
"""
_diagview(A) = view(A, diagind(A))
_diagview(A::Diagonal) = A.diag
_diagview(A::Union{<:Bidiagonal,<:SymTridiagonal}) = A.dv
_diagview(A::Tridiagonal) = A.d

"""
    _diagview(A::UniformScaling)

Return the scalar value of the uniform scaling matrix ``A``
"""
_diagview(A::UniformScaling) = A.λ

"""
    _muladd!!(C, λ::Number, A, B)

Overwrite ``C`` with ``λA + B` and return ``C``
"""
function _muladd!!(C, λ::Number, A, B)
    C .= muladd.(λ, A, B)
    return C
end

function _muladd!!(C, λ::Number, A::UniformScaling, B)
    copyto!(C, B)
    _muladd_diag!!(C, λ, A, B)  # muladd for accuracy rather than _diagview(C) .+= λ * A.λ
    return C
end

function _muladd!!(C, λ::Number, A, B::UniformScaling)
    C .= λ .* A
    _muladd_diag!!(C, λ, A, B)  # muladd for accuracy rather than _diagview(C) .+= B.λ
    return C
end

function _muladd!!(C, λ::Number, A::UniformScaling, B::UniformScaling)
    fill!(C, zero(eltype(C)))
    fill!(_diagview(C), muladd(λ, A.λ, B.λ))
    return C
end

"""
    _muladd!!(::Immutable, λ, A, B)

Return ``λA + B``
"""
_muladd!!(::Immutable, λ::Number, A, B) = muladd(λ, A, B)
_muladd!!(::Immutable, λ::Number, A::UniformScaling, B) = muladd(λ, A, B)
_muladd!!(::Immutable, λ::Number, A, B::UniformScaling) = muladd(λ, A, B)
_muladd!!(::Immutable, λ::Number, A::UniformScaling, B::UniformScaling) = muladd(λ, A, B)

"""
    _muladd_diag!!(C, λ::Number, A, B)

Overwrite the diagonal of ``C`` with `λ * diag(A) + diag(B)`, and return ``C``
"""
function _muladd_diag!!(C, λ::Number, A, B)
    _diagview(C) .= muladd.(λ, _diagview(A), _diagview(B))
    return C
end

"""
    _muladd_diag!!(C::Immutable, λ::Number, A::DiagonalOrUniform, B)

Return ``λA + B``

This is intended as an out-of-place equivalent of the in-place ``_muladd_diag!!``
implementation. This only makes sense when ``C`` and ``B`` only differ on the diagonal,
which is the case of interest in this package.
"""
_muladd_diag!!(::Immutable, λ::Number, A::DiagonalOrUniform, B) = muladd(λ, A, B)

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

function set_PL⁻¹R⁻¹!(status::WorkspaceStatus, PL⁻¹R⁻¹)
    PL⁻¹R⁻¹ && set_QPQ!(status, false)
    status.PL⁻¹R⁻¹ = PL⁻¹R⁻¹
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
