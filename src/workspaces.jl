module Workspaces

using EnumX: @enumx
using LinearAlgebra: Diagonal, Factorization, Hermitian

using ..Defaults: Defaults
using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer, shift, gap
using ..MathHelpers: AnySArray, decompose, decompose!!, eigmin!!, hsdot_normalized, invert,
    invert!!, ldiv!!, muladd!!, muladd_diag!!, neg, psdmul, psdmul!!, tr_normalized

public Workspace, Q!, QQh!, P!, P_L_inv_R_inv!, shift!, gap!, q_rhs!, k!, E!

# Abstract supertype that can also be subtyped by GPU workspaces later
abstract type AbstractWorkspace{F<:AbstractFloat} end

# What are the computational dependencies between the auxiliary matrices?
#                                 ↙ P
# G0_inv <- Q <- QQh <- P_inv_fact
#                                 ↖ P_L_inv_R_inv
# So the possible states are
# - NONE
# - G0_inv
# - G0_inv & Q
# - G0_inv & Q & QQh
# - G0_inv & Q & QQh & P_inv_fact
# - G0_inv & Q & QQh & P_inv_fact & P
# - G0_inv & Q & QQh & P_inv_fact & P_L_inv_R_inv
# - ALL
#
# We use an enum to keep track of this so we only recalculate what is necessary. Note that
# since P and P_L_int_R_inv are independent, we can treat their subspace as a bitfield: When
# P_inv_fact is clean, the third bit is set and the first two bits indicate whether each of
# P and P_L_inv_R_inv are also clean. See comments within the enum definition.
@enumx Clean::UInt8 begin
    none = 0
    G0_inv = 1
    Q = 2
    QQh = 3
    P_inv_fact = 4     # 0b100
    P = 5              # 0b101
    P_L_inv_R_inv = 6  # 0b110
    all = 7            # 0b111
end

Base.:|(a::Clean.T, b::Clean.T) = Clean.T(UInt8(a) | UInt8(b))
Base.:&(a::Clean.T, b::Clean.T) = Clean.T(UInt8(a) & UInt8(b))

"""
    Workspace(ensemble::MatrixEnsemble)

Workspace holding intermediate arrays used to compute the average resolvent from an ensemble
"""
mutable struct Workspace{
    F,
    TE<:MatrixEnsemble{F},
    TG0_inv<:AbstractMatrix{Complex{F}},
    TQ<:AbstractMatrix{Complex{F}},
    TQQh<:Union{Diagonal{F},Hermitian{Complex{F}}},
    TP_inv_fact<:Union{Diagonal{F},Factorization{<:Complex}},  # factorizations can promote
    TP<:Union{Diagonal{F},Hermitian{<:Complex}},
    TP_L_inv_R_inv<:AbstractMatrix{<:Number},
} <: AbstractWorkspace{F}
    ensemble::TE
    z::Complex{F}
    q::Complex{F}
    reg::Regularizer{F}
    G0_inv::TG0_inv
    Q::TQ
    QQh::TQQh
    P_inv_fact::TP_inv_fact
    P::TP
    P_L_inv_R_inv::TP_L_inv_R_inv
    baregap::F
    clean::Clean.T
    normalizeQ::Bool
    # PQ::TQ
    # QP::TQ
    # PL_inv_R_inv_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}

    function Workspace(ensemble::E) where {F,E<:MatrixEnsemble{F}}
        z = zero(Complex{F})
        q = convert(Complex{F}, Defaults.q0)
        reg = Regularizer{F}(
            first(Defaults.regularizer_schedule_values), Defaults.regularizer_schedule_type
        )
        G0_inv = muladd(z, ensemble.L_inv_R_inv, neg(ensemble.L_inv_M_R_inv))
        Q = muladd(-q, ensemble.T, G0_inv)
        QQh = psdmul(Q)
        P_inv_fact = decompose(QQh)
        baregap = eigmin!!(P_inv_fact, QQh)
        P_inv_fact = decompose!!(P_inv_fact, QQh; shift=shift(reg, baregap))
        P = invert(P_inv_fact)
        P_L_inv_R_inv = P_inv_fact \ ensemble.L_inv_R_inv
        return new{F,E,typeof.((G0_inv, Q, QQh, P_inv_fact, P, P_L_inv_R_inv))...}(
            ensemble,
            z,
            q,
            reg,
            G0_inv,
            Q,
            QQh,
            P_inv_fact,
            P,
            P_L_inv_R_inv,
            baregap,
            Clean.all,
            false,
            # PQ,
            # QhPQ,
            # QPQ_PL_inv_R_inv,
            # QP,
        )
    end
end

## Auxiliary matrix getters
## ==============
# TODO: delete this if the need never arises
# As of now there are no direct consumers of G0_inv itself, not even downstream in this
# module. It's just a computational step. So for now we don't include it in the getter API.
# """
#     G0_inv!(ws, eq)
#
# Calculate the bare inverse resolvent `G0_inv(z) = L \\ (z * I - M) / R`
# """
# function G0_inv!(ws, eq)
#     set_G0_inv_params(ws, eq)
#     update_G0_inv(ws)
#     return ws.G0_inv
# end

function set_G0_inv_params(ws, eq)
    if ws.ensemble !== eq.ensemble
        if size(ws.ensemble) != size(eq.ensemble)
            throw(ArgumentError("new ensemble size does not match workspace"))
        end
        ws.ensemble = eq.ensemble
        ws.clean = Clean.none
    end
    if ws.z != eq.z
        ws.z = eq.z
        ws.clean = Clean.none
    end
    return nothing
end

function update_G0_inv(ws)
    (ws.clean >= Clean.G0_inv) && return nothing
    (; L_inv_R_inv, L_inv_M_R_inv) = ws.ensemble
    ws.G0_inv = muladd!!(ws.G0_inv, ws.z, L_inv_R_inv, neg(L_inv_M_R_inv))
    if !(ws.Q isa Diagonal)
        ws.Q = copy!!(ws.Q, ws.G0_inv)  # set off-diagonal elements of Q
    end
    ws.clean = Clean.G0_inv
    return nothing
end

"""
    Q!(ws, eq, q)

Calculate `Q(z, q) = G0_inv(z) - q * T`
"""
function Q!(ws, eq, q)
    set_Q_params(ws, eq, q)
    update_Q(ws)
    return ws.Q
end

function set_Q_params(ws, eq, q)
    set_G0_inv_params(ws, eq)
    (ws.q == q) && return nothing
    ws.q = q
    ws.clean = min(ws.clean, Clean.G0_inv)
    return nothing
end

function update_Q(ws)
    (ws.clean >= Clean.Q) && return nothing
    update_G0_inv(ws)
    ws.Q = muladd_diag!!(ws.Q, -ws.q, ws.ensemble.T, ws.G0_inv)
    ws.clean = Clean.Q
    return nothing
end

"""
    QQh!(ws, eq, q)

Calculate `Q(z, q) * Q'(z, q)`

If `eq.normalizeQ` is `true`, this actually calculates `(Q * Q' + Q' * Q) / 2`. See the
`ResolventProblem` documentation for further explanation.
"""
function QQh!(ws, eq, q)
    set_QQh_params(ws, eq, q)
    update_QQh(ws)
    return ws.QQh
end

function set_QQh_params(ws, eq, q)
    set_Q_params(ws, eq, q)
    (ws.normalizeQ == eq.normalizeQ) && return nothing
    ws.normalizeQ = eq.normalizeQ
    ws.clean = min(ws.clean, Clean.Q)
    return nothing
end

function update_QQh(ws)
    (ws.clean >= Clean.QQh) && return nothing
    update_Q(ws)
    ws.QQh = psdmul!!(ws.QQh, ws.Q)
    if (!(ws.Q isa Diagonal)) && ws.normalizeQ
        half = one(real(eltype(ws.QQh))) / 2
        ws.QQh = psdmul!!(ws.QQh, ws.Q', half, half)
    end
    ws.baregap = eigmin!!(ws.P_inv_fact, ws.QQh)
    ws.clean = Clean.QQh
    return nothing
end

# TODO: delete this if the need never arises
# As of now there are no direct consumers of P_inv_fact itself, not even downstream in this
# module. It's just a computational step. So for now we don't include it in the getter API.
# """
#     P_inv_fact!(ws, eq, q, reg)
# 
# Factorize `P⁻¹ = Q * Q' + γ² * I`
# """
# function P_inv_fact!(ws, eq, q, reg)
#     set_P_inv_fact_params(ws, eq, q, reg)
#     update_P_inv_fact(ws)
#     return ws.P_inv_fact
# end

function set_P_inv_fact_params(ws, eq, q, reg)
    set_QQh_params(ws, eq, q)
    (ws.reg == reg) && return nothing
    ws.reg = reg
    ws.clean = min(ws.clean, Clean.QQh)
    return nothing
end

function update_P_inv_fact(ws)
    (ws.clean >= Clean.P_inv_fact) && return nothing
    update_QQh(ws)
    ws.P_inv_fact = decompose!!(ws.P_inv_fact, ws.QQh; shift=shift(ws.reg, ws.baregap))
    ws.clean = Clean.P_inv_fact
    return nothing
end

"""
    P!(ws, eq, q, reg)

Calculate `P(z, q, γ²) = inv(Q * Q' + γ² * I)`

The value of `γ²` is provided through `reg::Regularizer`, which can hold either the value of
`γ²` itself or `δ² = γ² + eigmin(Q * Q')`.
"""
function P!(ws, eq, q, reg)
    set_P_params(ws, eq, q, reg)
    update_P(ws)
    return ws.P
end

set_P_params(ws, eq, q, reg) = set_P_inv_fact_params(ws, eq, q, reg)

function update_P(ws)
    # The states of P and P_L_inv_R_inv are a bitfield, see Clean comments
    ((ws.clean & Clean.P) >= Clean.P) && return nothing
    update_P_inv_fact(ws)
    ws.P = invert!!(ws.P, ws.P_inv_fact)
    ws.clean |= Clean.P
    return nothing
end

"""
    P_L_inv_R_inv!(ws, eq, q, reg)

Calculate `P / (RL) = (Q * Q' + γ² * I) * (RL)`

The arguments are the same as for `P!`.
"""
function P_L_inv_R_inv!(ws, eq, q, reg)
    set_P_L_inv_R_inv_params(ws, eq, q, reg)
    update_P_L_inv_R_inv(ws)
    return ws.P_L_inv_R_inv
end

set_P_L_inv_R_inv_params(ws, eq, q, reg) = set_P_inv_fact_params(ws, eq, q, reg)

function update_P_L_inv_R_inv(ws)
    # The states of P and P_L_inv_R_inv are a bitfield, see Clean comments
    ((ws.clean & Clean.P_L_inv_R_inv) >= Clean.P_L_inv_R_inv) && return nothing
    update_P_inv_fact(ws)
    ws.P_L_inv_R_inv = ldiv!!(ws.P_inv_fact, copy!!(ws.P_L_inv_R_inv, ws.ensemble.L_inv_R_inv))
    ws.clean |= Clean.P_L_inv_R_inv
    return nothing
end

## Scalar getters
## ==============
"""
    q_rhs!(ws, eq, q, reg)

Calculate the RHS of the self-consistency relation for ``q``: `tr(T * Q'P) / size(P, 2)`

The arguments are the same as for `P!`
"""
function q_rhs!(ws, eq, q, reg)
    return hsdot_normalized(eq.ensemble.T, Q!(ws, eq, q), P!(ws, eq, q, reg))
end

"""
    shift!(ws, eq, q, reg)

Calculate the regularizer shift ``γ²``

The arguments are the same as for `P!`
"""
function shift!(ws, eq, q, reg)
    baregap = let ws = ws, eq = eq, q = q
        # Use a closure to defer computation of baregap in case it's not needed for this reg
        baregap!(ws, eq, q)
    end
    return shift(reg, baregap)
end

"""
    gap!(ws, eq, q, reg)

Calculate the regularized gap `eigmin(QQ†) + γ²`

The arguments are the same as for `P!`
"""
function gap!(ws, eq, q, reg)
    baregap = let ws = ws, eq = eq, q = q
        # Use a closure to defer computation of baregap in case it's not needed for this reg
        baregap!(ws, eq, q)
    end
    return gap(reg, baregap)
end

function baregap!(ws, eq, q)
    QQh!(ws, eq, q)
    return ws.baregap
end

"""
    k!(ws, eq, q, reg)

Calculate the normalized trace of `P(z, q, γ²)`, i.e., `tr(P) / size(P, 2)`

The arguments are the same as for `P!`
"""
k!(ws, eq, q, reg) = tr_normalized(P!(ws, eq, q, reg))

"""
    E!(ws, eq, q, reg)

Calculate the normalized resolvent trace, i.e., trace of `(RL)⁻¹ * Q'(z, q) * P(z, q, γ²)`

The arguments are the same as for `P!`
"""
E!(ws, eq, q, reg) = hsdot_normalized(Q!(ws, eq, q), P_L_inv_R_inv!(ws, eq, q, reg))

## Array copy helper
## =================
"""
    copy!!(dest, src)

Overwrite `dest` with the contents of `src` and return `dest`
"""
copy!!(dest, src) = copyto!(dest, src)

"""
    copy!!(::AnySArray, src)

Return a copy of `src`
"""
copy!!(::AnySArray, src) = deepcopy(src)

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

end  # module Workspaces

@testitem "Workspace" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace, shift!, gap!, q_rhs!, k!, E!

    using LinearAlgebra: I, Diagonal, Hermitian, eigmin, tr
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25*cispi(1 // 3)]

    z64 = -(2.0 + im) / 4
    q64 = (1.0 - im) / 2

    shift64 = 0.25
    gap64 = 0.5

    # These are just used for comparison, so we don't fuss about eltypes and array types
    Dabs = Diagonal([0.75, 0.25])
    Dphase = Diagonal([1.0, cispi(1 // 3)])

    # Ground truth; only for comparison, so we don't fuss about eltypes
    R_actual = Dphase
    L_inv_R_inv = map(conj, Dphase)
    T_actual = Dabs
    G0_inv = (z64 * I - A64data) / R_actual
    Q = G0_inv - q64 * T_actual

    @testset "normalizeQ = $normalizeQ" for normalizeQ in [false, true]
        # Ground truth; only for comparison, so we don't fuss about eltypes
        QQh = normalizeQ ? ((Q * Q') + (Q' * Q)) / 2 : Q * Q'
        baregap = eigmin(Hermitian(QQh))
        Pshift, Pgap = inv(QQh + shift64 * I), inv(QQh + (gap64 - baregap) * I)
        q_rhs_shift, q_rhs_gap = tr(T_actual * Q' * Pshift) / 2, tr(T_actual * Q' * Pgap) / 2
        kshift, kgap = tr(Pshift) / 2, tr(Pgap) / 2
        Eshift, Egap = tr(L_inv_R_inv * Q' * Pshift) / 2, tr(L_inv_R_inv * Q' * Pgap) / 2

        @testset "$F" for F in [Float32, Float64]
            AFdata = map(Complex{F}, A64data)
            DFdata = map(Complex{F}, D64data)

            zF, qF = Complex{F}.((z64, q64))
            shiftF, gapF = F.((shift64, gap64))
            reg_shiftF, reg_gapF = Regularizer(shiftF, :shift), Regularizer(gapF, :gap)

            @testset "$label" for (label, AF, DF) in [
                ("Array", AFdata, Diagonal(DFdata)),
                ("SArray", SMatrix{2,2}(AFdata), Diagonal(SVector{2}(DFdata))),
                ("SparseMatrixCSC", sparse(AFdata), Diagonal(DFdata)),
            ]
                ensembleF = MatrixEnsemble(AF, I, I, DF)

                eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)

                wsF = Workspace(ensembleF)

                @test wsF isa Workspace{F}

                @test shift!(wsF, eqF, qF, reg_shiftF) ≈ reg_shiftF.value
                @test shift!(wsF, eqF, qF, reg_gapF) ≈ reg_gapF.value - baregap

                @test gap!(wsF, eqF, qF, reg_shiftF) ≈ reg_shiftF.value + baregap
                @test gap!(wsF, eqF, qF, reg_gapF) ≈ reg_gapF.value

                @test q_rhs!(wsF, eqF, qF, reg_shiftF) ≈ q_rhs_shift
                @test q_rhs!(wsF, eqF, qF, reg_gapF) ≈ q_rhs_gap

                @test k!(wsF, eqF, qF, reg_shiftF) ≈ kshift
                @test k!(wsF, eqF, qF, reg_gapF) ≈ kgap

                @test E!(wsF, eqF, qF, reg_shiftF) ≈ Eshift
                @test E!(wsF, eqF, qF, reg_gapF) ≈ Egap
            end
        end
    end
end

@testitem "Workspace errors" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation
    using GaussianSpectralDensities.Workspaces: Workspace, Q!

    using LinearAlgebra: I, Diagonal

    A = ComplexF64[1 -1; 1 1]
    D = Diagonal(ComplexF64[0.75, 0.25*cispi(1 // 3)])
    z = -(2.0 + im) / 4
    q = (1.0 - im) / 2

    ensemble = MatrixEnsemble(A, I, I, D)
    ws = Workspace(ensemble)

    other_ensemble = MatrixEnsemble(I, I, I, I(1) / 2)
    other_eq = LocalResolventEquation(other_ensemble, z, false)

    @test_throws ArgumentError Q!(ws, other_eq, q)
    @test_throws "size does not match" Q!(ws, other_eq, q)
end
