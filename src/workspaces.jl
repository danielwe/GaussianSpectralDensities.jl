module Workspaces

using EnumX: @enumx
using LinearAlgebra: Diagonal, Factorization, Hermitian

using ..Defaults: Defaults
using ..Problems: ResolventEquation
using ..Regularizers: Regularizer, gap, shift

using ..ArrayTools: copy!!
using ..MathTools:
    decompose,
    decompose!!,
    eigmin!!,
    frobenius_normalized,
    herk,
    herk!!,
    inverse,
    inverse!!,
    ldiv!!,
    minus,
    muladd!!,
    muladd_diagonal!!,
    tr_normalized

public P!, P_Li_Ri!, Q!, QQh!, Workspace, gap!, greens!, ptrace!, qFP!, shift!

# Abstract supertype that can also be subtyped by GPU workspaces later
abstract type AbstractWorkspace{F<:AbstractFloat} end

# What are the computational dependencies between the auxiliary matrices?
#                            ↙ P
# G0i <- Q <- QQh <- Pi_fact
#                            ↖ P_Li_Ri
# So the possible states are
# - NONE
# - G0i
# - G0i & Q
# - G0i & Q & QQh
# - G0i & Q & QQh & Pi_fact
# - G0i & Q & QQh & Pi_fact & P
# - G0i & Q & QQh & Pi_fact & P_Li_Ri
# - ALL
#
# We use an enum to keep track of this so we only recalculate what is necessary. Note that
# since P and P_L_int_Ri are independent, we can treat their subspace as a bitfield: When
# Pi_fact is clean, the third bit is set and the first two bits indicate whether each of P
# and P_Li_Ri are also clean. See comments within the enum definition.
@enumx Clean::UInt8 begin
    none = 0
    G0i = 1
    Q = 2
    QQh = 3
    Pi_fact = 4  # 0b100
    P = 5        # 0b101
    P_Li_Ri = 6  # 0b110
    all = 7      # 0b111
end

Base.:|(a::Clean.T, b::Clean.T) = Clean.T(UInt8(a) | UInt8(b))
Base.:&(a::Clean.T, b::Clean.T) = Clean.T(UInt8(a) & UInt8(b))

"""
    Workspace(eq::ResolventEquation)

Workspace holding intermediate arrays used to compute the average resolvent from an ensemble
"""
mutable struct Workspace{
    F,
    TE<:ResolventEquation{F},
    TG0i<:AbstractMatrix{Complex{F}},
    TQ<:AbstractMatrix{Complex{F}},
    TQQh<:Union{Diagonal{F},Hermitian{Complex{F}}},
    TPi_fact<:Union{Diagonal{F},Factorization{<:Complex}},  # factorizations can promote
    TP<:Union{Diagonal{F},Hermitian{<:Complex}},
    TP_Li_Ri<:AbstractMatrix{<:Number},
} <: AbstractWorkspace{F}
    const eq::TE
    z::Complex{F}
    q::Complex{F}
    reg::Regularizer{F}
    G0i::TG0i
    Q::TQ
    QQh::TQQh
    Pi_fact::TPi_fact
    P::TP
    P_Li_Ri::TP_Li_Ri
    baregap::F
    clean::Clean.T
    # PQ::TQ
    # QP::TQ
    # PLi_Ri_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}

    function Workspace(eq::E) where {F,E<:ResolventEquation{F}}
        (; ensemble) = eq
        z = zero(Complex{F})
        q = convert(Complex{F}, Defaults.q0)
        reg = Regularizer{F}(
            first(Defaults.regularizer_schedule_values), Defaults.regularizer_schedule_type
        )
        G0i = muladd(z, ensemble.Li_Ri, minus(ensemble.Li_M_Ri))
        Q = muladd(-q, ensemble.S, G0i)
        QQh = herk(Q)
        Pi_fact = decompose(QQh)
        baregap = eigmin!!(Pi_fact, QQh)
        Pi_fact = decompose!!(Pi_fact, QQh; shift=shift(reg, baregap))
        P = inverse(Pi_fact)
        P_Li_Ri = Pi_fact \ ensemble.Li_Ri
        return new{F,E,typeof.((G0i, Q, QQh, Pi_fact, P, P_Li_Ri))...}(
            eq,
            z,
            q,
            reg,
            G0i,
            Q,
            QQh,
            Pi_fact,
            P,
            P_Li_Ri,
            baregap,
            Clean.all,
            # PQ,
            # QhPQ,
            # QPQ_PLi_Ri,
            # QP,
        )
    end
end

# Private clean state management: use setfield! to bypass setproperty! restrictions
function set_clean!(ws::Workspace, value::Clean.T)
    setfield!(ws, :clean, value)
    return nothing
end

function Base.setproperty!(ws::W, name::Symbol, value) where {W<:Workspace}
    if name === :clean
        throw(FieldError(W, name))
    elseif name === :z
        set_clean!(ws, Clean.none)
    elseif name === :q
        set_clean!(ws, min(ws.clean, Clean.G0i))
    elseif name === :reg
        set_clean!(ws, min(ws.clean, Clean.QQh))
    end
    # Replicate the default convert logic
    T = fieldtype(W, name)
    valueT = (value isa T) ? value : convert(T, value)
    return setfield!(ws, name, valueT)
end

## Auxiliary matrix getters
## ==============
# TODO: delete this if the need never arises
# As of now there are no direct consumers of G0i itself, not even downstream in this module.
# It's just a computational step. So for now we don't include it in the getter API.
# """
#     G0i!(ws, z)
#
# Calculate the bare inverse resolvent `G0i(z) = L \\ (z * I - M) / R`
# """
# function G0i!(ws, z)
#     set_G0i_params!(ws, z)
#     update_G0i!(ws)
#     return ws.G0i
# end

function set_G0i_params!(ws::Workspace{F}, z) where {F}
    (ws.z == z) && return nothing
    ws.z = z
    return nothing
end

function update_G0i!(ws::Workspace)
    (ws.clean >= Clean.G0i) && return nothing
    (; Li_Ri, Li_M_Ri) = ws.eq.ensemble
    setfield!(ws, :G0i, muladd!!(ws.G0i, ws.z, Li_Ri, minus(Li_M_Ri)))
    if !(ws.Q isa Diagonal)
        ws.Q, copy!!(ws.Q, ws.G0i)  # set off-diagonal elements of Q
    end
    set_clean!(ws, Clean.G0i)
    return nothing
end

"""
    Q!(ws, z, q)

Calculate `Q(z, q) = G0i(z) - q * S`
"""
function Q!(ws::Workspace{F}, z, q) where {F}
    set_Q_params!(ws, z, q)
    update_Q!(ws)
    return ws.Q
end

function set_Q_params!(ws::Workspace{F}, z, q) where {F}
    set_G0i_params!(ws, z)
    (ws.q == q) && return nothing
    ws.q = q
    return nothing
end

function update_Q!(ws::Workspace)
    (ws.clean >= Clean.Q) && return nothing
    update_G0i!(ws)
    ws.Q = muladd_diagonal!!(ws.Q, -ws.q, ws.eq.ensemble.S, ws.G0i)
    set_clean!(ws, Clean.Q)
    return nothing
end

"""
    QQh!(ws, z, q)

Calculate `Q(z, q) * Q'(z, q)`

If `ws.eq.normalizeQ` is `true`, this actually calculates `(Q * Q' + Q' * Q) / 2`. See the
`ResolventEquation` documentation for further explanation.
"""
function QQh!(ws::Workspace{F}, z, q) where {F}
    set_QQh_params!(ws, z, q)
    update_QQh!(ws)
    return ws.QQh
end

function set_QQh_params!(ws::Workspace{F}, z, q) where {F}
    return set_Q_params!(ws, z, q)
end

function update_QQh!(ws::Workspace)
    (ws.clean >= Clean.QQh) && return nothing
    update_Q!(ws)
    ws.QQh = herk!!(ws.QQh, ws.Q)
    if (!(ws.Q isa Diagonal)) && ws.eq.normalizeQ
        ws.QQh = herk!!(ws.QQh, ws.Q', 1 // 2, 1 // 2)
    end
    ws.baregap = eigmin!!(ws.Pi_fact, ws.QQh)
    set_clean!(ws, Clean.QQh)
    return nothing
end

# TODO: delete this if the need never arises
# As of now there are no direct consumers of Pi_fact itself, not even downstream in this
# module. It's just a computational step. So for now we don't include it in the getter API.
# """
#     Pi_fact!(ws, z, q, reg)
# 
# Factorize `P⁻¹ = Q * Q' + γ² * I`
# """
# function Pi_fact!(ws, z, q, reg)
#     set_Pi_fact_params!(ws, z, q, reg)
#     update_Pi_fact!(ws)
#     return ws.Pi_fact
# end

function set_Pi_fact_params!(ws::Workspace{F}, z, q, reg) where {F}
    set_QQh_params!(ws, z, q)
    (ws.reg == reg) && return nothing
    ws.reg = reg
    return nothing
end

function update_Pi_fact!(ws::Workspace)
    (ws.clean >= Clean.Pi_fact) && return nothing
    update_QQh!(ws)
    ws.Pi_fact = decompose!!(ws.Pi_fact, ws.QQh; shift=shift(ws.reg, ws.baregap))
    set_clean!(ws, Clean.Pi_fact)
    return nothing
end

"""
    P!(ws, z, q, reg)

Calculate `P(z, q, γ²) = inv(Q * Q' + γ² * I)`

The value of `γ²` is provided through `reg::Regularizer`, which can hold either the value of
`γ²` itself or `δ² = γ² + eigmin(Q * Q')`.
"""
function P!(ws::Workspace{F}, z, q, reg) where {F}
    set_P_params!(ws, z, q, reg)
    update_P!(ws)
    return ws.P
end

function set_P_params!(ws::Workspace{F}, z, q, reg) where {F}
    return set_Pi_fact_params!(ws, z, q, reg)
end

function update_P!(ws::Workspace)
    # The states of P and P_Li_Ri are a bitfield, see Clean comments
    ((ws.clean & Clean.P) >= Clean.P) && return nothing
    update_Pi_fact!(ws)
    ws.P = inverse!!(ws.P, ws.Pi_fact)
    set_clean!(ws, ws.clean | Clean.P)
    return nothing
end

"""
    P_Li_Ri!(ws, z, q, reg)

Calculate `P / (RL) = (Q * Q' + γ² * I) * (RL)`

The arguments are the same as for `P!`.
"""
function P_Li_Ri!(ws::Workspace{F}, z, q, reg) where {F}
    set_P_Li_Ri_params!(ws, z, q, reg)
    update_P_Li_Ri!(ws)
    return ws.P_Li_Ri
end

set_P_Li_Ri_params!(ws, z, q, reg) = set_Pi_fact_params!(ws, z, q, reg)

function update_P_Li_Ri!(ws::Workspace)
    # The states of P and P_Li_Ri are a bitfield, see Clean comments
    ((ws.clean & Clean.P_Li_Ri) >= Clean.P_Li_Ri) && return nothing
    update_Pi_fact!(ws)
    ws.P_Li_Ri = ldiv!!(ws.Pi_fact, copy!!(ws.P_Li_Ri, ws.eq.ensemble.Li_Ri))
    set_clean!(ws, ws.clean | Clean.P_Li_Ri)
    return nothing
end

## Scalar getters
## ==============
"""
    qFP!(ws, z, q, reg)

Calculate the function for which the self-consistent value of ``q`` is a fixed point

The function value is `tr(S * Q'P) / size(P, 2)`. The arguments are the same as for `P!`
"""
function qFP!(ws::Workspace{F}, z, q, reg) where {F}
    (; S) = ws.eq.ensemble
    return frobenius_normalized(S, Q!(ws, z, q), P!(ws, z, q, reg))
end

"""
    shift!(ws, z, q, reg)

Calculate the regularizer shift ``γ²``

The arguments are the same as for `P!`
"""
function shift!(ws::Workspace{F}, z, q, reg) where {F}
    baregap = let ws = ws, z = z, q = q
        # Use a closure to defer computation of baregap in case it's not needed for this reg
        baregap!(ws, z, q)
    end
    return shift(convert(Regularizer{F}, reg), baregap)::F
end

"""
    gap!(ws, z, q, reg)

Calculate the regularized gap `eigmin(QQ†) + γ²`

The arguments are the same as for `P!`
"""
function gap!(ws::Workspace{F}, z, q, reg) where {F}
    baregap = let ws = ws, z = z, q = q
        # Use a closure to defer computation of baregap in case it's not needed for this reg
        baregap!(ws, z, q)
    end
    return gap(convert(Regularizer{F}, reg), baregap)::F
end

function baregap!(ws::Workspace{F}, z, q) where {F}
    QQh!(ws, z, q)
    return ws.baregap
end

"""
    ptrace!(ws, z, q, reg)

Calculate the normalized trace of `P(z, q, γ²)`, i.e., `tr(P) / size(P, 2)`

The arguments are the same as for `P!`
"""
function ptrace!(ws::Workspace{F}, z, q, reg) where {F}
    return tr_normalized(P!(ws, z, q, reg))
end

"""
    greens!(ws, z, q, reg)

Calculate the Green's function, i.e., normalized trace of `(RL)⁻¹ * Q'(z, q) * P(z, q, γ²)`

The arguments are the same as for `P!`
"""
function greens!(ws::Workspace{F}, z, q, reg) where {F}
    return frobenius_normalized(Q!(ws, z, q), P_Li_Ri!(ws, z, q, reg))
end

end  # module Workspaces

@testitem "Workspace" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces:
        Clean, Workspace, gap!, greens!, ptrace!, qFP!, shift!

    using LinearAlgebra: Diagonal, Hermitian, I, eigmin, tr
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]

    # No need to fuss about eltypes for the rest, constructors take care of it
    z = -(2.0 + im) / 4
    q = (1.0 - im) / 2
    sscale = 2 // 3
    qscaled = sscale * q

    shift = 0.25
    gap = 0.5
    reg_shift, reg_gap = Regularizer(0.25, :shift), Regularizer(0.5, :gap)

    Dabs = Diagonal([0.75, 0.25])
    Dphase = Diagonal([1.0, cispi(1 // 3)])

    # Ground truth
    R_actual = Dphase
    Li_Ri = map(conj, Dphase)
    S_actual = Dabs
    G0i = (z * I - A64data) / R_actual
    Q = G0i - qscaled * S_actual

    @testset "normalizeQ = $normalizeQ" for normalizeQ in [false, true]
        # Ground truth; only for comparison, so we don't fuss about eltypes
        QQh = normalizeQ ? ((Q * Q') + (Q' * Q)) / 2 : Q * Q'
        baregap = eigmin(Hermitian(QQh))
        Pshift, Pgap = inv(QQh + shift * I), inv(QQh + (gap - baregap) * I)
        qFP_shift = tr(S_actual * Q' * Pshift) / 2
        qFP_gap = tr(S_actual * Q' * Pgap) / 2
        ptrace_shift, ptrace_gap = tr(Pshift) / 2, tr(Pgap) / 2
        greens_shift, greens_gap = tr(Li_Ri * Q' * Pshift) / 2, tr(Li_Ri * Q' * Pgap) / 2

        @testset "$F" for F in [Float32, Float64]
            AFdata = map(Complex{F}, A64data)
            DFdata = map(Complex{F}, D64data)

            @testset "$label" for (label, AF, DF) in [
                ("Array", AFdata, Diagonal(DFdata)),
                ("SArray", SMatrix{2,2}(AFdata), Diagonal(SVector{2}(DFdata))),
                ("SparseMatrixCSC", sparse(AFdata), Diagonal(DFdata)),
            ]
                ensembleF = MatrixEnsemble(AF, I, I, DF)
                eqF = ResolventEquation(ensembleF; normalizeQ)
                wsF = Workspace(eqF)

                @test wsF isa Workspace{F}

                @test shift!(wsF, z, qscaled, reg_shift) ≈ reg_shift.value
                @test shift!(wsF, z, qscaled, reg_gap) ≈ reg_gap.value - baregap

                @test gap!(wsF, z, qscaled, reg_shift) ≈ reg_shift.value + baregap
                @test gap!(wsF, z, qscaled, reg_gap) ≈ reg_gap.value

                @test qFP!(wsF, z, qscaled, reg_shift) ≈ qFP_shift
                @test qFP!(wsF, z, qscaled, reg_gap) ≈ qFP_gap

                @test ptrace!(wsF, z, qscaled, reg_shift) ≈ ptrace_shift
                @test ptrace!(wsF, z, qscaled, reg_gap) ≈ ptrace_gap

                @test greens!(wsF, z, qscaled, reg_shift) ≈ greens_shift
                @test greens!(wsF, z, qscaled, reg_gap) ≈ greens_gap

                # clean not settable
                @test_throws FieldError wsF.clean = Clean.none
            end
        end
    end
end
