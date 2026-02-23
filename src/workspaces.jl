module Workspaces

using EnumX: @enumx
using LinearAlgebra: Diagonal, Factorization, Hermitian

using ..Defaults: Defaults
using ..Problems: ResolventEquation
using ..Spreads: Spread, gap, shift

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

public P!, P_Li_Ri!, Q!, QQh!, Valid, Workspace, gap!, greens!, ptrace!, shapeFP!, shift!

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
# Pi_fact is valid, the third bit is set and the first two bits indicate whether each of P
# and P_Li_Ri are also valid. See comments within the enum definition.
@enumx Valid::UInt8 begin
    none = 0
    G0i = 1
    Q = 2
    QQh = 3
    Pi_fact = 4  # 0b100
    P = 5        # 0b101
    P_Li_Ri = 6  # 0b110
    all = 7      # 0b111
end

Base.:|(a::Valid.T, b::Valid.T) = Valid.T(UInt8(a) | UInt8(b))
Base.:&(a::Valid.T, b::Valid.T) = Valid.T(UInt8(a) & UInt8(b))

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
    eq::TE
    z::Complex{F}
    shape_scaled::Complex{F}
    spread::Spread{F}
    G0i::TG0i
    Q::TQ
    QQh::TQQh
    Pi_fact::TPi_fact
    P::TP
    P_Li_Ri::TP_Li_Ri
    baregap::F
    valid::Valid.T
    # PQ::TQ
    # QP::TQ
    # PLi_Ri_or_QPQ::Matrix{Complex{F}}
    # QhPQ::Hermitian{Complex{F},Matrix{Complex{F}}}

    function Workspace(eq::E) where {F,E<:ResolventEquation{F}}
        (; ensemble) = eq
        z = zero(Complex{F})
        shape_scaled = eq.sscale * zero(Complex{F})
        spread_val = first(Defaults.spread_annealing_schedule)
        spread = Spread{F}(spread_val, Defaults.spread_annealing_type)
        G0i = muladd(z, ensemble.Li_Ri, minus(ensemble.Li_M_Ri))
        Q = muladd(-shape_scaled, ensemble.S, G0i)
        QQh = herk(Q)
        Pi_fact = decompose(QQh)
        baregap = eigmin!!(Pi_fact, QQh)
        Pi_fact = decompose!!(Pi_fact, QQh; shift=shift(spread, baregap))
        P = inverse(Pi_fact)
        P_Li_Ri = Pi_fact \ ensemble.Li_Ri
        return new{F,E,typeof.((G0i, Q, QQh, Pi_fact, P, P_Li_Ri))...}(
            eq,
            z,
            shape_scaled,
            spread,
            G0i,
            Q,
            QQh,
            Pi_fact,
            P,
            P_Li_Ri,
            baregap,
            Valid.all,
            # PQ,
            # QhPQ,
            # QPQ_PLi_Ri,
            # QP,
        )
    end
end

# Private validity management: use setfield! to bypass setproperty! restrictions
function set_valid!(ws::Workspace, val::Valid.T)
    setfield!(ws, :valid, val)
    return nothing
end

function Base.setproperty!(ws::W, name::Symbol, val) where {W<:Workspace}
    if name === :valid
        throw(FieldError(W, name))
    elseif (name === :eq) || (name === :z)  # G0i obsolete => none valid
        set_valid!(ws, Valid.none)
    elseif name === :shape_scaled  # Q obsolete => at most G0i valid
        set_valid!(ws, min(ws.valid, Valid.G0i))
    elseif name === :spread  # Pi_fact obsolete => at most QQh valid
        set_valid!(ws, min(ws.valid, Valid.QQh))
    end
    # Replicate the default convert logic
    T = fieldtype(W, name)
    valT = (val isa T) ? val : convert(T, val)
    return setfield!(ws, name, valT)
end

## Auxiliary matrix getters
## ==============
# TODO: delete this if the need never arises
# As of now there are no direct consumers of G0i itself, not even downstream in this module.
# It's just a computational step. So for now we don't include it in the getter API.
# """
#     G0i!(ws, eq, z)
#
# Calculate the bare inverse resolvent `G0i(z) = L \\ (z * I - M) / R`
# """
# function G0i!(ws, eq, z)
#     set_G0i_params!(ws, eq, z)
#     update_G0i!(ws)
#     return ws.G0i
# end

function set_G0i_params!(ws::Workspace{F}, eq, z) where {F}
    ((ws.eq === eq) && (ws.z == z)) && return nothing
    ws.eq = eq
    ws.z = z
    return nothing
end

function update_G0i!(ws::Workspace)
    (ws.valid >= Valid.G0i) && return nothing
    (; Li_Ri, Li_M_Ri) = ws.eq.ensemble
    setfield!(ws, :G0i, muladd!!(ws.G0i, ws.z, Li_Ri, minus(Li_M_Ri)))
    if !(ws.Q isa Diagonal)
        ws.Q, copy!!(ws.Q, ws.G0i)  # set off-diagonal elements of Q
    end
    set_valid!(ws, Valid.G0i)
    return nothing
end

"""
    Q!(ws, eq, z, shape)

Calculate `Q(z, q) = G0i(z) - q * S`
"""
function Q!(ws::Workspace{F}, eq, z, shape) where {F}
    set_Q_params!(ws, eq, z, shape)
    update_Q!(ws)
    return ws.Q
end

function set_Q_params!(ws::Workspace{F}, eq, z, shape) where {F}
    set_G0i_params!(ws, eq, z)
    shape_scaled = eq.sscale * shape
    (ws.shape_scaled == shape_scaled) && return nothing
    ws.shape_scaled = shape_scaled
    return nothing
end

function update_Q!(ws::Workspace)
    (ws.valid >= Valid.Q) && return nothing
    update_G0i!(ws)
    ws.Q = muladd_diagonal!!(ws.Q, -ws.shape_scaled, ws.eq.ensemble.S, ws.G0i)
    set_valid!(ws, Valid.Q)
    return nothing
end

"""
    QQh!(ws, eq, z, shape)

Calculate `Q(z, q) * Q'(z, q)`

If `eq.normalizeQ` is `true`, this actually calculates `(Q * Q' + Q' * Q) / 2`. See the
`ResolventEquation` documentation for further explanation.
"""
function QQh!(ws::Workspace{F}, eq, z, shape) where {F}
    set_QQh_params!(ws, eq, z, shape)
    update_QQh!(ws)
    return ws.QQh
end

set_QQh_params!(ws::Workspace{F}, eq, z, shape) where {F} = set_Q_params!(ws, eq, z, shape)

function update_QQh!(ws::Workspace)
    (ws.valid >= Valid.QQh) && return nothing
    update_Q!(ws)
    ws.QQh = herk!!(ws.QQh, ws.Q)
    if (!(ws.Q isa Diagonal)) && ws.eq.normalizeQ
        ws.QQh = herk!!(ws.QQh, ws.Q', 1 // 2, 1 // 2)
    end
    ws.baregap = eigmin!!(ws.Pi_fact, ws.QQh)
    set_valid!(ws, Valid.QQh)
    return nothing
end

# TODO: delete this if the need never arises
# As of now there are no direct consumers of Pi_fact itself, not even downstream in this
# module. It's just a computational step. So for now we don't include it in the getter API.
# """
#     Pi_fact!(ws, eq, z, shape, spread)
# 
# Factorize `P⁻¹ = Q * Q' + γ² * I`
# """
# function Pi_fact!(ws, eq, z, shape, spread)
#     set_Pi_fact_params!(ws, eq, z, shape, spread)
#     update_Pi_fact!(ws)
#     return ws.Pi_fact
# end

function set_Pi_fact_params!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    set_QQh_params!(ws, eq, z, shape)
    (ws.spread == spread) && return nothing
    ws.spread = spread
    return nothing
end

function update_Pi_fact!(ws::Workspace)
    (ws.valid >= Valid.Pi_fact) && return nothing
    update_QQh!(ws)
    ws.Pi_fact = decompose!!(ws.Pi_fact, ws.QQh; shift=shift(ws.spread, ws.baregap))
    set_valid!(ws, Valid.Pi_fact)
    return nothing
end

"""
    P!(ws, eq, z, shape, spread)

Calculate `P(z, q, γ²) = inv(Q * Q' + γ² * I)`

The value of `γ²` is provided through `spread::Spread`, which can hold either the value of
`γ²` itself or the spectral gap `eigmin(Q * Q') + γ²`.
"""
function P!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    set_P_params!(ws, eq, z, shape, spread)
    update_P!(ws)
    return ws.P
end

function set_P_params!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    return set_Pi_fact_params!(ws, eq, z, shape, spread)
end

function update_P!(ws::Workspace)
    # The states of P and P_Li_Ri are a bitfield, see Valid comments
    ((ws.valid & Valid.P) >= Valid.P) && return nothing
    update_Pi_fact!(ws)
    ws.P = inverse!!(ws.P, ws.Pi_fact)
    set_valid!(ws, ws.valid | Valid.P)
    return nothing
end

"""
    P_Li_Ri!(ws, eq, z, shape, spread)

Calculate `P / (RL) = (Q * Q' + γ² * I) * (RL)`

The arguments are the same as for `P!`.
"""
function P_Li_Ri!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    set_P_Li_Ri_params!(ws, eq, z, shape, spread)
    update_P_Li_Ri!(ws)
    return ws.P_Li_Ri
end

function set_P_Li_Ri_params!(ws, eq, z, shape, spread)
    return set_Pi_fact_params!(ws, eq, z, shape, spread)
end

function update_P_Li_Ri!(ws::Workspace)
    # The states of P and P_Li_Ri are a bitfield, see Valid comments
    ((ws.valid & Valid.P_Li_Ri) >= Valid.P_Li_Ri) && return nothing
    update_Pi_fact!(ws)
    ws.P_Li_Ri = ldiv!!(ws.Pi_fact, copy!!(ws.P_Li_Ri, ws.eq.ensemble.Li_Ri))
    set_valid!(ws, ws.valid | Valid.P_Li_Ri)
    return nothing
end

## Scalar getters
## ==============
"""
    shapeFP!(ws, eq, z, shape, spread)

Fixed-point function for the self-consistent value of the shape field ``q``

The function value is `tr(S * Q'P) / size(P, 2)`. The arguments are the same as for `P!`
"""
function shapeFP!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    Q, P = Q!(ws, eq, z, shape), P!(ws, eq, z, shape, spread)
    return eq.sscale * frobenius_normalized(eq.ensemble.S, Q, P)
end

"""
    shift!(ws, eq, z, shape, spread)

Calculate the spectral shift ``γ²``

The arguments are the same as for `P!`
"""
function shift!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    baregap = let ws = ws, eq=eq, z = z, shape = shape
        # Use closure to defer baregap calculation in case it's not needed
        baregap() = baregap!(ws, eq, z, shape)
    end
    return shift(convert(Spread{F}, spread), baregap)::F
end

"""
    gap!(ws, eq, z, shape, spread)

Calculate the spectral gap `eigmin(QQ†) + γ²`

The arguments are the same as for `P!`
"""
function gap!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    baregap = let ws = ws, eq=eq, z = z, shape = shape
        # Use closure to defer baregap calculation in case it's not needed
        baregap() = baregap!(ws, eq, z, shape)
    end
    return gap(convert(Spread{F}, spread), baregap)::F
end

function baregap!(ws::Workspace{F}, eq, z, shape) where {F}
    QQh!(ws, eq, z, shape)
    return ws.baregap
end

"""
    ptrace!(ws, eq, z, shape, spread)

Calculate the normalized trace of `P(z, q, γ²)`, i.e., `tr(P) / size(P, 2)`

The arguments are the same as for `P!`
"""
function ptrace!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    return tr_normalized(P!(ws, eq, z, shape, spread))
end

"""
    greens!(ws, eq, z, shape, spread)

Calculate the Green's function, i.e., normalized trace of `(RL)⁻¹ * Q'(z, q) * P(z, q, γ²)`

The arguments are the same as for `P!`
"""
function greens!(ws::Workspace{F}, eq, z, shape, spread) where {F}
    return frobenius_normalized(Q!(ws, eq, z, shape), P_Li_Ri!(ws, eq, z, shape, spread))
end

end  # module Workspaces

@testitem "Workspace" begin
    using GaussianSpectralDensities: MatrixEnsemble, ResolventEquation, Spread
    using GaussianSpectralDensities.Workspaces:
        Valid, Workspace, gap!, greens!, ptrace!, shapeFP!, shift!

    using LinearAlgebra: Diagonal, Hermitian, I, eigmin, tr
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    A64data = ComplexF64[1 -1; 1 1]
    D64data = ComplexF64[0.75, 0.25cispi(1//3)]

    # No need to fuss about eltypes for the rest, constructors take care of it
    z = -(2.0 + im) / 4
    shape = (1.0 - im) / 2
    sscale = 2 // 3
    shape_scaled = sscale * shape

    shift = 0.25
    gap = 0.5
    spread_shift, spread_gap = Spread(0.25, :shift), Spread(0.5, :gap)

    S_actual = Diagonal(abs.(D64data))
    R_actual = Diagonal(sign.(D64data))

    # Ground truth
    Li_Ri = inv(R_actual)
    G0i = (z * I - A64data) / R_actual
    Q = G0i - shape_scaled * S_actual

    @testset "normalizeQ = $normalizeQ" for normalizeQ in [false, true]
        # Ground truth; only for comparison, so we don't fuss about eltypes
        QQh = Hermitian(normalizeQ ? ((Q * Q') + (Q' * Q)) / 2 : Q * Q')
        baregap = eigmin(QQh)
        Pshift, Pgap = inv(QQh + shift * I), inv(QQh + (gap - baregap) * I)
        shapeFP_shift = sscale * tr(S_actual * Q' * Pshift) / 2
        shapeFP_gap = sscale * tr(S_actual * Q' * Pgap) / 2
        ptrace_shift, ptrace_gap = tr(Pshift) / 2, tr(Pgap) / 2
        greens_shift, greens_gap = tr(Li_Ri * Q' * Pshift) / 2, tr(Li_Ri * Q' * Pgap) / 2

        @testset "$F" for F in [Float32, Float64]
            AFdata = map(Complex{F}, A64data)
            DFdata = map(Complex{F}, D64data)

            @testset "$label" for (label, AF, SF) in [
                ("Array", AFdata, Diagonal(DFdata)),
                ("SArray", SMatrix{2,2}(AFdata), Diagonal(SVector{2}(DFdata))),
                ("SparseMatrixCSC", sparse(AFdata), Diagonal(DFdata)),
            ]
                ensembleF = MatrixEnsemble(AF, I, I, SF)
                eqF = ResolventEquation(ensembleF; sscale, normalizeQ)
                wsF = Workspace(eqF)

                @test wsF isa Workspace{F}

                @test shift!(wsF, eqF, z, shape, spread_shift) ≈ spread_shift.val
                @test shift!(wsF, eqF, z, shape, spread_gap) ≈ spread_gap.val - baregap

                @test gap!(wsF, eqF, z, shape, spread_shift) ≈ spread_shift.val + baregap
                @test gap!(wsF, eqF, z, shape, spread_gap) ≈ spread_gap.val

                @test shapeFP!(wsF, eqF, z, shape, spread_shift) ≈ shapeFP_shift
                @test shapeFP!(wsF, eqF, z, shape, spread_gap) ≈ shapeFP_gap

                @test ptrace!(wsF, eqF, z, shape, spread_shift) ≈ ptrace_shift
                @test ptrace!(wsF, eqF, z, shape, spread_gap) ≈ ptrace_gap

                @test greens!(wsF, eqF, z, shape, spread_shift) ≈ greens_shift
                @test greens!(wsF, eqF, z, shape, spread_gap) ≈ greens_gap

                # valid not settable
                @test_throws FieldError wsF.valid = Valid.none
            end
        end
    end
end
