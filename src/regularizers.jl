module Regularizers

using AutoHashEquals: @auto_hash_equals
using EnumX: @enumx

using ..ArrayTools: ArrayTools

public Regularizer, RegularizerType, gap, regularizertype, shift

@enumx RegularizerType::UInt8 shift gap

function Base.convert(::Type{RegularizerType.T}, type::Symbol)
    try
        return @eval RegularizerType $type
    catch
        throw(ArgumentError("$(repr(type)) is not a valid Regularizer type"))
    end
end

"""
    Regularizer(val, type)
    Regularizer{F}(val, type) where {F<:Real}
    Regularizer{F}(reg::Regularizer) where {F<:Real}

Represent a regularizer value and type in the averaged resolvent equations.

The available types of regularizers are:

  - `:shift`: specifies the additive regularizer ``γ²`` in the denominator ``QQ† + γ²I``
  - `:gap`: specifies the spectral gap of the denominator, that is, `eigmin(QQ†) + γ²`
"""
@auto_hash_equals struct Regularizer{F<:Real}
    val::F
    type::RegularizerType.T
end

Regularizer(val::F, type) where {F} = Regularizer{F}(val, type)

Regularizer{F}(reg::Regularizer) where {F} = Regularizer{F}(reg.val, reg.type)

"""
    Regularizer(; shift=nothing, gap=nothing)
    Regularizer{F}(; shift=nothing, gap=nothing) where {F}

Keyword constructor for Regularizer. Exactly one of `shift` or `gap` must be provided.
"""
function Regularizer(; shift=nothing, gap=nothing)
    if isnothing(shift) && isnothing(gap)
        throw(ArgumentError("must specify either `shift` or `gap`"))
    elseif !isnothing(shift) && !isnothing(gap)
        throw(ArgumentError("cannot specify both `shift` and `gap`"))
    elseif !isnothing(shift)
        return Regularizer(shift, :shift)
    else
        return Regularizer(gap, :gap)
    end
end

function Regularizer{F}(; shift=nothing, gap=nothing) where {F}
    if isnothing(shift) && isnothing(gap)
        throw(ArgumentError("must specify either `shift` or `gap`"))
    elseif !isnothing(shift) && !isnothing(gap)
        throw(ArgumentError("cannot specify both `shift` and `gap`"))
    elseif !isnothing(shift)
        return Regularizer{F}(shift, :shift)
    else
        return Regularizer{F}(gap, :gap)
    end
end

## Traits
## ======
regularizertype(reg::Regularizer) = Symbol(reg.type)

## Conversions
## ===========
Base.convert(::Type{P}, r::Regularizer) where {P<:Regularizer} = (r isa P) ? r : P(r)
ArrayTools.adaptrealtype(::Type{F}, r::Regularizer) where {F} = convert(Regularizer{F}, r)

"""
    shift(reg::Regularizer, baregap::Real)
    shift(reg::Regularizer, () -> baregap)

Get the regularizer shift ``γ²`` given a Regularizer and the bare gap `eigmin(QQ†)`

`baregap` can be given either as a numerical value or as a callable returning the value. The
latter is useful for deferring any computation necessary to only be invoked if needed, that
is, if the regularizer type of `reg` is `:gap`.
"""
function shift(reg::Regularizer{F}, baregap) where {F}
    (reg.type === RegularizerType.shift) && return reg.val
    bg = (baregap isa Real) ? baregap : baregap()
    return F(reg.val - bg)::F
end

"""
    gap(reg::Regularizer, baregap::Real)
    gap(reg::Regularizer, () -> baregap)

Get the regularized gap `eigmin(QQ†) + γ²` given a Regularizer and the bare gap `eigmin(QQ†)`

`baregap` can be given either as a numerical value or as a callable returning the value. The
latter is useful for deferring any computation necessary to only be invoked if needed, that
is, if the regularizer type of `reg` is `:shift`.
"""
function gap(reg::Regularizer{F}, baregap) where {F}
    (reg.type === RegularizerType.gap) && return reg.val
    bg = (baregap isa Real) ? baregap : baregap()
    return F(reg.val + bg)::F
end

end  # module Regularizers

@testitem "Regularizer" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Regularizers: Regularizer, gap, regularizertype, shift

    shift64, gap64 = 0.0, 0.5
    baregap64 = gap64 - shift64
    reg_shift64 = Regularizer(shift64, :shift)
    reg_gap64 = Regularizer(gap64, :gap)

    # Traits
    @test reg_shift64 isa Regularizer{Float64}
    @test reg_gap64 isa Regularizer{Float64}
    @test regularizertype(reg_shift64) == :shift
    @test regularizertype(reg_gap64) == :gap

    @testset "$F" for F in [Float32, Float64]
        shiftF, gapF, baregapF = F.((shift64, gap64, baregap64))
        reg_shiftF, reg_gapF = Regularizer(shiftF, :shift), Regularizer(gapF, :gap)

        # Realtype inferring constructor
        @test reg_shiftF isa Regularizer{F}
        @test reg_gapF isa Regularizer{F}

        # Realtype enforcing constructor
        @test Regularizer{F}(shift64, :shift) === reg_shiftF
        @test Regularizer{F}(gap64, :gap) === reg_gapF
        @test Regularizer{Float64}(shiftF, :shift) === reg_shift64
        @test Regularizer{Float64}(gapF, :gap) === reg_gap64

        # Realtype conversions
        @test Regularizer{F}(reg_shift64) === reg_shiftF
        @test Regularizer{F}(reg_gap64) === reg_gapF
        @test Regularizer{Float64}(reg_shiftF) === reg_shift64
        @test Regularizer{Float64}(reg_gapF) === reg_gap64
        @test convert(Regularizer{F}, reg_shift64) === reg_shiftF
        @test convert(Regularizer{F}, reg_gap64) === reg_gapF
        @test convert(Regularizer{Float64}, reg_shiftF) === reg_shift64
        @test convert(Regularizer{Float64}, reg_gapF) === reg_gap64
        @test adaptrealtype(F, reg_shift64) === reg_shiftF
        @test adaptrealtype(F, reg_gap64) === reg_gapF
        @test adaptrealtype(Float64, reg_shiftF) === reg_shift64
        @test adaptrealtype(Float64, reg_gapF) === reg_gap64

        # Interface
        @test shift(reg_shiftF, baregapF) === shiftF
        @test shift(reg_shiftF, () -> baregapF) === shiftF
        @test gap(reg_shiftF, baregapF) === gapF
        @test gap(reg_shiftF, () -> baregapF) === gapF
        @test shift(reg_gapF, baregapF) === shiftF
        @test shift(reg_gapF, () -> baregapF) === shiftF
        @test gap(reg_gapF, baregapF) === gapF
        @test gap(reg_gapF, () -> baregapF) === gapF
    end
end

@testitem "Regularizer keyword constructor" begin
    using GaussianSpectralDensities.Regularizers: Regularizer, regularizertype

    # Valid calls
    @test Regularizer(; shift=0.5) == Regularizer(0.5, :shift)
    @test Regularizer(; gap=0.25) == Regularizer(0.25, :gap)
    @test regularizertype(Regularizer(; shift=0.5)) == :shift
    @test regularizertype(Regularizer(; gap=0.25)) == :gap

    # Type-enforcing
    @test Regularizer{Float32}(; shift=0.5) isa Regularizer{Float32}
    @test Regularizer{Float32}(; gap=0.25) isa Regularizer{Float32}

    # Errors
    @test_throws ArgumentError Regularizer()
    @test_throws "must specify" Regularizer()
    @test_throws ArgumentError Regularizer(; shift=0.5, gap=0.25)
    @test_throws "cannot specify both" Regularizer(; shift=0.5, gap=0.25)
end

@testitem "Regularizer errors" begin
    using GaussianSpectralDensities.Regularizers: Regularizer
    @test_throws ArgumentError Regularizer(0.5, :foo)
    @test_throws "not a valid Regularizer" Regularizer(0.5, :foo)
end
