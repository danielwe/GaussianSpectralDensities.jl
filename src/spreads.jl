module Spreads

using AutoHashEquals: @auto_hash_equals
using EnumX: @enumx

using ..ArrayTools: ArrayTools

public Spread, SpreadType, gap, shift

@enumx SpreadType::UInt8 shift gap

function Base.convert(::Type{SpreadType.T}, type::Symbol)
    try
        return @eval SpreadType $type
    catch
        throw(ArgumentError("$(repr(type)) is not a valid Spread type"))
    end
end

"""
    Spread(val, type)
    Spread{F}(val, type) where {F<:Real}
    Spread{F}(spread::Spread) where {F<:Real}

Represent the spread parameter in averaged resolvent equations.

The available types of spread parameters are:

  - `:shift`: the spectral shift ``γ²`` of the resolvent denominator ``QQ† + γ²I``
  - `:gap`: the spectral gap of the resolvent denominator: `eigmin(QQ†) + γ²`
"""
@auto_hash_equals struct Spread{F<:Real}
    val::F
    type::SpreadType.T
end

Spread(val::F, type) where {F} = Spread{F}(val, type)

Spread{F}(spread::Spread) where {F} = Spread{F}(spread.val, spread.type)

"""
    Spread(; shift=nothing, gap=nothing)
    Spread{F}(; shift=nothing, gap=nothing) where {F}

Keyword constructor for Spread. Exactly one of `shift` or `gap` must be provided.
"""
function Spread(; shift=nothing, gap=nothing)
    if !xor(isnothing(shift), isnothing(gap))
        throw(ArgumentError("must specify exactly one of `shift` or `gap`"))
    elseif !isnothing(shift)
        return Spread(shift, :shift)
    else
        return Spread(gap, :gap)
    end
end

Spread{F}(; kwargs...) where {F} = Spread{F}(Spread(; kwargs...))

## Conversions
## ===========
Base.convert(::Type{P}, r::Spread) where {P<:Spread} = (r isa P) ? r : P(r)
ArrayTools.adaptrealtype(::Type{F}, r::Spread) where {F} = convert(Spread{F}, r)

"""
    shift(spread::Spread, baregap::Real)
    shift(spread::Spread, () -> baregap)

Return the spectral shift ``γ²`` given a spread parameter and the bare gap `eigmin(QQ†)`

`baregap` can be either a numerical value or as a callable returning the value. The callable
form is useful for deferring the eigmin calculation in case it's not needed, which is the
case if `spread.type` is `:shift`.
"""
function shift(spread::Spread{F}, baregap) where {F}
    (spread.type === SpreadType.shift) && return spread.val
    bg = (baregap isa Number) ? baregap : baregap()
    return F(spread.val - bg)::F
end

"""
    gap(spread::Spread, baregap::Real)
    gap(spread::Spread, () -> baregap)

Return the spectral gap ``eigmin(QQ†) + γ²`` given a Spread and the bare gap `eigmin(QQ†)`

`baregap` can be either a numerical value or as a callable returning the value. The callable
form is useful for deferring the eigmin calculation in case it's not needed, which is the
case if `spread.type` is `:gap`.
"""
function gap(spread::Spread{F}, baregap) where {F}
    (spread.type === SpreadType.gap) && return spread.val
    bg = (baregap isa Number) ? baregap : baregap()
    return F(spread.val + bg)::F
end

end  # module Spreads

@testitem "Spread" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Spreads: Spread, gap, shift

    shift64, gap64 = 0.0, 0.5
    baregap64 = gap64 - shift64
    spread_shift64 = Spread(shift64, :shift)
    spread_gap64 = Spread(gap64, :gap)

    # Traits
    @test spread_shift64 isa Spread{Float64}
    @test spread_gap64 isa Spread{Float64}

    @testset "$F" for F in [Float32, Float64]
        shiftF, gapF, baregapF = F.((shift64, gap64, baregap64))
        spread_shiftF, spread_gapF = Spread(shiftF, :shift), Spread(gapF, :gap)

        # Realtype inferring constructor
        @test spread_shiftF isa Spread{F}
        @test spread_gapF isa Spread{F}

        # Realtype enforcing constructor
        @test Spread{F}(shift64, :shift) === spread_shiftF
        @test Spread{F}(gap64, :gap) === spread_gapF
        @test Spread{Float64}(shiftF, :shift) === spread_shift64
        @test Spread{Float64}(gapF, :gap) === spread_gap64

        # Realtype conversions
        @test Spread{F}(spread_shift64) === spread_shiftF
        @test Spread{F}(spread_gap64) === spread_gapF
        @test Spread{Float64}(spread_shiftF) === spread_shift64
        @test Spread{Float64}(spread_gapF) === spread_gap64
        @test convert(Spread{F}, spread_shift64) === spread_shiftF
        @test convert(Spread{F}, spread_gap64) === spread_gapF
        @test convert(Spread{Float64}, spread_shiftF) === spread_shift64
        @test convert(Spread{Float64}, spread_gapF) === spread_gap64
        @test adaptrealtype(F, spread_shift64) === spread_shiftF
        @test adaptrealtype(F, spread_gap64) === spread_gapF
        @test adaptrealtype(Float64, spread_shiftF) === spread_shift64
        @test adaptrealtype(Float64, spread_gapF) === spread_gap64

        # Interface
        @test shift(spread_shiftF, baregapF) === shiftF
        @test shift(spread_shiftF, () -> baregapF) === shiftF
        @test gap(spread_shiftF, baregapF) === gapF
        @test gap(spread_shiftF, () -> baregapF) === gapF
        @test shift(spread_gapF, baregapF) === shiftF
        @test shift(spread_gapF, () -> baregapF) === shiftF
        @test gap(spread_gapF, baregapF) === gapF
        @test gap(spread_gapF, () -> baregapF) === gapF
    end
end

@testitem "Spread keyword constructor" begin
    using GaussianSpectralDensities.Spreads: Spread

    # Valid calls
    @test Spread(; shift=0.5) == Spread(0.5, :shift)
    @test Spread(; gap=0.25) == Spread(0.25, :gap)

    # Type-enforcing
    @test Spread{Float32}(; shift=0.5) isa Spread{Float32}
    @test Spread{Float32}(; gap=0.25) isa Spread{Float32}

    # Errors
    @test_throws ArgumentError Spread()
    @test_throws "exactly one" Spread()
    @test_throws ArgumentError Spread(; shift=0.5, gap=0.25)
    @test_throws "exactly one" Spread(; shift=0.5, gap=0.25)
end

@testitem "Spread errors" begin
    using GaussianSpectralDensities.Spreads: Spread
    @test_throws ArgumentError Spread(0.5, :foo)
    @test_throws "not a valid Spread" Spread(0.5, :foo)
end
