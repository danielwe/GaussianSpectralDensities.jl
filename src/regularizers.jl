module Regularizers

using EnumX: @enumx
using TestItems: @testitem

using ..Public: @public

@public Regularizer shift gap

@enumx RegularizerType::UInt8 shift gap

function Base.convert(::Type{RegularizerType.T}, type::Symbol)
    try
        return @eval RegularizerType $type
    catch
        throw(ArgumentError("$(repr(type)) is not a valid Regularizer type"))
    end
end

"""
    Regularizer(value, type)
    Regularizer{F}(value, type) where {F<:Real}
    Regularizer{F}(reg::Regularizer) where {F<:Real}

Represent the regularizer in the averaged resolvent equations

The available types of regularizers are:

- `:shift`: specifies the additive regularizer ``γ²`` in the denominator ``QQ† + γ²I``
- `:gap`: specifies the spectral gap of the denominator, that is, `eigmin(QQ†) + γ²`
"""
struct Regularizer{F<:Real}
    value::F
    type::RegularizerType.T
end

Regularizer(value::F, type) where {F} = Regularizer{F}(value, type)

Regularizer{F}(reg::Regularizer) where {F} = Regularizer{F}(reg.value, reg.type)

## Conversions
## ===========
Base.convert(::Type{P}, r::Q) where {P<:Regularizer,Q<:Regularizer} = P(r)

"""
    shift(reg::Regularizer, nullgap::Number)
    shift(reg::Regularizer, () -> nullgap)

Return the value of the shift ``γ²`` given a Regularizer and the unregularized spectral gap `nullgap = eigmin(QQ†)`

`nullgap` can be given either as a numerical value or as a callable returning the value. The
latter is useful for deferring any heavy computation to only be invoked if needed, that is,
if the regularizer type of `reg` is `:gap`.
"""
function shift(reg::Regularizer, nullgap)
    (reg.type === RegularizerType.shift) && return reg.value
    (nullgap isa Number) && return reg.value - nullgap
    return reg.value - nullgap()
end

"""
    gap(reg::Regularizer, nullgap::Number)
    gap(reg::Regularizer, () -> nullgap)

Return the value of the gap `eigmin(QQ†) + γ²` given a Regularizer and the unregularized spectral gap `nullgap = eigmin(QQ†)`

`nullgap` can be given either as a numerical value or as a callable returning the value. The
latter is useful for deferring any heavy computation to only be invoked if needed, that is,
if the regularizer type of `reg` is `:shift`.
"""
function gap(reg::Regularizer, nullgap)
    (reg.type === RegularizerType.gap) && return reg.value
    (nullgap isa Number) && return reg.value + nullgap
    return reg.value + nullgap()
end

#=
## solution score
## ==============

function score(solution, r::Regularizer)
    for reg in instances(Regularizers.T)
        (r.type === reg) && return score(getproperty(solution, reg), r.value)
    end
end
=#

@testitem "Regularizer" begin
    using GaussianSpectralDensities.Regularizers: Regularizer, shift, gap

    shift64, gap64 = 0.0, 0.5
    nullgap64 = gap64 - shift64
    reg_shift64 = Regularizer(shift64, :shift)
    reg_gap64 = Regularizer(gap64, :gap)

    # Traits
    @test reg_shift64 isa Regularizer{Float64}
    @test reg_gap64 isa Regularizer{Float64}

    @testset "$F" for F in [Float32, Float64]
        shiftF, gapF, nullgapF = F.((shift64, gap64, nullgap64))
        reg_shiftF, reg_gapF = Regularizer(shiftF, :shift), Regularizer(gapF, :gap)

        # Floattype inferring constructor
        @test reg_shiftF isa Regularizer{F}
        @test reg_gapF isa Regularizer{F}

        # Floattype enforcing constructor
        @test Regularizer{F}(shift64, :shift) === reg_shiftF
        @test Regularizer{F}(gap64, :gap) === reg_gapF
        @test Regularizer{Float64}(shiftF, :shift) === reg_shift64
        @test Regularizer{Float64}(gapF, :gap) === reg_gap64

        # Floattype conversions
        @test convert(Regularizer{F}, reg_shift64) === reg_shiftF
        @test convert(Regularizer{F}, reg_gap64) === reg_gapF
        @test convert(Regularizer{Float64}, reg_shiftF) === reg_shift64
        @test convert(Regularizer{Float64}, reg_gapF) === reg_gap64
        @test Regularizer{F}(reg_shift64) === reg_shiftF
        @test Regularizer{F}(reg_gap64) === reg_gapF
        @test Regularizer{Float64}(reg_shiftF) === reg_shift64
        @test Regularizer{Float64}(reg_gapF) === reg_gap64

        # Interface
        @test shift(reg_shiftF, nullgapF) === shiftF
        @test shift(reg_shiftF, () -> nullgapF) === shiftF
        @test gap(reg_shiftF, nullgapF) === gapF
        @test gap(reg_shiftF, () -> nullgapF) === gapF
        @test shift(reg_gapF, nullgapF) === shiftF
        @test shift(reg_gapF, () -> nullgapF) === shiftF
        @test gap(reg_gapF, nullgapF) === gapF
        @test gap(reg_gapF, () -> nullgapF) === gapF
    end
end

end  # module Regularizers
