@enumx Regularizers γ δ

function Base.convert(::Type{Regularizers.T}, r::Symbol)
    try
        return @eval Regularizers $r
    catch
        throw(ArgumentError("$(repr(r)) is not a valid kind of Regularizer"))
    end
end

"""
    Regularizer(kind, value)
    Regularizer{F}(kind, value) where {F<:Real}

Represent the regularizer in the averaged resolvent equations

The available kinds of regularizers are:

- `:γ`: the simple additive regularizer in the denominator ``[QQ^† + γ²I]⁻¹``
- `:δ`: refers to the smallest eigenvalue of the entire denominator, that is,
    `δ^2 == γ^2 + eigmin(Q * Q')`

The value of the regularizer is the value of `γ²` or `δ²`, respectively
"""
struct Regularizer{F<:Real}
    kind::Regularizers.T
    value::F
end

Regularizer(kind, value::F) where {F} = Regularizer{F}(kind, value)

Regularizer{F}(reg::Regularizer) where {F} = Regularizer{F}(reg.kind, reg.value)

## Conversions
## ===========
Base.convert(::Type{P}, r::Q) where {P<:Regularizer,Q<:Regularizer} = P(r)

"""
    γ²(reg::Regularizer, s²min::Number)
    γ²(reg::Regularizer, () -> s²min)

Return the value of ``γ²`` given a regularizer and a value for `s²min = eigmin(Q * Q')`

The minimum eigenvalue `s²min` can either be given as a numerical value, or as a callable
returning the value. That way, any heavy computation involved in determining `s²min` can be
deferred and only invoked if needed, that is, if the `kind` of `reg` is `:δ`.
"""
function γ²(reg::Regularizer, s²min)
    (reg.kind === Regularizers.γ) && return reg.value
    if (reg.kind === Regularizers.δ)
        (s²min isa Number) && return reg.value - s²min
        return reg.value - s²min()
    end
end

"""
    δ²(reg::Regularizer, s²min::Number)
    δ²(reg::Regularizer, () -> s²min)

Return the value of ``δ²`` given a regularizer and a value for `s²min = eigmin(Q * Q')`

The minimum eigenvalue `s²min` can either be given as a numerical value, or as a callable
returning the value. That way, any heavy computation involved in determining `s²min` can be
deferred and only invoked if needed, that is, if the `kind` of `reg` is `:γ`.
"""
function δ²(reg::Regularizer, s²min)
    (reg.kind === Regularizers.δ) && return reg.value
    if (reg.kind === Regularizers.γ)
        (s²min isa Number) && return reg.value + s²min
        return reg.value + s²min()
    end
end

#=
## solution score
## ==============

function score(solution, r::Regularizer)
    for reg in instances(Regularizers.T)
        (r.kind === reg) && return score(getproperty(solution, reg), r.value)
    end
end
=#

@testitem "Regularizer" begin
    using GaussianSpectralDensities: Regularizer, γ², δ²

    γ²value, δ²value = 0.0, 0.5
    s²min = δ²value - γ²value
    γreg, δreg = Regularizer(:γ, γ²value), Regularizer(:δ, δ²value)

    # Traits
    @test γreg isa Regularizer{Float64}
    @test δreg isa Regularizer{Float64}

    @testset "$F" for F in [Float32, Float64]
        γ²valueF, δ²valueF, s²minF = F.((γ²value, δ²value, s²min))
        γregF, δregF = Regularizer(:γ, γ²valueF), Regularizer(:δ, δ²valueF)

        # Floattype inferring constructor
        @test γregF isa Regularizer{F}
        @test δregF isa Regularizer{F}

        # Floattype enforcing constructor
        @test Regularizer{F}(:γ, γ²value) === γregF
        @test Regularizer{F}(:δ, δ²value) === δregF
        @test Regularizer{Float64}(:γ, γ²valueF) === γreg
        @test Regularizer{Float64}(:δ, δ²valueF) === δreg

        # Floattype conversions
        @test convert(Regularizer{F}, γreg) === γregF
        @test convert(Regularizer{F}, δreg) === δregF
        @test convert(Regularizer{Float64}, γregF) === γreg
        @test convert(Regularizer{Float64}, δregF) === δreg
        @test Regularizer{F}(γreg) === γregF
        @test Regularizer{F}(δreg) === δregF
        @test Regularizer{Float64}(γregF) === γreg
        @test Regularizer{Float64}(δregF) === δreg

        # Interface
        @test γ²(γregF, s²minF) === γ²valueF
        @test γ²(γregF, () -> s²minF) === γ²valueF
        @test δ²(γregF, s²minF) === δ²valueF
        @test δ²(γregF, () -> s²minF) === δ²valueF
        @test γ²(δregF, s²minF) === γ²valueF
        @test γ²(δregF, () -> s²minF) === γ²valueF
        @test δ²(δregF, s²minF) === δ²valueF
        @test δ²(δregF, () -> s²minF) === δ²valueF
    end
end
