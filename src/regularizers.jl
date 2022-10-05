module Regularizers
    # Like EnumX.jl but manual so LanguageServer.jl can understand, plus making instances
    # behave like symbols in certain contexts
    @enum T γ δ

    function Base.convert(::Type{T}, r::Symbol)
        reg = @eval $r
        (reg isa T) || throw(ArgumentError("$reg is not a $T"))  # catch convert(T, :T)
        return reg
    end

    Base.:(==)(reg::T, r::Symbol) = (Symbol(reg) == r)
    Base.:(==)(r::Symbol, reg::T) = (reg == r)

    # Base.getproperty(value, reg::T) = getproperty(value, Symbol(reg))  # experimental
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

## Conversions
## ===========
Base.convert(::Type{P}, r::Q) where {P<:Regularizer,Q<:Regularizer} = P(r.kind, r.value)

#=
## solution score
## ==============

function score(solution, r::Regularizer)
    for reg in instances(Regularizers.T)
        (r.kind === reg) && return score(getproperty(solution, reg), r.value)
    end
end
=#
