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

Represent the regularizer in the averaged resolvent equations

The available kinds of regularizers are:

- `:γ`: the simple additive regularizer in the denominator ``[QQ^† + γ²I]⁻¹``
- `:δ`: refers to the smallest eigenvalue of the entire denominator, that is,
    ` δ^2 == γ^2 + eigmin(Q * Q')`
"""
struct Regularizer{T<:Real}
    kind::Regularizers.T
    value::T
end

Regularizer(kind, value::T) where {T} = Regularizer{T}(kind, value)


## value type conversion and adaption
## ==================================

Base.convert(::Type{P}, r::Q) where {P<:Regularizer,Q<:Regularizer} = P(r.kind, r.value)
adapt(::Type{T}, r::Regularizer) where {T} = convert(Regularizer{T}, r)

#=
## solution score
## ==============

function score(solution, r::Regularizer)
    for reg in instances(Regularizers.T)
        (r.kind === reg) && return score(getproperty(solution, reg), r.value)
    end
end
=#
