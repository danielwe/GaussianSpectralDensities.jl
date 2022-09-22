"""
    LocalResolventEquation(ensemble::MatrixEnsemble, z, μ, normalizeQ)
    LocalResolventEquation{F}(ensemble, z, μ, normalizeQ) where {F}

Specify the averaged resolvent equation for an ensemble at a point in the complex plane

# Arguments

- `ensemble`: random matrix ensemble over which to average the resolvent
- `z`: point in the complex plane at which to calculate the averaged resolvent
- `μ`: an overall scaling factor for the degree of symmetry `ensemble.T`. This parameter is
    inserted by hand into the equations to facilitate a solution strategy whereby the
    symmetry is annealed.
- `normalizeQ=false`: whether to replace ``QQ^†`` with ``(QQ^† + Q^†Q) / 2`` in the
    self-consistent equations for the averaged resolvent, mimicking the case where
    ``QQ^† = Q^†Q``, i.e., where ``Q`` is normal. This is an attempt to get around an issue
    relating to order of limits that can arise for highly non-normal `Q` (see Ahmadian,
    Fumarola, and Miller (2015)). Performing this normalization is not mathematically sound,
    and in most cases gives answers that are subtly but unambiguously wrong; however, in
    certain experiments it seems to do the trick in terms of removing regions of spurious
    spectral support. The option is provided here to support further experimenting.
"""
struct LocalResolventEquation{F,E<:MatrixEnsemble{F}}
    ensemble::E
    z::Complex{F}
    μ::F
    normalizeQ::Bool

    function LocalResolventEquation{F}(ensemble, z, μ, normalizeQ) where {F}
        if !((μ isa Real) && (0 <= μ <= 1))
            throw(ArgumentError("symmetry scale factor μ not in [0, 1]"))
        end
        ensemble = adapt(F, ensemble)
        return new{F,typeof(ensemble)}(ensemble, z, μ, normalizeQ)
    end
end

function LocalResolventEquation(ensemble::MatrixEnsemble{F}, z, μ, normalizeQ) where {F}
    return LocalResolventEquation{F}(ensemble, z, μ, normalizeQ)
end

function Base.convert(
    ::Type{P}, eq::Q
) where {F,P<:LocalResolventEquation{F},Q<:LocalResolventEquation}
    (eq isa P) && return eq
    return LocalResolventEquation{F}(eq.ensemble, eq.z, eq.μ, eq.normalizeQ)
end

abstract type AbstractLocalResolventProblem{F<:AbstractFloat} end

"""
    LocalQProblem(equation::LocalResolventEquation, regularizer, q0) where {F}
    LocalQProblem{F}(equation, regularizer, q0)

Specify the problem of finding ``q`` given a regularizer in the averaged resolvent equation

`q0` is a starting guess for `q`
"""
struct LocalQProblem{F,E<:LocalResolventEquation{F}} <: AbstractLocalResolventProblem{F}
    equation::E
    regularizer::Regularizer{F}
    q0::Complex{F}

    function LocalQProblem{F}(equation, regularizer, q0) where {F}
        equation = adapt(F, equation)
        return new{F,typeof(equation)}(equation, regularizer, q0)
    end
end

function LocalQProblem(equation::LocalResolventEquation{F}, regularizer, q0) where {F}
    return LocalQProblem{F}(equation, regularizer, q0)
end

function Base.convert(::Type{P}, p::Q) where {F,P<:LocalQProblem{F},Q<:LocalQProblem}
    (p isa P) && return p
    return LocalQProblem{F}(p.equation, p.regularizer, p.q0)
end

"""
    LocalQRegularizerProblem(equation::LocalResolventEquation, k, q0, regularizer0)
    LocalQRegularizerProblem{F}(equation, k, q0, regularizer0)

Specify the problem of finding ``q,γ²`` given ``k`` in the averaged resolvent equation

Here, ``k`` is the target value of the normalized trace ``k = \\tr[(QQ^† + γ²I)⁻¹] / N``.

`q0` and `regularizer0` represent starting guesses for ``q`` and one of the regularizing
parameters ``γ²,δ²``
"""
struct LocalQRegularizerProblem{
    F,E<:LocalResolventEquation{F}
} <: AbstractLocalResolventProblem{F}
    equation::E
    k::F
    q0::Complex{F}
    regularizer0::Regularizer{F}

    function LocalQRegularizerProblem{F}(equation, k, q0, regularizer0) where {F}
        equation = adapt(F, equation)
        return new{F,typeof(equation)}(equation, k, q0, regularizer0)
    end
end

function LocalQRegularizerProblem(
    equation::LocalResolventEquation{F}, k, q0, regularizer0
) where {F}
    return LocalQRegularizerProblem{F}(equation, k, q0, regularizer0)
end

function Base.convert(
    ::Type{P}, p::Q
) where {F,P<:LocalQRegularizerProblem{F},Q<:LocalQRegularizerProblem}
    (p isa P) && return p
    return LocalQRegularizerProblem{F}(p.equation, p.k, p.q0, p.regularizer0)
end

#=
"""
    ResolventProblem(
        ensemble::MatrixEnsemble,
        z::Union{Number,AbstractMatrix},
        objective=TraceObjective();
        normalizeQ=false,
    )
    ResolventProblem{F}(ensemble, z, objective, normalizeQ) where {F}

Specification for the resolvent averaging problem at a single point in the complex plane

# Arguments

- `ensemble`: random matrix ensemble over which to average the resolvent
- `z`: coordinates in the complex plane at which to calculate the averaged resolvent. This
    can be either a single complex number or a matrix representing a rectangular grid in the
    complex plane. Use `missing` to exclude grid points from calculation.
- `objective`: objective to solve for. The default trace-based
    objective obtains the true averaged resolvent, but other objectives, including
    regularizer-based ones, are relevant for certain purposes.
- `normalizeQ`: whether to replace ``QQ^†`` with ``(QQ^† + Q^†Q) / 2`` in the
    self-consistent equations for the averaged resolvent, mimicking the case where
    ``QQ^† = Q^†Q``, i.e., where ``Q`` is normal. This is an attempt to get around an issue
    relating to order of limits that can arise for highly non-normal `Q` (see Ahmadian,
    Fumarola, and Miller (2015)). Performing this normalization is not mathematically sound,
    and in most cases gives answers that are subtly but unambiguously wrong; however, in
    certain experiments it seems to do the trick in terms of removing regions of spurious
    spectral support. The option is provided here to support further experimenting.
"""
struct ResolventProblem{
    # TODO: what's a better name than ResolventProblem?
    F,
    E<:MatrixEnsemble{F},
    Z<:Union{Complex{F},Matrix{Union{Missing,Complex{F}}}},
    O<:AbstractObjective{F},
}
    ensemble::E
    z::Z
    objective::O
    normalizeQ::Bool
    # TODO: would a list of points, i.e., unstructured meshes, be better than a rectangular
    # grid represented by a matrix? Big change, and it seems harder to approximate
    # derivatives and hence the spectral density, but finite element packages might have
    # built-in solutions for that

    function ResolventProblem{F}(ensemble, z, objective, normalizeQ) where {F}
        ensemble = adapt(F, ensemble)
        if z isa Number
            z = convert(Complex{F}, z)
        elseif z isa AbstractMatrix
            z = convert(Matrix{Union{Missing,Complex{F}}}, z)
        end
        objective = adapt(F, objective)
        return new{F,typeof(ensemble),typeof(z),typeof(objective)}(
            ensemble, z, objective, normalizeQ
        )
    end
end

function ResolventProblem(
    ensemble::MatrixEnsemble{F}, z, objective=TraceObjective(); normalizeQ=false
) where {F}
    return ResolventProblem{F}(ensemble, z, objective, normalizeQ)
end
=#

#= WIP automesh is not ready for this yet, and this would depend on LazySets.jl

ResolventProblem(objective, ensemble; kwargs...)
    return SpectralProblem(objective, automesh(ensemble), ensemble; kwargs...)
end
=#
