"""
    LocalResolventEquation(ensemble::MatrixEnsemble, z, normalizeQ=false)
    LocalResolventEquation{F}(ensemble, z, normalizeQ) where {F}

Specify the averaged resolvent equation for an ensemble at a point in the complex plane

# Arguments

- `ensemble`: random matrix ensemble over which to average the resolvent
- `z`: point in the complex plane at which to calculate the averaged resolvent
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
    normalizeQ::Bool

    function LocalResolventEquation{F}(ensemble, z, normalizeQ) where {F}
        ensemble = adapt(F, ensemble)
        return new{F,typeof(ensemble)}(ensemble, z, normalizeQ)
    end
end

function LocalResolventEquation(ensemble::MatrixEnsemble{F}, z, normalizeQ=false) where {F}
    return LocalResolventEquation{F}(ensemble, z, normalizeQ)
end

function Base.convert(
    ::Type{P}, eq::Q
) where {F,P<:LocalResolventEquation{F},Q<:LocalResolventEquation}
    (eq isa P) && return eq
    return LocalResolventEquation{F}(eq.ensemble, eq.z, eq.normalizeQ)
end

abstract type AbstractLocalResolventProblem{F<:AbstractFloat,E<:LocalResolventEquation{F}} end

"""
    LocalRegularizerProblem(eq::LocalResolventEquation, reg, q0, μ=1) where {F}
    LocalRegularizerProblem{F}(eq, reg, q0, μ)

Specify the problem of finding ``q`` given a regularizer in the averaged resolvent equation

`q0` is a starting guess for `q`. `μ` is an overall scaling factor for the degree of symmetry given by `eq.ensemble.T`. This
parameter is inserted by hand into the equations to facilitate a solution strategy whereby
the symmetry is annealed.
"""
struct LocalRegularizerProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    reg::Regularizer{F}
    q0::Complex{F}
    μ::F

    function LocalRegularizerProblem{F}(eq, reg, q0, μ) where {F}
        _checkμ(μ)
        eq = adapt(F, eq)
        return new{F,typeof(eq)}(eq, reg, q0, μ)
    end
end

function LocalRegularizerProblem(eq::LocalResolventEquation{F}, reg, q0, μ=one(F)) where {F}
    return LocalRegularizerProblem{F}(eq, reg, q0, μ)
end

function Base.convert(
    ::Type{P}, p::Q
) where {F,P<:LocalRegularizerProblem{F},Q<:LocalRegularizerProblem}
    (p isa P) && return p
    return LocalRegularizerProblem{F}(p.eq, p.reg, p.q0, p.μ)
end

"""
    LocalTraceProblem(eq::LocalResolventEquation, k, q0, reg0, μ=1)
    LocalTraceProblem{F}(eq, k, q0, reg0, μ)

Specify the problem of finding ``q,γ²`` given ``k`` in the averaged resolvent equation

Here, ``k`` is the target value of the normalized trace ``k = \\tr[(QQ^† + γ²I)⁻¹] / N``.

`q0` and `reg0` are starting guesses for ``q`` and one of the regularizing parameters
``γ²,δ²``. `μ` is an overall scaling factor for the degree of symmetry given by
`eq.ensemble.T`. This parameter is inserted by hand into the equations to facilitate
a solution strategy whereby the symmetry is annealed.
"""
struct LocalTraceProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    k::F
    q0::Complex{F}
    reg0::Regularizer{F}
    μ::F

    function LocalTraceProblem{F}(eq, k, q0, reg0, μ) where {F}
        _checkμ(μ)
        eq = adapt(F, eq)
        return new{F,typeof(eq)}(eq, k, q0, reg0)
    end
end

function LocalTraceProblem(eq::LocalResolventEquation{F}, k, q0, reg0, μ=one(F)) where {F}
    return LocalTraceProblem{F}(eq, k, q0, reg0, μ)
end

function Base.convert(
    ::Type{P}, p::Q
) where {F,P<:LocalTraceProblem{F},Q<:LocalTraceProblem}
    (p isa P) && return p
    return LocalTraceProblem{F}(p.eq, p.k, p.q0, p.reg0, p.μ)
end

function _checkμ(μ)
    msg = "symmetry scale factor μ not in [0, 1]"
    (!((μ isa Real) && (0 <= μ <= 1))) && throw(ArgumentError(msg))
    return nothing
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
