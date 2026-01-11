module Problems

using TestItems: @testitem

using ..Ensembles: MatrixEnsemble
using ..Public: @public
using ..Regularizers: Regularizer

@public(
    AbstractLocalResolventProblem,
    LocalResolventEquation,
    LocalRegularizerProblem,
    LocalTraceProblem,
)

"""
    LocalResolventEquation(ensemble::MatrixEnsemble, z, normalizeQ=false)
    LocalResolventEquation{F}(ensemble, z, normalizeQ) where {F}
    LocalResolventEquation{F}(eq::LocalResolventEquation) where {F}

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
        ensembleF = MatrixEnsemble{F}(ensemble)
        return new{F,typeof(ensembleF)}(ensembleF, z, normalizeQ)
    end
end

function LocalResolventEquation(ensemble::MatrixEnsemble{F}, z, normalizeQ=false) where {F}
    return LocalResolventEquation{F}(ensemble, z, normalizeQ)
end

function LocalResolventEquation{F}(eq::LocalResolventEquation) where {F}
    return LocalResolventEquation{F}(eq.ensemble, eq.z, eq.normalizeQ)
end

function Base.convert(
    ::Type{P}, eq::LocalResolventEquation
) where {F,P<:LocalResolventEquation{F}}
    return (eq isa P) ? eq : LocalResolventEquation{F}(eq)
end

abstract type AbstractLocalResolventProblem{F<:AbstractFloat,E<:LocalResolventEquation{F}} end

"""
    LocalRegularizerProblem(eq::LocalResolventEquation, reg, q0, qscale=1)
    LocalRegularizerProblem{F}(eq, reg, q0, qscale) where {F}
    LocalRegularizerProblem{F}(problem::LocalRegularizerProblem) where {F}

Specify the problem of finding ``q`` given a regularizer in the averaged resolvent equation

`q0` is a starting guess for `q`. `qscale` is an overall scaling factor for the degree of
symmetry given by `eq.ensemble.T`. This parameter is inserted by hand into the equations to
facilitate a solution strategy where the symmetry is annealed from below.
"""
struct LocalRegularizerProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    reg::Regularizer{F}
    q0::Complex{F}
    qscale::F

    function LocalRegularizerProblem{F}(eq, reg, q0, qscale) where {F}
        _checkqscale(qscale)
        eqF = LocalResolventEquation{F}(eq)
        return new{F,typeof(eqF)}(eqF, reg, q0, qscale)
    end
end

function LocalRegularizerProblem(
    eq::LocalResolventEquation{F}, reg, q0, qscale=one(F)
) where {F}
    return LocalRegularizerProblem{F}(eq, reg, q0, qscale)
end

function LocalRegularizerProblem{F}(problem::LocalRegularizerProblem) where {F}
    return LocalRegularizerProblem{F}(problem.eq, problem.reg, problem.q0, problem.qscale)
end

function Base.convert(
    ::Type{P}, p::LocalRegularizerProblem
) where {F,P<:LocalRegularizerProblem{F}}
    return (p isa P) ? p : LocalRegularizerProblem{F}(p)
end

function Base.convert(
    ::Type{AbstractLocalResolventProblem{F}}, p::LocalRegularizerProblem
) where {F}
    return convert(LocalRegularizerProblem{F}, p)
end

"""
    LocalTraceProblem(eq::LocalResolventEquation, k, reg0, q0, qscale=1)
    LocalTraceProblem{F}(eq, k, reg0, q0, qscale) where {F}
    LocalTraceProblem{F}(problem::LocalTraceProblem) where {F}

Specify the problem of finding ``γ²,q`` given ``k`` in the averaged resolvent equation

Here, ``k`` is the target value of the normalized trace ``k = \\tr[(QQ^† + γ²I)⁻¹] / N``.

`reg0` and `q0` are starting guesses for ``γ²,q``. The initial guess for the regularizer can
be specified in terms of either `Regularizer` type. `qscale` is an overall scaling factor
for the degree of symmetry given by `eq.ensemble.T`. This parameter is inserted by hand into
the equations to facilitate a solution strategy where the symmetry is annealed from below.
"""
struct LocalTraceProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    k::F
    reg0::Regularizer{F}
    q0::Complex{F}
    qscale::F

    function LocalTraceProblem{F}(eq, k, reg0, q0, qscale) where {F}
        _checkqscale(qscale)
        eqF = LocalResolventEquation{F}(eq)
        return new{F,typeof(eqF)}(eqF, k, reg0, q0, qscale)
    end
end

function LocalTraceProblem(eq::LocalResolventEquation{F}, k, reg0, q0, qscale=one(F)) where {F}
    return LocalTraceProblem{F}(eq, k, reg0, q0, qscale)
end

function LocalTraceProblem{F}(problem::LocalTraceProblem) where {F}
    p = LocalTraceProblem{F}(problem.eq, problem.k, problem.reg0, problem.q0, problem.qscale)
    return p
end

function Base.convert(::Type{P}, p::LocalTraceProblem) where {F,P<:LocalTraceProblem{F}}
    return (p isa P) ? p : LocalTraceProblem{F}(p)
end

function Base.convert(::Type{AbstractLocalResolventProblem{F}}, p::LocalTraceProblem) where {F}
    return convert(LocalTraceProblem{F}, p)
end

function _checkqscale(qscale)
    if (qscale isa Real) && (0 <= qscale <= 1)
        return nothing
    end
    throw(ArgumentError("symmetry scale factor qscale not in [0, 1]"))
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

@testitem "LocalResolventEquation" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation

    using LinearAlgebra: I

    ensemble64, z64, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, false
    eq64 = LocalResolventEquation(ensemble64, z64, normalizeQ)

    # Traits
    @test eq64 isa LocalResolventEquation{Float64}

    # Defaults
    @test (
        LocalResolventEquation(ensemble64, z64)
        ===
        LocalResolventEquation(ensemble64, z64, false)
    )

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I)
        zF = F(z64)
        eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)

        # Floattype inferring constructor
        @test eqF isa LocalResolventEquation{F}

        # Floattype enforcing constructor
        @test LocalResolventEquation{F}(ensemble64, z64, normalizeQ) === eqF
        @test LocalResolventEquation{Float64}(ensembleF, zF, normalizeQ) === eq64

        # Floattype conversions
        @test convert(LocalResolventEquation{F}, eq64) === eqF
        @test convert(LocalResolventEquation{Float64}, eqF) === eq64
        @test LocalResolventEquation{F}(eq64) === eqF
        @test LocalResolventEquation{Float64}(eqF) === eq64
    end
end

@testitem "Local*Problems" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: AbstractLocalResolventProblem,
        LocalResolventEquation, LocalRegularizerProblem, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    ensemble64, z64, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, false
    eq64 = LocalResolventEquation(ensemble64, z64, normalizeQ)
    reg_value64, reg_type = 0.5, :gap
    reg64 = Regularizer(reg_value64, reg_type)
    k64, q64, qscale64 = 1.0, complex(eps(), eps()), 1.0

    @testset "$(nameof(P))" for (P, params64) in [
        (LocalRegularizerProblem, ()),
        (LocalTraceProblem, (k64,))
    ]
        problem64 = P(eq64, params64..., reg64, q64, qscale64)

        # Traits
        @test problem64 isa P{Float64}

        # Defaults
        @test P(eq64, params64..., reg64, q64) === P(eq64, params64..., reg64, q64, 1)

        # Errors: qscale ∈ [0, 1]
        @test_throws ArgumentError P(eq64, params64..., reg64, q64, -0.1)
        @test_throws ArgumentError P(eq64, params64..., reg64, q64, 1.1)
        @test_throws ArgumentError P(eq64, params64..., reg64, q64, 0.5im)

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(I, I, I, I)
            zF = F(z64)
            eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)
            reg_valueF = F(reg_value64)
            regF = Regularizer(reg_valueF, reg_type)
            qF = Complex{F}(q64)
            qscaleF = F(qscale64)
            paramsF = F.(params64)
            problemF = P(eqF, paramsF..., regF, qF, qscaleF)

            # Floattype inferring constructor
            @test problemF isa P{F}

            # Floattype enforcing constructor
            @test P{F}(eq64, params64..., reg64, q64, qscale64) === problemF
            @test P{Float64}(eqF, paramsF..., regF, qF, qscaleF) === problem64

            # Floattype conversions
            @test convert(P{F}, problem64) === problemF
            @test convert(P{Float64}, problemF) === problem64
            @test convert(AbstractLocalResolventProblem{F}, problem64) === problemF
            @test convert(AbstractLocalResolventProblem{Float64}, problemF) === problem64
            @test P{F}(problem64) === problemF
            @test P{Float64}(problemF) === problem64
        end
    end
end

end  # module Problems
