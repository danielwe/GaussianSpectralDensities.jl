module Problems

using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer

public AbstractLocalResolventProblem,
    LocalRegularizerProblem, LocalResolventEquation, LocalTraceProblem, update

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

## Comparisons, conversions
## ========================
function Base.:(==)(a::E, b::E) where {E<:LocalResolventEquation}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(
    ::Type{P}, eq::LocalResolventEquation
) where {F,P<:LocalResolventEquation{F}}
    return (eq isa P) ? eq : LocalResolventEquation{F}(eq)
end

abstract type AbstractLocalResolventProblem{F<:AbstractFloat,E<:LocalResolventEquation{F}} end

"""
    LocalRegularizerProblem(eq::LocalResolventEquation, reg, init, qscale=1)
    LocalRegularizerProblem{F}(eq, reg, init, qscale) where {F}
    LocalRegularizerProblem{F}(problem::LocalRegularizerProblem) where {F}

Specify the problem of finding ``q`` given a regularizer in the averaged resolvent equation

`init.q` is a starting guess for `q`, and `init` can be anything as long as it has the
property `q` and this can be converted to the appropriate type. `qscale` is an overall
scaling factor for the degree of symmetry given by `eq.ensemble.T`. This parameter is
inserted by hand into the equations to facilitate a solution strategy where the symmetry is
annealed from below.
"""
struct LocalRegularizerProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    reg::Regularizer{F}
    init::@NamedTuple{q::Complex{F}}
    qscale::F

    function LocalRegularizerProblem{F}(eq, reg, init, qscale) where {F}
        _checkqscale(qscale)
        eqF = LocalResolventEquation{F}(eq)
        init_filtered = (; q=init.q)
        return new{F,typeof(eqF)}(eqF, reg, init_filtered, qscale)
    end
end

function LocalRegularizerProblem(
    eq::LocalResolventEquation{F}, reg, init, qscale=one(F)
) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return LocalRegularizerProblem{F}(eq, reg, init, qscale)
end

function LocalRegularizerProblem{F}(problem::LocalRegularizerProblem) where {F}
    return LocalRegularizerProblem{F}(problem.eq, problem.reg, problem.init, problem.qscale)
end

function update(
    problem::LocalRegularizerProblem{F};
    eq=problem.eq,
    reg=problem.reg,
    init=problem.init,
    qscale=problem.qscale,
) where {F}
    return LocalRegularizerProblem{F}(problem.eq, reg, init, qscale)
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
    LocalTraceProblem(eq::LocalResolventEquation, k, init, qscale=1)
    LocalTraceProblem{F}(eq, k, init, qscale) where {F}
    LocalTraceProblem{F}(problem::LocalTraceProblem) where {F}

Specify the problem of finding ``γ²,q`` given ``k`` in the averaged resolvent equation

Here, ``k`` is the target value of the normalized trace ``k = \\tr[(QQ^† + γ²I)⁻¹] / N``.

`init.reg` and `init.q` are starting guesses for ``γ²,q``, and `init` can be anything as
long as it has these two properties property `q` and they can be converted to the
appropriate types. The initial guess for the regularizer can be specified in terms of either
`Regularizer` type. `qscale` is an overall scaling factor for the degree of symmetry given
by `eq.ensemble.T`. This parameter is inserted by hand into the equations to facilitate a
solution strategy where the symmetry is annealed from below.
"""
struct LocalTraceProblem{F,E} <: AbstractLocalResolventProblem{F,E}
    eq::E
    k::F
    init::@NamedTuple{reg::Regularizer{F}, q::Complex{F}}
    qscale::F

    function LocalTraceProblem{F}(eq, k, init, qscale) where {F}
        _checkqscale(qscale)
        eqF = LocalResolventEquation{F}(eq)
        init_filtered = (; reg=init.reg, q=init.q)
        return new{F,typeof(eqF)}(eqF, k, init_filtered, qscale)
    end
end

function LocalTraceProblem(eq::LocalResolventEquation{F}, k, init, qscale=one(F)) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return LocalTraceProblem{F}(eq, k, init, qscale)
end

function LocalTraceProblem{F}(problem::LocalTraceProblem) where {F}
    p = LocalTraceProblem{F}(problem.eq, problem.k, problem.init, problem.qscale)
    return p
end

function update(
    problem::LocalTraceProblem{F};
    eq=problem.eq,
    k=problem.k,
    init=problem.init,
    qscale=problem.qscale,
) where {F}
    return LocalTraceProblem{F}(eq, k, init, qscale)
end

function Base.convert(::Type{P}, p::LocalTraceProblem) where {F,P<:LocalTraceProblem{F}}
    return (p isa P) ? p : LocalTraceProblem{F}(p)
end

function Base.convert(
    ::Type{AbstractLocalResolventProblem{F}}, p::LocalTraceProblem
) where {F}
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

end  # module Problems

@testitem "LocalResolventEquation" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation

    using LinearAlgebra: I

    ensemble64, z64, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, false
    eq64 = LocalResolventEquation(ensemble64, z64, normalizeQ)

    # Traits
    @test eq64 isa LocalResolventEquation{Float64}

    # Defaults
    @test LocalResolventEquation(ensemble64, z64) ==
        LocalResolventEquation(ensemble64, z64, false)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I)
        zF = F(z64)
        eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)

        # Floattype inferring constructor
        @test eqF isa LocalResolventEquation{F}

        # Floattype enforcing constructor
        @test LocalResolventEquation{F}(ensemble64, z64, normalizeQ) == eqF
        @test LocalResolventEquation{Float64}(ensembleF, zF, normalizeQ) == eq64

        # Floattype conversions
        @test convert(LocalResolventEquation{F}, eq64) == eqF
        @test convert(LocalResolventEquation{Float64}, eqF) == eq64
        @test LocalResolventEquation{F}(eq64) == eqF
        @test LocalResolventEquation{Float64}(eqF) == eq64
    end
end

@testitem "Local*Problems" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        AbstractLocalResolventProblem,
        LocalRegularizerProblem,
        LocalResolventEquation,
        LocalTraceProblem,
        update
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I)
    z64 = 0.0im
    eq64 = LocalResolventEquation(ensemble64, z64)
    reg64 = Regularizer(0.5, :gap)
    reg_alt64 = Regularizer(0.5, :shift)
    k64, q0_64, qscale64 = 1.0, complex(eps(), eps()), 0.5
    k_alt64 = 0.5
    init64 = (; reg=reg64, q=q0_64)
    init_alt64 = (; reg=reg_alt64, q=q0_64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I)
        zF = F(z64)
        eqF = LocalResolventEquation(ensembleF, zF)
        regF = Regularizer(F(reg64.value), reg64.type)
        reg_altF = Regularizer(F(reg_alt64.value), reg64.type)
        kF = F(k64)
        k_altF = F(k64)
        q0_F = Complex{F}(q0_64)
        qscaleF = F(qscale64)
        initF = (; reg=regF, q=q0_F)
        init_altF = (; reg=reg_altF, q=q0_F)

        @testset "$(nameof(P))" for (P, pname, param64, param_alt64, paramF, param_altF) in
                                    [
            (LocalRegularizerProblem, :reg, reg64, reg_alt64, regF, reg_altF),
            (LocalTraceProblem, :k, k64, k_alt64, kF, k_altF),
        ]
            problem64 = P(eq64, param64, init64, qscale64)
            problemF = P(eqF, paramF, initF, qscaleF)

            # Traits
            @test problem64 isa P{Float64}
            @test problemF isa P{F}

            # Defaults
            problem_alt64 = P(eq64, param_alt64, init_alt64, 1)
            problem_altF = P(eqF, param_altF, init_altF, 1)
            @test P(eq64, param_alt64, init_alt64) === problem_alt64
            @test P(eqF, param_altF, init_altF) === problem_altF

            # Update
            @test problem_alt64 ===
                update(problem64; pname => param_alt64, init=init_alt64, qscale=1)
            @test problem_altF ===
                update(problemF; pname => param_altF, init=init_altF, qscale=1)

            # Floattype inferring constructor
            @test problemF isa P{F}

            # Floattype enforcing constructor
            @test P{F}(eq64, param64, init64, qscale64) === problemF
            @test P{Float64}(eqF, paramF, initF, qscaleF) === problem64

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

@testitem "Local*Problem errors" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        AbstractLocalResolventProblem,
        LocalRegularizerProblem,
        LocalResolventEquation,
        LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    eq = LocalResolventEquation(MatrixEnsemble(I, I, I, I), 0.0im, false)
    reg = Regularizer(0.5, :gap)
    init = (; reg, q=complex(eps(), eps()))
    k = 1.0

    # qscale ∈ [0, 1]
    @test_throws ArgumentError LocalRegularizerProblem(eq, reg, init, -0.1)
    @test_throws ArgumentError LocalRegularizerProblem(eq, reg, init, 1.1)
    @test_throws ArgumentError LocalRegularizerProblem(eq, reg, init, 0.5im)
    @test_throws ArgumentError LocalTraceProblem(eq, k, init, -0.1)
    @test_throws ArgumentError LocalTraceProblem(eq, k, init, 1.1)
    @test_throws ArgumentError LocalTraceProblem(eq, k, init, 0.5im)
end
