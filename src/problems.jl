module Problems

using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer

using ..ArrayTools: ArrayTools

public AbstractResolventProblem,
    PTraceProblem, RegularizerProblem, ResolventEquation, issameproblem, update

"""
    ResolventEquation(ensemble::MatrixEnsemble, z; sscale=1, shiftmin=-Inf, normalizeQ=false)
    ResolventEquation{F}(ensemble, z, sscale, shiftmin, normalizeQ) where {F}
    ResolventEquation{F}(eq::ResolventEquation) where {F}

Specify the averaged resolvent equation for an ensemble at a point in the complex plane

# Arguments

  - `ensemble`: random matrix ensemble over which to average the resolvent
  - `z`: point in the complex plane at which to calculate the averaged resolvent
  - `sscale`: overall scaling factor for the symmetry parameter `eq.ensemble.S`. Inserted by
    hand into the equations to facilitate a solution strategy where the symmetry is annealed
    from zero to the target value.
  - `shiftmin=-Inf`: lower bound on the value of the regularizer shift ``γ²``. Solutions with
    `shift < shiftmin` are marked as failed. Strictly speaking, the averaged resolvent
    equations are derived assuming ``γ² ≥ 0``, suggesting `shiftmin = 0`, but in practice it
    is useful to allow analytic continuation to negative values such that the averaged
    resolvent can be interpolated between grid points in regions where ``γ²`` vanishes
    rapidly. However, it's best to abort before reaching too large negative values, as the
    annealing may encounter a bifurcation that annihilates the solution branch.
  - `normalizeQ=false`: whether to replace ``QQ^†`` with ``(QQ^† + Q^†Q) / 2`` in the
    self-consistent equations for the averaged resolvent, mimicking the case where
    ``QQ^† = Q^†Q``, i.e., where ``Q`` is normal. This is an attempt to get around an issue
    relating to order of limits that can arise for highly non-normal `Q` (see Ahmadian,
    Fumarola, and Miller (2015)). Performing this normalization is not mathematically sound,
    and in most cases gives answers that are subtly but unambiguously wrong; however, in
    certain experiments it seems to do the trick in terms of removing regions of spurious
    spectral support. The option is provided here to support further experimenting.
"""
struct ResolventEquation{F,E<:MatrixEnsemble{F}}
    ensemble::E
    z::Complex{F}
    sscale::F
    shiftmin::F
    normalizeQ::Bool

    function common(::Type{F}, ensemble, z, sscale, shiftmin, normalizeQ) where {F}
        ensembleF = convert(MatrixEnsemble{F}, ensemble)
        return new{F,typeof(ensembleF)}(ensembleF, z, sscale, shiftmin, normalizeQ)
    end

    function ResolventEquation{F}(ensemble, z, sscale, shiftmin, normalizeQ) where {F}
        if !((sscale isa Real) && (0 <= sscale <= 1))
            throw(ArgumentError("symmetry scale factor sscale not in [0, 1]"))
        end
        return common(F, ensemble, z, sscale, shiftmin, normalizeQ)
    end

    function ResolventEquation{F}(eq::ResolventEquation) where {F}
        return common(F, eq.ensemble, eq.z, eq.sscale, eq.shiftmin, eq.normalizeQ)
    end
end

function ResolventEquation(
    ensemble::MatrixEnsemble{F}, z; sscale=one(F), shiftmin=F(-Inf), normalizeQ=false
) where {F}
    return ResolventEquation{F}(ensemble, z, sscale, shiftmin, normalizeQ)
end

function update(
    e::ResolventEquation{F};
    ensemble=e.ensemble,
    z=e.z,
    sscale=e.sscale,
    shiftmin=e.shiftmin,
    normalizeQ=e.normalizeQ,
) where {F}
    return ResolventEquation{F}(ensemble, z, sscale, shiftmin, normalizeQ)
end

## Comparisons, conversions
## ========================
function Base.:(==)(a::E, b::E) where {E<:ResolventEquation}
    return all(p -> getproperty(a, p) == getproperty(b, p), propertynames(a))
end

function Base.convert(::Type{P}, eq::ResolventEquation) where {F,P<:ResolventEquation{F}}
    return (eq isa P) ? eq : ResolventEquation{F}(eq)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::ResolventEquation) where {F}
    return convert(ResolventEquation{F}, eq)
end

## Problem types
## =============
abstract type AbstractResolventProblem{F,E<:ResolventEquation{F}} end

# True if two problems are equal except for their initial values
issameproblem(::AbstractResolventProblem, ::AbstractResolventProblem) = false

"""
    RegularizerProblem(eq::ResolventEquation, reg::Regularizer, init)
    RegularizerProblem{F}(eq, reg, init) where {F}
    RegularizerProblem{F}(problem::RegularizerProblem) where {F}

Specify the problem of finding ``q`` given an objective for the regularizer

`init.q` is a starting guess for `q`, and `init` can be any type as long as it has the
property `q` and this can be converted to the appropriate type.
"""
struct RegularizerProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    reg::Regularizer{F}
    init::@NamedTuple{q::Complex{F}}

    function common(::Type{F}, eq, reg, init) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, reg, init)
    end

    function RegularizerProblem{F}(eq, reg, init) where {F}
        return common(F, eq, reg, (; q=init.q))
    end

    function RegularizerProblem{F}(p::RegularizerProblem) where {F}
        return common(F, p.eq, p.reg, p.init)
    end
end

function RegularizerProblem(eq::ResolventEquation{F}, reg, init) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return RegularizerProblem{F}(eq, reg, init)
end

function update(p::RegularizerProblem{F}; eq=p.eq, reg=p.reg, init=p.init) where {F}
    return RegularizerProblem{F}(eq, reg, init)
end

## Comparisons, conversions
## ========================
function issameproblem(p::P, q::P) where {P<:RegularizerProblem}
    return (p.eq === q.eq) && (p.reg === q.reg)
end

function Base.convert(::Type{P}, p::RegularizerProblem) where {F,P<:RegularizerProblem{F}}
    return (p isa P) ? p : RegularizerProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::RegularizerProblem) where {F}
    return convert(RegularizerProblem{F}, eq)
end

"""
    PTraceProblem(eq::ResolventEquation, ptrace::Real, init)
    PTraceProblem{F}(eq, ptrace, init) where {F}
    PTraceProblem{F}(problem::PTraceProblem) where {F}

Specify the problem of finding ``γ²,q`` given an objective for the P-trace

Here, P-trace refers to the normalized trace ``k = \\tr[P] / N``, where
``P = (QQ^† + γ²I)⁻¹``.

`init.reg` and `init.q` are starting guesses for ``γ²,q``, and `init` can be any type as
long as it has these two properties and they can be converted to the appropriate types. The
initial guess for the regularizer can be specified in terms of either `Regularizer` type.
"""
struct PTraceProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    ptrace::F
    init::@NamedTuple{reg::Regularizer{F}, q::Complex{F}}

    function common(::Type{F}, eq, ptrace, init) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, ptrace, init)
    end

    function PTraceProblem{F}(eq, ptrace, init) where {F}
        return common(F, eq, ptrace, (; reg=init.reg, q=init.q))
    end

    function PTraceProblem{F}(p::PTraceProblem) where {F}
        return common(F, p.eq, p.ptrace, p.init)
    end
end

function PTraceProblem(eq::ResolventEquation{F}, ptrace, init) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return PTraceProblem{F}(eq, ptrace, init)
end

function update(p::PTraceProblem{F}; eq=p.eq, ptrace=p.ptrace, init=p.init) where {F}
    return PTraceProblem{F}(eq, ptrace, init)
end

## Comparisons, conversions
## ========================
function issameproblem(p::P, q::P) where {P<:PTraceProblem}
    return (p.eq === q.eq) && (p.ptrace === q.ptrace)
end

function Base.convert(::Type{P}, p::PTraceProblem) where {F,P<:PTraceProblem{F}}
    return (p isa P) ? p : PTraceProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::PTraceProblem) where {F}
    return convert(PTraceProblem{F}, eq)
end

end  # module Problems

@testitem "ResolventEquation" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: ResolventEquation, update

    using LinearAlgebra: I

    ensemble64, z64, sscale64, shiftmin64, normalizeQ = MatrixEnsemble(I, I, I, I / 2),
    0.0im, 0.5, 0.0,
    false
    eq64 = ResolventEquation{Float64}(ensemble64, z64, sscale64, shiftmin64, normalizeQ)

    # Traits
    @test eq64 isa ResolventEquation{Float64}

    # Defaults
    @test ResolventEquation(ensemble64, z64) ==
        ResolventEquation{Float64}(ensemble64, z64, 1.0, -Inf, false)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        zF = Complex{F}(z64)
        sscaleF = F(sscale64)
        shiftminF = F(shiftmin64)
        eqF = ResolventEquation{F}(ensembleF, zF, sscaleF, shiftminF, normalizeQ)

        # Update
        eqF_updated = update(eqF; sscale=1)
        @test eqF_updated isa ResolventEquation{F}
        @test eqF_updated.sscale == 1
        @test eqF_updated.ensemble === eqF.ensemble
        @test eqF_updated.z === eqF.z
        @test eqF_updated.shiftmin === eqF.shiftmin
        @test eqF_updated.normalizeQ === eqF.normalizeQ

        # Floattype inferring constructor
        @test ResolventEquation(
            ensembleF, zF; sscale=sscaleF, shiftmin=shiftminF, normalizeQ
        ) isa ResolventEquation{F}

        # Floattype enforcing constructor
        @test ResolventEquation{F}(ensemble64, z64, sscale64, shiftmin64, normalizeQ) == eqF
        @test ResolventEquation{Float64}(ensembleF, zF, sscaleF, shiftminF, normalizeQ) ==
            eq64

        # Floattype conversions
        @test ResolventEquation{F}(eq64) == eqF
        @test ResolventEquation{Float64}(eqF) == eq64
        @test convert(ResolventEquation{F}, eq64) == eqF
        @test convert(ResolventEquation{Float64}, eqF) == eq64
        @test adaptrealtype(F, eq64) == eqF
        @test adaptrealtype(Float64, eqF) == eq64
    end

    # sscale ∈ [0, 1]
    @test_throws ArgumentError ResolventEquation(ensemble64, z64; sscale=-0.1)
    @test_throws ArgumentError ResolventEquation(ensemble64, z64; sscale=1.1)
    @test_throws ArgumentError ResolventEquation(ensemble64, z64; sscale=0.5im)
end

@testitem "*Problems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceProblem, RegularizerProblem, ResolventEquation, issameproblem, update
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    z64 = 0.0im
    eq64 = ResolventEquation(ensemble64, z64)
    reg64 = Regularizer(0.5, :gap)
    ptrace64, q0_64 = 1.0, complex(eps(), eps())
    init64 = (; reg=reg64, q=q0_64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        zF = Complex{F}(z64)
        eqF = ResolventEquation(ensembleF, zF)
        regF = Regularizer(F(reg64.value), reg64.type)
        ptraceF = F(ptrace64)
        q0_F = Complex{F}(q0_64)
        initF = (; reg=regF, q=q0_F)

        @testset "$(nameof(P))" for (P, obj64, objF, objprob) in [
            (RegularizerProblem, reg64, regF, :reg),
            (PTraceProblem, ptrace64, ptraceF, :ptrace),
        ]
            problem64 = P(eq64, obj64, init64)
            problemF = P(eqF, objF, initF)

            # Traits
            @test problem64 isa P{Float64}
            @test problemF isa P{F}

            # Update and comparison
            init_updated = merge(init64, (; q=(-q0_64)))
            problemF_updated = update(problemF; init=init_updated)
            @test problemF_updated isa P{F}
            @test problemF_updated.init.q == -q0_F
            @test problemF_updated.eq === problemF.eq
            @test getproperty(problemF_updated, objprob) === getproperty(problemF, objprob)
            @test issameproblem(problemF, problemF_updated)

            # Different type, same value => not the same
            @test issameproblem(problemF, problem64) == (F === Float64)

            # Floattype inferring constructor
            @test problemF isa P{F}

            # Floattype enforcing constructor
            @test P{F}(eq64, obj64, init64) === problemF
            @test P{Float64}(eqF, objF, initF) === problem64

            # Floattype conversions
            @test P{F}(problem64) === problemF
            @test P{Float64}(problemF) === problem64
            @test convert(P{F}, problem64) === problemF
            @test convert(P{Float64}, problemF) === problem64
            @test adaptrealtype(F, problem64) === problemF
            @test adaptrealtype(Float64, problemF) === problem64
        end
    end
end
