module Problems

using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer

using ..ArrayTools: ArrayTools

public AbstractResolventProblem,
    PTraceProblem, RegularizerProblem, ResolventEquation, issameproblem, update

"""
    ResolventEquation(ensemble::MatrixEnsemble; shiftmin=-Inf, normalizeQ=false)
    ResolventEquation{F}(ensemble, shiftmin, normalizeQ) where {F}
    ResolventEquation{F}(eq::ResolventEquation) where {F}

Specify the shared context for an averaged resolvent equation over an ensemble

# Arguments

  - `ensemble`: random matrix ensemble over which to average the resolvent
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
    shiftmin::F
    normalizeQ::Bool

    function common(::Type{F}, ensemble, shiftmin, normalizeQ) where {F}
        ensembleF = convert(MatrixEnsemble{F}, ensemble)
        return new{F,typeof(ensembleF)}(ensembleF, shiftmin, normalizeQ)
    end

    function ResolventEquation{F}(ensemble, shiftmin, normalizeQ) where {F}
        return common(F, ensemble, shiftmin, normalizeQ)
    end

    function ResolventEquation{F}(eq::ResolventEquation) where {F}
        return common(F, eq.ensemble, eq.shiftmin, eq.normalizeQ)
    end
end

function ResolventEquation(
    ensemble::MatrixEnsemble{F}; shiftmin=F(-Inf), normalizeQ=false
) where {F}
    return ResolventEquation{F}(ensemble, shiftmin, normalizeQ)
end

function update(
    e::ResolventEquation{F};
    ensemble=e.ensemble,
    shiftmin=e.shiftmin,
    normalizeQ=e.normalizeQ,
) where {F}
    return ResolventEquation{F}(ensemble, shiftmin, normalizeQ)
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

# Constructor helper
function checksscale(sscale)
    ((sscale isa Real) && (0 <= sscale <= 1)) && return nothing
    throw(ArgumentError("symmetry scale factor sscale not in [0, 1]"))
end

"""
    RegularizerProblem(eq::ResolventEquation, z, reg, init; sscale=1)
    RegularizerProblem{F}(eq, z, reg, init, sscale) where {F}
    RegularizerProblem{F}(problem::RegularizerProblem) where {F}

Specify the problem of finding ``q`` given an objective for the regularizer

`init.q` is a starting guess for `q`, and `init` can be any type as long as it has the
property `q` and this can be converted to the appropriate type.

# Arguments

  - `eq`: the resolvent equation (shared context)
  - `z`: point in the complex plane at which to calculate the averaged resolvent
  - `reg`: target regularizer value
  - `init`: initial guess for the solution
  - `sscale=1`: overall scaling factor for the symmetry parameter `eq.ensemble.S`. Inserted by
    hand into the equations to facilitate a solution strategy where the symmetry is annealed
    from zero to the target value.
"""
struct RegularizerProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    z::Complex{F}
    reg::Regularizer{F}
    init::@NamedTuple{q::Complex{F}}
    sscale::F

    function common(::Type{F}, eq, z, reg, init, sscale) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, z, reg, init, sscale)
    end

    function RegularizerProblem{F}(eq, z, reg, init, sscale) where {F}
        checksscale(sscale)
        return common(F, eq, z, reg, (; q=init.q), sscale)
    end

    function RegularizerProblem{F}(p::RegularizerProblem) where {F}
        return common(F, p.eq, p.z, p.reg, p.init, p.sscale)
    end
end

function RegularizerProblem(eq::ResolventEquation{F}, z, reg, init; sscale=one(F)) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return RegularizerProblem{F}(eq, z, reg, init, sscale)
end

function update(
    p::RegularizerProblem{F}; eq=p.eq, z=p.z, reg=p.reg, init=p.init, sscale=p.sscale
) where {F}
    return RegularizerProblem{F}(eq, z, reg, init, sscale)
end

## Comparisons, conversions
## ========================
function issameproblem(p::P, q::P) where {P<:RegularizerProblem}
    return (p.eq === q.eq) && (p.z === q.z) && (p.reg === q.reg) && (p.sscale === q.sscale)
end

function Base.convert(::Type{P}, p::RegularizerProblem) where {F,P<:RegularizerProblem{F}}
    return (p isa P) ? p : RegularizerProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::RegularizerProblem) where {F}
    return convert(RegularizerProblem{F}, eq)
end

"""
    PTraceProblem(eq::ResolventEquation, z, ptrace::Real, init; sscale=1)
    PTraceProblem{F}(eq, z, ptrace, init, sscale) where {F}
    PTraceProblem{F}(problem::PTraceProblem) where {F}

Specify the problem of finding ``γ²,q`` given an objective for the P-trace

Here, P-trace refers to the normalized trace ``k = \\tr[P] / N``, where
``P = (QQ^† + γ²I)⁻¹``.

`init.reg` and `init.q` are starting guesses for ``γ²,q``, and `init` can be any type as
long as it has these two properties and they can be converted to the appropriate types. The
initial guess for the regularizer can be specified in terms of either `Regularizer` type.

# Arguments

  - `eq`: the resolvent equation (shared context)
  - `z`: point in the complex plane at which to calculate the averaged resolvent
  - `ptrace`: target P-trace value
  - `init`: initial guess for the solution
  - `sscale=1`: overall scaling factor for the symmetry parameter `eq.ensemble.S`. Inserted by
    hand into the equations to facilitate a solution strategy where the symmetry is annealed
    from zero to the target value.
"""
struct PTraceProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    z::Complex{F}
    ptrace::F
    init::@NamedTuple{reg::Regularizer{F}, q::Complex{F}}
    sscale::F

    function common(::Type{F}, eq, z, ptrace, init, sscale) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, z, ptrace, init, sscale)
    end

    function PTraceProblem{F}(eq, z, ptrace, init, sscale) where {F}
        checksscale(sscale)
        return common(F, eq, z, ptrace, (; reg=init.reg, q=init.q), sscale)
    end

    function PTraceProblem{F}(p::PTraceProblem) where {F}
        return common(F, p.eq, p.z, p.ptrace, p.init, p.sscale)
    end
end

function PTraceProblem(eq::ResolventEquation{F}, z, ptrace, init; sscale=one(F)) where {F}
    # TODO: make init.q optional, use large shift asymptotics to calculate default value
    return PTraceProblem{F}(eq, z, ptrace, init, sscale)
end

function update(
    p::PTraceProblem{F}; eq=p.eq, z=p.z, ptrace=p.ptrace, init=p.init, sscale=p.sscale
) where {F}
    return PTraceProblem{F}(eq, z, ptrace, init, sscale)
end

## Comparisons, conversions
## ========================
function issameproblem(p::P, q::P) where {P<:PTraceProblem}
    return (p.eq === q.eq) &&
           (p.z === q.z) &&
           (p.ptrace === q.ptrace) &&
           (p.sscale === q.sscale)
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

    ensemble64, shiftmin64, normalizeQ = MatrixEnsemble(I, I, I, I / 2), 0.0, false
    eq64 = ResolventEquation{Float64}(ensemble64, shiftmin64, normalizeQ)

    # Traits
    @test eq64 isa ResolventEquation{Float64}

    # Defaults
    @test ResolventEquation(ensemble64) ==
        ResolventEquation{Float64}(ensemble64, -Inf, false)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        shiftminF = F(shiftmin64)
        eqF = ResolventEquation{F}(ensembleF, shiftminF, normalizeQ)

        # Update
        eqF_updated = update(eqF; normalizeQ=true)
        @test eqF_updated isa ResolventEquation{F}
        @test eqF_updated.normalizeQ == true
        @test eqF_updated.ensemble === eqF.ensemble
        @test eqF_updated.shiftmin === eqF.shiftmin

        # Floattype inferring constructor
        @test ResolventEquation(ensembleF; shiftmin=shiftminF, normalizeQ) isa
            ResolventEquation{F}

        # Floattype enforcing constructor
        @test ResolventEquation{F}(ensemble64, shiftmin64, normalizeQ) == eqF
        @test ResolventEquation{Float64}(ensembleF, shiftminF, normalizeQ) == eq64

        # Floattype conversions
        @test ResolventEquation{F}(eq64) == eqF
        @test ResolventEquation{Float64}(eqF) == eq64
        @test convert(ResolventEquation{F}, eq64) == eqF
        @test convert(ResolventEquation{Float64}, eqF) == eq64
        @test adaptrealtype(F, eq64) == eqF
        @test adaptrealtype(Float64, eqF) == eq64
    end
end

@testitem "*Problems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceProblem, RegularizerProblem, ResolventEquation, issameproblem, update
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    z64 = 0.5 + 0.5im
    sscale64 = 0.5
    reg64 = Regularizer(0.5, :gap)
    ptrace64, q0_64 = 1.0, complex(eps(), eps())
    init64 = (; reg=reg64, q=q0_64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        eqF = ResolventEquation(ensembleF)
        zF = Complex{F}(z64)
        sscaleF = F(sscale64)
        regF = Regularizer(F(reg64.value), reg64.type)
        ptraceF = F(ptrace64)
        q0_F = Complex{F}(q0_64)
        initF = (; reg=regF, q=q0_F)

        @testset "$(nameof(P))" for (P, obj64, objF, objprob) in [
            (RegularizerProblem, reg64, regF, :reg),
            (PTraceProblem, ptrace64, ptraceF, :ptrace),
        ]
            problem64 = P(eq64, z64, obj64, init64; sscale=sscale64)
            problemF = P(eqF, zF, objF, initF; sscale=sscaleF)

            # Traits
            @test problem64 isa P{Float64}
            @test problemF isa P{F}

            # Update and comparison
            init_updated = merge(init64, (; q=(-q0_64)))
            problemF_updated = update(problemF; init=init_updated)
            @test problemF_updated isa P{F}
            @test problemF_updated.init.q == -q0_F
            @test problemF_updated.eq === problemF.eq
            @test problemF_updated.z === problemF.z
            @test problemF_updated.sscale === problemF.sscale
            @test getproperty(problemF_updated, objprob) === getproperty(problemF, objprob)
            @test issameproblem(problemF, problemF_updated)

            # Different type, same value => not the same
            @test issameproblem(problemF, problem64) == (F === Float64)

            # Floattype inferring constructor
            @test problemF isa P{F}

            # Floattype enforcing constructor
            @test P{F}(eq64, z64, obj64, init64, sscale64) === problemF
            @test P{Float64}(eqF, zF, objF, initF, sscaleF) === problem64

            # Floattype conversions
            @test P{F}(problem64) === problemF
            @test P{Float64}(problemF) === problem64
            @test convert(P{F}, problem64) === problemF
            @test convert(P{Float64}, problemF) === problem64
            @test adaptrealtype(F, problem64) === problemF
            @test adaptrealtype(Float64, problemF) === problem64
        end
    end

    # sscale ∈ [0, 1]
    @test_throws ArgumentError RegularizerProblem(eq64, z64, reg64, init64; sscale=-0.1)
    @test_throws ArgumentError RegularizerProblem(eq64, z64, reg64, init64; sscale=1.1)
    @test_throws ArgumentError RegularizerProblem(eq64, z64, reg64, init64; sscale=0.5im)
    @test_throws ArgumentError PTraceProblem(eq64, z64, ptrace64, init64; sscale=-0.1)
    @test_throws ArgumentError PTraceProblem(eq64, z64, ptrace64, init64; sscale=1.1)
    @test_throws ArgumentError PTraceProblem(eq64, z64, ptrace64, init64; sscale=0.5im)
end
