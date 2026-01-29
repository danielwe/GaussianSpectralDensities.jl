module Problems

using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer

using ..ArrayTools: ArrayTools, adaptrealtype

public AbstractResolventProblem,
    PTraceProblem, PointProblem, RegularizerProblem, ResolventEquation, update

"""
    ResolventEquation(ensemble::MatrixEnsemble; shiftmin=-Inf, normalizeQ=false, sscale=1)
    ResolventEquation{F}(ensemble, shiftmin, normalizeQ, sscale) where {F}
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
  - `sscale=1`: overall scaling factor for the symmetry parameter `ensemble.S`. Inserted by
    hand into the equations to facilitate a solution strategy where the symmetry is annealed
    from zero to the target value.
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
    sscale::F
    normalizeQ::Bool

    function common(::Type{F}, ensemble, shiftmin, sscale, normalizeQ) where {F}
        ensembleF = convert(MatrixEnsemble{F}, ensemble)
        return new{F,typeof(ensembleF)}(ensembleF, shiftmin, sscale, normalizeQ)
    end

    function ResolventEquation{F}(ensemble, shiftmin, sscale, normalizeQ) where {F}
        if !((sscale isa Real) && (0 <= sscale <= 1))
            throw(ArgumentError("symmetry scale factor sscale not in [0, 1]"))
        end
        return common(F, ensemble, shiftmin, sscale, normalizeQ)
    end

    function ResolventEquation{F}(eq::ResolventEquation) where {F}
        return common(F, eq.ensemble, eq.shiftmin, eq.sscale, eq.normalizeQ)
    end
end

function ResolventEquation(
    ensemble::MatrixEnsemble{F}; shiftmin=F(-Inf), sscale=one(F), normalizeQ=false
) where {F}
    return ResolventEquation{F}(ensemble, shiftmin, sscale, normalizeQ)
end

function update(
    e::ResolventEquation{F};
    ensemble=e.ensemble,
    shiftmin=e.shiftmin,
    sscale=e.sscale,
    normalizeQ=e.normalizeQ,
) where {F}
    return ResolventEquation{F}(ensemble, shiftmin, sscale, normalizeQ)
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

"""
    RegularizerProblem(eq::ResolventEquation, reg)
    RegularizerProblem{F}(eq, reg) where {F}
    RegularizerProblem{F}(problem::RegularizerProblem) where {F}

Specify the problem of finding ``q`` given an objective for the regularizer

# Arguments

  - `eq`: the resolvent equation (shared context)
  - `reg`: target regularizer value
"""
struct RegularizerProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    reg::Regularizer{F}

    function RegularizerProblem{F}(eq, reg) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, reg)
    end
end

function RegularizerProblem(eq::ResolventEquation{F}, reg) where {F}
    return RegularizerProblem{F}(eq, reg)
end

function RegularizerProblem{F}(p::RegularizerProblem) where {F}
    return RegularizerProblem{F}(p.eq, p.reg)
end

function update(p::RegularizerProblem{F}; eq=p.eq, reg=p.reg) where {F}
    return RegularizerProblem{F}(eq, reg)
end

## Conversions
## ========================
function Base.convert(::Type{P}, p::RegularizerProblem) where {F,P<:RegularizerProblem{F}}
    return (p isa P) ? p : RegularizerProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::RegularizerProblem) where {F}
    return convert(RegularizerProblem{F}, eq)
end

"""
    PTraceProblem(eq::ResolventEquation, ptrace::Real)
    PTraceProblem{F}(eq, ptrace) where {F}
    PTraceProblem{F}(problem::PTraceProblem) where {F}

Specify the problem of finding ``γ²,q`` given an objective for the P-trace

Here, P-trace refers to the normalized trace ``k = \\tr[P] / N``, where
``P = (QQ^† + γ²I)⁻¹``.

# Arguments

  - `eq`: the resolvent equation (shared context)
  - `ptrace`: target P-trace value
"""
struct PTraceProblem{F,E} <: AbstractResolventProblem{F,E}
    eq::E
    ptrace::F

    function PTraceProblem{F}(eq, ptrace) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, ptrace)
    end
end

function PTraceProblem(eq::ResolventEquation{F}, ptrace) where {F}
    return PTraceProblem{F}(eq, ptrace)
end

function PTraceProblem{F}(p::PTraceProblem) where {F}
    return PTraceProblem{F}(p.eq, p.ptrace)
end

function update(p::PTraceProblem{F}; eq=p.eq, ptrace=p.ptrace) where {F}
    return PTraceProblem{F}(eq, ptrace)
end

## Conversions
## ========================
function Base.convert(::Type{P}, p::PTraceProblem) where {F,P<:PTraceProblem{F}}
    return (p isa P) ? p : PTraceProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::PTraceProblem) where {F}
    return convert(PTraceProblem{F}, eq)
end

## PointProblem
## ============
"""
    PointProblem(problem::AbstractResolventProblem, z, init)

Bundle a resolvent problem with a point and initial values for solving at a specific point.

# Fields

  - `problem`: the target problem (eq + objective)
  - `z`: point in the complex plane
  - `init`: initial guess NamedTuple (`:q` for RegularizerProblem; `:reg, :q` for PTraceProblem)
"""
struct PointProblem{F,P<:AbstractResolventProblem{F},I<:NamedTuple}
    problem::P
    z::Complex{F}
    init::I

    function PointProblem{F}(problem, z, init) where {F}
        problemF = adaptrealtype(F, problem)
        initF = getinit(problemF, init)
        return new{F,typeof(problemF),typeof(initF)}(problemF, Complex{F}(z), initF)
    end
end

function PointProblem(problem::AbstractResolventProblem{F}, z, init) where {F}
    return PointProblem{F}(problem, z, init)
end

function PointProblem{F}(pp::PointProblem) where {F}
    return PointProblem{F}(pp.problem, pp.z, pp.init)
end

function update(pp::PointProblem{F}; problem=pp.problem, z=pp.z, init=pp.init) where {F}
    return PointProblem{F}(problem, z, init)
end

## Conversions
## ========================
function Base.convert(::Type{P}, p::PointProblem) where {F,P<:PointProblem{F}}
    return (p isa P) ? p : PointProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::PointProblem) where {F}
    return convert(PointProblem{F}, eq)
end

# inithelper
getinit(::RegularizerProblem{F}, init) where {F} = (; q=Complex{F}(init.q))
function getinit(::PTraceProblem{F}, init) where {F}
    return (; reg=Regularizer{F}(init.reg), q=Complex{F}(init.q))
end

end  # module Problems

@testitem "ResolventEquation" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: ResolventEquation, update

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    shiftmin64, sscale64, normalizeQ = 0.0, 0.5, false
    eq64 = ResolventEquation{Float64}(ensemble64, shiftmin64, sscale64, normalizeQ)

    # Traits
    @test eq64 isa ResolventEquation{Float64}

    # Defaults
    @test ResolventEquation(ensemble64) ==
        ResolventEquation{Float64}(ensemble64, -Inf, 1.0, false)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        shiftminF = F(shiftmin64)
        sscaleF = F(sscale64)
        eqF = ResolventEquation{F}(ensembleF, shiftminF, sscaleF, normalizeQ)

        # Update
        eqF_updated = update(eqF; normalizeQ=true)
        @test eqF_updated isa ResolventEquation{F}
        @test eqF_updated.normalizeQ == true
        @test eqF_updated.ensemble === eqF.ensemble
        @test eqF_updated.shiftmin === eqF.shiftmin
        @test eqF_updated.sscale === eqF.sscale

        # sscale parameter
        eqF_sscale_updated = update(eqF; sscale=F(0.3))
        @test eqF_sscale_updated.sscale == F(0.3)

        # Floattype inferring constructor
        @test ResolventEquation(
            ensembleF; shiftmin=shiftminF, sscale=sscaleF, normalizeQ
        ) isa ResolventEquation{F}

        # Floattype enforcing constructor
        @test ResolventEquation{F}(ensemble64, shiftmin64, sscale64, normalizeQ) == eqF
        @test ResolventEquation{Float64}(ensembleF, shiftminF, sscaleF, normalizeQ) == eq64

        # Floattype conversions
        @test ResolventEquation{F}(eq64) == eqF
        @test ResolventEquation{Float64}(eqF) == eq64
        @test convert(ResolventEquation{F}, eq64) == eqF
        @test convert(ResolventEquation{Float64}, eqF) == eq64
        @test adaptrealtype(F, eq64) == eqF
        @test adaptrealtype(Float64, eqF) == eq64
    end

    # sscale ∈ [0, 1]
    @test_throws ArgumentError ResolventEquation(ensemble64; sscale=-0.1)
    @test_throws ArgumentError ResolventEquation(ensemble64; sscale=1.1)
    @test_throws ArgumentError ResolventEquation(ensemble64; sscale=0.5im)
end

@testitem "*Problems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceProblem, RegularizerProblem, ResolventEquation, update
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    reg64 = Regularizer(0.5, :gap)
    ptrace64 = 1.0

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        eqF = ResolventEquation(ensembleF)
        regF = Regularizer(F(reg64.value), reg64.type)
        ptraceF = F(ptrace64)

        @testset "$(nameof(P))" for (P, obj64, objF, objprob) in [
            (RegularizerProblem, reg64, regF, :reg),
            (PTraceProblem, ptrace64, ptraceF, :ptrace),
        ]
            problem64 = P(eq64, obj64)
            problemF = P(eqF, objF)

            # Traits
            @test problem64 isa P{Float64}
            @test problemF isa P{F}

            # Update
            problem_updated64 = update(problem64; eq=eq64)
            problemF_updated = update(problemF; eq=eqF)
            @test problemF_updated isa P{F}
            @test problemF_updated.eq === problemF.eq
            @test getproperty(problemF_updated, objprob) === getproperty(problemF, objprob)

            # Floattype inferring constructor
            @test problemF isa P{F}

            # Floattype enforcing constructor
            @test P{F}(eq64, obj64) == problemF
            @test P{Float64}(eqF, objF) == problem64

            # Floattype conversions
            @test P{F}(problem64) == problemF
            @test P{Float64}(problemF) == problem64
            @test convert(P{F}, problem64) == problemF
            @test convert(P{Float64}, problemF) == problem64
            @test adaptrealtype(F, problem64) == problemF
            @test adaptrealtype(Float64, problemF) == problem64
        end
    end
end
