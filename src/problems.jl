module Problems

using AutoHashEquals: @auto_hash_equals

using ..Constraints: AbstractConstraint, PTrace
using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer, Regularizers, regularizertype

using ..ArrayTools: ArrayTools, adaptrealtype

public AbstractInit,
    PTraceInit,
    PointProblem,
    RegularizerInit,
    ResolventEquation,
    adaptinit,
    issameproblem,
    update

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
@auto_hash_equals typearg=true struct ResolventEquation{F,E<:MatrixEnsemble{F}}
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
    e::ResolventEquation{F,E};
    ensemble::E=e.ensemble,
    shiftmin=e.shiftmin,
    sscale=e.sscale,
    normalizeQ=e.normalizeQ,
) where {F,E}
    return ResolventEquation{F}(ensemble, shiftmin, sscale, normalizeQ)
end

## Conversions
## ========================
function Base.convert(::Type{P}, eq::ResolventEquation) where {F,P<:ResolventEquation{F}}
    return (eq isa P) ? eq : ResolventEquation{F}(eq)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::ResolventEquation) where {F}
    return convert(ResolventEquation{F}, eq)
end

## Init types
## ==========
abstract type AbstractInit{F<:AbstractFloat} end

"""
    PTraceInit(q, reg)
    PTraceInit{F}(q, reg) where {F}
    PTraceInit{F}(init::PTraceInit) where {F}

Initial values for solving a P-trace-constrained point problem.
"""
@auto_hash_equals struct PTraceInit{F} <: AbstractInit{F}
    q::Complex{F}
    reg::Regularizer{F}
end

PTraceInit{F}(init::PTraceInit) where {F} = PTraceInit{F}(init.q, init.reg)

function Base.convert(::Type{PTraceInit{F}}, init::PTraceInit) where {F}
    return (init isa PTraceInit{F}) ? init : PTraceInit{F}(init)
end

function ArrayTools.adaptrealtype(::Type{F}, init::PTraceInit) where {F}
    return convert(PTraceInit{F}, init)
end

"""
    RegularizerInit(q)
    RegularizerInit{F}(q) where {F}
    RegularizerInit{F}(init::RegularizerInit) where {F}

Initial values for solving a regularizer-constrained point problem.
"""
@auto_hash_equals struct RegularizerInit{F} <: AbstractInit{F}
    q::Complex{F}
end

RegularizerInit{F}(init::RegularizerInit) where {F} = RegularizerInit{F}(init.q)

function Base.convert(::Type{RegularizerInit{F}}, init::RegularizerInit) where {F}
    return (init isa RegularizerInit{F}) ? init : RegularizerInit{F}(init)
end

function ArrayTools.adaptrealtype(::Type{F}, init::RegularizerInit) where {F}
    return convert(RegularizerInit{F}, init)
end

## Problem type
## ============
"""
    PointProblem(equation::ResolventEquation, constraint, z, init)
    PointProblem{F}(equation, constraint, z, init) where {F}
    PointProblem{F}(problem::PointProblem) where {F}

Bundle a resolvent equation with a constraint, a point, and point-specific initialization.
"""
@auto_hash_equals typearg=true struct PointProblem{
    F,E<:ResolventEquation{F},C<:AbstractConstraint{F},I<:AbstractInit{F}
}
    equation::E
    constraint::C
    z::Complex{F}
    init::I

    function PointProblem{F}(equation, constraint, z, init) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        constraintF = adaptrealtype(F, constraint)
        initF = adaptinit(constraintF, init)
        return new{F,typeof(eqF),typeof(constraintF),typeof(initF)}(
            eqF, constraintF, z, initF
        )
    end
end

function PointProblem(equation::ResolventEquation{F}, constraint, z, init) where {F}
    return PointProblem{F}(equation, constraint, z, init)
end

function PointProblem{F}(problem::PointProblem) where {F}
    return PointProblem{F}(problem.equation, problem.constraint, problem.z, problem.init)
end

function update(
    problem::PointProblem{F,E};
    equation::E=problem.equation,
    constraint=problem.constraint,
    z=problem.z,
    init=problem.init,
) where {F,E}
    return PointProblem{F}(equation, constraint, z, init)
end

## Traits
## ======
function Regularizers.regularizertype(problem::PointProblem)
    return regularizertype_inner(problem.constraint, problem.init)
end

regularizertype_inner(reg::Regularizer, ::RegularizerInit) = regularizertype(reg)
regularizertype_inner(::PTrace, init::PTraceInit) = regularizertype(init.reg)

## Conversions
## ===========
function Base.convert(::Type{P}, p::PointProblem) where {F,P<:PointProblem{F}}
    return (p isa P) ? p : PointProblem{F}(p)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::PointProblem) where {F}
    return convert(PointProblem{F}, eq)
end

## Comparisons
## ===========
# NOTE: we could drop this and use `@auto_hash_equals fields=(equation,constraint,z)`, but I
# don't want to. Different inits constitute a significant structural difference that may
# lead to different behavior/output, even if they don't change the underlying math problem.
issameproblem(::PointProblem, ::PointProblem) = false

function issameproblem(a::P, b::P) where {P<:PointProblem}
    return (a.equation == b.equation) && (a.constraint == b.constraint) && (a.z == b.z)
end

# One-way downconversion
# We can construct a Regularizer-constrained problem using a PTraceInit, but not vice versa
function adaptinit(::AbstractConstraint, ::AbstractInit)
    throw(ArgumentError("incompatible init type for constraint"))
end

function adaptinit(::Regularizer{F}, init::RegularizerInit) where {F}
    return convert(RegularizerInit{F}, init)
end

adaptinit(::Regularizer{F}, init::PTraceInit) where {F} = RegularizerInit{F}(init.q)
adaptinit(::PTrace{F}, init::PTraceInit) where {F} = convert(PTraceInit{F}, init)

end  # module Problems

@testitem "ResolventEquation" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: ResolventEquation, update

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    shiftmin, sscale, normalizeQ = 0.0, 0.5, false
    eq64 = ResolventEquation{Float64}(ensemble64, shiftmin, sscale, normalizeQ)

    sscale_updated, normalizeQ_updated = 0.25, true

    # Traits
    @test eq64 isa ResolventEquation{Float64}

    # Defaults
    @test ResolventEquation(ensemble64) ==
        ResolventEquation{Float64}(ensemble64, -Inf, 1.0, false)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        eqF = ResolventEquation{F}(ensembleF, shiftmin, sscale, normalizeQ)

        # Update
        eqF_updated = update(eqF; sscale=sscale_updated, normalizeQ=normalizeQ_updated)
        @test eqF_updated isa ResolventEquation{F}
        @test eqF_updated.ensemble === eqF.ensemble
        @test eqF_updated.shiftmin === eqF.shiftmin
        @test eqF_updated.normalizeQ == normalizeQ_updated
        @test eqF_updated.sscale == sscale_updated

        # Floattype inferring constructor
        @test ResolventEquation(ensembleF; shiftmin, sscale, normalizeQ) isa
            ResolventEquation{F}

        # Floattype enforcing constructor
        @test ResolventEquation{F}(ensemble64, shiftmin, sscale, normalizeQ) == eqF
        @test ResolventEquation{Float64}(ensembleF, shiftmin, sscale, normalizeQ) == eq64

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

@testitem "Init types" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Problems: PTraceInit, RegularizerInit
    using GaussianSpectralDensities.Regularizers: Regularizer

    q64 = complex(0.25)
    reg64 = Regularizer(0.5, :gap)
    reg_init64 = RegularizerInit(q64)
    ptrace_init64 = PTraceInit(q64, reg64)

    @testset "$F" for F in [Float32, Float64]
        qF = Complex{F}(q64)
        regF = Regularizer{F}(reg64)
        reg_initF = RegularizerInit(qF)
        ptrace_initF = PTraceInit(qF, regF)

        # Floattype inferring constructor
        @test reg_initF isa RegularizerInit{F}
        @test ptrace_initF isa PTraceInit{F}

        # Floattype enforcing constructor
        @test RegularizerInit{F}(q64) == reg_initF
        @test RegularizerInit{Float64}(qF) == reg_init64
        @test PTraceInit{F}(q64, reg64) == ptrace_initF
        @test PTraceInit{Float64}(qF, regF) == ptrace_init64

        # Copy-converting constructor
        @test RegularizerInit{F}(reg_init64) == reg_initF
        @test RegularizerInit{Float64}(reg_initF) == reg_init64
        @test PTraceInit{F}(ptrace_init64) == ptrace_initF
        @test PTraceInit{Float64}(ptrace_initF) == ptrace_init64

        # Floattype conversions
        @test convert(RegularizerInit{F}, reg_init64) == reg_initF
        @test convert(RegularizerInit{Float64}, reg_initF) == reg_init64
        @test convert(PTraceInit{F}, ptrace_init64) == ptrace_initF
        @test convert(PTraceInit{Float64}, ptrace_initF) == ptrace_init64
        @test adaptrealtype(F, reg_init64) == reg_initF
        @test adaptrealtype(Float64, reg_initF) == reg_init64
        @test adaptrealtype(F, ptrace_init64) == ptrace_initF
        @test adaptrealtype(Float64, ptrace_initF) == ptrace_init64
    end
end

@testitem "PointProblems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Constraints: PTrace
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceInit, PointProblem, RegularizerInit, ResolventEquation, issameproblem, update
    using GaussianSpectralDensities.Regularizers: Regularizer, regularizertype

    using LinearAlgebra: I

    eq64 = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    reg_type = :gap
    reg = Regularizer(0.5, reg_type)
    ptrace = PTrace(1.0)
    z, q = complex(0.25), complex(0.25)
    z_updated = z / q
    q_updated = 2q

    reg_init = RegularizerInit(q)
    reg_init_updated = RegularizerInit(q_updated)
    ptrace_init = PTraceInit(q, reg)
    ptrace_init_updated = PTraceInit(q_updated, reg)

    @testset "$(nameof(typeof(constraint)))" for (
        constraint, init, init_updated, init_wrong
    ) in [
        (reg, reg_init, reg_init_updated, ptrace_init),
        (ptrace, ptrace_init, ptrace_init_updated, reg_init),
    ]
        problem64 = PointProblem(eq64, constraint, z, init)

        @testset "$F" for F in [Float32, Float64]
            eqF = ResolventEquation{F}(eq64)
            problemF = PointProblem(eqF, constraint, z, init)

            # Init check/adaptation
            if constraint isa Regularizer
                # Constructing Regularizer-constrained problem using PTraceInit is OK
                @test PointProblem(eqF, constraint, z, init_wrong) == problemF
            else
                @test_throws ArgumentError PointProblem(eqF, constraint, z, init_wrong)
            end

            # Traits
            @test problemF isa PointProblem{F}
            @test regularizertype(problemF) == reg_type

            # Same problem?
            problemF_newinit = PointProblem(eqF, constraint, z, init_updated)
            @test issameproblem(problemF, problemF_newinit)

            # Update
            problemF_updated = update(problemF; z=z_updated, init=init_updated)
            @test problemF_updated isa PointProblem{F}
            @test problemF_updated.equation === problemF.equation
            @test problemF_updated.constraint === problemF.constraint
            @test problemF_updated.z == z_updated
            @test problemF_updated.init == init_updated
            @test !issameproblem(problemF, problemF_updated)

            # Floattype enforcing constructor
            @test PointProblem{F}(problem64) == problemF
            @test PointProblem{Float64}(problemF) == problem64

            # Floattype conversions
            @test convert(PointProblem{F}, problem64) == problemF
            @test convert(PointProblem{Float64}, problemF) == problem64
            @test adaptrealtype(F, problem64) == problemF
            @test adaptrealtype(Float64, problemF) == problem64
        end
    end
end
