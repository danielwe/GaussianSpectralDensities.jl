module Problems

using AutoHashEquals: @auto_hash_equals

using ..Ensembles: MatrixEnsemble
using ..Regularizers: Regularizer, Regularizers

using ..ArrayTools: ArrayTools, adaptrealtype

public AbstractConstraint,
    PTraceConstraint,
    PointProblem,
    RegularizerConstraint,
    ResolventEquation,
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

## Constraint types
## ================

"""
    AbstractConstraint{F<:Real}

Abstract supertype for constraint specifications in point problems.

Concrete subtypes: [`RegularizerConstraint`](@ref), [`PTraceConstraint`](@ref)
"""
abstract type AbstractConstraint{F<:Real} end

"""
    RegularizerConstraint(val, q0)
    RegularizerConstraint(val; q0)
    RegularizerConstraint(; shift, gap, q0)
    RegularizerConstraint{F}(val, q0) where {F}
    RegularizerConstraint{F}(val; q0) where {F}
    RegularizerConstraint{F}(; shift, gap, q0) where {F}
    RegularizerConstraint{F}(c::RegularizerConstraint) where {F}

Constraint specifying a target regularizer value, with initial guess for solving.

Keyword constructors require exactly one of `shift` or `gap` to specify the regularizer.

# Fields

  - `val::Regularizer{F}`: the target regularizer value
  - `q0::Complex{F}`: initial guess for the resolvent parameter q
"""
@auto_hash_equals struct RegularizerConstraint{F} <: AbstractConstraint{F}
    val::Regularizer{F}
    q0::Complex{F}
end

RegularizerConstraint(val::Regularizer{F}, q0) where {F} = RegularizerConstraint{F}(val, q0)
RegularizerConstraint(val; q0) = RegularizerConstraint(val, q0)
RegularizerConstraint{F}(val; q0) where {F} = RegularizerConstraint{F}(val, q0)
RegularizerConstraint(; q0, kwargs...) = RegularizerConstraint(Regularizer(; kwargs...), q0)

function RegularizerConstraint{F}(; q0, kwargs...) where {F}
    return RegularizerConstraint{F}(Regularizer(; kwargs...), q0)
end

function RegularizerConstraint{F}(c::RegularizerConstraint) where {F}
    return RegularizerConstraint{F}(c.val, c.q0)
end

function update(
    constraint::RegularizerConstraint{F};
    shift=nothing,
    gap=nothing,
    val=(isnothing(shift) && isnothing(gap)) ? constraint.val : nothing,
    q0=constraint.q0,
) where {F}
    return RegularizerConstraint{F}(someregularizer(; shift, gap, reg=val), q0)
end

function Base.convert(::Type{RegularizerConstraint{F}}, c::RegularizerConstraint) where {F}
    return (c isa RegularizerConstraint{F}) ? c : RegularizerConstraint{F}(c)
end

function ArrayTools.adaptrealtype(::Type{F}, c::RegularizerConstraint) where {F}
    return convert(RegularizerConstraint{F}, c)
end

"""
    PTraceConstraint(val, q0, reg0)
    PTraceConstraint(val; q0, shift0, gap0, reg0)
    PTraceConstraint{F}(val, q0, reg0) where {F}
    PTraceConstraint{F}(val; q0, shift0, gap0, reg0) where {F}
    PTraceConstraint{F}(c::PTraceConstraint) where {F}

Constraint specifying a target value of ``\tr P``, with initial guesses for solving.

Keyword constructors require exactly one of `shift0`, `gap0`, or `reg0` to specify the
initial regularizer guess.

# Fields

  - `val::F`: the target value of `tr(P)`
  - `q0::Complex{F}`: initial guess for the resolvent parameter q
  - `reg0::Regularizer{F}`: initial guess for the regularizer
"""
@auto_hash_equals struct PTraceConstraint{F} <: AbstractConstraint{F}
    val::F
    q0::Complex{F}
    reg0::Regularizer{F}
end

PTraceConstraint(val::F, q0, reg0) where {F} = PTraceConstraint{F}(val, q0, reg0)

function PTraceConstraint(val; q0, shift0=nothing, gap0=nothing, reg0=nothing)
    return PTraceConstraint(val, q0, someregularizer(; shift=shift0, gap=gap0, reg=reg0))
end

function PTraceConstraint{F}(val; q0, shift0=nothing, gap0=nothing, reg0=nothing) where {F}
    somereg0 = someregularizer(; shift=shift0, gap=gap0, reg=reg0)
    return PTraceConstraint{F}(val, q0, somereg0)
end

function PTraceConstraint{F}(c::PTraceConstraint) where {F}
    return PTraceConstraint{F}(c.val, c.q0, c.reg0)
end

function update(
    constraint::PTraceConstraint{F};
    val=constraint.val,
    q0=constraint.q0,
    shift0=nothing,
    gap0=nothing,
    reg0=(isnothing(shift0) && isnothing(gap0)) ? constraint.reg0 : nothing,
) where {F}
    return PTraceConstraint{F}(val; q0, shift0, gap0, reg0)
end

function Base.convert(::Type{PTraceConstraint{F}}, c::PTraceConstraint) where {F}
    return (c isa PTraceConstraint{F}) ? c : PTraceConstraint{F}(c)
end

function ArrayTools.adaptrealtype(::Type{F}, c::PTraceConstraint) where {F}
    return convert(PTraceConstraint{F}, c)
end

# Regularizer constructor/selector helper for update methods
function someregularizer(; shift=nothing, gap=nothing, reg=nothing)
    if count(isnothing, (shift, gap, reg)) != 2
        throw(ArgumentError("must specify exactly one of `shift`, `gap`, or `reg`"))
    end
    return isnothing(reg) ? Regularizer(; shift, gap) : reg
end

## Problem type
## ============
"""
    PointProblem(equation::ResolventEquation, constraint, z)
    PointProblem{F}(equation, constraint, z) where {F}
    PointProblem{F}(problem::PointProblem) where {F}

Bundle a resolvent equation with a constraint and a point in the complex plane.

The constraint carries both the target specification and initial guesses for solving.
"""
@auto_hash_equals typearg=true struct PointProblem{
    F,E<:ResolventEquation{F},C<:AbstractConstraint{F}
}
    equation::E
    constraint::C
    z::Complex{F}

    function PointProblem{F}(equation, constraint, z) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        constraintF = adaptrealtype(F, constraint)
        return new{F,typeof(eqF),typeof(constraintF)}(eqF, constraintF, Complex{F}(z))
    end
end

function PointProblem(equation::ResolventEquation{F}, constraint, z) where {F}
    return PointProblem{F}(equation, constraint, z)
end

function PointProblem{F}(problem::PointProblem) where {F}
    return PointProblem{F}(problem.equation, problem.constraint, problem.z)
end

function update(
    problem::PointProblem{F,E};
    equation::E=problem.equation,
    constraint=problem.constraint,
    z=problem.z,
) where {F,E}
    return PointProblem{F}(equation, constraint, z)
end

## Conversions
## ===========
function Base.convert(::Type{P}, p::PointProblem) where {F,P<:PointProblem{F}}
    return (p isa P) ? p : PointProblem{F}(p)
end

ArrayTools.adaptrealtype(::Type{F}, p::PointProblem) where {F} = convert(PointProblem{F}, p)

## Comparisons
## ===========
issameproblem(::PointProblem, ::PointProblem) = false

function issameproblem(a::P, b::P) where {P<:PointProblem}
    ((a.equation == b.equation) && (a.z == b.z)) || return false
    return a.constraint.val == b.constraint.val
end

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

@testitem "Constraint types" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Problems:
        AbstractConstraint, PTraceConstraint, RegularizerConstraint
    using GaussianSpectralDensities.Regularizers: Regularizer

    q0_64 = complex(0.25)
    reg64 = Regularizer(0.5, :gap)
    ptrace64 = 1.0
    regc64 = RegularizerConstraint(reg64, q0_64)
    ptracec64 = PTraceConstraint(ptrace64, q0_64, reg64)

    # Traits
    @test regc64 isa AbstractConstraint{Float64}
    @test ptracec64 isa AbstractConstraint{Float64}

    @testset "$F" for F in [Float32, Float64]
        q0_F = Complex{F}(q0_64)
        regF = Regularizer{F}(reg64)
        regcF = RegularizerConstraint(regF, q0_F)
        ptracecF = PTraceConstraint(F(ptrace64), q0_F, regF)

        # Floattype inferring constructor (from Regularizer)
        @test regcF isa RegularizerConstraint{F}
        @test ptracecF isa PTraceConstraint{F}

        # User-friendly constructor (keyword args)
        @test RegularizerConstraint(; gap=F(0.5), q0=q0_F) == regcF
        @test PTraceConstraint(F(ptrace64); q0=q0_F, gap0=F(0.5)) == ptracecF

        # Floattype enforcing constructor
        @test RegularizerConstraint{F}(reg64, q0_64) == regcF
        @test RegularizerConstraint{Float64}(regF, q0_F) == regc64
        @test PTraceConstraint{F}(ptrace64, q0_64, reg64) == ptracecF
        @test PTraceConstraint{Float64}(F(ptrace64), q0_F, regF) == ptracec64

        # Copy-converting constructor
        @test RegularizerConstraint{F}(regc64) == regcF
        @test RegularizerConstraint{Float64}(regcF) == regc64
        @test PTraceConstraint{F}(ptracec64) == ptracecF
        @test PTraceConstraint{Float64}(ptracecF) == ptracec64

        # Floattype conversions
        @test convert(RegularizerConstraint{F}, regc64) == regcF
        @test convert(RegularizerConstraint{Float64}, regcF) == regc64
        @test convert(PTraceConstraint{F}, ptracec64) == ptracecF
        @test convert(PTraceConstraint{Float64}, ptracecF) == ptracec64
        @test adaptrealtype(F, regc64) == regcF
        @test adaptrealtype(Float64, regcF) == regc64
        @test adaptrealtype(F, ptracec64) == ptracecF
        @test adaptrealtype(Float64, ptracecF) == ptracec64
    end
end

@testitem "Constraint keyword constructors" begin
    using GaussianSpectralDensities.Problems: PTraceConstraint, RegularizerConstraint
    using GaussianSpectralDensities.Regularizers: Regularizer

    # Keyword constructors
    reg_shift = Regularizer(0.5, :shift)
    reg_gap = Regularizer(0.25, :gap)
    regc_shift = RegularizerConstraint(; shift=reg_shift.val, q0=complex(0.1))
    regc_gap = RegularizerConstraint(; gap=reg_gap.val, q0=complex(0.1))
    ptracec_shift = PTraceConstraint(1.0; q0=complex(0.1), shift0=reg_shift.val)
    ptracec_gap = PTraceConstraint(1.0; q0=complex(0.1), gap0=reg_gap.val)
    @test regc_shift == RegularizerConstraint(reg_shift, complex(0.1))
    @test regc_gap == RegularizerConstraint(reg_gap, complex(0.1))
    @test ptracec_shift == PTraceConstraint(1.0, complex(0.1), reg_shift)
    @test ptracec_gap == PTraceConstraint(1.0, complex(0.1), reg_gap)

    # Type-enforcing keyword constructors
    @test RegularizerConstraint{Float32}(; shift=0.5, q0=complex(0.1)) isa
        RegularizerConstraint{Float32}
    @test PTraceConstraint{Float32}(1.0; q0=complex(0.1), gap0=0.25) isa
        PTraceConstraint{Float32}

    # Errors propagate through constructors
    @test_throws ArgumentError RegularizerConstraint(; q0=complex(0.1))
    @test_throws ArgumentError RegularizerConstraint(; shift=0.5, gap=0.25, q0=complex(0.1))
    @test_throws ArgumentError PTraceConstraint(1.0; q0=complex(0.1))
    @test_throws ArgumentError PTraceConstraint(1.0; q0=complex(0.1), shift0=0.5, gap0=0.25)
end

@testitem "PointProblems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint,
        PointProblem,
        RegularizerConstraint,
        ResolventEquation,
        issameproblem,
        update
    using GaussianSpectralDensities.Regularizers: Regularizer

    using LinearAlgebra: I

    eq64 = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    reg_type = :gap
    reg = Regularizer(0.5, reg_type)
    q0, z = complex(0.25), complex(0.25)
    z_new = 2z
    q0_new = 2q0

    regc = RegularizerConstraint(reg, q0)
    regc_newinit = RegularizerConstraint(reg, q0_new)
    ptracec = PTraceConstraint(1.0, q0, reg)
    ptracec_newinit = PTraceConstraint(1.0, q0_new, reg)

    @testset "$(nameof(typeof(constraint)))" for (constraint, constraint_newinit) in [
        (regc, regc_newinit), (ptracec, ptracec_newinit)
    ]
        problem64 = PointProblem(eq64, constraint, z)

        @testset "$F" for F in [Float32, Float64]
            eqF = ResolventEquation{F}(eq64)
            problemF = PointProblem(eqF, constraint, z)

            # Traits
            @test problemF isa PointProblem{F}

            # Same problem? (same target, different init)
            problemF_newinit = PointProblem(eqF, constraint_newinit, z)
            @test issameproblem(problemF, problemF_newinit)

            # Update
            problemF_updated = update(problemF; z=z_new, constraint=constraint_newinit)
            @test problemF_updated isa PointProblem{F}
            @test problemF_updated.equation === problemF.equation
            @test problemF_updated.z == z_new
            @test problemF_updated.constraint == constraint_newinit
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
