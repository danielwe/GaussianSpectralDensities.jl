module Problems

using AutoHashEquals: @auto_hash_equals

using ..Ensembles: MatrixEnsemble
using ..Spreads: Spread, Spreads

using ..ArrayTools: ArrayTools, adaptrealtype

public AbstractConstraint,
    PTraceConstraint,
    PointProblem,
    ResolventEquation,
    SpreadConstraint,
    issameproblem,
    update

"""
    ResolventEquation(ensemble::MatrixEnsemble; shiftmin=-Inf, sscale=1, normalizeQ=false)
    ResolventEquation{F}(ensemble, shiftmin, sscale, normalizeQ) where {F}
    ResolventEquation{F}(eq::ResolventEquation) where {F}

Specify the shared context for an averaged resolvent equation over an ensemble

# Arguments

  - `ensemble`: random matrix ensemble over which to average the resolvent
  - `shiftmin=-Inf`: lower bound on the value of the shift ``γ²``. Solutions with
    `shift < shiftmin` are marked as failed. Strictly speaking, the averaged resolvent
    equations are derived assuming ``γ² ≥ 0``, suggesting `shiftmin = 0`, but in practice it
    is useful to allow analytic continuation to negative values such that the averaged
    resolvent can be interpolated between points in regions where ``γ²`` vanishes rapidly.
    However, it's best to abort before reaching excessively large negative values, as the
    conditioning of the equations deteriorates when the spectral gap shrinks.
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

Concrete subtypes: [`SpreadConstraint`](@ref), [`PTraceConstraint`](@ref)
"""
abstract type AbstractConstraint{F<:AbstractFloat} end

"""
    SpreadConstraint(val, shape0)
    SpreadConstraint(val; shape0)
    SpreadConstraint(; shift, gap, shape0)
    SpreadConstraint{F}(val, shape0) where {F}
    SpreadConstraint{F}(val; shape0) where {F}
    SpreadConstraint{F}(; shift, gap, shape0) where {F}
    SpreadConstraint{F}(c::SpreadConstraint) where {F}

Constraint specifying a target value for the spread parameter

Includes an initial guess for the stretch parameter to use when solving for this constraint.

Keyword constructors require exactly one of `shift` or `gap`.

# Fields

  - `val::Spread{F}`: the target value for the spread parameter
  - `shape0::Complex{F}`: initial guess for the stretch parameter
"""
@auto_hash_equals struct SpreadConstraint{F} <: AbstractConstraint{F}
    val::Spread{F}
    shape0::Complex{F}
end

SpreadConstraint(val::Spread{F}, shape0) where {F} = SpreadConstraint{float(F)}(val, shape0)
SpreadConstraint(val; shape0) = SpreadConstraint(val, shape0)
SpreadConstraint{F}(val; shape0) where {F} = SpreadConstraint{F}(val, shape0)
SpreadConstraint(; shape0, kwargs...) = SpreadConstraint(Spread(; kwargs...), shape0)

function SpreadConstraint{F}(; shape0, kwargs...) where {F}
    return SpreadConstraint{F}(Spread(; kwargs...), shape0)
end

SpreadConstraint{F}(c::SpreadConstraint) where {F} = SpreadConstraint{F}(c.val, c.shape0)

function update(
    constraint::SpreadConstraint{F};
    shift=nothing,
    gap=nothing,
    val=(isnothing(shift) && isnothing(gap)) ? constraint.val : nothing,
    shape0=constraint.shape0,
) where {F}
    return SpreadConstraint{F}(somespread(; shift, gap, spread=val), shape0)
end

function Base.convert(::Type{SpreadConstraint{F}}, c::SpreadConstraint) where {F}
    return (c isa SpreadConstraint{F}) ? c : SpreadConstraint{F}(c)
end

function ArrayTools.adaptrealtype(::Type{F}, c::SpreadConstraint) where {F}
    return convert(SpreadConstraint{F}, c)
end

"""
    PTraceConstraint(val, shape0, spread0)
    PTraceConstraint(val; shape0, shift0, gap0, spread0)
    PTraceConstraint{F}(val, shape0, spread0) where {F}
    PTraceConstraint{F}(val; shape0, shift0, gap0, spread0) where {F}
    PTraceConstraint{F}(c::PTraceConstraint) where {F}

Constraint specifying a target value for ``\tr P``

Includes initial guesses for the spread and stretch parameters to use when solving for this
constraint.

Keyword constructors require exactly one of `shift0`, `gap0`, or `spread0`.

# Fields

  - `val::F`: the target value for `tr(P)`
  - `shape0::Complex{F}`: initial guess for the stretch parameter
  - `spread0::Spread{F}`: initial guess for the spread parameter
"""
@auto_hash_equals struct PTraceConstraint{F} <: AbstractConstraint{F}
    val::F
    shape0::Complex{F}
    spread0::Spread{F}
end

function PTraceConstraint(val::F, shape0, spread0) where {F}
    return PTraceConstraint{float(F)}(val, shape0, spread0)
end

function PTraceConstraint(val; shape0, shift0=nothing, gap0=nothing, spread0=nothing)
    spread0_actual = somespread(; shift=shift0, gap=gap0, spread=spread0)
    return PTraceConstraint(val, shape0, spread0_actual)
end

function PTraceConstraint{F}(
    val; shape0, shift0=nothing, gap0=nothing, spread0=nothing
) where {F}
    somespread0 = somespread(; shift=shift0, gap=gap0, spread=spread0)
    return PTraceConstraint{float(F)}(val, shape0, somespread0)
end

function PTraceConstraint{F}(c::PTraceConstraint) where {F}
    return PTraceConstraint{F}(c.val, c.shape0, c.spread0)
end

function update(
    constraint::PTraceConstraint{F};
    val=constraint.val,
    shape0=constraint.shape0,
    shift0=nothing,
    gap0=nothing,
    spread0=(isnothing(shift0) && isnothing(gap0)) ? constraint.spread0 : nothing,
) where {F}
    return PTraceConstraint{F}(val; shape0, shift0, gap0, spread0)
end

function Base.convert(::Type{PTraceConstraint{F}}, c::PTraceConstraint) where {F}
    return (c isa PTraceConstraint{F}) ? c : PTraceConstraint{F}(c)
end

function ArrayTools.adaptrealtype(::Type{F}, c::PTraceConstraint) where {F}
    return convert(PTraceConstraint{F}, c)
end

# Spread type selector for update methods
function somespread(; shift=nothing, gap=nothing, spread=nothing)
    if count(isnothing, (shift, gap, spread)) != 2
        throw(ArgumentError("must specify exactly one of `shift`, `gap`, or `spread`"))
    end
    return isnothing(spread) ? Spread(; shift, gap) : spread
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
        AbstractConstraint, PTraceConstraint, SpreadConstraint
    using GaussianSpectralDensities.Spreads: Spread

    shape0_64 = complex(0.25)
    spread64 = Spread(0.5, :gap)
    ptrace64 = 1.0
    spreadc64 = SpreadConstraint(spread64, shape0_64)
    ptracec64 = PTraceConstraint(ptrace64, shape0_64, spread64)

    # Traits
    @test spreadc64 isa AbstractConstraint{Float64}
    @test ptracec64 isa AbstractConstraint{Float64}

    @testset "$F" for F in [Float32, Float64]
        shape0_F = Complex{F}(shape0_64)
        spreadF = Spread{F}(spread64)
        spreadcF = SpreadConstraint(spreadF, shape0_F)
        ptracecF = PTraceConstraint(F(ptrace64), shape0_F, spreadF)

        # Floattype inferring constructor (from Spread)
        @test spreadcF isa SpreadConstraint{F}
        @test ptracecF isa PTraceConstraint{F}

        # Type-inferring constructor: SpreadConstraint(val::Spread{F}, shape0)
        @test SpreadConstraint(spreadF, shape0_64) == spreadcF
        @test SpreadConstraint(spread64, shape0_F) == spreadc64

        # Type-inferring constructor: PTraceConstraint(val::F, shape0, spread0)
        @test PTraceConstraint(F(ptrace64), shape0_64, spread64) == ptracecF
        @test PTraceConstraint(ptrace64, shape0_F, spreadF) == ptracec64

        # User-friendly constructor (keyword args)
        @test SpreadConstraint(; gap=F(0.5), shape0=shape0_F) == spreadcF
        @test PTraceConstraint(F(ptrace64); shape0=shape0_F, gap0=F(0.5)) == ptracecF

        # Floattype enforcing constructor
        @test SpreadConstraint{F}(spread64, shape0_64) == spreadcF
        @test SpreadConstraint{Float64}(spreadF, shape0_F) == spreadc64
        @test PTraceConstraint{F}(ptrace64, shape0_64, spread64) == ptracecF
        @test PTraceConstraint{Float64}(F(ptrace64), shape0_F, spreadF) == ptracec64

        # Copy-converting constructor
        @test SpreadConstraint{F}(spreadc64) == spreadcF
        @test SpreadConstraint{Float64}(spreadcF) == spreadc64
        @test PTraceConstraint{F}(ptracec64) == ptracecF
        @test PTraceConstraint{Float64}(ptracecF) == ptracec64

        # Floattype conversions
        @test convert(SpreadConstraint{F}, spreadc64) == spreadcF
        @test convert(SpreadConstraint{Float64}, spreadcF) == spreadc64
        @test convert(PTraceConstraint{F}, ptracec64) == ptracecF
        @test convert(PTraceConstraint{Float64}, ptracecF) == ptracec64
        @test adaptrealtype(F, spreadc64) == spreadcF
        @test adaptrealtype(Float64, spreadcF) == spreadc64
        @test adaptrealtype(F, ptracec64) == ptracecF
        @test adaptrealtype(Float64, ptracecF) == ptracec64
    end
end

@testitem "Constraint keyword constructors" begin
    using GaussianSpectralDensities.Problems: PTraceConstraint, SpreadConstraint
    using GaussianSpectralDensities.Spreads: Spread

    shift = 0.5
    gap = 0.25
    ptrace = 1.0
    shape0 = complex(0.1)

    spread_shift = Spread(shift, :shift)
    spread_gap = Spread(gap, :gap)

    spreadc_shift = SpreadConstraint(spread_shift, shape0)
    spreadc_gap = SpreadConstraint(spread_gap, shape0)
    ptracec_shift = PTraceConstraint(ptrace, shape0, spread_shift)
    ptracec_gap = PTraceConstraint(ptrace, shape0, spread_gap)

    # Keyword constructors
    @test SpreadConstraint(; shift=shift, shape0) == spreadc_shift
    @test SpreadConstraint(; gap=gap, shape0) == spreadc_gap
    @test PTraceConstraint(ptrace; shape0, shift0=shift) == ptracec_shift
    @test PTraceConstraint(ptrace; shape0, gap0=gap) == ptracec_gap

    # Keyword constructor from Spread: SpreadConstraint(val; shape0)
    @test SpreadConstraint(spread_shift; shape0) == spreadc_shift
    @test SpreadConstraint(spread_gap; shape0) == spreadc_gap

    # Type-enforcing keyword constructors
    @test SpreadConstraint{Float32}(; shift, shape0) isa SpreadConstraint{Float32}
    @test PTraceConstraint{Float32}(ptrace; shape0, gap0=gap) isa PTraceConstraint{Float32}

    # Type-enforcing keyword constructor from Spread: SpreadConstraint{F}(val; shape0)
    @test SpreadConstraint{Float32}(spread_shift; shape0) isa SpreadConstraint{Float32}
    @test SpreadConstraint{Float64}(spread_gap; shape0) isa SpreadConstraint{Float64}

    # Errors propagate through constructors
    @test_throws ArgumentError SpreadConstraint(; shape0)
    @test_throws ArgumentError SpreadConstraint(; shift, gap, shape0)
    @test_throws ArgumentError PTraceConstraint(ptrace; shape0)
    @test_throws ArgumentError PTraceConstraint(ptrace; shape0, shift0=shift, gap0=gap)
end

@testitem "Constraint update methods" begin
    using GaussianSpectralDensities.Problems: PTraceConstraint, SpreadConstraint, update
    using GaussianSpectralDensities.Spreads: Spread

    @testset "update(::SpreadConstraint)" begin
        shape0 = complex(0.25)
        spreadc = SpreadConstraint(Spread(0.5, :gap), shape0)

        # Update shape0 only
        spreadc_newshape0 = update(spreadc; shape0=complex(0.5))
        @test spreadc_newshape0.val == spreadc.val
        @test spreadc_newshape0.shape0 == complex(0.5)

        # Update to shift spread
        spreadc_newshift = update(spreadc; shift=0.75)
        @test spreadc_newshift.val == Spread(0.75, :shift)
        @test spreadc_newshift.shape0 == shape0

        # Update to gap spread
        spreadc_newgap = update(spreadc; gap=0.75)
        @test spreadc_newgap.val == Spread(0.75, :gap)
        @test spreadc_newgap.shape0 == shape0

        # Update both spread and shape0
        spreadc_newboth = update(spreadc; shift=0.75, shape0=complex(0.5))
        @test spreadc_newboth.val == Spread(0.75, :shift)
        @test spreadc_newboth.shape0 == complex(0.5)

        # Error when specifying both shift and gap
        @test_throws ArgumentError update(spreadc; shift=0.5, gap=0.25)
    end

    @testset "update(::PTraceConstraint)" begin
        shape0 = complex(0.25)
        spread = Spread(0.5, :gap)
        ptracec = PTraceConstraint(1.0, shape0, spread)

        # Update val only
        ptracec_newval = update(ptracec; val=2.0)
        @test ptracec_newval.val == 2.0
        @test ptracec_newval.shape0 == shape0
        @test ptracec_newval.spread0 == spread

        # Update shape0 only
        ptracec_newshape0 = update(ptracec; shape0=complex(0.5))
        @test ptracec_newshape0.val == ptracec.val
        @test ptracec_newshape0.shape0 == complex(0.5)
        @test ptracec_newshape0.spread0 == spread

        # Update spread0 via shift0
        ptracec_newshift = update(ptracec; shift0=0.75)
        @test ptracec_newshift.val == ptracec.val
        @test ptracec_newshift.shape0 == shape0
        @test ptracec_newshift.spread0 == Spread(0.75, :shift)

        # Update spread0 via gap0
        ptracec_newgap = update(ptracec; gap0=0.75)
        @test ptracec_newgap.val == ptracec.val
        @test ptracec_newgap.shape0 == shape0
        @test ptracec_newgap.spread0 == Spread(0.75, :gap)

        # Update spread0 directly
        spread_new = Spread(0.25, :shift)
        ptracec_newspread = update(ptracec; spread0=spread_new)
        @test ptracec_newspread.val == ptracec.val
        @test ptracec_newspread.shape0 == shape0
        @test ptracec_newspread.spread0 == spread_new

        # Update everything
        ptracec_newall = update(ptracec; val=2.0, shape0=complex(0.5), shift0=0.75)
        @test ptracec_newall.val == 2.0
        @test ptracec_newall.shape0 == complex(0.5)
        @test ptracec_newall.spread0 == Spread(0.75, :shift)

        # Error when specifying multiple of shift0, gap0, spread0
        @test_throws ArgumentError update(ptracec; shift0=0.5, gap0=0.25)
        @test_throws ArgumentError update(ptracec; shift0=0.5, spread0=spread)
        @test_throws ArgumentError update(ptracec; gap0=0.25, spread0=spread)
        @test_throws ArgumentError update(ptracec; shift0=0.5, gap0=0.25, spread0=spread)
    end
end

@testitem "PointProblems" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint,
        PointProblem,
        ResolventEquation,
        SpreadConstraint,
        issameproblem,
        update
    using GaussianSpectralDensities.Spreads: Spread

    using LinearAlgebra: I

    eq64 = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    spread_type = :gap
    spread = Spread(0.5, spread_type)
    shape0, z = complex(0.25), complex(0.25)
    z_new = 2z
    shape0_new = 2shape0

    spreadc = SpreadConstraint(spread, shape0)
    spreadc_newinit = SpreadConstraint(spread, shape0_new)
    ptracec = PTraceConstraint(1.0, shape0, spread)
    ptracec_newinit = PTraceConstraint(1.0, shape0_new, spread)

    @testset "$(nameof(typeof(constraint)))" for (constraint, constraint_newinit) in [
        (spreadc, spreadc_newinit), (ptracec, ptracec_newinit)
    ]
        problem64 = PointProblem(eq64, constraint, z)

        @testset "$F" for F in [Float32, Float64]
            eqF = ResolventEquation{F}(eq64)
            problemF = PointProblem(eqF, constraint, z)

            # Traits
            @test problemF isa PointProblem{F}

            # Same problem? (same target, different init or type)
            problemF_newinit = PointProblem(eqF, constraint_newinit, z)
            @test issameproblem(problemF, problemF_newinit)
            if F === Float32
                @test !issameproblem(problemF, problem64)
            end

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
