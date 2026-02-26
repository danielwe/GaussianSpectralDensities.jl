module GaussianSpectralDensities

using CommonSolve: solve
using TestItems: @testitem

public Boundaries,
    Defaults,
    Densities,
    DirectSolve,
    MatrixEnsemble,
    Meshing,
    PTraceConstraint,
    PointProblem,
    ResolventEquation,
    Spread,
    SpectralProblem,
    SpreadAnnealing,
    SpreadConstraint,
    Surrogates,
    SymmetryAnnealing,
    evaluate,
    evaluatebatch,
    gap,
    greens,
    issuccess,
    pointsolve,
    ptrace,
    sample_eigenvalues,
    sample_matrix,
    shape,
    shift,
    solve,
    spectral_boundary,
    spectral_density,
    status,
    zvals

module Defaults

    public approxequal,
        atol,
        intheballpark,
        rtol,
        rtol_ballpark,
        spread_annealing_schedule,
        spread_annealing_type,
        spread_target_type,
        spread_target_val,
        symmetry_annealing_schedule,
        trace_target

    const trace_target = 1.0
    const spread_target_type = :shift
    const spread_target_val = 0.0
    const spread_annealing_type = :gap
    const spread_annealing_schedule = 2.0 .^ (7:(-(1 // 2)):-28)
    const symmetry_annealing_schedule = range(0, 1, 9)

    # isapprox but with option to explicitly pass reference type for tolerances
    approxequal(x, y) = approxequal(promote(x, y)...)
    approxequal(x::T, y::T) where {T} = approxequal(real(T), x, y)

    function approxequal(::Type{F}, x, y) where {F<:AbstractFloat}
        return isapprox(x, y; atol=atol(F), rtol=rtol(F))
    end

    rtol(::Type{T}) where {T} = eps(one(real(T)))^(3 // 4)
    atol(::Type{T}) where {T} = rtol(T) * oneunit(real(T))

    intheballpark(x, y) = intheballpark(promote(x, y)...)
    intheballpark(x::T, y::T) where {T} = intheballpark(real(T), x, y)

    function intheballpark(::Type{F}, x, y) where {F<:AbstractFloat}
        return isapprox(x, y; atol=atol(F), rtol=rtol_ballpark(F))
    end

    rtol_ballpark(::Type{T}) where {T} = (2one(real(T)))^(-2)

end

include("arraytools.jl")
include("mathtools.jl")
include("ensembles.jl")
include("samples.jl")
include("spreads.jl")
include("problems.jl")
include("workspaces.jl")
include("solutions.jl")
include("newton.jl")
include("equationsolvers.jl")
include("algorithms.jl")
include("rbfinterpolants.jl")
include("surrogates.jl")
include("meshing.jl")
include("spectralmodels.jl")
include("densities.jl")
include("boundaries.jl")

using .Algorithms: DirectSolve, SpreadAnnealing, SymmetryAnnealing, pointsolve
using .Boundaries: spectral_boundary
using .Densities: spectral_density
using .Ensembles: MatrixEnsemble
using .Problems: PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
using .Samples: sample_eigenvalues, sample_matrix
using .Solutions: greens, issuccess, ptrace, shape, status
using .SpectralModels: SpectralProblem
using .Spreads: Spread, gap, shift
using .Surrogates: evaluate, evaluatebatch, zvals

end
