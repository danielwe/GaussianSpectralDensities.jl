module GaussianSpectralDensities

using TestItems: @testitem

# using Adapt
# using ArnoldiMethod
# using ArnoldiMethod: LM
# using CommonSolve: CommonSolve, init, solve!, solve
# using LinearMaps
# using MappedArrays
# using StaticArrays
# using SparseArrays
# using SuiteSparse
# using StructArrays
# using Random
# using ThreadPools
# using Printf

#=
* Consider defining plot recipes to simplify notebooks. This should perhaps also go in
  a separate module with optional dependency on RecipesBase. I don't know exactly what kinds
  of plots it makes sense to write recipes for, however.
=#

# public MatrixEnsemble, TraceObjective, RegularizerObjective

# export init, solve!, solve
# export DirectSolve, RegularizerSchedule, SymmetrySchedule
# export partially_hermitize, randiid, randpartherm, transform, transform!, randsample
# export SpectralProblem, SpectralSolver, set_ensemble!, solve, solve!
# export spectral_density
# export automesh, automesh!

# const MissingOr{T} = Union{Missing, T}

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
    const spread_annealing_schedule = 2.0 .^ (2:-1:-2)
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

    rtol_ballpark(::Type{T}) where {T} = one(real(T)) / 4

end

include("arraytools.jl")
include("mathtools.jl")
# include("matrix_utils.jl")
include("ensembles.jl")
include("samples.jl")
include("spreads.jl")
include("problems.jl")
include("workspaces.jl")
include("solutions.jl")
include("newton.jl")
include("equationsolvers.jl")
include("algorithms.jl")
include("solvers.jl")
# include("globaldata.jl")  # XXX: possibly superseded by batches.jl and surrogates.jl
include("batches.jl")
include("rbfinterpolants.jl")
include("surrogates.jl")
include("boundaries.jl")

# include("solve.jl")
# include("density.jl")
# include("sampling_old.jl")
# include("automesh.jl")
# include("printing.jl")

# adapt(::Type{F}, x::U) where {F,U} = convert(@eval($(U.name.name){$F}), x)

# score(value, reference) = inv(abs(value - reference))

# using Requires
#
# function __init__()
#     @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
#         export CuMatrixEnsemble, CuHybridMatrixEnsemble
#         include("cuda.jl")
#         include("cudahybrid.jl")
#     end
#     @require LazySets="b4f0291d-fe17-52bc-9479-3d1a343d9043" begin
#         export spectral_region, spectral_region!
#         include("spectral_region.jl")
#     end
# end

end
