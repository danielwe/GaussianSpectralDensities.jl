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

    public trace_target,
        regularizer_target_value,
        regularizer_target_type,
        q0,
        regularizer_schedule_values,
        regularizer_schedule_type,
        symmetry_schedule_values

    const trace_target = 1.0
    const regularizer_target_value = 0.0
    const regularizer_target_type = :shift
    const q0 = ComplexF64(sqrt(eps(Float32)))
    const regularizer_schedule_values = 2.0 .^ (4:-0.5:-4)
    const regularizer_schedule_type = :gap
    const symmetry_schedule_values = range(0, 1, 11)
# const regularizer_init = Regularizer(1.0, :gap)
# const margin = 0.1

end

include("mathhelpers.jl")
# include("matrix_utils.jl")
include("ensembles.jl")
include("regularizers.jl")
include("objectives.jl")
include("problems.jl")
include("workspaces.jl")
include("solutions.jl")
include("newton.jl")
include("equationsolvers.jl")
include("algorithms.jl")

# include("localsolve.jl")
# include("solve.jl")
# include("density.jl")
# include("sampling.jl")
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
