module GaussianSpectralDensities

using TestItems

using EnumX: @enumx
using LinearAlgebra: I

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

export MatrixEnsemble

# export init, solve!, solve
# export TraceObjective, RegularizerObjective
# export DirectSolve, RegularizerSchedule, SymmetrySchedule
# export partially_hermitize, randiid, randpartherm, transform, transform!, randsample
# export SpectralProblem, SpectralSolver, set_ensemble!, solve, solve!
# export spectral_density
# export automesh, automesh!

# const MissingOr{T} = Union{Missing, T}

include("internals.jl")
using .Internals

# include("matrix_utils.jl")
include("ensemble.jl")
include("regularizers.jl")
# include("objectives.jl")
# include("problems.jl")
# include("workspace.jl")
# include("solutions.jl")
# include("newton.jl")
# include("eqsolve.jl")
# include("algorithms.jl")
# include("localsolve.jl")
# include("linalg.jl")
# include("solve.jl")
# include("density.jl")
# include("sampling.jl")
# include("automesh.jl")
# include("printing.jl")

# Base.@kwdef mutable struct Defaults
#     verbosity::Int = 0
#     store_trace::Bool = false
#     kobjective::Float64 = 1.0
#     regobjective::Regularizer{Float64} = Regularizer(:γ, 0.0)
#     q0::ComplexF64 = sqrt(eps(Float32))
#     reg0::Regularizer{Float64} = Regularizer(:δ, 1.0)
#     regschedule::Vector{Regularizer{Float64}} = Regularizer.(:δ, 2.0 .^ (4:-0.5:-4))
#     μschedule::typeof(0.0:1.0) = range(0, 1, 11)
#     γ²min::Float64 = -Inf
#     margin::Float64 = 0.1
# end
#
# const DEFAULTS = Defaults()
#
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
