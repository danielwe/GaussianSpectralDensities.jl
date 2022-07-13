module GaussianSpectralDensities

# using CommonSolve
using LinearAlgebra
using StaticArrays
using EnumX
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
export TraceObjective, RegularizerObjective, Regularizers
export ResolventProblem
# export partially_hermitize, randiid, randpartherm, transform, transform!, randsample
# export Rampγ², Rampδ², Rampμ, StraightShot
# export SpectralProblem, SpectralSolver, set_ensemble!, solve, solve!
# export spectral_density
# export automesh, automesh!

# const MissingOr{T} = Union{Missing, T}

# include("matrix_utils.jl")
include("ensemble.jl")
include("objectives.jl")
include("problems.jl")
# include("sampling.jl")
# include("workspace.jl")
# include("algorithms.jl")
# include("solutions.jl")
include("newton.jl")
# include("qsolve.jl")
# include("solve.jl")
# include("density.jl")
# include("automesh.jl")
# include("printing.jl")

Base.@kwdef mutable struct Defaults
    verbosity::Int = 0
    store_trace::Bool = false
    k::Float64 = 1.0
    γ::Float64 = 0.0
    μ::Float64 = 1.0
    regularizerschedule::Vector{Float64} = 2.0.^(4:-0.5:-4)
    μschedule::Vector{Float64} = LinRange(0.0, 1.0, 9)
    γ²min::Float64 = -Inf
    q0::ComplexF64 = sqrt(eps(Float32))
    δ0::Float64 = 1.0
    margin::Float64 = 0.1
end

const DEFAULTS = Defaults()

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
