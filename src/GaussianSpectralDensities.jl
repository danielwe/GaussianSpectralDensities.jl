module GaussianSpectralDensities

using LinearAlgebra
using StaticArrays
using StructArrays
using Random
using ThreadPools
using Printf

#=
* Consider defining plot recipes to simplify notebooks. This should perhaps also go in
  a separate module with optional dependency on RecipesBase. I don't know exactly what kinds
  of plots it makes sense to write recipes for, however.
=#

export AbstractMatrixEnsemble, MatrixEnsemble
export partially_hermitize, randiid, randpartherm, transform, transform!, randsample
export Rampγ², Rampδ², Rampμ, StraightShot
export Fixk, Fixγ², Fixδ²
export SpectralProblem, SpectralSolver, solve, solve!
export spectral_density
export automesh, automesh!

const MissingOr{T} = Union{Missing, T}

include("matrix_utils.jl")
include("ensemble.jl")
include("sampling.jl")
include("objectives.jl")
include("workspace.jl")
include("algorithms.jl")
include("solutions.jl")
include("newton.jl")
include("qsolve.jl")
include("solve.jl")
include("density.jl")
include("automesh.jl")
include("printing.jl")
include("settings.jl")

using Requires

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        export CuMatrixEnsemble, CuHybridMatrixEnsemble
        include("cuda.jl")
        include("cudahybrid.jl")
    end
    @require LazySets="b4f0291d-fe17-52bc-9479-3d1a343d9043" begin
        export spectral_region, spectral_region!
        include("spectral_region.jl")
    end
end

end
