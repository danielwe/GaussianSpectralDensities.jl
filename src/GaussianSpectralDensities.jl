module GaussianSpectralDensities

using LinearAlgebra
using StaticArrays
using StructArrays
using ThreadPools
using Printf

#=
* An elegant way to generate a masked grid would be to use LazySets, but I don't think that
  belongs in this package. This can be done in demo notebooks.
* Or, actually, it woule be nice to automate the process of defining the domain by computing
  the convex hull of the spectrum of a randomly sampled instance,  and bloating it a little.
  Ideally this functionality should be implemented in a submodule with optional
  dependencies.
* Define plot recipes to simplify notebooks. This should perhaps also go in a separate
  module with optional dependency on RecipesBase.
=#

export MatrixEnsemble, SpectralProblem, SpectralSolver
export Rampγ², Rampδ², Rampμ, StraightShot
export Fixk, Fixγ², Fixδ²
export solve, solve!
export partially_hermitize, sample

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
include("printing.jl")
include("settings.jl")

using Requires

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        export CuMatrixEnsemble, CuHybridMatrixEnsemble
        include("cuda.jl")
        include("cudahybrid.jl")
    end
end

end
