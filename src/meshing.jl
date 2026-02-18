module Meshing

using GeometryBasics: Mesh, Point2, TriangleFace
using DistMesh: distmesh2d, huniform
using Statistics: mean

using ..Solutions: issuccess
using ..Spreads: shift
using ..Surrogates: SurrogateModel, Surrogates, evaluate

public boundingbox, mesh

function mesh(model::SurrogateModel; hcoeff::Real=1, shiftoffset::Real=0)
    h = hcoeff * mean(model.rbf.nn_dists)
    bbox = boundingbox(model.rbf.x)
    msh = distmesh2d(huniform, h, bbox) do p
        sol = evaluate(model, complex(p...); refine=false)
        s = shift(sol)
        return issuccess(sol) ? -(s + shiftoffset) : oftype(s, Inf)
    end
    return Mesh(map(Point2, msh.p), map(TriangleFace, msh.t))
end

function mesh(eq, constraint, alg, args...; kwargs...)
    return mesh(SurrogateModel(eq, constraint, alg, args...); kwargs...)
end

function boundingbox(ps::AbstractVector)
    ffirst, flast = (eltype(ps) <: Complex) ? (real, imag) : (first, last)
    xmin, xmax = extrema(ffirst, ps)
    ymin, ymax = extrema(flast, ps)
    return ((xmin, ymin), (xmax, ymax))
end

Surrogates.zvals(msh::Mesh) = map(splat(complex), msh.position)

end  # module Meshing

@testitem "meshing" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Meshing: mesh
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel, zvals

    using GeometryBasics: Mesh
    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    alg = DirectSolve()

    coarse = mesh(eq, constraint, alg, 16; hcoeff=1 // 2, shiftoffset=1 // 6)
    @test coarse isa Mesh{2,Float64}
    @test !isempty(coarse.position)
    @test !isempty(coarse.faces)

    coarse_model = SurrogateModel(eq, constraint, alg, zvals(coarse))
    fine = mesh(coarse_model; hcoeff=1 // 2)
    @test fine isa Mesh{2,Float64}
    @test !isempty(fine.position)
    @test length(fine.position) > length(coarse.position)

    nv = length(fine.position)
    @test all(f -> all(i -> 1 <= i <= nv, f), fine.faces)
end
