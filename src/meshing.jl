module Meshing

using DistMesh: DistMesh, distmesh2d, huniform
using StaticArrays: SVector
using Statistics: mean

using ..Solutions: issuccess
using ..Spreads: shift
using ..Surrogates: SurrogateModel, evaluate

public SupportMesh, dmesh, mesh, refine

struct SupportMesh{F<:AbstractFloat,I<:Integer,M<:SurrogateModel{F}}
    vertices::Vector{Complex{F}}
    faces::Vector{SVector{3,I}}
    # TODO: What to put here, actually? For fine meshes, the model _used to build it_ (like
    # here) is the one you're interested in; for coarse meshes, the model _built from the
    # mesh_ is more useful.
    model::M
end

function dmesh(support_mesh::SupportMesh)
    return DistMesh.DMesh(map(SVector ∘ reim, support_mesh.vertices), support_mesh.faces)
end

function mesh(
    model::SurrogateModel;
    h_coeff::Real=3 // 2,
    s_offset::Real=1 // 6,
    refinement::Union{Nothing,NamedTuple}=(;),
)
    bbox = bounding_box(model.rbf.x)
    msh = run_distmesh(model, h_coeff, bbox, s_offset)
    vertices = map(splat(complex), msh.p)
    support_mesh = SupportMesh(vertices, msh.t, model)
    isnothing(refinement) || return refine(support_mesh; refinement...)
    return support_mesh
end

function mesh(eq, constraint, alg, args...; kwargs...)
    return mesh(SurrogateModel(eq, constraint, alg, args...); kwargs...)
end

function refine(coarse::SupportMesh; h_coeff::Real=1 // 4, s_offset::Real=0)
    model = SurrogateModel(coarse.model, coarse.vertices)
    return mesh(model; h_coeff, s_offset, refinement=nothing)
end

function run_distmesh(model, h_coeff, bbox, s_offset)
    h = h_coeff * mean(model.rbf.nn_dists)
    return distmesh2d(huniform, h, bbox) do p
        sol = evaluate(model, complex(p...); refine=false)
        s = shift(sol)
        return issuccess(sol) ? -(s + s_offset) : oftype(s, Inf)
    end
end

function bounding_box(ps::AbstractVector{SVector{2,F}}) where {F}
    xmin, xmax = extrema(first, ps)
    ymin, ymax = extrema(last, ps)
    return ((xmin, ymin), (xmax, ymax))
end

end  # module Meshing

@testitem "mesh and refine" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Meshing: SupportMesh, mesh, refine
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I
    using StaticArrays: SVector

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    alg = DirectSolve()
    model = SurrogateModel(eq, constraint, alg)

    coarse = mesh(model; h_coeff=3 // 2, refinement=nothing)
    @test coarse isa SupportMesh{Float64}
    @test !isempty(coarse.vertices)
    @test !isempty(coarse.faces)

    fine = refine(coarse; h_coeff=1 // 2)
    @test fine isa SupportMesh{Float64}
    @test !isempty(fine.vertices)
    @test length(fine.vertices) > length(coarse.vertices)

    nv = length(fine.vertices)
    @test all(f -> all(i -> 1 <= i <= nv, f), fine.faces)
end

@testitem "multi-stage meshing" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Meshing: SupportMesh, mesh
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))

    support_mesh = mesh(
        eq,
        constraint,
        DirectSolve(),
        16;
        h_coeff=1,
        s_offset=1 // 6,
        refinement=(; h_coeff=1 // 2),
    )
    @test support_mesh isa SupportMesh{Float64}
    @test !isempty(support_mesh.vertices)
    @test !isempty(support_mesh.faces)
end
