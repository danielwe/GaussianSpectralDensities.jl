module Densities

using GeometryBasics: FaceView, Mesh, per_face
using LinearAlgebra: dot
using StaticArrays: SVector, StaticVector

using ..Meshing: Meshing, updatemesh
using ..SpectralModels: SpectralModel

public area_weighted_average, grad, spectral_density, wirtinger_derivatives

"""
    spectral_density(model::SpectralModel; hcoeff=1//4, kwargs...) -> Mesh

Build a fine mesh from `model.surrogate`, compute spectral density, and add an
area-weighted vertex density attribute.
"""
function spectral_density(model::SpectralModel; hcoeff=1 // 4, kwargs...)
    msh = Meshing.mesh(model.surrogate; hcoeff, params=true, kwargs...)
    msh = spectral_density(msh)
    msh = area_weighted_average(msh, :density)
    return msh
end

"""
    spectral_density(mesh, s=:greens; label=:density)

Compute the spectral density of a matrix ensemble from a mesh with resolvent parameters.

    density = |∂t/∂z̄| / π; here t is the ensemble's Green's function (normalized trace of
                           the ensemble-averaged resolvent).

Returns a new mesh with an additional FaceView attribute `mesh.vertex_attributes[label]`
containing the spectral density.
"""
function spectral_density(msh::Mesh{2}, s=:greens; label=:density)
    wirt_label = Symbol(s, "_wirt")
    wirt_msh = wirtinger_derivatives(msh, s; label=wirt_label)
    wirt = wirt_msh.vertex_attributes[wirt_label]::FaceView
    density = real.(last.(wirt.data)) ./ π
    return updatemesh(msh; label => per_face(density, wirt.faces))
end

"""
    grad(msh, s; label=Symbol(s, "_grad")) -> Mesh

Compute per-face gradients of a field over a 2D mesh using linear FEM basis functions.
"""
function grad(msh::Mesh, s::Symbol; label=Symbol(s, "_grad"))
    g = grad(msh.vertex_attributes[s], msh.position, msh.faces)
    return updatemesh(msh; label => per_face(g, msh))
end

function grad(
    values::AbstractArray{<:Any,N},
    vertices::AbstractArray{<:StaticVector{2,<:Real},N},
    faces::AbstractVector{<:StaticVector{3,<:Integer}},
) where {N}
    return map(faces) do face
        vals = face_vertex_vals(values, face)

        dps = face_edge_dvals(vertices, face)
        dxs, dys = first.(dps), last.(dps)
        signed_area2 = cross1(dxs, dys)

        # Important: pass the coordinate triple, always real, as first arg to avoid dot's conjugation
        dfdx = dot(dys, vals) / signed_area2
        dfdy = -dot(dxs, vals) / signed_area2
        return SVector(dfdx, dfdy)
    end
end

function signed_face_area(
    vertices::AbstractArray{<:StaticVector{2,<:Real}}, face::StaticVector{3,<:Integer}
)
    dps = face_edge_dvals(vertices, face)
    return cross1(first.(dps), last.(dps)) / 2
end

cross1(a, b) = (a[2] * b[3]) - (a[3] * b[2])

function face_edge_dvals(vertex_vals::AbstractArray, face::StaticVector{3,<:Integer})
    face_vvals = face_vertex_vals(vertex_vals, face)
    return (
        face_vvals[2] - face_vvals[3],
        face_vvals[3] - face_vvals[1],
        face_vvals[1] - face_vvals[2],
    )
end

function face_vertex_vals(vertex_vals::AbstractArray, face::StaticVector{3,<:Integer})
    return ntuple(i -> vertex_vals[face[i]], Val(3))
end

"""
    area_weighted_average(msh, s; label=Symbol("vertex_", s)) -> Mesh

Map a per-face mesh attribute to vertices using face-area weighting.
"""
function area_weighted_average(msh::Mesh, s::Symbol; label=Symbol("vertex_", s))
    face_values = msh.vertex_attributes[s]::FaceView
    vertex_values = area_weighted_average(face_values.data, msh.position, msh.faces)
    return updatemesh(msh; label => vertex_values)
end

function area_weighted_average(
    face_values::AbstractVector{T},
    vertices::AbstractArray{<:StaticVector{2,F}},
    faces::AbstractVector{<:StaticVector{3,<:Integer}},
) where {F<:Real,T}
    zero_area = zero(F)^2
    zero_value = zero(T) * zero_area
    aw_values, aweights = fill(zero_value, size(vertices)), fill(zero_area, size(vertices))
    for (face, face_val) in zip(faces, face_values)
        area = abs(signed_face_area(vertices, face))
        for I in face
            aw_values[I] += area * face_val
            aweights[I] += area
        end
    end
    return aw_values ./ aweights
end

"""
    wirtinger_derivatives(mesh, s; label=Symbol(s, "_wirt") -> Mesh

Compute the Wirtinger derivatives of a scalar field on a 2D mesh over the complex plane.
"""
function wirtinger_derivatives(
    mesh::Mesh{2,F}, s::Symbol; label=Symbol(s, "_wirt")
) where {F<:Real}
    grad_label = Symbol(s, "_grad")
    grad_msh = grad(mesh, s; label=grad_label)
    grd = grad_msh.vertex_attributes[grad_label]::FaceView
    # ∂f/∂z = (∂f/∂x - i∂f/∂y) / 2
    # ∂f/∂z̄ = (∂f/∂x + i∂f/∂y) / 2
    wirt_data = map(grd.data) do (gx, gy)::SVector{2}
        gx_2, igy_2 = gx / 2, im * gy / 2
        return SVector(gx_2 - igy_2, gx_2 + igy_2)
    end
    return updatemesh(grad_msh; label => per_face(wirt_data, grd.faces))
end

end  # module Densities

@testitem "grad with real-valued linear field" begin
    using GaussianSpectralDensities.Densities: grad
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace
    using StaticArrays: SVector

    # Unit square split into two triangles
    #   4--3
    #   |\ |
    #   | \|
    #   1--2
    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    # f(x, y) = 2x + 3y — gradient should be exactly [2, 3] on every face
    vals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    msh_g = grad(msh_f, :scalar)
    gdata = msh_g.vertex_attributes[:scalar_grad]
    @test gdata isa FaceView
    @testset "face = $i" for (i, g) in pairs(gdata.data)
        @test g ≈ SVector(2.0, 3.0)
    end
end

@testitem "grad with complex-valued linear field" begin
    using GaussianSpectralDensities.Densities: grad
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace
    using StaticArrays: SVector

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    vals = [(2.0 + 1.0im) * p[1] + (3.0 - 1.0im) * p[2] for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    msh_g = grad(msh_f, :scalar)
    gdata = msh_g.vertex_attributes[:scalar_grad]
    @test gdata isa FaceView
    @testset "face = $i" for (i, g) in pairs(gdata.data)
        @test g ≈ SVector(2.0 + 1.0im, 3.0 - 1.0im)
    end
end

@testitem "wirtinger_derivatives of anti-analytic function" begin
    using GaussianSpectralDensities.Densities: wirtinger_derivatives
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    # f(z) = z̄ = x - iy, exactly linear => ∂f/∂z = 0, ∂f/∂z̄ = 1 exactly
    vals = [p[1] - im * p[2] for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    msh_w = wirtinger_derivatives(msh_f, :scalar)
    wdata = msh_w.vertex_attributes[:scalar_wirt]
    @test wdata isa FaceView
    @testset "face = $i" for (i, w) in pairs(wdata.data)
        @test w[1] ≈ 0  # ∂f/∂z
        @test w[2] ≈ 1  # ∂f/∂z̄
    end
end

@testitem "wirtinger_derivatives of analytic function" begin
    using GaussianSpectralDensities.Densities: wirtinger_derivatives
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    # f(z) = z = x + iy => ∂f/∂z = 1, ∂f/∂z̄ = 0 exactly
    vals = [complex(p[1], p[2]) for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    msh_w = wirtinger_derivatives(msh_f, :scalar)
    wdata = msh_w.vertex_attributes[:scalar_wirt]
    @test wdata isa FaceView
    @testset "face = $i" for (i, w) in pairs(wdata.data)
        @test w[1] ≈ 1  # ∂f/∂z
        @test w[2] ≈ 0  # ∂f/∂z̄
    end
end

@testitem "area_weighted_average round-trip" begin
    using GaussianSpectralDensities.Densities: area_weighted_average, grad
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: Mesh, Point2, TriangleFace
    using StaticArrays: SVector

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    # For a linear field the per-face gradient is constant, so area-weighted
    # averaging back to vertices should recover that same constant at every vertex.
    vals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    msh_g = grad(msh_f, :scalar)
    msh_a = area_weighted_average(msh_g, :scalar_grad)
    vg = msh_a.vertex_attributes[:vertex_scalar_grad]
    @testset "vertex = $i" for (i, g) in pairs(vg)
        @test g ≈ SVector(2.0, 3.0)
    end
end

@testitem "spectral_density low-level (Mesh)" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Densities: spectral_density
    using GaussianSpectralDensities.Meshing: addresolventparams, mesh
    using GaussianSpectralDensities.Surrogates: SurrogateModel, zvals

    using GeometryBasics: FaceView, Mesh
    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    alg = DirectSolve()

    coarse = mesh(eq, constraint, alg, 16; hcoeff=1 // 2, shiftoffset=1 // 6)
    model = SurrogateModel(eq, constraint, alg, zvals(coarse))
    fine = mesh(model; hcoeff=1 // 2, params=true)

    density_msh = spectral_density(fine)
    @test density_msh isa Mesh{2,Float64}
    @test haskey(density_msh.vertex_attributes, :density)
    @test density_msh.vertex_attributes[:density] isa FaceView

    # Density values should be non-negative (spectral density ≥ 0)
    density_vals = density_msh.vertex_attributes[:density].data
    @test all(d -> d >= -1e-10, density_vals)

    # At least some density values should be positive (inside support)
    @test any(d -> d > 0.01, density_vals)
end

@testitem "spectral_density high-level (SpectralModel)" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, SpectralProblem, solve
    using GaussianSpectralDensities.Densities: spectral_density

    using GeometryBasics: FaceView, Mesh
    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    prob = SpectralProblem(ensemble, constraint)
    model = solve(prob, DirectSolve(); nsamples=16)

    density_msh = spectral_density(model)
    @test density_msh isa Mesh{2,Float64}

    # Should have density FaceView and vertex_density
    @test haskey(density_msh.vertex_attributes, :density)
    @test density_msh.vertex_attributes[:density] isa FaceView
    @test haskey(density_msh.vertex_attributes, :vertex_density)

    # Density values should be non-negative
    density_vals = density_msh.vertex_attributes[:density].data
    @test all(d -> d >= -1e-10, density_vals)
    @test any(d -> d > 0.01, density_vals)

    # Vertex density should also be non-negative
    vertex_density = density_msh.vertex_attributes[:vertex_density]
    @test all(d -> d >= -1e-10, vertex_density)
    @test any(d -> d > 0.01, vertex_density)
end
