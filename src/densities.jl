module Densities

using GeometryBasics: FaceView, Mesh, per_face

using ..Meshing: Meshing

public spectral_density

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
    wirt_msh = Meshing.wirtinger_derivatives(msh, s; label=wirt_label)
    wirt = wirt_msh.vertex_attributes[wirt_label]::FaceView
    density = real.(last.(wirt.data)) ./ π
    return Meshing.updatemesh(msh; label => per_face(density, wirt.faces))
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

@testitem "grad low-level method" begin
    using GaussianSpectralDensities.Densities: grad

    using GeometryBasics: Point2, TriangleFace
    using StaticArrays: SVector

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    values = [2.0 * p[1] + 3.0 * p[2] for p in vertices]

    g = grad(values, vertices, faces)
    @test length(g) == length(faces)
    @testset "face = $i" for (i, gi) in pairs(g)
        @test gi ≈ SVector(2.0, 3.0)
    end
end

@testitem "area_weighted_average low-level method" begin
    using GaussianSpectralDensities.Densities: area_weighted_average

    using GeometryBasics: Point2, TriangleFace

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    face_values = [1.0, 3.0]

    vertex_values = area_weighted_average(face_values, vertices, faces)
    @test length(vertex_values) == length(vertices)
    @test vertex_values[1] == 2.0
    @test vertex_values[2] == 1.0
    @test vertex_values[3] == 2.0
    @test vertex_values[4] == 3.0
end

@testitem "remove_faceviews" begin
    using GaussianSpectralDensities.Densities: grad, remove_faceviews
    using GaussianSpectralDensities.Meshing: updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    vals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
    msh_f = updatemesh(msh; scalar=vals)

    # Add a FaceView via grad
    msh_g = grad(msh_f, :scalar)
    @test msh_g.vertex_attributes[:scalar_grad] isa FaceView

    # remove_faceviews should strip FaceViews but keep non-FaceView attrs
    msh_clean = remove_faceviews(msh_g)
    @test !haskey(msh_clean.vertex_attributes, :scalar_grad)
    @test haskey(msh_clean.vertex_attributes, :scalar)
end

@testitem "spectral_density low-level (Mesh)" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Densities: spectral_density
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Meshing: addresolventparams, mesh
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using CommonSolve: solve
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Densities: spectral_density
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint
    using GaussianSpectralDensities.SpectralModels: SpectralProblem

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
