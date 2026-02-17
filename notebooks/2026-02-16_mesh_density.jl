### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c2c3cda0-06ee-11f1-9285-551483cbf55e
begin
    if occursin("intel", lowercase(first(Sys.cpu_info()).model))
        using MKL  # From global env
    elseif Sys.isapple()
        using AppleAccelerate  # From global env
    end
    using BenchmarkTools, Revise  # From global env

    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
    using Pkg: Pkg
    Pkg.activate(temp=true)
    Pkg.develop(path="..")
    Pkg.add([
		"CairoMakie",
		"DistMesh",
		"GeometryBasics",
		"LinearAlgebra",
		"StaticArrays",
		"Statistics",
	])
    Pkg.precompile()
end

# ╔═╡ 07c3373f-fefb-49f3-8384-0f6826ec2a4a
begin
	using GaussianSpectralDensities.Algorithms: SpreadAnnealing
	using GaussianSpectralDensities.Boundaries: extract_boundaries
	using GaussianSpectralDensities.Spreads: shift
	using GaussianSpectralDensities.Densities:
		mesh_gradient_per_face, mesh_gradient_per_vertex, signed_face_area
	using GaussianSpectralDensities.Ensembles: MatrixEnsemble
	using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
	using GaussianSpectralDensities.Samples: sample_eigenvalues
	using GaussianSpectralDensities.Solutions: greens, issuccess
	using GaussianSpectralDensities.Surrogates:
		SurrogateModel, evaluate, evaluatebatch, unpack_args

	using CairoMakie:
		Axis, Colorbar, DataAspect, Figure, hlines!, lines!, mesh!, plot, scatter!, vlines!, @L_str
	using DistMesh: distmesh2d
	using GeometryBasics: GLTriangleFace, Mesh, Point, per_face
	using LinearAlgebra: I, Diagonal, dot
	using StaticArrays: SVector
	using Statistics: mean
end

# ╔═╡ 8266b399-d711-473e-9a55-ed837e65db0d
e = let
	#MatrixEnsemble(0.0I, I, I, 0.0I)
    MatrixEnsemble(Diagonal([1.05, 1.05im, -1.05, -1.05im]), I, I, 0.5cispi(1//6)I)
	#MatrixEnsemble(Diagonal([1.25, 1.25im, -1.25, -1.25im]), I, I, 0.5cispi(1//3)I)
end

# ╔═╡ d2733d97-69ce-4ecd-9e0a-e2f040912b82
eq = ResolventEquation(e; shiftmin=-0.5)

# ╔═╡ c076126d-98a9-45aa-a144-1bc4803a87b5
constraint = PTraceConstraint(1; shape0=0, shift0=1)

# ╔═╡ 6ed5f7f7-5175-4e34-b76b-94a282d652a7
Npre = 64

# ╔═╡ d311b9b0-36fb-4446-95b7-fd323edd3853
premodel = SurrogateModel(eq, constraint, SpreadAnnealing(), Npre)

# ╔═╡ f260bf48-a72e-4857-a7d9-e1da13af5d4e
prepoints = map(splat(complex), premodel.rbf.x)

# ╔═╡ 86074959-e77b-4b96-ad68-5c12480422b7
bbox = (minimum.(reim(prepoints)), maximum.(reim(prepoints)))

# ╔═╡ 3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	vlines!(ax, collect(first.(bbox)))
	hlines!(ax, collect(last.(bbox)))
	scatter!(ax, real(prepoints), imag(prepoints))
	fig
end

# ╔═╡ c14d5c71-57e6-421c-ba7a-1015eb5675d8
premsh_h = (3 // 2)mean(premodel.rbf.nn_dists)

# ╔═╡ 057a9734-46e7-4ea9-b945-5316a2f97889
premsh_s_offset = 1 // 6

# ╔═╡ b5e038f2-8004-43fa-98ae-713a796679a2
premsh = distmesh2d(one ∘ eltype, premsh_h, bbox) do p
	sol = evaluate(premodel, complex(p...); refine=false)
	s = shift(sol)
	return issuccess(sol) ? -(s + premsh_s_offset) : oftype(s, Inf)
end

# ╔═╡ 0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
function plot_boundaries!(ax, boundaries)
	for b in boundaries
		(; points, internal) = b.result
		lines!(
			ax, real(points), imag(points);
			color=internal ? :green : :red, linewidth=2,
		)
	end
	return nothing
end

# ╔═╡ 908325bc-6a3f-4bbc-af04-d7ca73f4d6e1
model = SurrogateModel(eq, constraint, SpreadAnnealing(), map(splat(complex), premsh.p))

# ╔═╡ 9b71d29d-6b7d-4a3c-a34f-251a9248bbfd
step_coeff = 1 // 4

# ╔═╡ 4f9946e3-feda-404a-8316-2525848c0579
boundaries = extract_boundaries(model, step_coeff)

# ╔═╡ 4274301c-18c5-4329-a1f4-3f5a69c6574b
let fig=plot(premsh), ax=fig.current_axis[]
	vlines!(ax, collect(first.(bbox)))
	hlines!(ax, collect(last.(bbox)))
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 3b2ae836-9d7d-416a-844b-ddae1b8cfb4d
msh_h = step_coeff * mean(model.rbf.nn_dists)

# ╔═╡ 70608acf-2495-48d7-9e7d-ff9a85c7fe0a
msh = distmesh2d(one ∘ eltype, msh_h, bbox) do p
	sol = evaluate(model, complex(p...); refine=false)
	s = shift(sol)
	return issuccess(sol) ? -s : oftype(s, Inf)
end

# ╔═╡ dd30581a-ab0f-4fcd-a167-38d585717beb
let fig=plot(msh), ax=fig.current_axis[]
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 67e7082c-9747-4fc3-bbc6-9842dce52bf6
eigs = sample_eigenvalues(e, 1536);

# ╔═╡ 192115b0-68a8-430a-bdc8-cafd3aaf227d
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	plot_boundaries!(ax, boundaries)
	scatter!(ax, real(eigs), imag(eigs); markersize=4)
	fig
end

# ╔═╡ 652d89b6-218a-4d04-b3f9-b37ef82f792c
msh_zs = map(splat(complex), msh.p);

# ╔═╡ 3d1a53dc-dd97-4429-b754-dee7a67955f0
msh_sols = evaluatebatch(model, msh_zs; refine=false);

# ╔═╡ b8080bc0-b92c-4c22-82b0-1d76aff6842d
msh_shift = shift(msh_sols);

# ╔═╡ 0e6f9e9c-f353-4368-82fa-a097f5391613
msh_greens = greens(msh_sols);

# ╔═╡ 3fb17c5b-6191-41d0-a709-873f739bf984
msh_greens_re, msh_greens_im = reim(msh_greens);

# ╔═╡ c1abcb97-ccb9-423a-8345-28e915a9d38f
msh_greens_abs, msh_greens_arg = abs.(msh_greens), angle.(msh_greens);

# ╔═╡ 41e9aa36-6756-4425-89c5-7c03508449e9
msh_faces = reduce(vcat, transpose.(msh.t));

# ╔═╡ 9d41a80c-8e3f-49a6-98bf-a71a02557e4c
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(ax, msh.p, msh_faces; color=msh_shift, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(ax, msh.p, msh_faces; color=msh_greens_re, colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ aa86d73f-554f-499b-a593-490644042b2f
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(msh.p, msh_faces; color=msh_greens_im, colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 869a8407-9c6e-4a7b-9840-74ea47fc6ca7
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(ax, msh.p, msh_faces; color=msh_greens_abs, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ f6c3584c-8f44-4839-a5e4-45bcbe357768
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(ax, msh.p, msh_faces; color=msh_greens_arg, colorrange=(-π, π))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ c9e97073-ec93-487b-89a5-6f51192d11c2
msh_greens_grad = mesh_gradient_per_vertex(msh_greens, msh_zs, msh.t)

# ╔═╡ 6bf63dbb-043c-4350-b82c-5703c226436f
# NOTE: First vector in dot conjugated, hence the minus sign to get d_dzc
msh_dgreens_dzc = dot.([SVector(1 // 2, -im // 2)], msh_greens_grad)

# ╔═╡ 92cd0183-2a6e-4415-94ba-74b26ae32e81
msh_density = abs.(msh_dgreens_dzc) ./ π

# ╔═╡ bca18e3b-67fc-4a7d-803c-1bb18a5c6926
msh_face_area = map(ti -> abs(signed_face_area(msh_zs, ti)), msh.t)

# ╔═╡ f9d059f4-464f-4f58-85c6-39327eea9b5f
msh_area = sum(msh_face_area)

# ╔═╡ e45d65ff-d035-4dc5-b019-135dd51cb358
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(
		ax, msh.p, msh_faces; color=msh_density, colorrange=(0.0, 2inv(msh_area))
	)
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(ax, real(eigs), imag(eigs); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ cab4874c-365f-4f84-b5c7-0f07291bf2dc
msh_greens_grad_pf = mesh_gradient_per_face(msh_greens, msh_zs, msh.t)

# ╔═╡ 21d04fd4-b4e7-4c2c-974b-300d5fb4f940
msh_dgreens_dzc_pf = dot.([SVector(1 // 2, -im // 2)], msh_greens_grad_pf)

# ╔═╡ 3329c5d6-8a7f-47d4-9699-83105998d9f2
msh_density_pf = abs.(msh_dgreens_dzc_pf) ./ π

# ╔═╡ 230cf7e2-78e5-4533-b498-6c4cfb5ea463
dot(msh_face_area, msh_density_pf)  # density integral, should be == 1

# ╔═╡ 7f75e1ec-301e-4001-9bbf-571fab308b96
gmsh_points, gmsh_faces = Point.(msh.p), GLTriangleFace.(msh.t)

# ╔═╡ 975a513d-b38f-452f-9849-208923a3c412
gmsh_density = Mesh(
	gmsh_points, gmsh_faces; color=per_face(msh_density_pf, gmsh_faces)
)

# ╔═╡ bf1493cf-452d-4a02-820e-7c21738192e3
let fig=Figure(), ax=Axis(fig[1, 1]; aspect=DataAspect())
	co = mesh!(ax, gmsh_density; colorrange=(0.0, 2inv(msh_area)))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(ax, real(eigs), imag(eigs); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ Cell order:
# ╠═c2c3cda0-06ee-11f1-9285-551483cbf55e
# ╠═07c3373f-fefb-49f3-8384-0f6826ec2a4a
# ╠═8266b399-d711-473e-9a55-ed837e65db0d
# ╠═d2733d97-69ce-4ecd-9e0a-e2f040912b82
# ╠═c076126d-98a9-45aa-a144-1bc4803a87b5
# ╠═6ed5f7f7-5175-4e34-b76b-94a282d652a7
# ╠═d311b9b0-36fb-4446-95b7-fd323edd3853
# ╠═f260bf48-a72e-4857-a7d9-e1da13af5d4e
# ╠═86074959-e77b-4b96-ad68-5c12480422b7
# ╠═3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
# ╠═c14d5c71-57e6-421c-ba7a-1015eb5675d8
# ╠═057a9734-46e7-4ea9-b945-5316a2f97889
# ╠═b5e038f2-8004-43fa-98ae-713a796679a2
# ╠═4f9946e3-feda-404a-8316-2525848c0579
# ╠═0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
# ╠═908325bc-6a3f-4bbc-af04-d7ca73f4d6e1
# ╠═4274301c-18c5-4329-a1f4-3f5a69c6574b
# ╠═9b71d29d-6b7d-4a3c-a34f-251a9248bbfd
# ╠═3b2ae836-9d7d-416a-844b-ddae1b8cfb4d
# ╠═70608acf-2495-48d7-9e7d-ff9a85c7fe0a
# ╠═dd30581a-ab0f-4fcd-a167-38d585717beb
# ╠═67e7082c-9747-4fc3-bbc6-9842dce52bf6
# ╠═192115b0-68a8-430a-bdc8-cafd3aaf227d
# ╠═652d89b6-218a-4d04-b3f9-b37ef82f792c
# ╠═3d1a53dc-dd97-4429-b754-dee7a67955f0
# ╠═b8080bc0-b92c-4c22-82b0-1d76aff6842d
# ╠═0e6f9e9c-f353-4368-82fa-a097f5391613
# ╠═3fb17c5b-6191-41d0-a709-873f739bf984
# ╠═c1abcb97-ccb9-423a-8345-28e915a9d38f
# ╠═41e9aa36-6756-4425-89c5-7c03508449e9
# ╠═9d41a80c-8e3f-49a6-98bf-a71a02557e4c
# ╠═e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
# ╠═aa86d73f-554f-499b-a593-490644042b2f
# ╠═869a8407-9c6e-4a7b-9840-74ea47fc6ca7
# ╠═f6c3584c-8f44-4839-a5e4-45bcbe357768
# ╠═c9e97073-ec93-487b-89a5-6f51192d11c2
# ╠═6bf63dbb-043c-4350-b82c-5703c226436f
# ╠═92cd0183-2a6e-4415-94ba-74b26ae32e81
# ╠═bca18e3b-67fc-4a7d-803c-1bb18a5c6926
# ╠═f9d059f4-464f-4f58-85c6-39327eea9b5f
# ╠═e45d65ff-d035-4dc5-b019-135dd51cb358
# ╠═cab4874c-365f-4f84-b5c7-0f07291bf2dc
# ╠═21d04fd4-b4e7-4c2c-974b-300d5fb4f940
# ╠═3329c5d6-8a7f-47d4-9699-83105998d9f2
# ╠═230cf7e2-78e5-4533-b498-6c4cfb5ea463
# ╠═7f75e1ec-301e-4001-9bbf-571fab308b96
# ╠═975a513d-b38f-452f-9849-208923a3c412
# ╠═bf1493cf-452d-4a02-820e-7c21738192e3
