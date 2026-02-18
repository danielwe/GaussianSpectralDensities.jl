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
	using GaussianSpectralDensities.Meshing: Meshing
	using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
	using GaussianSpectralDensities.Samples: sample_eigenvalues
	using GaussianSpectralDensities.Solutions: greens, issuccess
	using GaussianSpectralDensities.Surrogates:
		SurrogateModel, evaluate, evaluatebatch, unpack_args, zvals

	using CairoMakie:
		Axis, CairoMakie, Colorbar, DataAspect, Figure, hlines!, lines!, mesh!, poly!, scatter!, vlines!, @L_str
	using DistMesh: distmesh2d
	using GeometryBasics: GeometryBasics, per_face
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

# ╔═╡ a6337bdb-91ab-4fe5-8733-ebdeab64d44c
eigvals, coarse_mesh = let
	# The sampled eigenvalues aren't always good enough, so we catch and repeat
	local mod, msh
	while true
		mod = SurrogateModel(eq, constraint, SpreadAnnealing(), 64)
		try
			msh = Meshing.mesh(mod; hcoeff=3//2, shiftoffset=1//8)
			break
		catch
		end
	end
	zvals(mod), msh
end

# ╔═╡ 86074959-e77b-4b96-ad68-5c12480422b7
bbox = Meshing.boundingbox(eigvals)

# ╔═╡ aa75d5c3-cb38-4ffc-9283-f6a52cf3778a
coarse_zs = map(splat(complex), coarse_mesh.position)

# ╔═╡ 0e2a4852-0b80-4ff3-b5d0-2afe045044c0
model = SurrogateModel(eq, constraint, SpreadAnnealing(), coarse_zs)

# ╔═╡ 4f9946e3-feda-404a-8316-2525848c0579
boundaries = extract_boundaries(model, 1 // 4)

# ╔═╡ 875b4f94-99d9-412c-a7b3-d876a322d5b3
function figax()
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	return fig, ax
end

# ╔═╡ 3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
let
	fig, ax = figax()
	vlines!(ax, collect(first.(bbox)))
	hlines!(ax, collect(last.(bbox)))
	scatter!(ax, real(eigvals), imag(eigvals); color=:black, alpha=0.5)
	fig
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

# ╔═╡ 4274301c-18c5-4329-a1f4-3f5a69c6574b
let
	fig, ax = figax()
	poly!(ax, coarse_mesh; color=:grey90, strokewidth=1)
	vlines!(ax, collect(first.(bbox)))
	hlines!(ax, collect(last.(bbox)))
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ fc6e3681-2386-4a4f-be68-16d71606a7fe
msh = Meshing.mesh(model; hcoeff=1//4)

# ╔═╡ dd30581a-ab0f-4fcd-a167-38d585717beb
let
	fig, ax = figax()
	poly!(ax, msh; color=:gray90, strokewidth=1)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 67e7082c-9747-4fc3-bbc6-9842dce52bf6
eigvals_large = sample_eigenvalues(e, 1536);

# ╔═╡ 192115b0-68a8-430a-bdc8-cafd3aaf227d
let
	fig, ax = figax()
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals_large), imag(eigvals_large);
		markersize=4, color=:black, alpha=0.5,
	)
	fig
end

# ╔═╡ 67f5fd93-aad0-4e03-838b-f82060f5ad09
msh_zs = zvals(msh)

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

# ╔═╡ 9d41a80c-8e3f-49a6-98bf-a71a02557e4c
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh_shift, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh_greens_re, colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ aa86d73f-554f-499b-a593-490644042b2f
let
	fig, ax = figax()
	co = mesh!(msh; color=msh_greens_im, colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 869a8407-9c6e-4a7b-9840-74ea47fc6ca7
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh_greens_abs, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ f6c3584c-8f44-4839-a5e4-45bcbe357768
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh_greens_arg, colorrange=(-π, π))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ c9e97073-ec93-487b-89a5-6f51192d11c2
msh_greens_grad = mesh_gradient_per_vertex(msh_greens, msh.position, msh.faces)

# ╔═╡ 6bf63dbb-043c-4350-b82c-5703c226436f
# NOTE: First vector in dot conjugated, hence the minus sign to get d_dzc
msh_dgreens_dzc = dot.([SVector(1 // 2, -im // 2)], msh_greens_grad)

# ╔═╡ 92cd0183-2a6e-4415-94ba-74b26ae32e81
msh_density = abs.(msh_dgreens_dzc) ./ π

# ╔═╡ bca18e3b-67fc-4a7d-803c-1bb18a5c6926
msh_face_area = map(abs ∘ Base.Fix1(signed_face_area, msh.position), msh.faces)

# ╔═╡ f9d059f4-464f-4f58-85c6-39327eea9b5f
msh_area = sum(msh_face_area)

# ╔═╡ e45d65ff-d035-4dc5-b019-135dd51cb358
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh_density, colorrange=(0.0, 2inv(msh_area)))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals_large), imag(eigvals_large);
		markersize=4, color=:black, alpha=0.5,
	)
	fig
end

# ╔═╡ cab4874c-365f-4f84-b5c7-0f07291bf2dc
msh_greens_grad_pf = mesh_gradient_per_face(msh_greens, msh.position, msh.faces)

# ╔═╡ 21d04fd4-b4e7-4c2c-974b-300d5fb4f940
msh_dgreens_dzc_pf = dot.([SVector(1 // 2, -im // 2)], msh_greens_grad_pf)

# ╔═╡ 3329c5d6-8a7f-47d4-9699-83105998d9f2
msh_density_pf = abs.(msh_dgreens_dzc_pf) ./ π

# ╔═╡ 230cf7e2-78e5-4533-b498-6c4cfb5ea463
dot(msh_face_area, msh_density_pf)  # density integral, should be == 1

# ╔═╡ bb6f19a4-384a-4649-8cc3-30a0d3dab8d6
msh_with_density_pf = GeometryBasics.mesh(msh; color=per_face(msh_density_pf, msh))

# ╔═╡ bf1493cf-452d-4a02-820e-7c21738192e3
let
	fig, ax = figax()
	co = mesh!(ax, msh_with_density_pf; colorrange=(0.0, 2inv(msh_area)))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals_large), imag(eigvals_large);
		markersize=4, color=:black, alpha=0.5,
	)
	fig
end

# ╔═╡ Cell order:
# ╠═c2c3cda0-06ee-11f1-9285-551483cbf55e
# ╠═07c3373f-fefb-49f3-8384-0f6826ec2a4a
# ╠═8266b399-d711-473e-9a55-ed837e65db0d
# ╠═d2733d97-69ce-4ecd-9e0a-e2f040912b82
# ╠═c076126d-98a9-45aa-a144-1bc4803a87b5
# ╠═a6337bdb-91ab-4fe5-8733-ebdeab64d44c
# ╠═86074959-e77b-4b96-ad68-5c12480422b7
# ╠═3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
# ╠═aa75d5c3-cb38-4ffc-9283-f6a52cf3778a
# ╠═0e2a4852-0b80-4ff3-b5d0-2afe045044c0
# ╠═4f9946e3-feda-404a-8316-2525848c0579
# ╠═875b4f94-99d9-412c-a7b3-d876a322d5b3
# ╠═0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
# ╠═4274301c-18c5-4329-a1f4-3f5a69c6574b
# ╠═fc6e3681-2386-4a4f-be68-16d71606a7fe
# ╠═dd30581a-ab0f-4fcd-a167-38d585717beb
# ╠═67e7082c-9747-4fc3-bbc6-9842dce52bf6
# ╠═192115b0-68a8-430a-bdc8-cafd3aaf227d
# ╠═67f5fd93-aad0-4e03-838b-f82060f5ad09
# ╠═3d1a53dc-dd97-4429-b754-dee7a67955f0
# ╠═b8080bc0-b92c-4c22-82b0-1d76aff6842d
# ╠═0e6f9e9c-f353-4368-82fa-a097f5391613
# ╠═3fb17c5b-6191-41d0-a709-873f739bf984
# ╠═c1abcb97-ccb9-423a-8345-28e915a9d38f
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
# ╠═bb6f19a4-384a-4649-8cc3-30a0d3dab8d6
# ╠═bf1493cf-452d-4a02-820e-7c21738192e3
