### A Pluto.jl notebook ###
# v0.20.23

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
	using GaussianSpectralDensities:
		Densities,
		MatrixEnsemble,
		Meshing,
		PTraceConstraint,
		ResolventEquation,
		SpreadAnnealing,
		Surrogates,
		sample_eigenvalues,
		spectral_boundary,
		spectral_density,
		zvals

	using CairoMakie:
		Axis, Colorbar, DataAspect, Figure, hlines!, lines!, mesh!, poly!, scatter!, vlines!, @L_str
	using GeometryBasics: area
	using LinearAlgebra: I, Diagonal
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
coarse_eigvals, coarse_mesh = let
	# The sampled eigenvalues aren't always good enough, so we catch and repeat
	local mod, msh
	while true
		mod = Surrogates.SurrogateModel(eq, constraint, SpreadAnnealing(), 64)
		try
			msh = Meshing.mesh(mod; hcoeff=3//2, shiftoffset=1//12)
			break
		catch
		end
	end
	zvals(mod), msh
end

# ╔═╡ 86074959-e77b-4b96-ad68-5c12480422b7
bbox = Meshing.boundingbox(coarse_eigvals)

# ╔═╡ 0e2a4852-0b80-4ff3-b5d0-2afe045044c0
model = Surrogates.SurrogateModel(
	eq, constraint, SpreadAnnealing(), zvals(coarse_mesh)
)

# ╔═╡ 4f9946e3-feda-404a-8316-2525848c0579
boundaries = spectral_boundary(model, 1 // 4)

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
	scatter!(ax, real(coarse_eigvals), imag(coarse_eigvals); color=:black, alpha=0.5)
	fig
end

# ╔═╡ 0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
function plot_boundaries!(ax, boundaries)
	for b in boundaries
		lines!(
			ax, real(b.points), imag(b.points);
			color=b.internal ? :green : :red, linewidth=2,
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
msh = Meshing.mesh(model; hcoeff=1//4, params=true)

# ╔═╡ 5dfe8231-74ab-4b72-abfc-85276e4bc054
msh_area = area(msh.position, msh.faces)

# ╔═╡ dd30581a-ab0f-4fcd-a167-38d585717beb
let
	fig, ax = figax()
	poly!(ax, msh; color=:gray90, strokewidth=1)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 67e7082c-9747-4fc3-bbc6-9842dce52bf6
eigvals = sample_eigenvalues(e, 1536);

# ╔═╡ 192115b0-68a8-430a-bdc8-cafd3aaf227d
let
	fig, ax = figax()
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals), imag(eigvals); markersize=4, color=:black, alpha=0.5
	)
	fig
end

# ╔═╡ 9d41a80c-8e3f-49a6-98bf-a71a02557e4c
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=msh.shift, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=real.(msh.greens), colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ aa86d73f-554f-499b-a593-490644042b2f
let
	fig, ax = figax()
	co = mesh!(msh; color=imag.(msh.greens), colorrange=(-1.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 869a8407-9c6e-4a7b-9840-74ea47fc6ca7
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=abs.(msh.greens), colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ f6c3584c-8f44-4839-a5e4-45bcbe357768
let
	fig, ax = figax()
	co = mesh!(ax, msh; color=angle.(msh.greens), colorrange=(-π, π))
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	fig
end

# ╔═╡ 557d845a-8d25-4982-b97d-fa147f16d53f
msh_with_density = spectral_density(msh)

# ╔═╡ 324b06ed-0521-42b1-88cb-57b9ecbd6718
msh_colored_by_density = Meshing.updatemesh(
	msh_with_density; color=msh_with_density.density
)

# ╔═╡ bf1493cf-452d-4a02-820e-7c21738192e3
let
	fig, ax = figax()
	co = mesh!(ax, msh_colored_by_density; colorrange=(0.0, 2inv(msh_area)))	   
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals), imag(eigvals); markersize=4, color=:black, alpha=0.5
	)
	fig
end

# ╔═╡ 92cd0183-2a6e-4415-94ba-74b26ae32e81
msh_with_vertex_density = Densities.area_weighted_average(msh_with_density, :density)

# ╔═╡ e45d65ff-d035-4dc5-b019-135dd51cb358
let
	fig, ax = figax()
	co = mesh!(
		# When using a vertex attribute as color, the mesh can't contain face views.
		# Probably a Makie bug.
		ax, Meshing.remove_faceviews(msh_with_vertex_density);
		color=msh_with_vertex_density.vertex_density,
		colorrange=(0.0, 2inv(msh_area)),
	)
	Colorbar(fig[1, 2], co)
	plot_boundaries!(ax, boundaries)
	scatter!(
		ax, real(eigvals), imag(eigvals);markersize=4, color=:black, alpha=0.5
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
# ╠═0e2a4852-0b80-4ff3-b5d0-2afe045044c0
# ╠═4f9946e3-feda-404a-8316-2525848c0579
# ╠═875b4f94-99d9-412c-a7b3-d876a322d5b3
# ╠═0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
# ╠═4274301c-18c5-4329-a1f4-3f5a69c6574b
# ╠═fc6e3681-2386-4a4f-be68-16d71606a7fe
# ╠═5dfe8231-74ab-4b72-abfc-85276e4bc054
# ╠═dd30581a-ab0f-4fcd-a167-38d585717beb
# ╠═67e7082c-9747-4fc3-bbc6-9842dce52bf6
# ╠═192115b0-68a8-430a-bdc8-cafd3aaf227d
# ╠═9d41a80c-8e3f-49a6-98bf-a71a02557e4c
# ╠═e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
# ╠═aa86d73f-554f-499b-a593-490644042b2f
# ╠═869a8407-9c6e-4a7b-9840-74ea47fc6ca7
# ╠═f6c3584c-8f44-4839-a5e4-45bcbe357768
# ╠═557d845a-8d25-4982-b97d-fa147f16d53f
# ╠═324b06ed-0521-42b1-88cb-57b9ecbd6718
# ╠═bf1493cf-452d-4a02-820e-7c21738192e3
# ╠═92cd0183-2a6e-4415-94ba-74b26ae32e81
# ╠═e45d65ff-d035-4dc5-b019-135dd51cb358
