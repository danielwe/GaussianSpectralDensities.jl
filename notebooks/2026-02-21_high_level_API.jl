### A Pluto.jl notebook ###
# v0.20.22

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
    Pkg.add(["CairoMakie", "CommonSolve", "GeometryBasics", "LinearAlgebra"])
    Pkg.precompile()
end

# ╔═╡ 07c3373f-fefb-49f3-8384-0f6826ec2a4a
begin
	using GaussianSpectralDensities.Boundaries: spectral_boundary
	using GaussianSpectralDensities.Densities: spectral_density, remove_faceviews
	using GaussianSpectralDensities.Ensembles: MatrixEnsemble
	using GaussianSpectralDensities.Meshing: Meshing
	using GaussianSpectralDensities.Samples: sample_eigenvalues
	using GaussianSpectralDensities.SpectralModels: SpectralProblem

	using CairoMakie:
		Axis, Colorbar, DataAspect, Figure, lines!, mesh!, poly!, scatter!, @L_str
	using CommonSolve: solve  # TODO: probably reexport?
	using GeometryBasics: area
	using LinearAlgebra: I, Diagonal
end

# ╔═╡ 8266b399-d711-473e-9a55-ed837e65db0d
e = let
	#MatrixEnsemble(0.0I, I, I, 0.0I)
    MatrixEnsemble(Diagonal([1.05, 1.05im, -1.05, -1.05im]), I, I, 0.5cispi(1//6)I)
	#MatrixEnsemble(Diagonal([1.25, 1.25im, -1.25, -1.25im]), I, I, 0.5cispi(1//3)I)
end

# ╔═╡ ddb47ea4-b92e-409e-9e45-4bd0210167f5
prob = SpectralProblem(e)

# ╔═╡ d2733d97-69ce-4ecd-9e0a-e2f040912b82
model = solve(prob; hcoeff=3//2)

# ╔═╡ 4f9946e3-feda-404a-8316-2525848c0579
boundary = spectral_boundary(model; step_size_coeff=1//4)

# ╔═╡ 67e7082c-9747-4fc3-bbc6-9842dce52bf6
eigvals = sample_eigenvalues(e, 1536);

# ╔═╡ 875b4f94-99d9-412c-a7b3-d876a322d5b3
function figax()
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	return fig, ax
end

# ╔═╡ 0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
function plotboundary!(ax, boundaries)
	for segment in boundary
		(; points, internal) = segment.result
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
	plotboundary!(ax, boundary)
	scatter!(
		ax, real(eigvals), imag(eigvals); markersize=4, color=:black, alpha=0.5
	)
	fig
end

# ╔═╡ f15c2ae5-f9c3-40ea-8d0a-9d65abe078f6
msh = spectral_density(model; hcoeff=1//4)

# ╔═╡ dd30581a-ab0f-4fcd-a167-38d585717beb
let
	fig, ax = figax()
	poly!(ax, msh; color=:gray90, strokewidth=1)
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ 2dfdb54f-6ad8-4fbf-8125-78463ff9ed43
function plotmesh!(ax, msh; colorbar=nothing, kwargs...)
	# When using a vertex attribute as color, the mesh can't contain face views.
	# Probably a Makie bug.
	msh_fixed = (:color in keys(kwargs)) ? remove_faceviews(msh) : msh
	mshplot = mesh!(ax, msh_fixed; kwargs...)
	if !isnothing(colorbar)
		Colorbar(colorbar, mshplot)
	end
	return mshplot
end

# ╔═╡ 9d41a80c-8e3f-49a6-98bf-a71a02557e4c
let
	fig, ax = figax()
	plotmesh!(ax, msh; color=msh.shift, colorrange=(0.0, 1.0), colorbar=fig[1, 2])
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
let
	fig, ax = figax()
	co = plotmesh!(
		ax, msh; color=real.(msh.greens), colorrange=(-1.0, 1.0), colorbar=fig[1, 2]
	)
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ aa86d73f-554f-499b-a593-490644042b2f
let
	fig, ax = figax()
	co = plotmesh!(
		ax, msh; color=imag.(msh.greens), colorrange=(-1.0, 1.0), colorbar=fig[1, 2]
	)
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ 869a8407-9c6e-4a7b-9840-74ea47fc6ca7
let
	fig, ax = figax()
	co = plotmesh!(
		ax, msh; color=abs.(msh.greens), colorrange=(-1.0, 1.0), colorbar=fig[1, 2]
	)
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ f6c3584c-8f44-4839-a5e4-45bcbe357768
let
	fig, ax = figax()
	co = plotmesh!(
		ax, msh; color=angle.(msh.greens), colorrange=(-π, π), colorbar=fig[1, 2]
	)
	plotboundary!(ax, boundary)
	fig
end

# ╔═╡ 786515af-0b15-4f06-aba0-81f6d8d45e87
densityrange = (0.0, 2inv(area(msh.position, msh.faces)))

# ╔═╡ e45d65ff-d035-4dc5-b019-135dd51cb358
let
	fig, ax = figax()
	plotmesh!(
		ax, msh;
		color=msh.vertex_density, colorrange=densityrange, colorbar=fig[1, 2],
	)
	plotboundary!(ax, boundary)
	scatter!(ax, real(eigvals), imag(eigvals);markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ 324b06ed-0521-42b1-88cb-57b9ecbd6718
msh_colored_by_density = Meshing.updatemesh(msh; color=msh.density)

# ╔═╡ bf1493cf-452d-4a02-820e-7c21738192e3
let
	fig, ax = figax()
	plotmesh!(
		ax, msh_colored_by_density; colorrange=densityrange, colorbar=fig[1, 2]
	)
	plotboundary!(ax, boundary)
	scatter!(ax, real(eigvals), imag(eigvals); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ Cell order:
# ╠═c2c3cda0-06ee-11f1-9285-551483cbf55e
# ╠═07c3373f-fefb-49f3-8384-0f6826ec2a4a
# ╠═8266b399-d711-473e-9a55-ed837e65db0d
# ╠═ddb47ea4-b92e-409e-9e45-4bd0210167f5
# ╠═d2733d97-69ce-4ecd-9e0a-e2f040912b82
# ╠═4f9946e3-feda-404a-8316-2525848c0579
# ╠═67e7082c-9747-4fc3-bbc6-9842dce52bf6
# ╠═875b4f94-99d9-412c-a7b3-d876a322d5b3
# ╠═0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
# ╠═4274301c-18c5-4329-a1f4-3f5a69c6574b
# ╠═f15c2ae5-f9c3-40ea-8d0a-9d65abe078f6
# ╠═dd30581a-ab0f-4fcd-a167-38d585717beb
# ╠═2dfdb54f-6ad8-4fbf-8125-78463ff9ed43
# ╠═9d41a80c-8e3f-49a6-98bf-a71a02557e4c
# ╠═e8b72031-5b52-4afe-8ff0-5a3dbc0016e5
# ╠═aa86d73f-554f-499b-a593-490644042b2f
# ╠═869a8407-9c6e-4a7b-9840-74ea47fc6ca7
# ╠═f6c3584c-8f44-4839-a5e4-45bcbe357768
# ╠═786515af-0b15-4f06-aba0-81f6d8d45e87
# ╠═e45d65ff-d035-4dc5-b019-135dd51cb358
# ╠═324b06ed-0521-42b1-88cb-57b9ecbd6718
# ╠═bf1493cf-452d-4a02-820e-7c21738192e3
