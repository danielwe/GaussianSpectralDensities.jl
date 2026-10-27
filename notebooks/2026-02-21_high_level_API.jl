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
    Pkg.add(["CairoMakie", "GeometryBasics", "LinearAlgebra"])
    Pkg.precompile()
end

# ╔═╡ 07c3373f-fefb-49f3-8384-0f6826ec2a4a
begin
	using GaussianSpectralDensities:
		MatrixEnsemble, 
		Meshing,
		SpectralProblem,
		spectral_boundary,
		spectral_density,
		sample_eigenvalues,
		solve

	using CairoMakie
	using GeometryBasics: area
	using LinearAlgebra: I, Bidiagonal, Diagonal
end

# ╔═╡ 8266b399-d711-473e-9a55-ed837e65db0d
e = let
	#MatrixEnsemble(0.0I, I, I, 0.0I)
    MatrixEnsemble(Diagonal([1.05, 1.05im, -1.05, -1.05im]), I, I, 0.5cispi(1//6)I)
	# MatrixEnsemble(Matrix(Bidiagonal([1.05, 1.05im, -1.05, -1.05im], [0.0, 0.0, 0.0], :U)), I, I, 0.5cispi(1//6)I)
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
		aspect=DataAspect(), xlabel=L"\text{Re}(z)", ylabel=L"\text{Im}(z)",
	)
	return fig, ax
end

# ╔═╡ 0b02d2f7-b23b-4bc9-8e44-7a4241515ea6
function plotboundary!(ax, boundary)
	labeled = false
	for  segment in boundary
		labelkw = if (!labeled) && (!segment.internal)
			labeled = true
			(; label="Boundary")
		else
			(;)
		end
		lines!(
			ax, real(segment.points), imag(segment.points);
			color=segment.internal ? :green : :black, linewidth=1.5, labelkw...
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
function plotmesh!(ax, msh; colorbar=nothing, colorbarlabel=nothing, kwargs...)
	# When using a vertex attribute as color, the mesh can't contain face views.
	# Probably a Makie bug.
	msh_fixed = (:color in keys(kwargs)) ? Meshing.remove_faceviews(msh) : msh
	mshplot = mesh!(ax, msh_fixed; kwargs...)
	if !isnothing(colorbar)
		cbarkw = isnothing(colorbarlabel) ? (;) : (; label=colorbarlabel)
		Colorbar(colorbar, mshplot; cbarkw...)
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
# ╠═╡ disabled = true
#=╠═╡
densityrange = (0.0, 2inv(area(msh.position, msh.faces)))
  ╠═╡ =#

# ╔═╡ 4f91f7cb-0aa9-43d5-81d4-0373c9881b2f
densityrange = (0.0, maximum(msh.vertex_density))

# ╔═╡ c719e4da-e6fe-4584-96b0-20969add0f99
eJ = MatrixEnsemble(0I, e.L, e.R, e.S)

# ╔═╡ 6554d6e6-8f8a-454f-bcc4-25225d332288
eigvalsJ = sample_eigenvalues(eJ, 1536)

# ╔═╡ 8d6de195-0907-439d-8611-0e4fffc95916
probJ = SpectralProblem(eJ)

# ╔═╡ e71d6df7-bca0-421d-a3f7-a7c135842bc8
modelJ = solve(probJ; hcoeff=3//2)

# ╔═╡ e532ac67-b05c-4272-9f63-bf6a61246156
boundaryJ = spectral_boundary(modelJ; step_size_coeff=1//4)

# ╔═╡ 7cd479ca-c18d-4cee-a7f8-9d4c1eab61f0
mshJ = spectral_density(modelJ; hcoeff=1//4)

# ╔═╡ 1093e46f-ac96-4cc6-8b2e-8cc323a826bd
# ╠═╡ disabled = true
#=╠═╡
densityrangeJ = (0.0, 2inv(area(mshJ.position, mshJ.faces)))
  ╠═╡ =#

# ╔═╡ ae57960b-cede-4683-b87a-80a6230587b2
densityrangeJ = (0.0, maximum(mshJ.vertex_density))

# ╔═╡ 874f58ca-5a03-4f05-bf6e-e65fc9c211c1
let
	fig, ax = figax()
	plotmesh!(
		ax, mshJ;
		color=mshJ.vertex_density, colorrange=densityrangeJ, colorbar=fig[1, 2],
		colorbarlabel=L"\rho(z)",
	)
	plotboundary!(ax, boundaryJ)
	scatter!(
		ax, real(eigvalsJ), imag(eigvalsJ);
		markersize=4, color=:black, alpha=0.5, label=L"\lambda[J]",
	)
	axislegend(ax; position=:lt)
	fig
end

# ╔═╡ e45d65ff-d035-4dc5-b019-135dd51cb358
let Mev = parent(e.M)
	fig, ax = figax()
	plotmesh!(
		ax, msh;
		color=msh.vertex_density, colorrange=densityrangeJ, colorbar=fig[1, 2],
		colorbarlabel=L"\rho(z)",
	)
	plotboundary!(ax, boundary)
	scatter!(
		ax, real(eigvals), imag(eigvals);
		markersize=4, color=:black, alpha=0.5, label=L"\lambda[A]",
	)
	scatter!(
		ax, real(Mev), imag(Mev);
		markersize=10, marker=:diamond, color=:magenta, alpha=1, label=L"\lambda[M]",
	)
	axislegend(ax)
	
	ax_inset = Axis(
		fig[1, 1],
    	width=Relative(0.24),
    	height=Relative(0.24),
    	halign=0.95,
    	valign=0.08,
		aspect=DataAspect(),
		xticks=[-1.0, 0.0, 1.0],
		yticks=[-0.5, 0.0, 0.5],
    )
	plotmesh!(ax_inset, mshJ; color=mshJ.vertex_density, colorrange=densityrangeJ)
	plotboundary!(ax_inset, boundaryJ)
	scatter!(
		ax_inset, real(eigvalsJ), imag(eigvalsJ);
		markersize=2, color=:black, alpha=0.5, label=L"\lambda[J]",
	)
	# axislegend(ax_inset)
	save("/Users/daniel/Desktop/toy_example.pdf", fig)
	fig
end

# ╔═╡ 324b06ed-0521-42b1-88cb-57b9ecbd6718
msh_colored_by_density = Meshing.updatemesh(msh; color=msh.density)

# ╔═╡ bf1493cf-452d-4a02-820e-7c21738192e3
let
	fig, ax = figax()
	plotmesh!(
		ax, msh_colored_by_density;
		colorrange=densityrange, colorbar=fig[1, 2], colorbarlabel=L"\rho(z)",
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
# ╠═4f91f7cb-0aa9-43d5-81d4-0373c9881b2f
# ╠═c719e4da-e6fe-4584-96b0-20969add0f99
# ╠═6554d6e6-8f8a-454f-bcc4-25225d332288
# ╠═8d6de195-0907-439d-8611-0e4fffc95916
# ╠═e71d6df7-bca0-421d-a3f7-a7c135842bc8
# ╠═e532ac67-b05c-4272-9f63-bf6a61246156
# ╠═7cd479ca-c18d-4cee-a7f8-9d4c1eab61f0
# ╠═1093e46f-ac96-4cc6-8b2e-8cc323a826bd
# ╠═ae57960b-cede-4683-b87a-80a6230587b2
# ╠═874f58ca-5a03-4f05-bf6e-e65fc9c211c1
# ╠═e45d65ff-d035-4dc5-b019-135dd51cb358
# ╠═324b06ed-0521-42b1-88cb-57b9ecbd6718
# ╠═bf1493cf-452d-4a02-820e-7c21738192e3
