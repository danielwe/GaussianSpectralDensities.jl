### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 42838418-1275-11f1-b960-732f0011f5a1
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
		"GeometryBasics",
		"LinearAlgebra",
		"OrdinaryDiffEqTsit5",
	])
    Pkg.precompile()
end


# ╔═╡ 8264147d-7091-40c3-9010-a8fd21e63bc9
begin
	using GaussianSpectralDensities:
		Defaults,
		MatrixEnsemble, 
		Meshing,
		PTraceConstraint,
		SpectralProblem,
		SpreadAnnealing,
		spectral_boundary,
		spectral_density,
		sample_eigenvalues,
		sample_matrix,
		solve

	using CairoMakie
	using GeometryBasics: area
	using LinearAlgebra: I, Diagonal, diagind, eigvals, mul!, norm
	using OrdinaryDiffEqTsit5: ODEProblem, Tsit5
end

# ╔═╡ e40ad15b-0a3a-4cde-bae4-dea6b8718578
τ = 0.8

# ╔═╡ e97c14ed-3cf8-4cec-ad39-af066403f116
σ = sqrt(τ)

# ╔═╡ 3f74e29d-78de-4836-b2b0-c57dd1a67df5
N = 1024

# ╔═╡ 9986bbe6-59ad-4a2c-a6d9-6961f5c748c3
p = (; a=-0.75)

# ╔═╡ df376816-7316-44b7-aaba-0fcc997474e5
coupling_ensemble = MatrixEnsemble(0I, I, I, σ * I)

# ╔═╡ 062f96f9-6c5a-4152-a96a-eb06ca75d845
f!, Af_diag, Af = let J=sample_matrix(coupling_ensemble, N)
	for i in diagind(J)
		J[i] = 0
	end
	function f!(du, u, p, t)
		map!(ui -> (p.a - ui^2) * ui, du, u)
		mul!(du, J, u, true, true)
		return nothing
	end
	Af_diag(u, p, t) = Diagonal(map(ui -> (p.a - 3(ui^2)), u))
	Af(u, p, t) = J + Af_diag(u, p, t)
	f!, Af_diag, Af
end

# ╔═╡ 0e383bb5-3bf2-4f87-93de-554a4b46df98
u0 = randn(N) / 2

# ╔═╡ 60a75d92-2cdf-4a13-ae97-9a11c3dd7abf
tspan = (0.0, 4000.0)

# ╔═╡ 4dc77c2f-bc08-4521-992f-69d76bd3e8fb
prob = ODEProblem(f!, u0, tspan, p)

# ╔═╡ 2bddc493-54c4-463b-a54e-230d8343b23e
sol = solve(prob, Tsit5(); reltol=eps()^(3//4), abstol=eps())

# ╔═╡ 36164606-63ec-4f06-9225-7ced7b7330e7
let
	fig = Figure(; size=(600, 300))
	ax = Axis(fig[1, 1]; xlabel=L"t", ylabel=L"x_i(t)")
	for i in 1:8
		lines!(ax, tspan[1]..tspan[2], t -> sol(t)[i]; linewidth=1.5, alpha=0.75)
	end
	# xlims!(ax, 2500, 3500)
	ylims!(ax, -sqrt(2), sqrt(2))
	# save("/Users/daniel/Desktop/tau=$(τ)_trajectory.pdf", fig)
	fig
end

# ╔═╡ 9fa4e8c5-33d6-4101-9559-f980a1398477
let
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"t", ylabel=L"\Vert\mathbf{x}(t)\Vert")
	lines!(ax, tspan[1]..tspan[2], t -> norm(sol(t)); linewidth=1.5, alpha=0.75)
	ylims!(ax; low=0)
	fig
end

# ╔═╡ 42a33bb0-a544-466a-81cf-d83ef2dfb930
fp = true

# ╔═╡ 5a661917-95f2-41aa-a537-f30cb733f3ae
tend, uend = fp ? (sol.t[end], sol[end]) : (nothing, nothing)

# ╔═╡ 63da614b-0746-4608-93a0-06e30b0ea365
fp_dv = parent(Af_diag(uend, p, tend))

# ╔═╡ 1a45197b-3541-493a-b732-911a4087e091
fp_A = Af(uend, p, tend)

# ╔═╡ 0098616d-2bc5-4f97-8fc5-6a3198df0078
fp_ev = eigvals(fp_A)

# ╔═╡ 77a58a4e-99e4-498b-baba-ea398646f0eb
function figax()
	fig = Figure(; size=(600, 150))
	ax = Axis(
		fig[1, 1];
		autolimitaspect=1, xlabel=L"\text{Re}(z)", ylabel=L"\text{Im}(z)",
	)
	return fig, ax
end

# ╔═╡ 60e9d8a2-f6ee-4b64-9364-d1d689b75698
let
	fig, ax = figax()
	scatter!(ax, real(fp_ev), imag(fp_ev); markersize=3, color=:gray, alpha=3/4)
	scatter!(
		ax, real(fp_dv), imag(fp_dv);
		markersize=4.5, marker=:diamond, color=:blue, alpha=3/4,
	)
	fig
end

# ╔═╡ 7850387c-ee13-42e3-bed6-7704565639d3
maximum(real, fp_ev)

# ╔═╡ f3d4f9c7-dae1-482c-8792-461a3cb4fcb8
fp_ensemble = MatrixEnsemble(Af_diag(uend, p, tend), I, I, σ * I)

# ╔═╡ f49c9087-9607-4ab0-92a4-1d7ae90afcd4
fp_ev_mf = sample_eigenvalues(fp_ensemble, N);

# ╔═╡ c9113c12-fdd9-4fde-8c5d-dc4a3ebb0484
let
	fig, ax = figax()
	scatter!(
		ax, real(fp_ev_mf), imag(fp_ev_mf); markersize=3, color=:gray, alpha=3/4
	)
	scatter!(
		ax, real(fp_dv), imag(fp_dv);
		markersize=4.5, marker=:diamond, color=:blue, alpha=3/4,
	)
	fig
end

# ╔═╡ 15aeae3c-01f9-436c-aaac-6fb6142c2552
fp_constraint = PTraceConstraint(1.0; shape0=complex(0.0), shift0=1.0)

# ╔═╡ 065cda0f-fd3d-4661-8db9-73ef8e1201ed
fp_prob = SpectralProblem(
	fp_ensemble, fp_constraint; shiftmin=-1//2, shiftoffset=0
)

# ╔═╡ 3f7839a1-8429-4b74-841a-7909373109fe
fp_nsamples = 256

# ╔═╡ 105051de-1dde-4944-8cab-5aeb3b24cd58
# ╠═╡ disabled = true
#=╠═╡
for _ in 1:32
	fp_model_samples = solve(fp_prob; nsamples=fp_nsamples, meshing=false)
end
  ╠═╡ =#

# ╔═╡ cc59d7ea-2c48-4e29-a02f-31bec8124acd
fp_model_samples = solve(fp_prob; nsamples=fp_nsamples, meshing=false)

# ╔═╡ 25b3c906-be08-404b-838c-deca467adf37
length(fp_model_samples.surrogate.rbf.x)

# ╔═╡ 2ed1b23b-b651-4658-b421-1f84cd197cb9
let
	fig, ax = figax()
	points = fp_model_samples.surrogate.rbf.x
	scatter!(ax, first.(points), last.(points); markersize=3, color=:gray, alpha=3/4)
	scatter!(
		ax, real(fp_dv), imag(fp_dv);
		markersize=4.5, marker=:diamond, color=:blue, alpha=3/4,
	)
	fig
end

# ╔═╡ 0605d781-474c-4492-af30-639753500af3
fp_boundary = spectral_boundary(
	fp_model_samples; step_size_coeff=1//3, allow_open=false,
)

# ╔═╡ 1562c42c-f944-4819-bd75-e654a701abbc
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

# ╔═╡ 5a295b55-0ae6-413f-817b-8f5041e36551
let
	fig, ax = figax()
	plotboundary!(ax, fp_boundary)
	scatter!(
		ax, real(fp_ev), imag(fp_ev);
		markersize=3, color=:gray, alpha=3/4, label=L"\lambda[f'(x_i) \delta_{ij} + J_{ij}]",
	)
	scatter!(
		ax, real(fp_dv), imag(fp_dv);
		markersize=4.5, marker=:diamond, color=:blue, alpha=3/4, label=L"f'(x_i)",
	)
	# xlims!(ax, -6.6, -5.3)
	# ylims!(ax, -0.05, 0.05)
	Legend(fig[1, 2], ax; framevisible=false, labelsize=12)
	fig
end

# ╔═╡ 6f961191-e2e0-45b6-a493-6828366a789c
baremsh = let hmin=(3//4)median(fp_model_samples.surrogate.rbf.nn_dists)
	Meshing.mesh(fp_boundary, hmin)
end

# ╔═╡ 94627260-a9c7-4c34-8d74-00a1918d3a66
pmsh = Meshing.addresolventparams(baremsh, fp_model_samples.surrogate)

# ╔═╡ 92a085a1-6279-48e4-911a-41cd5c604623
count(!isfinite, pmsh.shift)

# ╔═╡ 31e9ba4e-e828-4e0a-a8c3-d7f4d0d3dc0d
msh = spectral_density(pmsh)

# ╔═╡ ac3c9656-ee92-4e93-a1b9-24beca13a83d
let
	fig, ax = figax()
	poly!(ax, msh; color=:gray90, strokewidth=1)
	# xlims!(ax, -5.2, -4.8)
	# ylims!(ax, -0.05, 0.05)
	fig
end

# ╔═╡ daac327b-c6ff-4c5c-bf1c-f0fbcc89a87a
function plotmesh!(ax, msh; colorbar=nothing, colorbarlabel=nothing, kwargs...)
	# When using a vertex attribute as color, the mesh can't contain face views.
	# Probably a Makie bug.
	msh_fixed = (:color in keys(kwargs)) ? Meshing.remove_faceviews(msh) : msh
	mshplot = mesh!(ax, msh_fixed; kwargs...)
	if !isnothing(colorbar)
		cbarkw = isnothing(colorbarlabel) ? (;) : (; label=colorbarlabel)
		Colorbar(colorbar, mshplot; alignmode=Outside(), cbarkw...)
	end
	return mshplot
end

# ╔═╡ 9daac647-4148-485a-b0d5-2121d7234e56
msh_colored_by_density = Meshing.updatemesh(msh; color=msh.density)

# ╔═╡ 7ca38a46-9e5b-4cab-9abb-d1ec5b37e625
# ╠═╡ disabled = true
#=╠═╡
colorrange = (0.0, 2inv(area(msh.position, msh.faces)))
  ╠═╡ =#

# ╔═╡ deb5357d-5738-4e1c-97a4-e6bd18c135fd
colorrange = (0.0, maximum(filter(isfinite, msh.density.data)))

# ╔═╡ abbfebc5-9afd-4e98-8aea-8ba2e58262b7
# ╠═╡ disabled = true
#=╠═╡
colorrange = (0.0, 0.8)
  ╠═╡ =#

# ╔═╡ 0cc2b836-2186-4a2e-b172-a2336bb98959
let
	fig, ax = figax()
	plotmesh!(
		ax, msh_colored_by_density;
		colorrange=colorrange, colorbar=fig[1, 2], colorbarlabel=L"\rho(z)",
	)
	plotboundary!(ax, fp_boundary)
	# scatter!(ax, real(fp_ev), imag(fp_ev); markersize=3, color=:gray, alpha=3/4)
	# scatter!(
	# 	ax, real(fp_dv), imag(fp_dv);
	# 	markersize=3, marker=:diamond, color=:blue, alpha=3/4
	# )
	fig
end

# ╔═╡ a834655f-ba09-4a8f-8ef6-93c8f4428117
function figax2()
	fig = Figure(; size=(600, 215))
	ax1 = Axis(
		fig[1, 1];
		autolimitaspect=1, ylabel=L"\text{Im}(z)",
	)
	ax2 = Axis(
		fig[2, 1];
		autolimitaspect=1, xlabel=L"\text{Re}(z)", ylabel=L"\text{Im}(z)",
	)
	hidexdecorations!(ax1; grid=false)
	linkxaxes!(ax1, ax2)
	return fig, ax1, ax2
end

# ╔═╡ f8b5eea0-7eb5-4b39-a7bd-66a8adfeee10
let
	fig, ax1, ax2 = figax2()
	plotboundary!(ax1, fp_boundary)
	scatter!(
		ax1, real(fp_ev), imag(fp_ev);
		markersize=3, color=:gray, alpha=3/4, label=L"\lambda[f'(x_i) \delta_{ij} + J_{ij}]",
	)
	scatter!(
		ax1, real(fp_dv), imag(fp_dv);
		markersize=4.5, marker=:diamond, color=:blue, alpha=3/4, label=L"f'(x_i)",
	)
	# xlims!(ax1, -5.2, -4.8)
	# ylims!(ax1, -0.05, 0.05)
	Legend(fig[1, 2], ax1; framevisible=false, labelsize=12)
	plotmesh!(
		ax2, msh_colored_by_density;
		colorrange=colorrange, colorbar=fig[2, 2], colorbarlabel=L"\rho(z)",
	)
	plotboundary!(ax2, fp_boundary)
	# save("/Users/daniel/Desktop/tau=$(τ)_spectrum.pdf", fig)
	fig
end

# ╔═╡ 76a6906a-2eb2-43fd-af12-4a38eee06fc9
# ╠═╡ disabled = true
#=╠═╡
fp_model_mesh = solve(fp_prob; nsamples=fp_nsamples, hcoeff=1, max_retries=4)
  ╠═╡ =#

# ╔═╡ e2b6b9b9-f4da-432b-ab03-4e4f7dd6aad9
#=╠═╡
length(fp_model_mesh.surrogate.rbf.x)
  ╠═╡ =#

# ╔═╡ 684c276b-8a1e-47d3-90e4-2f6f3e8f8673
#=╠═╡
let
	fig, ax = figax()
	points = fp_model_mesh.surrogate.rbf.x
	scatter!(ax, first.(points), last.(points); markersize=4, color=:black, alpha=0.5)
	fig
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═42838418-1275-11f1-b960-732f0011f5a1
# ╠═8264147d-7091-40c3-9010-a8fd21e63bc9
# ╠═e40ad15b-0a3a-4cde-bae4-dea6b8718578
# ╠═e97c14ed-3cf8-4cec-ad39-af066403f116
# ╠═3f74e29d-78de-4836-b2b0-c57dd1a67df5
# ╠═9986bbe6-59ad-4a2c-a6d9-6961f5c748c3
# ╠═df376816-7316-44b7-aaba-0fcc997474e5
# ╠═062f96f9-6c5a-4152-a96a-eb06ca75d845
# ╠═0e383bb5-3bf2-4f87-93de-554a4b46df98
# ╠═60a75d92-2cdf-4a13-ae97-9a11c3dd7abf
# ╠═4dc77c2f-bc08-4521-992f-69d76bd3e8fb
# ╠═2bddc493-54c4-463b-a54e-230d8343b23e
# ╠═36164606-63ec-4f06-9225-7ced7b7330e7
# ╠═9fa4e8c5-33d6-4101-9559-f980a1398477
# ╠═42a33bb0-a544-466a-81cf-d83ef2dfb930
# ╠═5a661917-95f2-41aa-a537-f30cb733f3ae
# ╠═63da614b-0746-4608-93a0-06e30b0ea365
# ╠═1a45197b-3541-493a-b732-911a4087e091
# ╠═0098616d-2bc5-4f97-8fc5-6a3198df0078
# ╠═77a58a4e-99e4-498b-baba-ea398646f0eb
# ╠═60e9d8a2-f6ee-4b64-9364-d1d689b75698
# ╠═7850387c-ee13-42e3-bed6-7704565639d3
# ╠═f3d4f9c7-dae1-482c-8792-461a3cb4fcb8
# ╠═f49c9087-9607-4ab0-92a4-1d7ae90afcd4
# ╠═c9113c12-fdd9-4fde-8c5d-dc4a3ebb0484
# ╠═15aeae3c-01f9-436c-aaac-6fb6142c2552
# ╠═065cda0f-fd3d-4661-8db9-73ef8e1201ed
# ╠═3f7839a1-8429-4b74-841a-7909373109fe
# ╠═105051de-1dde-4944-8cab-5aeb3b24cd58
# ╠═cc59d7ea-2c48-4e29-a02f-31bec8124acd
# ╠═25b3c906-be08-404b-838c-deca467adf37
# ╠═2ed1b23b-b651-4658-b421-1f84cd197cb9
# ╠═0605d781-474c-4492-af30-639753500af3
# ╠═1562c42c-f944-4819-bd75-e654a701abbc
# ╠═5a295b55-0ae6-413f-817b-8f5041e36551
# ╠═6f961191-e2e0-45b6-a493-6828366a789c
# ╠═94627260-a9c7-4c34-8d74-00a1918d3a66
# ╠═92a085a1-6279-48e4-911a-41cd5c604623
# ╠═31e9ba4e-e828-4e0a-a8c3-d7f4d0d3dc0d
# ╠═ac3c9656-ee92-4e93-a1b9-24beca13a83d
# ╠═daac327b-c6ff-4c5c-bf1c-f0fbcc89a87a
# ╠═9daac647-4148-485a-b0d5-2121d7234e56
# ╠═7ca38a46-9e5b-4cab-9abb-d1ec5b37e625
# ╠═deb5357d-5738-4e1c-97a4-e6bd18c135fd
# ╠═abbfebc5-9afd-4e98-8aea-8ba2e58262b7
# ╠═0cc2b836-2186-4a2e-b172-a2336bb98959
# ╠═a834655f-ba09-4a8f-8ef6-93c8f4428117
# ╠═f8b5eea0-7eb5-4b39-a7bd-66a8adfeee10
# ╠═76a6906a-2eb2-43fd-af12-4a38eee06fc9
# ╠═e2b6b9b9-f4da-432b-ab03-4e4f7dd6aad9
# ╠═684c276b-8a1e-47d3-90e4-2f6f3e8f8673
