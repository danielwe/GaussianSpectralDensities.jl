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
		"SciMLOperators",
	])
    Pkg.precompile()
end


# ╔═╡ 8264147d-7091-40c3-9010-a8fd21e63bc9
begin
	using GaussianSpectralDensities:
		MatrixEnsemble, 
		Meshing,
		PTraceConstraint,
		SpectralProblem,
		spectral_boundary,
		spectral_density,
		sample_eigenvalues,
		sample_matrix,
		solve

	using CairoMakie:
		Axis, Colorbar, DataAspect, Figure, (..), lines!, mesh!, poly!, scatter!, xlims!, ylims!, @L_str
	using GeometryBasics: area
	using LinearAlgebra: I, Diagonal, diagind, eigvals, mul!
	using OrdinaryDiffEqTsit5: ODEProblem, Tsit5
end

# ╔═╡ e40ad15b-0a3a-4cde-bae4-dea6b8718578
τ = 0.5

# ╔═╡ 3f74e29d-78de-4836-b2b0-c57dd1a67df5
N = 1024

# ╔═╡ 9986bbe6-59ad-4a2c-a6d9-6961f5c748c3
p = (; a=0.45)

# ╔═╡ df376816-7316-44b7-aaba-0fcc997474e5
coupling_ensemble = MatrixEnsemble(0I, I, I, sqrt(τ) * I)

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
	Af_diag(u, p, t) = Diagonal(map(ui -> (p.a - 3ui^2), u))
	Af(u, p, t) = J + Af_diag(u, p, t)
	f!, Af_diag, Af
end

# ╔═╡ 0e383bb5-3bf2-4f87-93de-554a4b46df98
u0 = randn(N) / 2

# ╔═╡ 60a75d92-2cdf-4a13-ae97-9a11c3dd7abf
tspan = (0.0, 3000.0)

# ╔═╡ 4dc77c2f-bc08-4521-992f-69d76bd3e8fb
prob = ODEProblem(f!, u0, tspan, p)

# ╔═╡ 2bddc493-54c4-463b-a54e-230d8343b23e
sol = solve(prob, Tsit5(); reltol=sqrt(eps()), abstol=sqrt(eps()))

# ╔═╡ 36164606-63ec-4f06-9225-7ced7b7330e7
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for i in 1:N
		lines!(ax, ..(tspan...), t -> sol(t)[i])
	end
	fig
end

# ╔═╡ 5a661917-95f2-41aa-a537-f30cb733f3ae
tend, uend = sol.t[end], sol[end]

# ╔═╡ 1a45197b-3541-493a-b732-911a4087e091
fp_A = Af(uend, p, tend)

# ╔═╡ 0098616d-2bc5-4f97-8fc5-6a3198df0078
fp_ev = eigvals(fp_A)

# ╔═╡ 77a58a4e-99e4-498b-baba-ea398646f0eb
function figax()
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	return fig, ax
end

# ╔═╡ 60e9d8a2-f6ee-4b64-9364-d1d689b75698
let
	fig, ax = figax()
	scatter!(ax, real(fp_ev), imag(fp_ev); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ f3d4f9c7-dae1-482c-8792-461a3cb4fcb8
fp_ensemble = MatrixEnsemble(Af_diag(uend, p, tend), I, I, sqrt(τ) * I)

# ╔═╡ 15aeae3c-01f9-436c-aaac-6fb6142c2552
fp_constraint = PTraceConstraint(1.0; shape0=complex(0.0), shift0=1.0)

# ╔═╡ 065cda0f-fd3d-4661-8db9-73ef8e1201ed
fp_prob = SpectralProblem(
	fp_ensemble, fp_constraint; shiftmin=-1//2, shiftoffset=-1//1024
)

# ╔═╡ 3f7839a1-8429-4b74-841a-7909373109fe
fp_nsamples = 640

# ╔═╡ cc59d7ea-2c48-4e29-a02f-31bec8124acd
fp_model_samples = solve(fp_prob; nsamples=fp_nsamples, meshing=false)

# ╔═╡ 2ed1b23b-b651-4658-b421-1f84cd197cb9
let
	fig, ax = figax()
	points = fp_model_samples.surrogate.rbf.x
	scatter!(ax, first.(points), last.(points); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ 0605d781-474c-4492-af30-639753500af3
fp_boundary = spectral_boundary(fp_model_samples; step_size_coeff=1//3)

# ╔═╡ 1562c42c-f944-4819-bd75-e654a701abbc
function plotboundary!(ax, boundary)
	for segment in boundary
		lines!(
			ax, real(segment.points), imag(segment.points);
			color=segment.internal ? :green : :red, linewidth=2,
		)
	end
	return nothing
end

# ╔═╡ 5a295b55-0ae6-413f-817b-8f5041e36551
let
	fig, ax = figax()
	plotboundary!(ax, fp_boundary)
	scatter!(ax, real(fp_ev), imag(fp_ev); markersize=4, color=:black, alpha=0.5)
	# xlims!(ax, -10, -7)
	# ylims!(ax, -0.1, 0.1)
	fig
end

# ╔═╡ 76a6906a-2eb2-43fd-af12-4a38eee06fc9
fp_model_mesh = solve(fp_prob; nsamples=fp_nsamples, hcoeff=1, max_retries=4)

# ╔═╡ 684c276b-8a1e-47d3-90e4-2f6f3e8f8673
let
	fig, ax = figax()
	points = fp_model_mesh.surrogate.rbf.x
	scatter!(ax, first.(points), last.(points); markersize=4, color=:black, alpha=0.5)
	fig
end

# ╔═╡ Cell order:
# ╠═42838418-1275-11f1-b960-732f0011f5a1
# ╠═8264147d-7091-40c3-9010-a8fd21e63bc9
# ╠═e40ad15b-0a3a-4cde-bae4-dea6b8718578
# ╠═3f74e29d-78de-4836-b2b0-c57dd1a67df5
# ╠═9986bbe6-59ad-4a2c-a6d9-6961f5c748c3
# ╠═df376816-7316-44b7-aaba-0fcc997474e5
# ╠═062f96f9-6c5a-4152-a96a-eb06ca75d845
# ╠═0e383bb5-3bf2-4f87-93de-554a4b46df98
# ╠═60a75d92-2cdf-4a13-ae97-9a11c3dd7abf
# ╠═4dc77c2f-bc08-4521-992f-69d76bd3e8fb
# ╠═2bddc493-54c4-463b-a54e-230d8343b23e
# ╠═36164606-63ec-4f06-9225-7ced7b7330e7
# ╠═5a661917-95f2-41aa-a537-f30cb733f3ae
# ╠═1a45197b-3541-493a-b732-911a4087e091
# ╠═0098616d-2bc5-4f97-8fc5-6a3198df0078
# ╠═77a58a4e-99e4-498b-baba-ea398646f0eb
# ╠═60e9d8a2-f6ee-4b64-9364-d1d689b75698
# ╠═f3d4f9c7-dae1-482c-8792-461a3cb4fcb8
# ╠═15aeae3c-01f9-436c-aaac-6fb6142c2552
# ╠═065cda0f-fd3d-4661-8db9-73ef8e1201ed
# ╠═3f7839a1-8429-4b74-841a-7909373109fe
# ╠═cc59d7ea-2c48-4e29-a02f-31bec8124acd
# ╠═2ed1b23b-b651-4658-b421-1f84cd197cb9
# ╠═0605d781-474c-4492-af30-639753500af3
# ╠═1562c42c-f944-4819-bd75-e654a701abbc
# ╠═5a295b55-0ae6-413f-817b-8f5041e36551
# ╠═76a6906a-2eb2-43fd-af12-4a38eee06fc9
# ╠═684c276b-8a1e-47d3-90e4-2f6f3e8f8673
