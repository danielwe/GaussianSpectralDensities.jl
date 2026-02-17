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
    Pkg.add(["CairoMakie", "LinearAlgebra"])
    Pkg.precompile()
end

# ╔═╡ 07c3373f-fefb-49f3-8384-0f6826ec2a4a
begin
	using GaussianSpectralDensities.Algorithms: SpreadAnnealing
	using GaussianSpectralDensities.Boundaries: extract_boundaries
	using GaussianSpectralDensities.Spreads: shift
	using GaussianSpectralDensities.Ensembles: MatrixEnsemble
	using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
	using GaussianSpectralDensities.Samples: sample_eigenvalues
	using GaussianSpectralDensities.Solutions: greens
	using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluatebatch

	using CairoMakie: Axis, Colorbar, DataAspect, Figure, contour!, contourf!, lines!, scatter!, @L_str
	using LinearAlgebra: I, Diagonal
end

# ╔═╡ 8266b399-d711-473e-9a55-ed837e65db0d
e = let M=Diagonal([1.05, 1.05im, -1.05, -1.05im])
    MatrixEnsemble(M, I, I, 0.5I)
end

# ╔═╡ d2733d97-69ce-4ecd-9e0a-e2f040912b82
eq = ResolventEquation(e; shiftmin=-0.5)

# ╔═╡ c076126d-98a9-45aa-a144-1bc4803a87b5
constraint = PTraceConstraint(1; shape0=0, shift0=1)

# ╔═╡ 6ed5f7f7-5175-4e34-b76b-94a282d652a7
N = 96

# ╔═╡ d311b9b0-36fb-4446-95b7-fd323edd3853
model = SurrogateModel(eq, constraint, SpreadAnnealing(), N)

# ╔═╡ f260bf48-a72e-4857-a7d9-e1da13af5d4e
eigs = map(zv -> complex(zv...), model.rbf.x)

# ╔═╡ 3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	scatter!(ax, real(eigs), imag(eigs))
	fig
end

# ╔═╡ 8ded32e7-0adf-4d98-9271-3409944f3446
begin
	xs, ys = -2.0:0.05:2.0, -1.55:0.05:1.55
	zs = complex.(xs, ys')
end;

# ╔═╡ 015427c7-3a98-4eec-96eb-f9ab72065023
sols = evaluatebatch(model, zs; refine=false);

# ╔═╡ 12a0c8f5-4254-4a1a-a8be-d904fa7159f3
begin
	shifts, greenss = shift(sols), greens(sols)
	@show count(isfinite, shifts)
	@show maximum(filter(isfinite, shifts))
	@show maximum(abs ∘ real, filter(isfinite, greenss))
	@show maximum(abs ∘ imag, filter(isfinite, greenss))
end;

# ╔═╡ 765b7e38-71ee-4297-869e-c0191c597e96
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	co = contourf!(ax, xs, ys, shifts; levels=-0.5:0.125:0.375)
	Colorbar(fig[1, 2], co)
	contour!(ax, xs, ys, shifts; levels=[0.0], color=:red, linewidth=2)
	fig
end

# ╔═╡ 465f7020-60bd-4dda-b2d9-05fc8fd936fa
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	co = contourf!(ax, xs, ys, real(greenss); levels=-1.125:0.125:1.125)
	Colorbar(fig[1, 2], co)
	contour!(ax, xs, ys, shifts; levels=[0.0], color=:red, linewidth=2)
	fig
end

# ╔═╡ e0077b5b-fed8-4a35-9dfa-d31fe019c98e
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	co = contourf!(ax, xs, ys, imag(greenss); levels=-1.125:0.125:1.125)
	Colorbar(fig[1, 2], co)
	contour!(ax, xs, ys, shifts; levels=[0.0], color=:red, linewidth=2)
	fig
end

# ╔═╡ 0b1d92d5-b09c-45b7-bf5a-f39337d73979
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	co = contourf!(ax, xs, ys, abs.(greenss); levels=0:0.125:1.125)
	Colorbar(fig[1, 2], co)
	contour!(ax, xs, ys, shifts; levels=[0.0], color=:red, linewidth=2)
	fig
end

# ╔═╡ c2fa9dd5-f0f6-4b59-998d-3ed4b017b2fe
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	co = contourf!(ax, xs, ys, angle.(greenss); levels=range(-π, π, 19))
	Colorbar(fig[1, 2], co)
	contour!(ax, xs, ys, shifts; levels=[0.0], color=:red, linewidth=2)
	fig
end

# ╔═╡ 272863c6-aa6b-435d-914b-f473dd66418e
# ╠═╡ disabled = true
#=╠═╡
let f=let model=model
		f(zs) = evaluatebatch(model, zs; refine=false)
	end
	@btime SurrogateModel($eq, $constraint, $(SpreadAnnealing()), $N)
	@btime SurrogateModel($eq, $constraint, $(SpreadAnnealing()), $eigs)
	@btime $f($zs)
end;
  ╠═╡ =#

# ╔═╡ 4f9946e3-feda-404a-8316-2525848c0579
boundaries = extract_boundaries(model, 1 / 3)

# ╔═╡ 67e7082c-9747-4fc3-bbc6-9842dce52bf6
eigs_big = sample_eigenvalues(e, 1536);

# ╔═╡ 192115b0-68a8-430a-bdc8-cafd3aaf227d
let
	fig = Figure()
	ax = Axis(
		fig[1, 1];
		aspect=DataAspect(), xlabel=L"\text{Re}(λ)", ylabel=L"\text{Im}(λ)",
	)
	scatter!(ax, real(eigs_big), imag(eigs_big))
	for b in boundaries
		(; points, internal) = b.result
		lines!(
			ax, real(points), imag(points);
			color=internal ? :green : :red, linewidth=2,
		)
	end
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
# ╠═3af9d3f8-17c0-4ef8-9bb6-cb03dd15c376
# ╠═8ded32e7-0adf-4d98-9271-3409944f3446
# ╠═015427c7-3a98-4eec-96eb-f9ab72065023
# ╠═12a0c8f5-4254-4a1a-a8be-d904fa7159f3
# ╠═765b7e38-71ee-4297-869e-c0191c597e96
# ╠═465f7020-60bd-4dda-b2d9-05fc8fd936fa
# ╠═e0077b5b-fed8-4a35-9dfa-d31fe019c98e
# ╠═0b1d92d5-b09c-45b7-bf5a-f39337d73979
# ╠═c2fa9dd5-f0f6-4b59-998d-3ed4b017b2fe
# ╠═272863c6-aa6b-435d-914b-f473dd66418e
# ╠═4f9946e3-feda-404a-8316-2525848c0579
# ╠═67e7082c-9747-4fc3-bbc6-9842dce52bf6
# ╠═192115b0-68a8-430a-bdc8-cafd3aaf227d
