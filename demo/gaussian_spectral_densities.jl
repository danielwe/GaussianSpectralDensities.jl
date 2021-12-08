using StableRNGs
import LazySets
using GaussianSpectralDensities
using CUDA

using LinearAlgebra: BLAS; BLAS.set_num_threads(1);

rng = StableRNG(93686);
CUDA.allowscalar(false);

include("example_problems.jl");

ftype = Float32;

σ = ftype(0.5);
τs = [ftype(0.3), ftype(0.8)];
λ = ftype(0.0);

N_big = 1024;
T_big = repeat(sqrt.(τs), N_big ÷ length(τs));
ensemble_large = jordan(N_big, λ, σ, T_big);
xs, ys, zs = automesh(spectral_region(rng, ftype, ensemble_large));

N = 128;
T = repeat(sqrt.(τs), N ÷ length(τs));
ensemble = jordan(N, λ, σ, T);
objective = Fixk();

problems = [
    SpectralProblem(ensemble, objective, zs),
    SpectralProblem(CuMatrixEnsemble(ensemble), objective, zs),
    SpectralProblem(CuHybridMatrixEnsemble(ensemble), objective, zs),
];
algorithms = [Rampδ²(), Rampγ²(), Rampγ²()];

i, j = 23, 46;
za = zs[i, j + 1];
zb = zs[i - 1, j];
zc = zs[i, j];

verbosity = 3;

for (problem, algorithm) in zip(problems, algorithms)
    solver = SpectralSolver(problem, algorithm);
    solutiona = @time solve!(solver, za; verbosity=verbosity);
    solutionb = @time solve!(solver, zb; verbosity=verbosity);
    qa, δa² = solutiona.q, solutiona.δ²;
    qb, δb² = solutionb.q, solutionb.δ²;
    qc, δc² = GaussianSpectralDensities.select(qa, qb, δa², δb²);
    GaussianSpectralDensities.print_q(qc);
    GaussianSpectralDensities.print_δ²(δc²);
    solutionc = @time solve!(
        solver, zc,
        qc, δc²;
        verbosity=verbosity,
    );
end

solutions = [
    @time solve(problem, algorithm; chunks_per_thread=1)
    for (problem, algorithm) in zip(problems, algorithms)
];

# Jacobian check
using GaussianSpectralDensities: workspace, qf!, qj!, @SVector, finite_difference_jacobian

ws = workspace(ensemble);
z = Complex{ftype}(-1.13, 0.02);
obj = Fixγ²{false}(ftype(0.2));
q = Complex{ftype}(0.23, 0.58);
qv = @SVector [real(q), imag(q)];

# Value
display(qf!(ws, obj, z, qv));

# Analytical Jacobian
display(qj!(ws, obj, z, qv));

# Numerical Jacobian
display(finite_difference_jacobian(qv -> qf!(ws, obj, z, qv), qv));
