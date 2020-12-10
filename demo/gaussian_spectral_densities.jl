using Revise

using LinearAlgebra: BLAS; BLAS.set_num_threads(1)

using CUDA

CUDA.allowscalar(false);

using GaussianSpectralDensities

include("example_problems.jl");

ftype = Float32

σ = ftype(0.5);
τ = ftype(0.8);
λ = ftype(0.0);

N_big = 1024;
ensemble_big = jordan(N_big, λ, σ, τ);

N = 128;
ensemble = jordan(N, λ, σ, τ);
ensemble_gpu = CuMatrixEnsemble(ensemble);
ensemble_hybrid = CuHybridMatrixEnsemble(ensemble);

M_dense = Array(ensemble_big.M);
Mevs = unique(eigvals(M_dense));

#J = ftype(1.0 / sqrt(N_big)) * (rand(ftype[-1.0, 1.0], N_big, N_big));
J = ftype(1.0 / sqrt(2N_big)) * (
    randn(ftype, N_big, N_big) + im * randn(ftype, N_big, N_big)
);

J = SamplingUtils.partially_hermitize(J, τ);
Jevs = eigvals(J);

LJR = ensemble_big.L * J * ensemble_big.R;
A = M_dense + LJR;
Aevs = eigvals(A);

using LazySets: VPolygon, Bloating, overapproximate, radius

coords = reinterpret(ftype, reshape(Aevs, 1, :));
region = VPolygon(coords);
region = Bloating(region, ftype(0.125) * radius(region));
region = overapproximate(region, ftype(0.01) * radius(region));
bounds = overapproximate(region);

xcenter, ycenter = bounds.center;
xradius, yradius = bounds.radius;
xmin, xmax = xcenter - xradius, xcenter + xradius;
ymin, ymax = ycenter - yradius, ycenter + yradius;
xs = range(xmin, xmax; length=round(Int, 42 * xradius));
ys = range(ymin, ymax; length=round(Int, 42 * yradius));
zs = [[x, y] in region ? complex(x, y) : missing for y in ys, x in xs];

objective = Fixk();
algorithm = Rampδ²();
algorithm_gpu = Rampγ²();
algorithm_hybrid = Rampγ²();

i, j = 23, 46;
za = zs[i, j + 1];
zb = zs[i - 1, j];
zc = zs[i, j];

verbosity = 3

for solver in [
    SpectralSolver(ensemble, objective, algorithm),
    SpectralSolver(ensemble_gpu, objective, algorithm_gpu),
    SpectralSolver(ensemble_hybrid, objective, algorithm_hybrid),
]
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

problem = SpectralProblem(zs, objective, ensemble);
solution = @time solve(problem, algorithm; chunks_per_thread=1);

problem_gpu = SpectralProblem(zs, objective, ensemble_gpu);
solution_gpu = @time solve(problem_gpu, algorithm_gpu; chunks_per_thread=1);

problem_hybrid = SpectralProblem(zs, objective, ensemble_hybrid);
solution_hybrid = @time solve(problem_hybrid, algorithm_hybrid; chunks_per_thread=1);

# Jacobian check
using GaussianSpectralDensities: workspace, qf!, qj!
using GaussianSpectralDensities: @SVector, finite_difference_jacobian

ws = workspace(ensemble);
z = Complex{ftype}(-1.13, 0.02);
obj = Fixγ²{false}(ftype(0.2));
q = Complex{ftype}(0.23, 0.58);
qv = @SVector [real(q), imag(q)];

# Value
println(repr("text/plain", qf!(ws, obj, z, qv)));

# Analytical Jacobian
println(repr("text/plain", qj!(ws, obj, z, qv)));

# Numerical Jacobian
println(repr("text/plain", finite_difference_jacobian(qv -> qf!(ws, obj, z, qv), qv)));
