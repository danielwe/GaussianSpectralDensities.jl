using CairoMakie: Axis, Axis3, DataAspect, Figure, axislegend, lines!, scatter!
# using GLMakie: Axis, Axis3, DataAspect, Figure, axislegend, lines!, scatter!

using GaussianSpectralDensities.Algorithms: SpreadAnnealing
using GaussianSpectralDensities.Ensembles: MatrixEnsemble
using GaussianSpectralDensities.GlobalData: GlobalProblem, GlobalSolution  #, zvals
using GaussianSpectralDensities.Problems: PTraceConstraint, PointProblem, ResolventEquation
using GaussianSpectralDensities.Spreads: Spread, shift
using GaussianSpectralDensities.Solutions: issuccess
using GaussianSpectralDensities.Solvers: pointsolve
using LinearAlgebra: I, diag
using Meshes:
    Ball,
    JarvisMarch,
    MaxLengthRefinement,
    Point,
    RegularDiscretization,
    SimpleMesh,
    coords,
    discretize,
    hull,
    refine,
    simplexify,
    viz!

disk = Ball(Point(0.0, 0.0), 1.5 * inv(cos(π / 6)))  # With this radius, the circle inscribed in the hexagon inscribed in the disk is the unit circle
hex = discretize(disk, RegularDiscretization(1, 6))
poly = hull(hex.vertices, JarvisMarch())  # GrahamScan is buggy and returns non-convex result
disc = discretize(poly)
mesh = simplexify(refine(disc, MaxLengthRefinement(0.4)))

ensemble = MatrixEnsemble(0I, I, I, 2I/3)
equation = ResolventEquation(ensemble; shiftmin=-2)

constraint = PTraceConstraint(1; shift0=1, shape0=0)
algorithm = SpreadAnnealing()

prob = GlobalProblem(equation, constraint, mesh)
sol = GlobalSolution(equation, mesh)

for i in eachindex(prob, sol)
    sol[i] = pointsolve(prob[i], algorithm)
end

maskedshift = map(shift(sol), issuccess(sol)) do s, success
    return success ? s : oftype(s, NaN)
end
shiftvertices = map(mesh.vertices, maskedshift) do v, s
    c = coords(v)
    return Point(c.x.val, c.y.val, s)
end
shiftmesh = SimpleMesh(shiftvertices, mesh.topology)

fig = Figure()
ax = Axis3(fig[1, 1]; xlabel="Re(z)", ylabel="Im(z)", zlabel="shift")
viz!(
    ax,
    shiftmesh;
    color=maskedshift,
    colorrange=(0.0, maximum(maskedshift[isfinite.(maskedshift)])),
)
display(fig)

# =============================================================================
# GP-based adaptive contour finding for shift = 0 (spectral support boundary)
# =============================================================================

using AbstractGPs, KernelFunctions, Optim
using Logging: @debug, @info, @warn

# =============================================================================
# GP Kernel Configuration
# =============================================================================

# Configuration for GP kernels with optional anisotropy (ARD)
struct KernelConfig
    kernel_type::Symbol  # :se for SquaredExponential, :matern52 for Matern5/2
    lengthscale_re::Float64  # Lengthscale for real axis
    lengthscale_im::Float64  # Lengthscale for imaginary axis
    isotropic::Bool  # Whether to use single lengthscale for both axes

    function KernelConfig(
        kernel_type::Symbol, l_re::Real, l_im::Real; isotropic::Bool=false
    )
        @assert kernel_type in (:se, :matern52) "kernel_type must be :se or :matern52"
        return new(kernel_type, float(l_re), float(l_im), isotropic)
    end
end

# Convenience constructors for common configurations
KernelConfig() = KernelConfig(:se, 1.0, 1.0; isotropic=true)  # Default isotropic SE
function KernelConfigAnisotropic(l_re::Real, l_im::Real)
    return KernelConfig(:se, l_re, l_im; isotropic=false)
end
KernelConfigMatern52(l::Real) = KernelConfig(:matern52, l, l; isotropic=true)
function KernelConfigMatern52ARD(l_re::Real, l_im::Real)
    return KernelConfig(:matern52, l_re, l_im; isotropic=false)
end

function make_kernel(config::KernelConfig)
    # Select base kernel
    base_kernel = config.kernel_type == :se ? SqExponentialKernel() : Matern52Kernel()

    # Apply transform (isotropic scale or ARD with per-dimension scales)
    if config.isotropic
        # Use geometric mean for single lengthscale to balance both dimensions
        l_mean = sqrt(config.lengthscale_re * config.lengthscale_im)
        return base_kernel ∘ ScaleTransform(1.0 / l_mean)
    else
        # ARD: independent lengthscales [1/Re, 1/Im] for each dimension
        return base_kernel ∘
               ARDTransform([1.0 / config.lengthscale_re, 1.0 / config.lengthscale_im])
    end
end

# Predefined configurations for typical elongation ratios
# 1:1 - Isotropic (circular/elliptic with modest aspect ratio)
const KERNEL_ISO = KernelConfig()

# 5:1 elongation - moderate stretching along real axis
const KERNEL_5to1 = KernelConfigAnisotropic(5.0, 1.0)

# 10:1 elongation - maximum expected stretching
const KERNEL_10to1 = KernelConfigAnisotropic(10.0, 1.0)

# Matérn variants for rougher landscapes (same aspect ratios)
const KERNEL_MATERN_ISO = KernelConfigMatern52(1.0)
const KERNEL_MATERN_5to1 = KernelConfigMatern52ARD(5.0, 1.0)
const KERNEL_MATERN_10to1 = KernelConfigMatern52ARD(10.0, 1.0)

# =============================================================================
# GP-based adaptive contour finding for shift = 0 (spectral support boundary)
# =============================================================================

# Extract successful samples from the mesh solution
valid_mask = issuccess(sol)
z_valid = [complex(coords(v).x.val, coords(v).y.val) for v in mesh.vertices[valid_mask]]
shift_valid = shift(sol)[valid_mask]

# Build GP posterior from existing samples
function build_gp(
    z_samples::Vector{ComplexF64},
    s_values::Vector{Float64},
    config::KernelConfig=KernelConfig(),
)
    # Treat complex plane as R^2 for kernel
    x_samples = [[real(z), imag(z)] for z in z_samples]
    # Target: shift = 0 contour, so we model shift values directly
    gp = GP(make_kernel(config))
    # Small noise for numerical stability (function is deterministic)
    return posterior(gp(x_samples, 1e-6), s_values)
end

# Mean and std prediction at complex point(s)
function predict(gp_post, z::ComplexF64)
    x = [real(z), imag(z)]
    μ = only(mean(gp_post, [x]))
    σ = sqrt(only(cov(gp_post, [x])))
    return μ, σ
end

# Batched prediction for efficiency
function predict_batch(gp_post, zs::Vector{ComplexF64})
    xs = [[real(z), imag(z)] for z in zs]
    μs = vec(mean(gp_post, xs))
    Σ = cov(gp_post, xs)
    σs = sqrt.(diag(Σ))
    return μs, σs
end

# Acquisition: high uncertainty near predicted zero crossing
function acquisition(gp_post, z::ComplexF64; bandwidth::Float64=0.5)
    μ, σ = predict(gp_post, z)
    # Weight uncertainty by proximity to zero (Gaussian decay from predicted contour)
    return σ * exp(-(μ^2) / (2 * bandwidth^2))
end

# Adaptive sampling loop: add points near the contour
function adaptive_contour_sampling(
    z_init::Vector{ComplexF64},
    s_init::Vector{Float64},
    prob::GlobalProblem,
    algorithm::SpreadAnnealing;
    n_adaptive::Int=30,
    candidate_budget::Int=200,  # XXX: this variable is not used in the function body
    kernel_config::KernelConfig=KernelConfig(),
)
    @info "Starting adaptive contour sampling" initial_samples=length(z_init) n_adaptive=n_adaptive kernel=kernel_config.kernel_type isotropic=kernel_config.isotropic l_re=kernel_config.lengthscale_re l_im=kernel_config.lengthscale_im
    z_samples = copy(z_init)
    s_samples = copy(s_init)
    n_failed = 0

    # Sampling bounds from initial data
    re_min, re_max = extrema(real.(z_init))
    im_min, im_max = extrema(imag.(z_init))
    @debug "Sampling bounds" re_min re_max im_min im_max

    for iter in 1:n_adaptive
        gp_post = build_gp(z_samples, s_samples, kernel_config)

        # Generate candidate points (stratified + random near promising regions)
        candidates = ComplexF64[]
        # Grid candidates
        for re in range(re_min, re_max; length=15), im in range(im_min, im_max; length=15)
            push!(candidates, complex(re, im))
        end
        # Random jitter around existing samples with small shift
        for z in z_samples[rand(1:length(z_samples), 50)]
            push!(candidates, z + 0.1 * cis(2π * rand()))
        end

        # Evaluate acquisition and pick best
        best_acq, best_z = -Inf, candidates[1]
        for z in candidates
            a = acquisition(gp_post, z)
            @debug "Acquisition evaluation" z a
            a > best_acq && (best_acq=a; best_z=z)
        end
        @debug "Selected best candidate" best_z best_acq

        # Evaluate expensive function at best point
        point_prob = PointProblem(prob.equation, prob.constraint, best_z)
        point_sol = pointsolve(point_prob, algorithm)
        if issuccess(point_sol)
            push!(z_samples, best_z)
            push!(s_samples, shift(point_sol))
            n_failed = 0
            @debug "Successful solve" iter best_z shift=shift(point_sol) total_samples=length(
                z_samples
            )
        else
            n_failed += 1
            @warn "pointsolve failed at candidate point" iter best_z n_consecutive_failures=n_failed
            if n_failed > 10
                @warn "Too many consecutive failures, stopping adaptive sampling early" iter
                break
            end
        end

        if iter % 10 == 0 || iter == n_adaptive
            @info "Adaptive sampling progress" iter total_samples=length(z_samples) n_failed_recent=n_failed
        end
    end

    @info "Adaptive sampling complete" final_samples=length(z_samples) total_evaluations=length(
        z_samples
    )-length(z_init)
    return z_samples, s_samples, build_gp(z_samples, s_samples, kernel_config)
end

# =============================================================================
# Select kernel configuration
# =============================================================================
#
# For isotropic/circular supports:
#   kernel = KERNEL_ISO                    # Squared exponential, isotropic
#   kernel = KERNEL_MATERN_ISO             # Matérn 5/2, isotropic (rougher landscapes)
#
# For elongated supports (use ARD with appropriate aspect ratio):
#   kernel = KERNEL_5to1                   # 5:1 elongation along real axis
#   kernel = KERNEL_10to1                  # 10:1 elongation (maximum expected)
#   kernel = KERNEL_MATERN_5to1            # Matérn with 5:1 ARD
#   kernel = KERNEL_MATERN_10to1           # Matérn with 10:1 ARD
#
# Custom configuration:
#   kernel = KernelConfigAnisotropic(8.0, 1.0)  # Custom 8:1 ratio
#   kernel = KernelConfig(:se, 3.0, 1.0; isotropic=false)  # Full control
#
# FUTURE: Automatic kernel selection from ensemble geometry
#   The aspect ratio can be estimated from the Girko ellipse dimensions convolved
#   with ensemble.M's spectrum. For example, if the spectrum of ensemble.M is real:
#     - Im extent: height of Girko ellipse (from symmetry parameter S)
#     - Re extent: extreme eigenvalues of M ± width of Girko ellipse
#   This gives a cheap O(1) heuristic for the elongation ratio without expensive
#   solves. Example for elongated negative-real spectra:
#     l_re = (abs(min_re(M)) + width_ellipse) / l_base
#     l_im = height_ellipse / l_base
#   where l_base is a characteristic lengthscale (e.g., 1.0 or mesh spacing).
#
# FAR FUTURE, MAYBE: correlation length(s) determined by maximizing marginal likelihood.
#   Expensive, may not be worth the flops.

kernel = KERNEL_ISO  # Default: isotropic SE kernel

# Run adaptive sampling
@info "=== Starting GP-based adaptive contour finding ==="
z_adaptive, s_adaptive, gp_final = adaptive_contour_sampling(
    z_valid, shift_valid, prob, algorithm; n_adaptive=40, kernel_config=kernel
)

@info "Adaptive sampling summary" initial=length(z_valid) final=length(z_adaptive) added=length(
    z_adaptive
)-length(z_valid)

# =============================================================================
# Continuation / edge tracking to trace the zero contour
# =============================================================================

# Find starting points on contour: edges where sign changes or GP predicts zero crossing
function find_contour_seeds(
    z_samples::Vector{ComplexF64}, gp_post; n_seeds::Int=8, grid_size::Int=50
)
    @info "Finding contour seeds from GP" n_seeds_target=n_seeds grid_size=grid_size
    # Evaluate GP on fine grid to locate approximate zero crossings
    re_range = range(extrema(real.(z_samples))...; length=grid_size)
    im_range = range(extrema(imag.(z_samples))...; length=grid_size)
    @debug "GP evaluation grid" n_re=length(re_range) n_im=length(im_range)

    # Build all grid points and batch-predict
    grid_points = [complex(re, im) for re in re_range, im in im_range]
    μs_grid, σs_grid = predict_batch(gp_post, vec(grid_points))
    μs_grid = reshape(μs_grid, size(grid_points))

    # Find cells where sign changes
    seeds = ComplexF64[]
    n_cells_checked = 0
    for i in 1:(length(re_range) - 1), j in 1:(length(im_range) - 1)
        n_cells_checked += 1
        μ_corners = [
            μs_grid[i, j], μs_grid[i + 1, j], μs_grid[i + 1, j + 1], μs_grid[i, j + 1]
        ]
        @debug "Cell evaluation" i j μ_corners
        # Check for sign change
        if maximum(μ_corners) > 0 && minimum(μ_corners) < 0
            # Linear interpolation to approximate zero crossing
            z_corners = [
                complex(re_range[i], im_range[j]),
                complex(re_range[i + 1], im_range[j]),
                complex(re_range[i + 1], im_range[j + 1]),
                complex(re_range[i], im_range[j + 1]),
            ]
            seed = sum(z_corners) / 4
            push!(seeds, seed)
            @debug "Found seed at cell" i j seed
        end
    end

    @info "Grid search complete" n_cells_checked n_seeds_found=length(seeds)

    # Return evenly spaced seeds around the contour
    if isempty(seeds)
        @warn "No contour seeds found! GP may not cross zero in sampled region."
        return ComplexF64[]
    end
    # Sort by angle around centroid
    centroid = sum(seeds) / length(seeds)
    sort!(seeds; by=z -> angle(z - centroid))
    @debug "Seeds sorted by angle around centroid" centroid n_seeds=length(seeds)

    # Select evenly spaced
    step = max(1, div(length(seeds), n_seeds))
    selected_seeds = seeds[1:step:length(seeds)]
    @info "Selected evenly spaced seeds" n_selected=length(selected_seeds)

    return selected_seeds
end

# Trace contour using predictor-corrector with finite-difference gradient
function trace_contour(
    seed::ComplexF64,
    prob::GlobalProblem,
    algorithm::SpreadAnnealing,
    gp_post;  # XXX: this variable is not used in the function body
    step_size::Float64=0.05,
    max_steps::Int=500,
    tol::Float64=1e-6,
)
    @info "Starting contour trace" seed step_size max_steps tol
    contour_points = ComplexF64[]
    z = seed
    direction = cis(π / 2)  # Initial tangent: rotate gradient 90°
    initial_direction = direction
    n_corrector_failures = 0
    n_gradient_failures = 0

    for step in 1:max_steps
        @debug "Contour trace step" step z direction

        # Predictor step: move along tangent
        z_pred = z + step_size * direction
        @debug "Predictor step" z_pred

        # Corrector: Newton step to find zero (move perpendicular to contour)
        z_curr = z_pred
        corrector_converged = false
        δ = 0.01  # Finite difference step
        for corr_iter in 1:10
            point_sol = pointsolve(
                PointProblem(prob.equation, prob.constraint, z_curr), algorithm
            )
            if !issuccess(point_sol)
                @debug "Corrector: solve failed" corr_iter z_curr
                break
            end
            s = shift(point_sol)
            @debug "Corrector iteration" corr_iter z_curr s
            if abs(s) < tol
                corrector_converged = true
                break
            end

            # Compute gradient using finite differences in x and y directions
            ps_xp = pointsolve(
                PointProblem(prob.equation, prob.constraint, z_curr + δ), algorithm
            )
            ps_xm = pointsolve(
                PointProblem(prob.equation, prob.constraint, z_curr - δ), algorithm
            )
            ps_yp = pointsolve(
                PointProblem(prob.equation, prob.constraint, z_curr + im * δ), algorithm
            )
            ps_ym = pointsolve(
                PointProblem(prob.equation, prob.constraint, z_curr - im * δ), algorithm
            )

            if !(
                issuccess(ps_xp) && issuccess(ps_xm) && issuccess(ps_yp) && issuccess(ps_ym)
            )
                @debug "Corrector: gradient evaluation failed"
                break
            end

            ∂s_∂x = (shift(ps_xp) - shift(ps_xm)) / (2δ)
            ∂s_∂y = (shift(ps_yp) - shift(ps_ym)) / (2δ)
            grad_s = complex(∂s_∂x, ∂s_∂y)
            grad_norm_sq = abs2(grad_s)

            if grad_norm_sq < 1e-10
                @debug "Corrector: gradient too small" grad_s
                break
            end

            # Newton step: move in direction of gradient to reach s=0
            # z_new = z - s * grad(s) / |grad(s)|^2
            z_curr = z_curr - s * grad_s / grad_norm_sq
        end

        if !corrector_converged
            n_corrector_failures += 1
            @warn "Corrector did not converge" step z_pred n_corrector_failures
            if n_corrector_failures > 5
                @warn "Too many corrector failures, stopping trace early" step
                break
            end
            continue
        end
        n_corrector_failures = 0

        # Check convergence and add point
        point_sol = pointsolve(
            PointProblem(prob.equation, prob.constraint, z_curr), algorithm
        )
        if !issuccess(point_sol)
            @warn "Final verification solve failed" step z_curr
            break
        end
        s = shift(point_sol)
        if abs(s) > 10 * tol
            @warn "Failed to converge to tolerance" step z_curr s target_tol=tol actual_tol=abs(
                s
            )
            break
        end

        push!(contour_points, z_curr)
        @debug "Added contour point" step z_curr n_points=length(contour_points)

        # Update direction: tangent perpendicular to gradient
        δ = 0.01
        ps_xp = pointsolve(
            PointProblem(prob.equation, prob.constraint, z_curr + δ), algorithm
        )
        ps_xm = pointsolve(
            PointProblem(prob.equation, prob.constraint, z_curr - δ), algorithm
        )
        ps_yp = pointsolve(
            PointProblem(prob.equation, prob.constraint, z_curr + im * δ), algorithm
        )
        ps_ym = pointsolve(
            PointProblem(prob.equation, prob.constraint, z_curr - im * δ), algorithm
        )

        if issuccess(ps_xp) && issuccess(ps_xm) && issuccess(ps_yp) && issuccess(ps_ym)
            ∂s_∂x = (shift(ps_xp) - shift(ps_xm)) / (2δ)
            ∂s_∂y = (shift(ps_yp) - shift(ps_ym)) / (2δ)
            grad = complex(∂s_∂x, ∂s_∂y)
            # Tangent is perpendicular to gradient
            new_direction = cis(angle(grad) + π / 2)
            # Maintain consistent orientation (counter-clockwise)
            dot_product = real(direction * conj(new_direction))
            direction = dot_product > 0 ? new_direction : -new_direction
            @debug "Direction updated" grad new_direction
            n_gradient_failures = 0
        else
            n_gradient_failures += 1
            @warn "Gradient evaluation failed, keeping previous direction" step n_gradient_failures
            if n_gradient_failures > 10
                @warn "Too many gradient failures, stopping trace" step
                break
            end
        end

        z = z_curr

        # Close loop if: (1) far enough along, (2) near first point, (3) heading back toward start
        # Check heading: dot product with initial direction should be positive (coming around)
        heading_back = real(direction * conj(initial_direction)) > 0
        if step > 50 && abs(z - first(contour_points)) < step_size && heading_back
            @info "Contour loop closed naturally" step n_points=length(contour_points) gap=abs(
                z - first(contour_points)
            )
            # Close the loop by appending first verified point
            push!(contour_points, first(contour_points))
            break
        end

        if step == max_steps
            @warn "Hit max_steps without closing loop" max_steps n_points=length(
                contour_points
            )
        end
    end

    # Report closure status - don't artificially close, as open contours are valid
    is_closed = !isempty(contour_points) && abs(contour_points[end] - seed) < 1e-10
    @info "Contour trace complete" n_points=length(contour_points) seed is_closed is_compact=is_closed
    return contour_points
end

# Find seeds and trace contours
@info "=== Starting contour tracing ==="
seeds = find_contour_seeds(z_adaptive, gp_final; n_seeds=6)
contours = Vector{ComplexF64}[]  # Always define, even if empty

if isempty(seeds)
    @warn "No seeds found, cannot trace contours"
else
    @info "Tracing contours from seeds" n_seeds=length(seeds)
    # Deduplication: skip seeds that lie on already-traced contours
    for (i, seed) in enumerate(seeds)
        # Check if this seed is already covered by a previous contour
        is_duplicate = false
        for existing in contours
            # Check if seed is close to any point in existing contour
            distances = abs.(existing .- seed)
            if any(d -> d < 2 * 0.05, distances)  # 2*step_size tolerance
                is_duplicate = true
                @info "Skipping duplicate seed" seed_number=i seed existing_contour_n_points=length(
                    existing
                )
                break
            end
        end
        is_duplicate && continue

        @info "Tracing contour" seed_number=i total_seeds=length(seeds) seed
        contour = trace_contour(seed, prob, algorithm, gp_final)
        if length(contour) > 10  # Only keep substantial contours
            push!(contours, contour)
        else
            @warn "Contour too short, discarding" seed n_points=length(contour)
        end
    end
    all_contour_points = reduce(vcat, contours; init=ComplexF64[])
    @info "Contour tracing summary" n_contours=length(contours) total_points=length(
        all_contour_points
    ) n_seeds_skipped=length(seeds)-length(contours)
end

# =============================================================================
# Visualization
# =============================================================================

fig2 = Figure()
ax2 = Axis(
    fig2[1, 1];
    xlabel="Re(z)",
    ylabel="Im(z)",
    aspect=DataAspect(),
    title="Spectral Support Boundary (shift = 0)",
)

# Plot adaptive samples (color by shift value)
scatter!(
    ax2,
    real.(z_adaptive),
    imag.(z_adaptive);
    color=s_adaptive,
    markersize=8,
    colormap=:RdBu,
    colorrange=(-2, 2),
    label="GP samples",
)

# Plot contour
for contour in contours
    isempty(contour) && continue
    lines!(ax2, real.(contour), imag.(contour); color=:black, linewidth=2, label="Contour")
end

# Plot expected ellipse for comparison (half-axes 1.25, 0.75 for τ = 1/4)
θ_ellipse = range(0, 2π; length=200)
ellipse_x = 1.25 * cos.(θ_ellipse)
ellipse_y = 0.75 * sin.(θ_ellipse)
lines!(
    ax2,
    ellipse_x,
    ellipse_y;
    color=:green,
    linestyle=:dash,
    linewidth=2,
    label="Expected ellipse",
)

axislegend(ax2; position=:rb)
display(fig2)

# NOTE: To see @debug log messages, run as:
#   JULIA_DEBUG=mesh julia --project=@. mesh.jl
# For @info and @warn only (default):
#   julia --project=@. mesh.jl
