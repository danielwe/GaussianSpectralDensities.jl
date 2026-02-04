using AbstractGPs
using CairoMakie: Axis, DataAspect, Figure, Colorbar, contourf!, lines!, scatter!
using KernelFunctions
using LinearAlgebra: Diagonal, I, UniformScaling, diag, eigvals
using Logging: @debug, @info, @warn
using Optim: Optim
using Optim: BFGS, Fminbox, minimizer, minimum, optimize
using Random: rand

using GaussianSpectralDensities.Algorithms: SpreadAnnealing
using GaussianSpectralDensities.Ensembles: MatrixEnsemble
using GaussianSpectralDensities.Problems: PTraceConstraint, PointProblem, ResolventEquation
using GaussianSpectralDensities.Spreads: shift
using GaussianSpectralDensities.Solutions: issuccess
using GaussianSpectralDensities.Solvers: pointsolve

using Meshes: JarvisMarch, Point, coords, hull

# =============================================================================
# Component state
# =============================================================================

"""
Per-component state for multi-component refinement.

Fields:

  - `id::Int`: component identifier.
  - `z_samples::Vector{ComplexF64}`: sample points in the complex plane.
  - `s_samples::Vector{Float64}`: corresponding shift values.
  - `gp_post`: GP posterior used for predictions.
  - `bounds`: active bounds for candidate generation.
  - `initial_bounds`: clamp bounds derived from initial radial probing.
  - `converged::Bool`: whether stopping criteria are met.
"""
mutable struct ComponentState
    id::Int
    z_samples::Vector{ComplexF64}
    s_samples::Vector{Float64}
    gp_post
    bounds::NamedTuple{
        (:re_min, :re_max, :im_min, :im_max),Tuple{Float64,Float64,Float64,Float64}
    }
    initial_bounds::NamedTuple{
        (:re_min, :re_max, :im_min, :im_max),Tuple{Float64,Float64,Float64,Float64}
    }
    converged::Bool
end

# =============================================================================
# Kriging-based spectral support sampler prototype
# =============================================================================

# =============================================================================
# GP Kernel Configuration
# =============================================================================

"""
Kernel configuration for the GP surrogate.

Fields:

  - `kernel_type::Symbol`: kernel family (currently only `:se`).
  - `lengthscale_re::Float64`: lengthscale for real axis.
  - `lengthscale_im::Float64`: lengthscale for imaginary axis.
  - `isotropic::Bool`: use isotropic lengthscale if true.
"""
struct KernelConfig
    kernel_type::Symbol  # :se for SquaredExponential
    lengthscale_re::Float64
    lengthscale_im::Float64
    isotropic::Bool

    function KernelConfig(
        kernel_type::Symbol, l_re::Real, l_im::Real; isotropic::Bool=false
    )
        @assert kernel_type in (:se,) "kernel_type must be :se"
        return new(kernel_type, float(l_re), float(l_im), isotropic)
    end
end

"""
Create the default isotropic squared-exponential configuration.
"""
KernelConfig() = KernelConfig(:se, 1.0, 1.0; isotropic=true)
"""
Create an anisotropic squared-exponential configuration.

Arguments:

  - `l_re`: lengthscale along the real axis.
  - `l_im`: lengthscale along the imaginary axis.
"""
function KernelConfigAnisotropic(l_re::Real, l_im::Real)
    return KernelConfig(:se, l_re, l_im; isotropic=false)
end

"""
Build a kernel instance from a `KernelConfig`.

Arguments:

  - `config`: kernel configuration.

Returns:

  - A `KernelFunctions` kernel with appropriate transforms.
"""
function make_kernel(config::KernelConfig)
    base_kernel = SqExponentialKernel()
    if config.isotropic
        l_mean = sqrt(config.lengthscale_re * config.lengthscale_im)
        return base_kernel ∘ ScaleTransform(1.0 / l_mean)
    else
        return base_kernel ∘
               ARDTransform([1.0 / config.lengthscale_re, 1.0 / config.lengthscale_im])
    end
end

# =============================================================================
# Utility helpers
# =============================================================================

"""
Solve the point problem and return `(success, shift)`.

Arguments:

  - `equation`: resolvent equation.
  - `constraint`: trace constraint.
  - `algorithm`: solver strategy.
  - `z`: complex point.
"""
function solve_shift(
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing,
    z::ComplexF64,
)
    point_sol = pointsolve(PointProblem(equation, constraint, z), algorithm)
    return issuccess(point_sol), shift(point_sol)
end

"""
Return eigenvalues of the mean matrix in the ensemble.

Arguments:

  - `ensemble`: matrix ensemble.
"""
function ensemble_eigvals(ensemble::MatrixEnsemble)
    M = ensemble.M
    if M isa UniformScaling
        n = first(size(ensemble))
        return fill(M.λ, n)
    end
    return eigvals(Matrix(M))
end

"""
Push a `(z, s)` sample if it is not already present.

Arguments:

  - `z_samples`: sample locations (mutated).
  - `s_samples`: sample shifts (mutated).
  - `z`: complex location.
  - `s`: shift value.
  - `tol`: duplicate tolerance.
"""
function push_sample!(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    z::ComplexF64,
    s::Float64;
    tol::Float64=1e-10,
)
    if any(abs(z - z_existing) < tol for z_existing in z_samples)
        return false
    end
    push!(z_samples, z)
    push!(s_samples, s)
    return true
end

"""
Convert a `Meshes.Point` to `ComplexF64`.
"""
function complex_from_point(p::Point)
    c = coords(p)
    return complex(c.x.val, c.y.val)
end

"""
Return convex hull vertices of the sample set.

Arguments:

  - `z_samples`: complex samples.
"""
function hull_vertices(z_samples::Vector{ComplexF64})
    if length(z_samples) < 3
        return ComplexF64[]
    end
    points = Point.(real.(z_samples), imag.(z_samples))
    poly = hull(points, JarvisMarch())
    vertices = first(poly.rings).vertices
    return complex_from_point.(vertices)
end

"""
Compute a bounding box for samples with a margin.

Arguments:

  - `z_samples`: complex samples.
  - `margin`: expansion added to each side.
"""
function component_bounds(z_samples::Vector{ComplexF64}; margin::Float64=0.5)
    re_min, re_max = extrema(real.(z_samples))
    im_min, im_max = extrema(imag.(z_samples))
    return (
        re_min=float(re_min - margin),
        re_max=float(re_max + margin),
        im_min=float(im_min - margin),
        im_max=float(im_max + margin),
    )
end

"""
Log basic distribution statistics for a sample set.

Arguments:

  - `label`: log label.
  - `z_samples`: complex samples.
"""
function log_sample_distribution(label::String, z_samples::Vector{ComplexF64})
    if isempty(z_samples)
        @info label n_samples=0
        return nothing
    end
    re_vals = real.(z_samples)
    im_vals = imag.(z_samples)
    counts = (
        re_pos=sum(re_vals .>= 0),
        re_neg=sum(re_vals .< 0),
        im_pos=sum(im_vals .>= 0),
        im_neg=sum(im_vals .< 0),
    )
    re_min, re_max = extrema(re_vals)
    im_min, im_max = extrema(im_vals)
    @info label n_samples=length(z_samples) re_min re_max im_min im_max counts
end

"""
Clamp `bounds` to lie within `limit`.

Arguments:

  - `bounds`: candidate bounds.
  - `limit`: bounding box to enforce.
"""
function clamp_bounds(bounds, limit)
    return (
        re_min=max(bounds.re_min, limit.re_min),
        re_max=min(bounds.re_max, limit.re_max),
        im_min=max(bounds.im_min, limit.im_min),
        im_max=min(bounds.im_max, limit.im_max),
    )
end

"""
Return the union bounding box of a list of bounds.

Arguments:

  - `bounds_list`: iterable of bounds.
"""
function bounds_union(bounds_list)
    isempty(bounds_list) && return (re_min=0.0, re_max=0.0, im_min=0.0, im_max=0.0)
    re_min = Base.minimum(b.re_min for b in bounds_list)
    re_max = Base.maximum(b.re_max for b in bounds_list)
    im_min = Base.minimum(b.im_min for b in bounds_list)
    im_max = Base.maximum(b.im_max for b in bounds_list)
    return (re_min=re_min, re_max=re_max, im_min=im_min, im_max=im_max)
end

"""
Return the area of a bounding box.
"""
function bounds_area(bounds)
    return max(bounds.re_max - bounds.re_min, 0.0) * max(bounds.im_max - bounds.im_min, 0.0)
end

"""
Compute overlap area divided by the smaller box area.

Arguments:

  - `a`, `b`: bounding boxes.
"""
function bounds_overlap_fraction(a, b)
    re_min = max(a.re_min, b.re_min)
    re_max = min(a.re_max, b.re_max)
    im_min = max(a.im_min, b.im_min)
    im_max = min(a.im_max, b.im_max)
    overlap = max(re_max - re_min, 0.0) * max(im_max - im_min, 0.0)
    denom = min(bounds_area(a), bounds_area(b))
    return denom <= 0 ? 0.0 : overlap / denom
end

"""
Return the centroid of complex samples (0+0im for empty).
"""
function component_centroid(z_samples::Vector{ComplexF64})
    isempty(z_samples) && return 0.0 + 0.0im
    return sum(z_samples) / length(z_samples)
end

"""
Merge undersized components into the nearest neighbor.

Arguments:

  - `components`: component list (mutated).
  - `min_samples`: minimum sample count to keep.
  - `kernel_config`: GP kernel configuration.
  - `margin`: bounds margin after merge.
"""
function ensure_min_samples!(
    components::Vector{ComponentState},
    min_samples::Int,
    kernel_config::KernelConfig;
    margin::Float64=0.5,
)
    length(components) <= 1 && return components
    keep = trues(length(components))
    kept = false
    for (idx, comp) in enumerate(components)
        length(comp.z_samples) >= min_samples && continue

        centroid = component_centroid(comp.z_samples)
        nearest_idx = nothing
        nearest_dist = Inf
        for (jdx, other) in enumerate(components)
            jdx == idx && continue
            if length(other.z_samples) < min_samples
                continue
            end
            dist = abs(centroid - component_centroid(other.z_samples))
            if dist < nearest_dist
                nearest_dist = dist
                nearest_idx = jdx
            end
        end
        if nearest_idx === nothing
            for (jdx, other) in enumerate(components)
                jdx == idx && continue
                dist = abs(centroid - component_centroid(other.z_samples))
                if dist < nearest_dist
                    nearest_dist = dist
                    nearest_idx = jdx
                end
            end
        end
        nearest_idx === nothing && continue

        target = components[nearest_idx]
        append!(target.z_samples, comp.z_samples)
        append!(target.s_samples, comp.s_samples)
        target.gp_post = build_gp(target.z_samples, target.s_samples, kernel_config)
        target.bounds = clamp_bounds(
            component_bounds(target.z_samples; margin=margin), target.initial_bounds
        )
        target.converged = false
        keep[idx] = false
    end
    kept = any(keep)
    return kept ? [components[i] for i in eachindex(components) if keep[i]] : components
end

"""
Return true if `z` lies inside `bounds`.
"""
function within_bounds(z::ComplexF64, bounds)
    re = real(z)
    im = imag(z)
    return bounds.re_min <= re <= bounds.re_max && bounds.im_min <= im <= bounds.im_max
end

"""
Create a `ComponentState` from samples and a kernel config.

Arguments:

  - `id`: component identifier.
  - `z_samples`: complex samples.
  - `s_samples`: shift samples.
  - `kernel_config`: GP kernel configuration.
  - `margin`: bounds margin.
  - `initial_bounds`: optional clamp bounds derived from initial seeding.
  - `converged`: initial convergence flag.
"""
function make_component(
    id::Int,
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    kernel_config::KernelConfig;
    margin::Float64=0.5,
    initial_bounds=nothing,
    converged::Bool=false,
)
    local_initial = if initial_bounds === nothing
        component_bounds(z_samples; margin=0.0)
    else
        initial_bounds
    end
    bounds = clamp_bounds(component_bounds(z_samples; margin=margin), local_initial)
    gp_post = build_gp(z_samples, s_samples, kernel_config)
    return ComponentState(
        id, z_samples, s_samples, gp_post, bounds, local_initial, converged
    )
end

"""
Run a local refinement pass for a component.

Arguments:

  - `comp`: component state (mutated).
  - `equation`, `constraint`, `algorithm`: solver inputs.
  - `kernel_config`: GP kernel configuration.
  - `n_iterations`: inner iterations to run.
  - `hull_tol`, `interior_tol`: stopping thresholds.
"""
function refine_component!(
    comp::ComponentState,
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing,
    kernel_config::KernelConfig;
    n_iterations::Int=3,
    hull_tol::Float64=0.01,
    interior_tol::Float64=hull_tol,
)
    gp_post, z_samples, s_samples = adaptive_refinement(
        comp.z_samples,
        comp.s_samples,
        equation,
        constraint,
        algorithm;
        bounds=comp.bounds,
        n_iterations=n_iterations,
        hull_tol=hull_tol,
        interior_tol=interior_tol,
        kernel_config=kernel_config,
    )
    comp.z_samples = z_samples
    comp.s_samples = s_samples
    comp.gp_post = gp_post
    comp.bounds = clamp_bounds(component_bounds(z_samples; margin=0.5), comp.initial_bounds)
    @info "Component bounds" id=comp.id bounds=comp.bounds
    vertices = hull_vertices(z_samples)
    if isempty(vertices)
        comp.converged = false
        return nothing
    end
    μs, σs = predict_batch(gp_post, vertices)
    max_hull_sigma = maximum(σs)
    candidates = candidate_points(z_samples; bounds=comp.bounds)
    μc, σc = predict_batch(gp_post, candidates)
    interior_mask = μc .> 0
    interior_sigma = isempty(findall(interior_mask)) ? 0.0 : maximum(σc[interior_mask])
    hull_is_exterior = all(
        s_samples[findfirst(i -> abs(z_samples[i] - v) < 1e-10, eachindex(z_samples))] < 0
        for v in vertices
    )
    return comp.converged =
        max_hull_sigma < hull_tol && hull_is_exterior && interior_sigma < interior_tol
end

"""
Split a component when multiple well-separated contours are detected.

Arguments:

  - `comp`: component to split.
  - `kernel_config`: GP kernel configuration.
  - `n_seeds`: contour seed count.
  - `min_loop_length`: minimum contour length to keep.
  - `min_component_samples`: minimum samples per component.
  - `min_centroid_separation`: minimum centroid distance to allow splitting.

Returns:

  - A vector of components (original if no split).
"""
function split_component_if_needed(
    comp::ComponentState,
    kernel_config::KernelConfig;
    n_seeds::Int=6,
    min_loop_length::Int=12,
    min_component_samples::Int=10,
    min_centroid_separation::Float64=0.0,
)
    contours = trace_contours(
        comp.gp_post,
        comp.z_samples;
        bounds=comp.bounds,
        n_seeds=n_seeds,
        min_length=min_loop_length,
    )
    if length(contours) <= 1
        return [comp]
    end
    @info "Split detected" id=comp.id n_contours=length(contours)
    centroids = [component_centroid(loop) for loop in contours]
    span_re = comp.bounds.re_max - comp.bounds.re_min
    span_im = comp.bounds.im_max - comp.bounds.im_min
    default_sep = 0.15 * max(span_re, span_im)
    min_sep = min_centroid_separation > 0 ? min_centroid_separation : default_sep
    if length(centroids) > 1
        closest = Base.minimum([
            abs(c1 - c2) for (i, c1) in enumerate(centroids) for
            (j, c2) in enumerate(centroids) if i < j
        ])
        if closest < min_sep
            @warn "Split suppressed: contours too close" id=comp.id closest min_sep
            return [comp]
        end
    end
    assigned = [ComplexF64[] for _ in centroids]
    assigned_s = [Float64[] for _ in centroids]
    for (z, s) in zip(comp.z_samples, comp.s_samples)
        idx = argmin(abs.(z .- centroids))
        push!(assigned[idx], z)
        push!(assigned_s[idx], s)
    end
    sizes = length.(assigned)
    large_indices = findall(size -> size >= min_component_samples, sizes)
    if length(large_indices) < 2
        @warn "Split suppressed: not enough large components" id=comp.id sizes
        return [comp]
    end

    merged_assigned = deepcopy(assigned)
    merged_assigned_s = deepcopy(assigned_s)
    for idx in eachindex(assigned)
        if idx in large_indices
            continue
        end
        nearest_large = argmin(abs.(centroids[idx] .- centroids[large_indices]))
        target_idx = large_indices[nearest_large]
        append!(merged_assigned[target_idx], merged_assigned[idx])
        append!(merged_assigned_s[target_idx], merged_assigned_s[idx])
        empty!(merged_assigned[idx])
        empty!(merged_assigned_s[idx])
    end

    components = ComponentState[]
    for (idx, (zs, ss)) in enumerate(zip(merged_assigned, merged_assigned_s))
        isempty(zs) && continue
        @info "Split component" parent=comp.id child_idx=idx n_samples=length(zs)
        push!(
            components,
            make_component(
                comp.id * 10 + idx,
                zs,
                ss,
                kernel_config;
                initial_bounds=comp.initial_bounds,
            ),
        )
    end
    total_samples = sum(length.(getfield.(components, :z_samples)))
    if total_samples != length(comp.z_samples)
        error(
            "Split lost samples: expected $(length(comp.z_samples)), got $(total_samples)"
        )
    end
    return components
end

"""
Merge components when overlap and bridge evidence indicate connectivity.

Arguments:

  - `components`: component list.
  - `kernel_config`: GP kernel configuration.
  - `overlap_threshold`: minimum overlap fraction.
  - `bridge_threshold`: GP mean threshold for connectivity.
  - `bridge_grid`: grid resolution used for bridge test.
"""
function merge_components_if_needed(
    components::Vector{ComponentState},
    kernel_config::KernelConfig;
    overlap_threshold::Float64=0.25,
    bridge_threshold::Float64=0.1,
    bridge_grid::Int=12,
)
    merged = ComponentState[]
    used = falses(length(components))
    for i in eachindex(components)
        used[i] && continue
        comp = components[i]
        for j in (i + 1):length(components)
            used[j] && continue
            other = components[j]
            overlap = bounds_overlap_fraction(comp.bounds, other.bounds)
            overlap < overlap_threshold && continue
            @info "Merge candidate" left=comp.id right=other.id overlap

            bounds = bounds_union((comp.bounds, other.bounds))
            re_grid = range(bounds.re_min, bounds.re_max; length=bridge_grid)
            im_grid = range(bounds.im_min, bounds.im_max; length=bridge_grid)
            bridge_found = false
            for re in re_grid, im in im_grid
                z = complex(re, im)
                μ1 = predict(comp.gp_post, z)[1]
                μ2 = predict(other.gp_post, z)[1]
                if max(μ1, μ2) > bridge_threshold
                    bridge_found = true
                    break
                end
            end
            if bridge_found
                @info "Merging components" left=comp.id right=other.id
                new_z = vcat(comp.z_samples, other.z_samples)
                new_s = vcat(comp.s_samples, other.s_samples)
                merged_initial = bounds_union((comp.initial_bounds, other.initial_bounds))
                comp = make_component(
                    comp.id, new_z, new_s, kernel_config; initial_bounds=merged_initial
                )
                used[j] = true
            end
        end
        push!(merged, comp)
    end
    return merged
end

"""
Run the full multi-component refinement loop.

Arguments:

  - `equation`, `constraint`, `algorithm`, `ensemble`: solver context.
  - `kernel_config`: GP kernel configuration.
  - `n_outer_iterations`: outer refinement iterations.
  - `per_component_iterations`: inner iterations per component.
  - `hull_tol`, `interior_tol`: stopping thresholds.
  - `cluster_radius`, `cluster_min_pts`: clustering controls.
  - `min_component_samples`: minimum samples per component.
  - `min_centroid_separation`: minimum centroid distance to allow splitting.

Returns:

  - Vector of final components.
"""
function multi_component_refinement(
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing,
    ensemble::MatrixEnsemble;
    kernel_config::KernelConfig=KernelConfig(),
    n_outer_iterations::Int=40,
    per_component_iterations::Int=2,
    hull_tol::Float64=0.01,
    interior_tol::Float64=hull_tol,
    cluster_radius::Float64=0.6,
    cluster_min_pts::Int=3,
    min_component_samples::Int=10,
    min_centroid_separation::Float64=0.0,
)
    z_samples, s_samples, boundary_candidates = initial_seeding(
        equation, constraint, algorithm, ensemble; n_directions=10, step=0.3
    )
    log_sample_distribution("Initial seeding samples", z_samples)
    components = initialize_components(
        z_samples,
        s_samples,
        boundary_candidates,
        kernel_config;
        cluster_radius=cluster_radius,
        cluster_min_pts=cluster_min_pts,
    )
    @info "Initialized components" n_components=length(components)
    for comp in components
        log_sample_distribution("Component samples", comp.z_samples)
    end

    for iter in 1:n_outer_iterations
        @info "Global refinement iteration" iter n_components=length(components)
        for comp in components
            refine_component!(
                comp,
                equation,
                constraint,
                algorithm,
                kernel_config;
                n_iterations=per_component_iterations,
                hull_tol=hull_tol,
                interior_tol=interior_tol,
            )
        end

        split_components = ComponentState[]
        split_happened = false
        for comp in components
            new_components = split_component_if_needed(
                comp,
                kernel_config;
                min_component_samples=min_component_samples,
                min_centroid_separation=min_centroid_separation,
            )
            split_happened |= length(new_components) > 1
            append!(split_components, new_components)
        end
        components = split_components

        for comp in components
            @info "Component status" id=comp.id n_samples=length(comp.z_samples) converged=comp.converged
        end

        merged_components = merge_components_if_needed(components, kernel_config)
        merge_happened = length(merged_components) < length(components)
        components = merged_components
        components = ensure_min_samples!(
            components, min_component_samples, kernel_config; margin=0.5
        )
        for comp in components
            log_sample_distribution("Post-refine samples", comp.z_samples)
        end

        all_converged = all(comp.converged for comp in components)
        if all_converged && !split_happened && !merge_happened
            @info "Global stopping criteria met" iter
            break
        end
    end
    return components
end

function cluster_boundary_candidates(
    candidates::Vector{ComplexF64}; radius::Float64=0.6, min_pts::Int=3
)
    n = length(candidates)
    n == 0 && return Vector{Vector{ComplexF64}}()
    visited = falses(n)
    clusters = Vector{Vector{ComplexF64}}()

    function neighbors(idx)
        z = candidates[idx]
        return [j for j in 1:n if abs(candidates[j] - z) <= radius]
    end

    for i in 1:n
        visited[i] && continue
        visited[i] = true
        neigh = neighbors(i)
        if length(neigh) < min_pts
            continue
        end
        cluster_indices = copy(neigh)
        queue = copy(neigh)
        while !isempty(queue)
            j = popfirst!(queue)
            if !visited[j]
                visited[j] = true
                neigh_j = neighbors(j)
                if length(neigh_j) >= min_pts
                    append!(queue, setdiff(neigh_j, queue))
                end
            end
            if !(j in cluster_indices)
                push!(cluster_indices, j)
            end
        end
        push!(clusters, candidates[cluster_indices])
    end
    return clusters
end

"""
Assign samples to the nearest cluster centroid.

Arguments:

  - `z_samples`: complex samples.
  - `s_samples`: shift samples.
  - `clusters`: vector of boundary clusters.

Returns:

  - Tuple `(assigned_z, assigned_s)` of per-cluster samples.
"""
function assign_samples_to_clusters(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    clusters::Vector{Vector{ComplexF64}},
)
    centroids = [component_centroid(cluster) for cluster in clusters]
    assigned = [ComplexF64[] for _ in centroids]
    assigned_s = [Float64[] for _ in centroids]
    for (z, s) in zip(z_samples, s_samples)
        idx = argmin(abs.(z .- centroids))
        push!(assigned[idx], z)
        push!(assigned_s[idx], s)
    end
    return assigned, assigned_s
end

"""
Initialize components from radial probing and boundary clustering.

Arguments:

  - `z_samples`, `s_samples`: initial samples.
  - `boundary_candidates`: boundary midpoints from probing.
  - `kernel_config`: GP kernel configuration.
  - `cluster_radius`, `cluster_min_pts`: clustering parameters.
  - `margin`: bounds margin for components.

Returns:

  - Vector of `ComponentState`.
"""
function initialize_components(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    boundary_candidates::Vector{ComplexF64},
    kernel_config::KernelConfig;
    cluster_radius::Float64=0.6,
    cluster_min_pts::Int=3,
    margin::Float64=0.5,
)
    clusters = cluster_boundary_candidates(
        boundary_candidates; radius=cluster_radius, min_pts=cluster_min_pts
    )
    @info "Clustered boundary candidates" n_clusters=length(clusters)
    initial_bounds = component_bounds(z_samples; margin=0.0)
    if length(clusters) <= 1
        return [
            make_component(
                1,
                copy(z_samples),
                copy(s_samples),
                kernel_config;
                margin=margin,
                initial_bounds=initial_bounds,
            ),
        ]
    end
    assigned, assigned_s = assign_samples_to_clusters(z_samples, s_samples, clusters)
    components = ComponentState[]
    for (idx, (zs, ss)) in enumerate(zip(assigned, assigned_s))
        if length(zs) < 5
            continue
        end
        push!(
            components,
            make_component(
                idx, zs, ss, kernel_config; margin=margin, initial_bounds=initial_bounds
            ),
        )
    end
    isempty(components) && return [
        make_component(
            1,
            copy(z_samples),
            copy(s_samples),
            kernel_config;
            margin=margin,
            initial_bounds=initial_bounds,
        ),
    ]
    return components
end

# =============================================================================
# Initial seeding + radial probing
# =============================================================================

function radial_probe!(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing,
    center::ComplexF64;
    n_directions::Int=12,
    step::Float64=0.15,
    max_steps::Int=80,
)
    """
    Radially probe from a center to collect samples and boundary candidates.

    Arguments:
    - `z_samples`, `s_samples`: sample buffers (mutated).
    - `equation`, `constraint`, `algorithm`: solver inputs.
    - `center`: starting complex point.
    - `n_directions`: number of radial directions.
    - `step`: step size along each ray.
    - `max_steps`: maximum steps per direction.

    Returns:
    - Vector of boundary candidate points.
    """
    s_center_ok, s_center = solve_shift(equation, constraint, algorithm, center)
    if !s_center_ok
        @warn "Seed solve failed" center
        return ComplexF64[]
    end
    push_sample!(z_samples, s_samples, center, s_center)

    boundary_candidates = ComplexF64[]
    for θ in range(0, 2π; length=n_directions + 1)[1:(end - 1)]
        z_prev = center
        s_prev = s_center
        direction_label = ""
        if abs(real(cis(θ))) >= abs(imag(cis(θ)))
            direction_label = real(cis(θ)) >= 0 ? "positive real" : "negative real"
        else
            direction_label = imag(cis(θ)) >= 0 ? "positive imag" : "negative imag"
        end
        for step_idx in 1:max_steps
            z_curr = center + step * step_idx * cis(θ)
            ok, s_curr = solve_shift(equation, constraint, algorithm, z_curr)
            if !ok
                @warn "Radial probe pointsolve failed" direction=direction_label θ step_idx z_curr
                break
            end
            push_sample!(z_samples, s_samples, z_curr, s_curr)
            if s_prev >= 0 && s_curr < 0
                mid = (z_prev + z_curr) / 2
                push!(boundary_candidates, mid)
                break
            end
            z_prev, s_prev = z_curr, s_curr
        end
    end
    return boundary_candidates
end

"""
Run initial seeding using eigenvalues and radial probing.

Arguments:

  - `equation`, `constraint`, `algorithm`, `ensemble`: solver context.
  - `n_directions`: number of probe directions per seed.
  - `step`: radial step size.

Returns:

  - `(z_samples, s_samples, boundary_candidates)`.
"""
function initial_seeding(
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing,
    ensemble::MatrixEnsemble;
    n_directions::Int=12,
    step::Float64=0.15,
)
    z_samples = ComplexF64[]
    s_samples = Float64[]
    boundary_candidates = ComplexF64[]

    for λ in ensemble_eigvals(ensemble)
        z0 = complex(λ)
        append!(
            boundary_candidates,
            radial_probe!(
                z_samples,
                s_samples,
                equation,
                constraint,
                algorithm,
                z0;
                n_directions=n_directions,
                step=step,
            ),
        )
    end
    return z_samples, s_samples, boundary_candidates
end

# =============================================================================
# GP construction and prediction
# =============================================================================

"""
Build a GP posterior for the shift surface.

Arguments:

  - `z_samples`: complex samples.
  - `s_values`: shift values.
  - `config`: kernel configuration.
"""
function build_gp(
    z_samples::Vector{ComplexF64},
    s_values::Vector{Float64},
    config::KernelConfig=KernelConfig(),
)
    x_samples = [[real(z), imag(z)] for z in z_samples]
    gp = GP(make_kernel(config))
    return posterior(gp(x_samples, 1e-6), s_values)
end

"""
Predict mean and standard deviation at a single point.

Arguments:

  - `gp_post`: GP posterior.
  - `z`: complex location.
"""
function predict(gp_post, z::ComplexF64)
    x = [real(z), imag(z)]
    μ = only(mean(gp_post, [x]))
    σ = sqrt(only(cov(gp_post, [x])))
    return μ, σ
end

"""
Predict means and standard deviations for a batch of points.

Arguments:

  - `gp_post`: GP posterior.
  - `zs`: vector of complex locations.
"""
function predict_batch(gp_post, zs::AbstractVector{ComplexF64})
    xs = [[real(z), imag(z)] for z in zs]
    μs = vec(mean(gp_post, xs))
    Σ = cov(gp_post, xs)
    σs = sqrt.(diag(Σ))
    return μs, σs
end

"""
Return a scalar mean function closure for the GP posterior.
"""
mean_function(gp_post) = x -> only(mean(gp_post, [x]))

# =============================================================================
# Acquisition strategy
# =============================================================================

"""
Acquisition score balancing boundary and interior exploration.

Arguments:

  - `gp_post`: GP posterior.
  - `z`: complex location.
  - `boundary_bandwidth`, `interior_bandwidth`: scale parameters.
  - `boundary_weight`, `interior_weight`: weighting factors.
"""
function acquisition(
    gp_post,
    z::ComplexF64;
    boundary_bandwidth::Float64=0.5,
    interior_bandwidth::Float64=0.5,
    boundary_weight::Float64=1.0,
    interior_weight::Float64=0.3,
)
    μ, σ = predict(gp_post, z)
    boundary_score = exp(-(μ^2) / (2 * boundary_bandwidth^2))
    interior_score = 1 / (1 + exp(-μ / interior_bandwidth))
    return σ * (boundary_weight * boundary_score + interior_weight * interior_score)
end

"""
Generate candidate points for acquisition.

Arguments:

  - `z_samples`: current samples.
  - `bounds`: optional bounding box.
  - `n_grid`: grid resolution.
  - `n_random`: number of random points.
  - `margin`: expansion when `bounds` is not provided.
"""
function candidate_points(
    z_samples::Vector{ComplexF64};
    bounds=nothing,
    n_grid::Int=25,
    n_random::Int=200,
    margin::Float64=0.5,
)
    if bounds === nothing
        re_min, re_max = extrema(real.(z_samples))
        im_min, im_max = extrema(imag.(z_samples))
        re_min -= margin
        re_max += margin
        im_min -= margin
        im_max += margin
    else
        re_min = bounds.re_min
        re_max = bounds.re_max
        im_min = bounds.im_min
        im_max = bounds.im_max
    end

    candidates = ComplexF64[]
    for re in range(re_min, re_max; length=n_grid),
        im in range(im_min, im_max; length=n_grid)

        z = complex(re, im)
        if bounds === nothing || within_bounds(z, bounds)
            push!(candidates, z)
        end
    end

    for _ in 1:n_random
        re = re_min + (re_max - re_min) * rand()
        im = im_min + (im_max - im_min) * rand()
        z = complex(re, im)
        if bounds === nothing || within_bounds(z, bounds)
            push!(candidates, z)
        end
    end

    for z in z_samples[rand(1:length(z_samples), min(50, length(z_samples)))]
        z_candidate = z + 0.2 * cis(2π * rand())
        if bounds === nothing || within_bounds(z_candidate, bounds)
            push!(candidates, z_candidate)
        end
    end
    return candidates
end

"""
Select the best acquisition candidate by sampling the candidate set.

Arguments:

  - `gp_post`: GP posterior.
  - `z_samples`: current samples.
  - `bounds`: optional bounding box.
  - `n_starts`: number of candidates to consider.
  - `margin`: expansion when `bounds` is not provided.
"""
function optimize_acquisition(
    gp_post,
    z_samples::Vector{ComplexF64};
    bounds=nothing,
    n_starts::Int=12,
    margin::Float64=0.5,
)
    candidates = candidate_points(
        z_samples; bounds=bounds, n_grid=6, n_random=max(n_starts, 20), margin=margin
    )

    if isempty(candidates)
        return nothing, -Inf
    end

    best_z = nothing
    best_acq = -Inf
    for z in candidates
        a = acquisition(gp_post, z)
        if a > best_acq
            best_acq = a
            best_z = z
        end
    end
    return best_z, best_acq
end

# =============================================================================
# Hull enforcement: all hull points exterior (shift < 0)
# =============================================================================

"""
Ensure all convex hull vertices lie in the exterior (shift < 0).

Arguments:

  - `z_samples`, `s_samples`: samples (mutated).
  - `equation`, `constraint`, `algorithm`: solver inputs.
  - `bounds`: optional clamp bounds.
  - `step`: step size along outward normal.
  - `max_steps`: maximum outward steps per vertex.
"""
function ensure_hull_exterior!(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing;
    bounds=nothing,
    step::Float64=0.2,
    max_steps::Int=30,
)
    vertices = hull_vertices(z_samples)
    if isempty(vertices)
        return false
    end
    center = mean(z_samples)
    added = false

    for v in vertices
        idx = findfirst(i -> abs(z_samples[i] - v) < 1e-10, eachindex(z_samples))
        idx === nothing && continue
        if s_samples[idx] < 0
            continue
        end
        direction = (v - center) / max(abs(v - center), 1e-12)
        for step_idx in 1:max_steps
            z_out = v + step * step_idx * direction
            if bounds !== nothing && !within_bounds(z_out, bounds)
                continue
            end
            ok, s_out = solve_shift(equation, constraint, algorithm, z_out)
            if ok && s_out < 0
                push_sample!(z_samples, s_samples, z_out, s_out)
                added = true
                break
            end
        end
    end
    return added
end

# =============================================================================
# Adaptive refinement loop
# =============================================================================

"""
Adaptive refinement loop for a single component.

Arguments:

  - `z_samples`, `s_samples`: samples (mutated).
  - `equation`, `constraint`, `algorithm`: solver inputs.
  - `bounds`: optional bounds for candidate generation.
  - `n_iterations`: iteration count.
  - `hull_tol`, `interior_tol`: stopping thresholds.
  - `kernel_config`: GP kernel configuration.
"""
function adaptive_refinement(
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64},
    equation::ResolventEquation,
    constraint::PTraceConstraint,
    algorithm::SpreadAnnealing;
    bounds=nothing,
    n_iterations::Int=60,
    hull_tol::Float64=0.01,
    interior_tol::Float64=hull_tol,
    kernel_config::KernelConfig=KernelConfig(),
)
    @info "Starting adaptive refinement" initial_samples=length(z_samples)

    gp_post = build_gp(z_samples, s_samples, kernel_config)
    for iter in 1:n_iterations
        while ensure_hull_exterior!(
            z_samples, s_samples, equation, constraint, algorithm; bounds=bounds
        )
            gp_post = build_gp(z_samples, s_samples, kernel_config)
        end

        vertices = hull_vertices(z_samples)
        if isempty(vertices)
            break
        end
        μs, σs = predict_batch(gp_post, vertices)
        max_hull_sigma = maximum(σs)
        hull_is_exterior = all(
            s_samples[findfirst(i -> abs(z_samples[i] - v) < 1e-10, eachindex(z_samples))] <
            0 for v in vertices
        )

        candidates = candidate_points(z_samples; bounds=bounds)
        μc, σc = predict_batch(gp_post, candidates)
        interior_mask = μc .> 0
        interior_sigma = isempty(findall(interior_mask)) ? 0.0 : maximum(σc[interior_mask])

        @info "Refinement status" iter n_samples=length(z_samples) max_hull_sigma interior_sigma hull_is_exterior

        if max_hull_sigma < hull_tol && hull_is_exterior && interior_sigma < interior_tol
            @info "Stopping criteria met" iter
            break
        end

        best_z, best_acq = optimize_acquisition(gp_post, z_samples; bounds=bounds)
        if best_z === nothing
            best_acq = -Inf
            best_z = candidates[1]
            for z in candidates
                a = acquisition(gp_post, z)
                if a > best_acq
                    best_acq = a
                    best_z = z
                end
            end
        end

        if bounds !== nothing && !within_bounds(best_z, bounds)
            @warn "candidate outside bounds" best_z bounds
            continue
        end

        ok, s_val = solve_shift(equation, constraint, algorithm, best_z)
        if ok
            push_sample!(z_samples, s_samples, best_z, s_val)
            gp_post = build_gp(z_samples, s_samples, kernel_config)
        else
            @warn "pointsolve failed at candidate" best_z
        end
    end

    return gp_post, z_samples, s_samples
end

# =============================================================================
# Boundary contour extraction (mean level set μ = 0)
# =============================================================================

"""
Find seed points for contour tracing of the GP mean level set.

Arguments:

  - `gp_post`: GP posterior.
  - `z_samples`: samples for range determination.
  - `bounds`: optional bounds for grid.
  - `n_seeds`: number of seeds to return.
  - `grid_size`: grid resolution.
"""
function find_contour_seeds(
    gp_post,
    z_samples::Vector{ComplexF64};
    bounds=nothing,
    n_seeds::Int=6,
    grid_size::Int=60,
)
    if bounds === nothing
        re_range = range(extrema(real.(z_samples))...; length=grid_size)
        im_range = range(extrema(imag.(z_samples))...; length=grid_size)
    else
        re_range = range(bounds.re_min, bounds.re_max; length=grid_size)
        im_range = range(bounds.im_min, bounds.im_max; length=grid_size)
    end
    grid_points = [complex(re, im) for re in re_range, im in im_range]
    μs_grid, _ = predict_batch(gp_post, vec(grid_points))
    μs_grid = reshape(μs_grid, size(grid_points))

    seeds = ComplexF64[]
    for i in 1:(length(re_range) - 1), j in 1:(length(im_range) - 1)
        μ_corners = (
            μs_grid[i, j], μs_grid[i + 1, j], μs_grid[i + 1, j + 1], μs_grid[i, j + 1]
        )
        if max(μ_corners...) > 0 && min(μ_corners...) < 0
            z_corners = [
                complex(re_range[i], im_range[j]),
                complex(re_range[i + 1], im_range[j]),
                complex(re_range[i + 1], im_range[j + 1]),
                complex(re_range[i], im_range[j + 1]),
            ]
            push!(seeds, sum(z_corners) / 4)
        end
    end

    if isempty(seeds)
        @warn "No contour seeds found"
        return ComplexF64[]
    end

    centroid = sum(seeds) / length(seeds)
    sort!(seeds; by=z -> angle(z - centroid))
    step = max(1, div(length(seeds), n_seeds))
    return seeds[1:step:length(seeds)]
end

"""
Trace a single contour loop of the GP mean level set.

Arguments:

  - `seed`: starting point.
  - `gp_post`: GP posterior.
  - `step_size`: prediction step length.
  - `max_steps`: maximum tracing steps.
  - `tol`: convergence tolerance for corrector.
"""
function trace_contour(
    seed::ComplexF64,
    gp_post;
    step_size::Float64=0.05,
    max_steps::Int=500,
    tol::Float64=1e-4,
)
    contour_points = ComplexF64[]
    z = seed
    δ = 0.01

    function gradient_at(z::ComplexF64)
        μ_xp = predict(gp_post, z + δ)[1]
        μ_xm = predict(gp_post, z - δ)[1]
        μ_yp = predict(gp_post, z + im * δ)[1]
        μ_ym = predict(gp_post, z - im * δ)[1]
        ∂μ_∂x = (μ_xp - μ_xm) / (2δ)
        ∂μ_∂y = (μ_yp - μ_ym) / (2δ)
        return complex(∂μ_∂x, ∂μ_∂y)
    end

    grad_seed = gradient_at(z)
    direction = abs2(grad_seed) < 1e-12 ? cis(π / 2) : cis(angle(grad_seed) + π / 2)
    initial_direction = direction
    n_corrector_failures = 0
    n_gradient_failures = 0

    for step in 1:max_steps
        z_pred = z + step_size * direction
        z_curr = z_pred
        corrector_converged = false

        for _ in 1:10
            μ = predict(gp_post, z_curr)[1]
            if abs(μ) < tol
                corrector_converged = true
                break
            end
            grad = gradient_at(z_curr)
            grad_norm_sq = abs2(grad)
            if grad_norm_sq < 1e-12
                break
            end
            z_curr = z_curr - μ * grad / grad_norm_sq
        end

        if !corrector_converged
            n_corrector_failures += 1
            if n_corrector_failures > 5
                break
            end
            continue
        end
        n_corrector_failures = 0

        μ_check = predict(gp_post, z_curr)[1]
        if abs(μ_check) > 10 * tol
            break
        end

        push!(contour_points, z_curr)

        grad = gradient_at(z_curr)
        if abs2(grad) < 1e-12
            n_gradient_failures += 1
            if n_gradient_failures > 10
                break
            end
        else
            new_direction = cis(angle(grad) + π / 2)
            dot_product = real(direction * conj(new_direction))
            direction = dot_product > 0 ? new_direction : -new_direction
            n_gradient_failures = 0
        end

        z = z_curr

        heading_back = real(direction * conj(initial_direction)) > 0
        if step > 50 && abs(z - first(contour_points)) < step_size && heading_back
            push!(contour_points, first(contour_points))
            break
        end
    end

    return contour_points
end

"""
Trace multiple contour loops from a set of seeds.

Arguments:

  - `gp_post`: GP posterior.
  - `z_samples`: sample set.
  - `bounds`: optional bounds for seeding.
  - `n_seeds`: number of seeds.
  - `min_length`: minimum contour length.
"""
function trace_contours(
    gp_post,
    z_samples::Vector{ComplexF64};
    bounds=nothing,
    n_seeds::Int=6,
    min_length::Int=10,
)
    seeds = find_contour_seeds(gp_post, z_samples; bounds=bounds, n_seeds=n_seeds)
    if isempty(seeds)
        return Vector{Vector{ComplexF64}}()
    end

    contours = Vector{ComplexF64}[]
    for seed in seeds
        contour = trace_contour(seed, gp_post)
        if length(contour) > min_length
            if bounds === nothing || all(z -> within_bounds(z, bounds), contour)
                push!(contours, contour)
            end
        end
    end
    return contours
end

"""
Extract the longest contour loop for plotting.

Arguments:

  - `gp_post`: GP posterior.
  - `z_samples`, `s_samples`: samples.
  - `bounds`: optional bounds for seeding.
  - `n_seeds`: number of seeds.
"""
function extract_boundary_contour(
    gp_post,
    z_samples::Vector{ComplexF64},
    s_samples::Vector{Float64};
    bounds=nothing,
    n_seeds::Int=6,
)
    contours = trace_contours(gp_post, z_samples; bounds=bounds, n_seeds=n_seeds)
    isempty(contours) && return ComplexF64[]
    lengths = map(length, contours)
    return contours[argmax(lengths)]
end

# =============================================================================
# Plotting helpers
# =============================================================================

function plot_components(components::Vector{ComponentState})
    """
    Plot components, their samples, and mean contours.

    Arguments:
    - `components`: vector of component states.
    """
    if isempty(components)
        @warn "No components to plot"
        return Figure()
    end
    combined_bounds = bounds_union([comp.bounds for comp in components])
    margin = 0.2
    re_limits = (combined_bounds.re_min - margin, combined_bounds.re_max + margin)
    im_limits = (combined_bounds.im_min - margin, combined_bounds.im_max + margin)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel="Re(z)",
        ylabel="Im(z)",
        aspect=DataAspect(),
        title="Kriging surrogate (mean) with components",
    )
    palette = [:white, :orange, :cyan, :magenta, :yellow, :green, :red]
    ax.limits = (re_limits..., im_limits...)

    for (idx, comp) in enumerate(components)
        comp_margin = 0.1
        re_grid = range(
            comp.bounds.re_min - comp_margin, comp.bounds.re_max + comp_margin; length=120
        )
        im_grid = range(
            comp.bounds.im_min - comp_margin, comp.bounds.im_max + comp_margin; length=120
        )
        values = [
            predict(comp.gp_post, complex(re, im))[1] for re in re_grid, im in im_grid
        ]
        contourf!(
            ax, re_grid, im_grid, values; colormap=:RdBu, levels=16, transparency=true
        )
        scatter!(
            ax,
            real.(comp.z_samples),
            imag.(comp.z_samples);
            color=:black,
            markersize=5,
            strokewidth=0.5,
        )
        contours = trace_contours(comp.gp_post, comp.z_samples; bounds=comp.bounds)
        for contour in contours
            lines!(
                ax,
                real.(contour),
                imag.(contour);
                color=palette[mod1(idx, length(palette))],
                linewidth=2,
            )
        end
    end
    return fig
end

# =============================================================================
# End-to-end prototype example run
# =============================================================================

ensemble = MatrixEnsemble(Diagonal([1, -1]), I, I, 0I)
equation = ResolventEquation(ensemble; shiftmin=-2)
constraint = PTraceConstraint(1; shift0=1, shape0=0)
algorithm = SpreadAnnealing()

kernel_config = KernelConfig()
components = multi_component_refinement(
    equation,
    constraint,
    algorithm,
    ensemble;
    kernel_config=kernel_config,
    min_centroid_separation=0.5,
    n_outer_iterations=60,
    per_component_iterations=3,
    hull_tol=0.002,
    interior_tol=0.002,
)

fig = plot_components(components)
display(fig)

# The GP mean is differentiable; use automatic differentiation on mean_function(gp_post)
# for analytic derivatives needed in spectral density computations.
