module Boundaries

using GeometryBasics: Mesh, Point2, TriangleFace, merge
using DistMesh: distmesh2d, huniform, dpoly
using Statistics: median
using StructArrays: components

using ..Defaults: Defaults
using ..Newton: increment
using ..Meshing: Meshing
using ..SpectralModels: SpectralModel
using ..Solutions: issuccess
using ..Spreads: Spreads
using ..Surrogates: SurrogateModel, evaluate

using Statistics: median

public BoundarySegment, SpectralModel, spectral_boundary, point_in_polygon

# TODO: Derive and implement analytic gradient (implicit differentiation)

"""
    BoundarySegment(points, shift_values, seed, closed, internal=false)

Container for traced boundary points and their shift values.

  - `points` are ordered along the contour.
  - `shift_values` aligns with `points` and should be near zero.
  - `gradients` holds the shift gradient at each point.
  - `seed` stores the (refined) starting point.
  - `closed` marks whether the contour loop closed.
  - `internal` marks whether this is an internal boundary (hole) of an annular support
    region. `false` for outer boundaries and open contours.
  - `nsolvercalls` is a diagnostic count of how many surrogate evaluations were made.
    Defaults to `-1` if not provided at construction.
"""
struct BoundarySegment{F<:AbstractFloat}
    points::Vector{Complex{F}}
    shift_values::Vector{F}
    gradients::Vector{Complex{F}}
    seed::Complex{F}
    closed::Bool
    internal::Bool
    nsolvercalls::Int

    function BoundarySegment(
        points, shift_values, gradients, seed, closed, internal, nsolvercalls=-1
    )
        F = real(eltype(points))
        return new{F}(points, shift_values, gradients, seed, closed, internal, nsolvercalls)
    end
end

"""
    solve_shift(model, z; counter) -> shift

Evaluate the surrogate model at `z` and return the shift value (or `NaN` on failure).
Increments `counter[]` if provided.
"""
function solve_shift(
    model::SurrogateModel{F}, z::Complex{F}; counter::Union{Ref{Int},Nothing}=nothing
) where {F}
    if !isnothing(counter)
        counter[] += 1
    end
    sol = evaluate(model, z; refine=false)
    return issuccess(sol) ? F(Spreads.shift(sol)) : F(NaN)
end

"""
    compute_gradient(model, z; counter) -> grad | nothing

Compute the shift gradient at `z` via centered finite differences.
"""
function compute_gradient(
    model::SurrogateModel{F}, z::Complex{F}; counter::Union{Ref{Int},Nothing}=nothing
) where {F}
    h = increment(real(z)) / 2
    shift_xp = solve_shift(model, z + h; counter)
    shift_xm = solve_shift(model, z - h; counter)
    shift_yp = solve_shift(model, z + im * h; counter)
    shift_ym = solve_shift(model, z - im * h; counter)
    if !all(isfinite, (shift_xp, shift_xm, shift_yp, shift_ym))
        return nothing
    end

    ds_dx = (shift_xp - shift_xm) / (2h)
    ds_dy = (shift_yp - shift_ym) / (2h)
    return complex(ds_dx, ds_dy)
end

# scale by -im => rotate 90 deg cw to get ccw tangent from inward-pointing gradient
tangent_from_gradient(grad) = -im * grad / abs(grad)

"""
    corrector_step(model, z; tol, max_iters, counter)

Project `z` onto the shift=0 contour via gradient correction.
"""
function corrector_step(
    model::SurrogateModel{F},
    z::Complex{F};
    tol::F=Defaults.atol(F),
    max_iters::Integer=48,
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    z_current = Complex{F}(z)
    for _ in 0:max_iters
        shift_val = solve_shift(model, z_current; counter)
        isfinite(shift_val) || return nothing
        grad = compute_gradient(model, z_current; counter)
        isnothing(grad) && return nothing
        (abs(shift_val) < tol) && return (z_current, shift_val, grad)
        (abs(grad) < tol) && return nothing  # too flat to correct
        z_current -= shift_val * grad / abs2(grad)
    end
    return nothing
end

"""
    predictor_corrector_step(model, z, direction, step_size; tol, max_iters, counter)

Take one step along the shift=0 contour: predict along `direction`, correct onto the
contour, and update the tangent direction. Returns `(new_z, shift_val)` on success,
`nothing` on failure.
"""
function predictor_corrector_step(
    model::SurrogateModel{F},
    z::Complex{F},
    grad::Complex{F},
    step_size::F;
    tol::F=Defaults.atol(F),
    max_iters::Integer=48,
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    tangent = tangent_from_gradient(grad)
    # step along tangent; correct back to contour; if corrector fails or step is not
    # accepted, halve step size and try again
    for i in 0:max_iters
        z_pred = z + (F(2)^(-i // 3)) * step_size * tangent
        corrected = corrector_step(model, z_pred; tol, counter)
        if !isnothing(corrected)
            new_z, _, new_grad = corrected
            new_tangent = tangent_from_gradient(new_grad)
            accept_step(new_z - z, tangent, new_tangent, step_size) && return corrected
        end
    end
    return nothing
end

function accept_step(dz, tangent_start, tangent_end, step_size)
    # Step acceptance criteria:
    # - projection of tangent_end onto tangent_start > 1/2 (less than 60° turn)
    # - projection of corrected step direction onto tangent_start > 1/2
    # - projection of corrected step onto tangent_start not longer than predicted step
    # - projection of corrected step direction onto tangent_end > 1/2
    # - projection of corrected step onto tangent_end not longer than predicted step
    absdz = abs(dz)
    return (
        (1 <= 2real(tangent_end'tangent_start)) &&
        (absdz <= 2real(dz'tangent_start) <= 2step_size) &&
        (absdz <= 2real(dz'tangent_end) <= 2step_size)
    )
end

"""
    try_close!(
        model, points, shift_values, gradients, step_size; tol, max_iters, counter
    ) -> Bool

Check whether the contour has looped back to its starting point and, if so, refine the
closure. Returns `true` if the contour was successfully closed.
"""
function try_close!(
    model::SurrogateModel{F},
    points::Vector{Complex{F}},
    shift_values::Vector{F},
    gradients::Vector{Complex{F}},
    step_size::F;
    tol::F=Defaults.atol(F),
    max_iters::Integer=48,
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    initial, final = points[begin], points[end]
    dclose = initial - final
    (abs(dclose) <= 2step_size) || return false
    final_pred = final
    prev = points[end - 1]
    dfinal = final - prev
    current_shift = shift_values[end]
    current_grad = gradients[end]
    initial_tangent = tangent_from_gradient(gradients[begin])
    for i in 0:max_iters
        final_tangent = tangent_from_gradient(current_grad)
        if accept_step(dclose, initial_tangent, final_tangent, step_size)
            points[end] = final
            shift_values[end] = current_shift
            gradients[end] = current_grad
            push!(points, initial)
            push!(shift_values, shift_values[begin])
            push!(gradients, gradients[begin])
            return true
        end
        final_pred = prev + (F(2)^(-(i + 1) // 3)) * dfinal
        corrected = corrector_step(model, final_pred; tol, counter)
        isnothing(corrected) && break  # corrector failed, can't proceed
        final, current_shift, current_grad = corrected
        dclose = initial - final
        (abs(dclose) <= 2step_size) || break # left the closure neighborhood, can't proceed
    end
    # We couldn't satisfy both the distance and direction criteria, give up
    return false
end

"""
    classify_interior(model, points) -> Bool

Determine whether a closed contour is an internal boundary (hole). Computes the shift
gradient at a contour point and offsets a test point in the gradient direction (toward
shift > 0). If the test point lies inside the polygon, the support is interior → outer
boundary → `false`. If outside, the support is exterior → hole → `true`.
"""
function classify_interior(
    points::Vector{Complex{F}}, gradients::Vector{Complex{F}}; tol::F=Defaults.atol(F)
) where {F}
    for (z, grad) in zip(points, gradients)
        test_z = z + increment(abs(z)) * grad / abs(grad)
        return !point_in_polygon(test_z, points; tol)
    end
    return false
end

"""
    spectral_boundary_segment(model, seed, step_size; tol, ...) -> BoundarySegment

Trace a single shift=0 contour starting from `seed` (assumed already refined onto the
contour). Returns a `BoundarySegment` with the traced points, contour properties, and
diagnostics.
"""
function spectral_boundary_segment(
    model::SurrogateModel{F},
    seed::Complex{F},
    step_size::F;
    tol::F=Defaults.atol(F),
    max_steps::Integer=2^16,
    min_steps_before_close::Integer=5,
) where {F}
    counter = Ref(0)

    points, shift_values, gradients, closed = trace_contour(
        model, seed, step_size; tol, counter, max_steps, min_steps_before_close
    )

    internal = closed ? classify_interior(points, gradients; tol) : false
    return BoundarySegment(
        points, shift_values, gradients, seed, closed, internal, counter[]
    )
end

"""
    trace_contour(model, seed_z, step_size; tol, counter, ...) -> closed

Run the predictor-corrector loop, appending traced points to `points` and `shift_values`.
Returns whether the final contour is closed.
"""
function trace_contour(
    model::SurrogateModel{F},
    seed_z,
    step_size;
    tol::F=Defaults.atol(F),
    counter::Union{Ref{Int},Nothing}=nothing,
    max_steps::Integer=2^16,
    min_steps_before_close::Integer=5,
) where {F}
    points, shift_values, gradients = Complex{F}[], F[], Complex{F}[]
    corrected_seed = corrector_step(model, seed_z; tol, counter)
    isnothing(corrected_seed) && return points, shift_values, gradients, false
    current_z, current_shift, current_gradient = corrected_seed
    push!(points, current_z)
    push!(shift_values, current_shift)
    push!(gradients, current_gradient)
    for step_index in 1:max_steps
        step = predictor_corrector_step(
            model, current_z, current_gradient, step_size; tol, counter
        )
        isnothing(step) && break

        current_z, current_shift, current_gradient = step
        push!(points, current_z)
        push!(shift_values, current_shift)
        push!(gradients, current_gradient)

        if (
            (step_index >= min_steps_before_close) &&
            try_close!(model, points, shift_values, gradients, step_size; tol, counter)
        )
            return points, shift_values, gradients, true
        end
    end
    return points, shift_values, gradients, false
end

"""
    point_in_polygon(z, polygon; tol) -> Bool

Return `true` if `z` is inside the closed polygon defined by `polygon`.
"""
function point_in_polygon(
    z::Complex{F}, polygon::AbstractVector{Complex{F}}; tol::F=Defaults.atol(F)
) where {F<:AbstractFloat}
    n = length(polygon)
    (n < 3) && return false

    vertices = polygon
    if (n > 1) && (polygon[end] == polygon[begin])
        vertices = polygon[begin:(end - 1)]
        n = length(vertices)
    end
    (n < 3) && return false
    length(unique(vertices)) < 3 && return false

    area = zero(F)
    for i in eachindex(vertices)
        p1 = vertices[i]
        p2 = (i == lastindex(vertices)) ? vertices[begin] : vertices[i + 1]
        area += real(p1) * imag(p2) - real(p2) * imag(p1)
    end
    (abs(area) <= tol) && return false

    total_angle = zero(F)
    for i in eachindex(vertices)
        p1 = vertices[i] - z
        p2 = ((i == lastindex(vertices)) ? vertices[begin] : vertices[i + 1]) - z
        dtheta = angle(p2) - angle(p1)
        dtheta = mod2pi(dtheta + pi) - pi
        total_angle += dtheta
    end
    return abs(total_angle) > pi
end

"""
    is_near_contour(z, contour, threshold) -> Bool

Return `true` if `z` is within `threshold` of any point in `contour`.
"""
function is_near_contour(
    z::Complex{F}, contour::BoundarySegment{F}, threshold::F
) where {F<:AbstractFloat}
    isempty(contour.points) && return false
    for p in contour.points
        (abs(z - p) < threshold) && return true
    end
    return false
end

"""
    is_near_any_contour(z, contours, threshold) -> Bool

Return `true` if `z` is near any contour in `contours`.
"""
function is_near_any_contour(
    z::Complex{F}, contours::AbstractVector{BoundarySegment{F}}, threshold::F
) where {F<:AbstractFloat}
    for contour in contours
        is_near_contour(z, contour, threshold) && return true
    end
    return false
end

"""
    spectral_boundary(
        model::SpectralModel; step_size_coeff=1//4, kwargs...
    ) -> Vector{BoundarySegment}

Extract all boundary segments from the model's surrogate.
"""
function spectral_boundary(model::SpectralModel; step_size_coeff=1 // 4, kwargs...)
    return spectral_boundary(model.surrogate, step_size_coeff; kwargs...)
end

"""
    spectral_boundary(model::SurrogateModel; kwargs...) -> Vector{BoundarySegment}

Trace all disconnected boundary components. Sample points are sorted by
`abs(shift)` and each is tried once as a potential seed for a new contour.
"""
function spectral_boundary(
    model::SurrogateModel{F},
    step_size_coeff::Real=one(F);
    tol::Real=Defaults.atol(F),
    allow_open::Bool=false,
    max_steps::Integer=2^16,
    min_steps_before_close::Integer=5,
    max_components::Union{Nothing,Integer}=nothing,
) where {F}
    step_size = F(step_size_coeff * median(model.rbf.nn_dists))
    tol = F(tol)
    if step_size <= 0
        throw(ArgumentError("step_size_coeff must be positive"))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive"))
    end
    if !isnothing(max_components) && (max_components <= 0)
        throw(ArgumentError("max_components must be positive"))
    end

    contours = BoundarySegment{F}[]

    candidates = map(model.rbf.x) do xv
        z = complex(xv[1], xv[2])
        s = solve_shift(model, z)
        return (z, abs(s))
    end
    filter!(isfinite ∘ last, candidates)
    sort!(candidates; by=last)

    for (z, _) in candidates
        refined = corrector_step(model, z; tol)
        isnothing(refined) && continue
        seed, _, _ = refined
        is_near_any_contour(seed, contours, step_size) && continue

        c = spectral_boundary_segment(
            model, seed, step_size; tol, max_steps, min_steps_before_close
        )
        if allow_open || c.closed
            push!(contours, c)
            (!isnothing(max_components) && (length(contours) >= max_components)) && break
        end
    end

    return contours
end

## Meshing methods
## ===============
function Meshing.mesh(contours::AbstractVector{<:BoundarySegment}, h::Real)
    return merge(map(c -> Meshing.mesh(c, h), contours))
end

function Meshing.mesh(contour::BoundarySegment{F}, h::Real) where {F}
    fd, bbox = let pv=map(reim, contour.points)
        fd(p) = dpoly(p, pv)
        fd, Meshing.boundingbox(pv)
    end
    bw, bh = bbox[2][1] - bbox[1][1], bbox[2][2] - bbox[1][2]
    bd = min(bw, bh)::F
    h0 = min(F(h), bd/3)::F
    dmsh = distmesh2d(
        # TODO: tweak kwargs in meshing.jl too
        fd,
        huniform,
        h0,
        bbox;
        maxiter=2^16,
        ttol=h0/(2^4),
        geps=2sqrt(eps(F)) * bd,
        deps=sqrt(eps(F)) * bd,
    )
    return Mesh(map(Point2, dmsh.p), map(TriangleFace, dmsh.t))
end

end  # module Boundaries

@testitem "solve_shift inside support" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Origin is well inside the elliptic support
    s = solve_shift(model, complex(0.0))
    @test isfinite(s)
    @test s > 0
end

@testitem "solve_shift outside support" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Well outside the elliptic support (semiaxes 1.25 × 0.75)
    s = solve_shift(model, complex(2.0))
    @test isfinite(s)
    @test s < 0
end

@testitem "solve_shift on boundary" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Right endpoint of ellipse: (1.25, 0)
    s = solve_shift(model, complex(1.25))
    @test isfinite(s)
    @test abs(s) < 1e-4

    # Top of ellipse: (0, 0.75)
    s = solve_shift(model, complex(0.0, 0.75))
    @test isfinite(s)
    @test abs(s) < 1e-4
end

@testitem "solve_shift failure" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Outside support with shiftmin = 0 should fail
    s = solve_shift(model, complex(3.0))
    @test isnan(s)
end

@testitem "compute_gradient on boundary" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # At right endpoint (1.25, 0), the gradient should point leftward (shift decreases outward)
    grad = compute_gradient(model, complex(1.25))
    @test !isnothing(grad)
    @test real(grad) < 0

    # At top of ellipse (0, 0.75), the gradient should point downward (shift decreases outward)
    grad = compute_gradient(model, complex(0.0, 0.75))
    @test !isnothing(grad)
    @test imag(grad) < 0
end

@testitem "compute_gradient inside support" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Inside the support, gradient should still be computable
    grad = compute_gradient(model, complex(0.5, 0.3))
    @test !isnothing(grad)
    @test abs(grad) > 0
end

@testitem "compute_gradient failure" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Finite-difference evaluations should fail outside support when shiftmin = 0
    grad = compute_gradient(model, complex(3.0))
    @test isnothing(grad)
end

@testitem "corrector_step convergence" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: corrector_step
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Start near the boundary (slightly inside at real axis, boundary at 1.25)
    result = corrector_step(model, complex(1.2))
    @test !isnothing(result)
    z_corrected, s_corrected = result
    @test abs(s_corrected) < 1e-6
    # The corrected point should be near the ellipse boundary
    @test abs(real(z_corrected)) > 1.0  # still near the right side

    # Start near the boundary (slightly outside)
    result = corrector_step(model, complex(1.3))
    @test !isnothing(result)
    z_corrected, s_corrected = result
    @test abs(s_corrected) < 1e-6
end

@testitem "corrector_step far from boundary" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: corrector_step
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Very far from any zero crossing; one iteration should not converge to tight tolerance
    result = corrector_step(model, complex(3.0); tol=1e-12, max_iters=1)
    @test isnothing(result)
end

@testitem "point_in_polygon unit square" begin
    using GaussianSpectralDensities.Boundaries: point_in_polygon

    @testset "$F" for F in [Float32, Float64]
        # Unit square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        square = Complex{F}[0, 1, 1 + im, im, 0]

        # Interior point
        @test point_in_polygon(Complex{F}(0.5 + 0.5im), square)

        # Exterior points
        @test !point_in_polygon(Complex{F}(2.0 + 0.5im), square)
        @test !point_in_polygon(Complex{F}(-1.0 + 0.5im), square)
        @test !point_in_polygon(Complex{F}(0.5 + 2.0im), square)
        @test !point_in_polygon(Complex{F}(0.5 - 1.0im), square)

        # Point far away
        @test !point_in_polygon(Complex{F}(100.0 + 100.0im), square)
    end
end

@testitem "point_in_polygon triangle" begin
    using GaussianSpectralDensities.Boundaries: point_in_polygon

    @testset "$F" for F in [Float32, Float64]
        # Triangle: (0,0), (2,0), (1,2), closed
        triangle = Complex{F}[0, 2, 1 + 2im, 0]

        # Centroid is inside
        @test point_in_polygon(Complex{F}(1.0 + 0.5im), triangle)

        # Outside the triangle
        @test !point_in_polygon(Complex{F}(-1.0 + 0.0im), triangle)
        @test !point_in_polygon(Complex{F}(3.0 + 0.0im), triangle)
        @test !point_in_polygon(Complex{F}(1.0 + 3.0im), triangle)
    end
end

@testitem "point_in_polygon implicitly closed" begin
    using GaussianSpectralDensities.Boundaries: point_in_polygon

    @testset "$F" for F in [Float32, Float64]
        # Square without explicit closure (last != first)
        square_open = Complex{F}[0, 1, 1 + im, im]

        # Interior point should still be detected
        @test point_in_polygon(Complex{F}(0.5 + 0.5im), square_open)

        # Exterior point
        @test !point_in_polygon(Complex{F}(2.0 + 0.5im), square_open)
    end
end

@testitem "point_in_polygon degenerate cases" begin
    using GaussianSpectralDensities.Boundaries: point_in_polygon

    @testset "$F" for F in [Float32, Float64]
        # Empty polygon
        @test !point_in_polygon(Complex{F}(0.5 + 0.5im), Complex{F}[])

        # Single point
        @test !point_in_polygon(Complex{F}(0.5 + 0.5im), Complex{F}[0])

        # Two points (line segment, < 3 distinct vertices)
        @test !point_in_polygon(Complex{F}(0.5 + 0.5im), Complex{F}[0, 1])

        # Collinear points (degenerate polygon)
        @test !point_in_polygon(Complex{F}(0.5 + 0.5im), Complex{F}[0, 1, 2])
    end
end

@testitem "point_in_polygon concave polygon" begin
    using GaussianSpectralDensities.Boundaries: point_in_polygon

    @testset "$F" for F in [Float32, Float64]
        # L-shaped polygon (concave)
        # Vertices: (0,0), (2,0), (2,1), (1,1), (1,2), (0,2), (0,0)
        lshape = Complex{F}[0, 2, 2 + im, 1 + im, 1 + 2im, 2im, 0]

        # Inside the L
        @test point_in_polygon(Complex{F}(0.5 + 0.5im), lshape)
        @test point_in_polygon(Complex{F}(0.5 + 1.5im), lshape)

        # In the "notch" of the L (outside)
        @test !point_in_polygon(Complex{F}(1.5 + 1.5im), lshape)

        # Outside entirely
        @test !point_in_polygon(Complex{F}(3.0 + 0.5im), lshape)
    end
end

@testitem "is_near_contour" begin
    using GaussianSpectralDensities.Boundaries: BoundarySegment, is_near_contour

    @testset "$F" for F in [Float32, Float64]
        # Contour with a few known points
        pts = Complex{F}[1, im, -1, -im, 1]
        shifts = zeros(F, 5)
        contour = BoundarySegment(pts, shifts, zero(pts), Complex{F}(1), true, false)

        # Point exactly on a contour point: distance = 0 < any positive threshold
        @test is_near_contour(Complex{F}(1), contour, F(0.1))

        # Point very close to a contour point
        @test is_near_contour(Complex{F}(1.05), contour, F(0.1))

        # Point far from all contour points
        @test !is_near_contour(Complex{F}(5.0 + 5.0im), contour, F(0.1))

        # Large threshold makes everything near
        @test is_near_contour(Complex{F}(2.0), contour, F(2.0))

        # Empty contour points
        empty_contour = BoundarySegment(
            Complex{F}[], F[], Complex{F}[], Complex{F}(0), true, false
        )
        @test !is_near_contour(Complex{F}(0), empty_contour, F(1.0))
    end
end

@testitem "is_near_contour threshold boundary" begin
    using GaussianSpectralDensities.Boundaries: BoundarySegment, is_near_contour

    @testset "$F" for F in [Float32, Float64]
        pts = Complex{F}[0]
        shifts = zeros(F, 1)
        contour = BoundarySegment(pts, shifts, zero(pts), Complex{F}(0), true, false)

        # Distance is exactly 0.5; threshold is 1.0 → near
        @test is_near_contour(Complex{F}(0.5), contour, F(1.0))

        # Distance is 0.5; threshold is 0.25 → not near
        @test !is_near_contour(Complex{F}(0.5), contour, F(0.25))
    end
end

@testitem "is_near_any_contour" begin
    using GaussianSpectralDensities.Boundaries: BoundarySegment, is_near_any_contour

    @testset "$F" for F in [Float32, Float64]
        pts1 = Complex{F}[1, im, -1, -im, 1]
        contour1 = BoundarySegment(
            pts1, zeros(F, 5), zero(pts1), Complex{F}(1), true, false
        )

        pts2 = Complex{F}[5, 5 + im, 5 + 2im]
        contour2 = BoundarySegment(
            pts2, zeros(F, 3), zero(pts2), Complex{F}(5), false, false
        )

        pts3 = Complex{F}[10, 10 + im, 10 + 2im, 10]
        contour3 = BoundarySegment(
            pts3, zeros(F, 4), zero(pts3), Complex{F}(10), true, false
        )

        contours = BoundarySegment{F}[contour1, contour2, contour3]
        threshold = F(0.5)

        # Near contour1
        @test is_near_any_contour(Complex{F}(1.1), contours, threshold)

        # Near contour3
        @test is_near_any_contour(Complex{F}(10.1), contours, threshold)

        # Near contour2
        @test is_near_any_contour(Complex{F}(5.1), contours, threshold)

        # Far from everything
        @test !is_near_any_contour(Complex{F}(50.0), contours, threshold)

        # Empty contours
        @test !is_near_any_contour(Complex{F}(1.0), BoundarySegment{F}[], threshold)
    end
end

@testitem "spectral_boundary single component" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: BoundarySegment, spectral_boundary
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    contours = spectral_boundary(model)
    @test contours isa Vector{BoundarySegment{Float64}}
    @test length(contours) == 1

    contour = contours[1]
    @test contour.closed
    @test !contour.internal  # outer boundary of simply connected support
    @test length(contour.points) > 10

    # All shift values should be near zero
    @test all(s -> abs(s) < 1e-3, contour.shift_values)

    # Each boundary point should lie approximately on the ellipse
    for z in contour.points
        ellipse_val = (real(z) / 1.25)^2 + (imag(z) / 0.75)^2
        @test abs(ellipse_val - 1.0) < 0.1
    end
end

@testitem "spectral_boundary annular support" begin
    using GaussianSpectralDensities:
        MatrixEnsemble, PTraceConstraint, ResolventEquation, SpreadAnnealing
    using GaussianSpectralDensities.Boundaries:
        BoundarySegment, spectral_boundary, point_in_polygon
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: Diagonal, I

    # Four connected lobes surrounding a hole
    M = Diagonal([1.05, 1.05im, -1.05, -1.05im])
    ensemble = MatrixEnsemble(M, I, I, 0.5I)
    eq = ResolventEquation(ensemble; shiftmin=-0.5)
    constraint = PTraceConstraint(1.0; shift0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, SpreadAnnealing(), 96)

    contours = spectral_boundary(model)
    @test contours isa Vector{BoundarySegment{Float64}}
    @test length(contours) == 2

    # Sort by contour size so outer comes first
    sort!(contours; by=c -> -length(c.points))

    outer = contours[1]
    inner = contours[2]

    @test outer.closed
    @test !outer.internal  # outer boundary

    @test inner.closed
    @test inner.internal   # inner boundary (hole)

    # All shift values should be near zero on both boundaries
    @test all(s -> abs(s) < 1e-3, outer.shift_values)
    @test all(s -> abs(s) < 1e-3, inner.shift_values)

    # A point inside the annular support region (between inner and outer boundaries)
    z_in_support = complex(0.8)
    # A point inside the hole (inside the inner boundary)
    z_in_hole = complex(0.1)
    # A point outside the outer boundary
    z_outside = complex(2.0)

    # Inside the outer boundary but outside the inner boundary → in the support
    @test point_in_polygon(z_in_support, outer.points)
    @test !point_in_polygon(z_in_support, inner.points)

    # Inside both boundaries → in the hole
    @test point_in_polygon(z_in_hole, outer.points)
    @test point_in_polygon(z_in_hole, inner.points)

    # Outside both boundaries → outside the support
    @test !point_in_polygon(z_outside, outer.points)
    @test !point_in_polygon(z_outside, inner.points)
end

@testitem "spectral_boundary max_components limit" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: spectral_boundary
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    contours = spectral_boundary(model; max_components=1)
    @test length(contours) <= 1
end

@testitem "spectral_boundary argument validation" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: spectral_boundary
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    @test_throws ArgumentError spectral_boundary(model, 0.0)
    @test_throws ArgumentError spectral_boundary(model, -0.1)
    @test_throws ArgumentError spectral_boundary(model; tol=0.0)
    @test_throws ArgumentError spectral_boundary(model; tol=-1e-6)
    @test_throws ArgumentError spectral_boundary(model; max_components=0)
    @test_throws ArgumentError spectral_boundary(model; max_components=-1)
end

@testitem "spectral_boundary keyword forwarding" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Boundaries: spectral_boundary
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # max_steps=2 should produce all open contours
    contours = spectral_boundary(model; max_steps=2)
    @test length(contours) > 0
    @testset "contour $i" for (i, c) in pairs(contours)
        @test !c.closed
    end
end

@testitem "spectral_boundary from SpectralModel" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, SpectralProblem, solve
    using GaussianSpectralDensities.Boundaries: BoundarySegment, spectral_boundary

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    prob = SpectralProblem(ensemble, constraint)
    model = solve(prob, DirectSolve(); nsamples=16)

    contours = spectral_boundary(model)
    @test contours isa Vector{BoundarySegment{Float64}}
    @test length(contours) == 1

    contour = contours[1]
    @test contour.closed
    @test !contour.internal
    @test length(contour.points) > 10
    @test all(s -> abs(s) < 1e-3, contour.shift_values)
end

@testitem "spectral_boundary SpectralModel wrapper equivalence" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, SpectralProblem, solve
    using GaussianSpectralDensities.Boundaries: spectral_boundary

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    prob = SpectralProblem(ensemble, constraint)
    model = solve(prob, DirectSolve(); nsamples=16)

    step_size_coeff = 1 // 4
    contours_from_model = spectral_boundary(model; step_size_coeff, max_components=1)
    contours_from_surrogate = spectral_boundary(
        model.surrogate, step_size_coeff; max_components=1
    )

    @test length(contours_from_model) == length(contours_from_surrogate)
    @testset "contour $i" for (i, (from_model, from_surrogate)) in
                              enumerate(zip(contours_from_model, contours_from_surrogate))
        @test from_model.closed == from_surrogate.closed
        @test from_model.internal == from_surrogate.internal
        @test length(from_model.points) == length(from_surrogate.points)
        @test length(from_model.shift_values) == length(from_surrogate.shift_values)
        @test length(from_model.gradients) == length(from_surrogate.gradients)
        @test all(
            abs(z1 - z2) < 1e-10 for
            (z1, z2) in zip(from_model.points, from_surrogate.points)
        )
        @test all(
            abs(s1 - s2) < 1e-10 for
            (s1, s2) in zip(from_model.shift_values, from_surrogate.shift_values)
        )
        @test all(
            abs(g1 - g2) < 1e-10 for
            (g1, g2) in zip(from_model.gradients, from_surrogate.gradients)
        )
    end
end
