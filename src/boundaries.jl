module Boundaries

using ..Defaults: Defaults
using ..Newton: increment
using ..Solutions: issuccess
using ..Spreads: Spreads
using ..Surrogates: SurrogateModel, evaluate

using Statistics: mean

public BoundaryResult, BoundaryTrace, extract_boundaries, extract_boundary, point_in_polygon

# TODO: Derive and implement analytic gradient (implicit differentiation)

"""
    BoundaryResult(points, shift_values, closed, seed)

Container for traced boundary points and their shift values.

  - `points` are ordered along the contour.
  - `shift_values` aligns with `points` and should be near zero.
  - `closed` marks whether the contour loop closed.
  - `seed` stores the (refined) starting point.
"""
struct BoundaryResult{F<:AbstractFloat}
    points::Vector{Complex{F}}
    shift_values::Vector{F}
    closed::Bool
    seed::Complex{F}
end

"""
    BoundaryTrace(result, n_solver_calls, n_failures)

Diagnostic wrapper for a boundary extraction run, including solver counts and failures.
"""
struct BoundaryTrace{F<:AbstractFloat}
    result::BoundaryResult{F}
    n_solver_calls::Int
    n_failures::Int
end

"""
    solve_shift(model, z[, counter]) -> shift

Evaluate the surrogate model at `z` and return the shift value (or `NaN` on failure).
Increments `counter[]` if provided.
"""
function solve_shift(
    model::SurrogateModel{F}, z::Complex{F}, counter::Union{Ref{Int},Nothing}=nothing
) where {F}
    if !isnothing(counter)
        counter[] += 1
    end
    sol = evaluate(model, z; refine=false)
    return issuccess(sol) ? F(Spreads.shift(sol)) : F(NaN)
end

"""
    compute_gradient(model, z[, counter]) -> grad | nothing

Compute the shift gradient at `z` via centered finite differences.
"""
function compute_gradient(
    model::SurrogateModel{F}, z::Complex{F}, counter::Union{Ref{Int},Nothing}=nothing
) where {F}
    hx = increment(real(z))
    hy = increment(imag(z))
    shift_xp = solve_shift(model, z + hx, counter)
    shift_xm = solve_shift(model, z - hx, counter)
    shift_yp = solve_shift(model, z + im * hy, counter)
    shift_ym = solve_shift(model, z - im * hy, counter)
    if !all(isfinite, (shift_xp, shift_xm, shift_yp, shift_ym))
        return nothing
    end

    ds_dx = (shift_xp - shift_xm) / (2 * hx)
    ds_dy = (shift_yp - shift_ym) / (2 * hy)
    grad = complex(ds_dx, ds_dy)
    (abs(grad) < Defaults.atol(F)) && return nothing
    return grad
end

"""
    corrector_step(model, z; tol, max_iters, counter)

Project `z` onto the shift=0 contour via gradient correction.
"""
function corrector_step(
    model::SurrogateModel{F},
    z::Complex{F};
    tol::F=Defaults.atol(F),
    max_iters::Int=10,
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    z_current = Complex{F}(z)
    for _ in 1:max_iters
        shift_val = solve_shift(model, z_current, counter)
        isfinite(shift_val) || return nothing
        if abs(shift_val) < tol
            return z_current, shift_val
        end
        grad = compute_gradient(model, z_current, counter)
        isnothing(grad) && return nothing
        z_current -= shift_val * grad / abs2(grad)
    end
    return nothing
end

"""
    find_seed(model; tol, counter)

Find a boundary seed by locating the surrogate sample point whose shift is closest to zero
and refining it with a corrector step.
"""
function find_seed(
    model::SurrogateModel{F};
    tol::F=Defaults.atol(F),
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    xs = model.rbf.x
    isempty(xs) && return nothing

    candidates = map(xs) do xv
        z = complex(xv[1], xv[2])
        s = solve_shift(model, z, counter)
        return (z, abs(s))
    end
    filter!(isfinite ∘ last, candidates)
    isempty(candidates) && return nothing
    sort!(candidates; by=last)

    for (z, _) in candidates
        refined = corrector_step(model, z; tol, counter)
        isnothing(refined) && continue
        return refined[1]
    end

    return nothing
end

"""
    predictor_corrector_step(model, z, direction, step_size; tol, counter)

Take one step along the shift=0 contour: predict along `direction`, correct onto the
contour, and update the tangent direction. Returns `(new_z, shift_val, new_direction)` on
success, `nothing` on failure.
"""
function predictor_corrector_step(
    model::SurrogateModel{F},
    z::Complex{F},
    direction::Complex{F},
    step_size::F;
    tol::F=Defaults.atol(F),
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    z_pred = z + step_size * direction
    corrected = corrector_step(model, z_pred; tol, counter)
    isnothing(corrected) && return nothing

    new_z, shift_val = corrected
    grad = compute_gradient(model, new_z, counter)
    isnothing(grad) && return nothing

    new_direction = tangent_from_gradient(grad, new_z)
    if real(direction * conj(new_direction)) < 0
        new_direction = -new_direction
    end
    return (new_z, shift_val, new_direction)
end

"""
    tangent_from_gradient(grad, z) -> direction

Return the counter-clockwise tangent direction inferred from the shift gradient.
"""
function tangent_from_gradient(grad::Complex{F}, z::Complex{F}) where {F}
    direction = cis(angle(grad) + pi / 2)
    if imag(conj(z) * direction) < 0
        direction = -direction
    end
    return direction
end

function try_close!(
    points::Vector{Complex{F}},
    shift_values::Vector{F},
    model::SurrogateModel{F},
    current_z::Complex{F},
    direction::Complex{F},
    initial_direction::Complex{F},
    step_size::F;
    tol::F,
    counter,
) where {F}
    (abs(current_z - points[1]) < step_size) || return false
    (real(direction * conj(initial_direction)) > 0) || return false
    refined = corrector_step(model, points[1]; tol, max_iters=5, counter)
    isnothing(refined) && return false
    final_z, final_shift = refined
    points[1] = final_z
    shift_values[1] = final_shift
    push!(points, final_z)
    push!(shift_values, final_shift)
    return true
end

"""
    extract_boundary(model; seed, step_size, max_steps, tol, max_failures,
        min_steps_before_close) -> BoundaryTrace

Trace the shift=0 contour using a predictor-corrector loop and return diagnostics.
"""
function extract_boundary(
    model::SurrogateModel{F};
    seed::Union{Complex,Nothing}=nothing,
    step_size::Real=mean(model.rbf.nn_dists) / 2,
    tol::Real=Defaults.atol(F),
    max_steps::Int=500,
    max_failures::Int=5,
    min_steps_before_close::Int=12,
) where {F}
    step_size = F(step_size)
    tol = F(tol)
    if step_size <= 0
        throw(ArgumentError("step_size must be positive"))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive"))
    end

    # Initialize diagnostics and counters.
    counter = Ref(0)
    n_failures = 0
    consecutive_failures = 0
    seed_z = isnothing(seed) ? find_seed(model; tol, counter) : Complex{F}(seed)
    isnothing(seed_z) && return trivial_trace(zero(Complex{F}), counter, n_failures)

    # Reject unsolvable seeds.
    seed_shift = solve_shift(model, seed_z, counter)
    isfinite(seed_shift) || return trivial_trace(Complex{F}(seed_z), counter, n_failures)

    # Refine the seed to land on the contour.
    refined = corrector_step(model, seed_z; tol, counter)
    isnothing(refined) && return trivial_trace(Complex{F}(seed_z), counter, n_failures)

    seed_z, seed_shift = refined
    points = Complex{F}[seed_z]
    shift_values = F[seed_shift]
    closed = false

    # Compute initial tangent direction from the seed gradient.
    grad = compute_gradient(model, seed_z, counter)
    if isnothing(grad)
        n_failures += 1
        return trace(points, shift_values, closed, seed_z, counter, n_failures)
    end

    direction = tangent_from_gradient(grad, seed_z)
    initial_direction = direction

    current_z = seed_z
    for step_index in 1:max_steps
        stepped = predictor_corrector_step(
            model, current_z, direction, step_size; tol, counter
        )
        if isnothing(stepped)
            n_failures += 1
            consecutive_failures += 1
            (consecutive_failures > max_failures) && break
            continue
        end

        consecutive_failures = 0
        current_z, current_shift, direction = stepped
        push!(points, current_z)
        push!(shift_values, current_shift)

        if (step_index > min_steps_before_close) && try_close!(
            points,
            shift_values,
            model,
            current_z,
            direction,
            initial_direction,
            step_size;
            tol,
            counter,
        )
            closed = true
            break
        end
    end
    return trace(points, shift_values, closed, seed_z, counter, n_failures)
end

"""
    point_in_polygon(z, polygon) -> Bool

Return `true` if `z` is inside the closed polygon defined by `polygon`.
"""
function point_in_polygon(
    z::Complex{F}, polygon::AbstractVector{Complex{F}}
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
    (abs(area) <= Defaults.atol(F)) && return false

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
    is_inside_any(z, contours) -> Bool

Return `true` if `z` is inside any closed contour in `contours`.
"""
function is_inside_any(
    z::Complex{F}, contours::AbstractVector{BoundaryResult{F}}
) where {F<:AbstractFloat}
    for contour in contours
        contour.closed || continue
        point_in_polygon(z, contour.points) && return true
    end
    return false
end

"""
    is_near_contour(z, contour, threshold) -> Bool

Return `true` if `z` is within `threshold` of any point in `contour`.
"""
function is_near_contour(
    z::Complex{F}, contour::BoundaryResult{F}, threshold::F
) where {F<:AbstractFloat}
    isempty(contour.points) && return false
    for p in contour.points
        if abs(z - p) < threshold
            return true
        end
    end
    return false
end

"""
    is_near_any_contour(z, contours, threshold) -> Bool

Return `true` if `z` is near any closed contour in `contours`.
"""
function is_near_any_contour(
    z::Complex{F}, contours::AbstractVector{BoundaryResult{F}}, threshold::F
) where {F<:AbstractFloat}
    for contour in contours
        contour.closed || continue
        is_near_contour(z, contour, threshold) && return true
    end
    return false
end

"""
    find_exterior_seed(model, contours; step_size, tol, counter) -> seed | nothing

Find a boundary seed outside all existing closed contours.
"""
function find_exterior_seed(
    model::SurrogateModel{F},
    contours::AbstractVector{BoundaryResult{F}};
    step_size::F,
    tol::F=Defaults.atol(F),
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    xs = model.rbf.x
    isempty(xs) && return nothing

    candidates = Tuple{Complex{F},F}[]
    for xv in xs
        z = complex(xv[1], xv[2])
        is_inside_any(z, contours) && continue
        shift_val = solve_shift(model, z, counter)
        isfinite(shift_val) || continue
        push!(candidates, (z, shift_val))
    end
    isempty(candidates) && return nothing

    primary = filter(candidate -> candidate[2] > 0, candidates)
    sort!(primary; by=last)
    for (z, _) in primary
        refined = corrector_step(model, z; tol, counter)
        isnothing(refined) && continue
        is_near_any_contour(refined[1], contours, step_size) && continue
        return refined[1]
    end

    fallback = filter(candidate -> candidate[2] <= 0, candidates)
    sort!(fallback; by=candidate -> abs(candidate[2]))
    for (z, _) in fallback
        refined = corrector_step(model, z; tol, counter)
        isnothing(refined) && continue
        is_near_any_contour(refined[1], contours, step_size) && continue
        return refined[1]
    end

    return nothing
end

"""
    extract_boundaries(model; kwargs...) -> Vector{BoundaryTrace}

Trace all disconnected boundary components.
"""
function extract_boundaries(
    model::SurrogateModel{F};
    step_size::Real=mean(model.rbf.nn_dists) / 2,
    tol::Real=Defaults.atol(F),
    max_steps::Int=500,
    max_failures::Int=5,
    min_steps_before_close::Int=12,
    max_components::Int=10,
) where {F}
    step_size = F(step_size)
    tol = F(tol)
    if step_size <= 0
        throw(ArgumentError("step_size must be positive"))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive"))
    end
    if max_components <= 0
        throw(ArgumentError("max_components must be positive"))
    end

    traces = BoundaryTrace{F}[]
    contours = BoundaryResult{F}[]

    seed = find_seed(model; tol)
    isnothing(seed) && return traces

    trace = extract_boundary(
        model; seed, step_size, tol, max_steps, max_failures, min_steps_before_close
    )
    push!(traces, trace)
    if trace.result.closed
        push!(contours, trace.result)
    else
        return traces
    end

    while length(traces) < max_components
        seed = find_exterior_seed(model, contours; step_size, tol)
        isnothing(seed) && break
        trace = extract_boundary(
            model; seed, step_size, tol, max_steps, max_failures, min_steps_before_close
        )
        push!(traces, trace)
        if trace.result.closed
            push!(contours, trace.result)
        else
            break
        end
    end

    return traces
end

function trivial_trace(seed_z::Complex{F}, counter, n_failures) where {F}
    return trace(Complex{F}[], F[], false, seed_z, counter, n_failures)
end

function trace(points, shift_values, closed, seed_z, counter, n_failures)
    result = BoundaryResult(points, shift_values, closed, seed_z)
    return BoundaryTrace(result, counter[], n_failures)
end

end  # module Boundaries

@testitem "BoundaryResult construction" begin
    using GaussianSpectralDensities.Boundaries: BoundaryResult

    @testset "closed = true" begin
        pts = ComplexF64[1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 1.0 + 0.0im]
        shifts = [0.0, 0.0, 0.0, 0.0]
        seed = 1.0 + 0.0im

        result = BoundaryResult(pts, shifts, true, seed)
        @test result.points === pts
        @test result.shift_values === shifts
        @test result.closed === true
        @test result.seed === seed
    end

    @testset "closed = false" begin
        pts = ComplexF64[1.0 + 0.0im, 0.0 + 1.0im]
        shifts = [0.01, -0.02]
        seed = 1.0 + 0.0im

        result = BoundaryResult(pts, shifts, false, seed)
        @test result.points === pts
        @test result.shift_values === shifts
        @test result.closed === false
        @test result.seed === seed
    end

    @testset "empty points" begin
        pts = ComplexF64[]
        shifts = Float64[]
        seed = 0.0 + 0.0im

        result = BoundaryResult(pts, shifts, false, seed)
        @test isempty(result.points)
        @test isempty(result.shift_values)
        @test result.closed === false
    end
end

@testitem "BoundaryTrace construction" begin
    using GaussianSpectralDensities.Boundaries: BoundaryResult, BoundaryTrace

    pts = ComplexF64[1.0 + 0.0im, 0.0 + 1.0im, 1.0 + 0.0im]
    shifts = [0.0, 0.0, 0.0]
    seed = 1.0 + 0.0im
    result = BoundaryResult(pts, shifts, true, seed)

    trace = BoundaryTrace(result, 150, 2)
    @test trace.result === result
    @test trace.n_solver_calls == 150
    @test trace.n_failures == 2
end

@testitem "solve_shift inside support" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: solve_shift
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # At right endpoint (1.25, 0), gradient should point outward (positive real direction)
    grad = compute_gradient(model, complex(1.25))
    @test !isnothing(grad)
    @test real(grad) < 0  # shift decreases outward, so gradient points outward (negative)

    # At top of ellipse (0, 0.75), gradient should point outward (positive imaginary direction)
    grad = compute_gradient(model, complex(0.0, 0.75))
    @test !isnothing(grad)
    @test imag(grad) < 0  # shift decreases outward, so gradient points in -y direction
end

@testitem "compute_gradient inside support" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: compute_gradient
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: corrector_step
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: corrector_step
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
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

@testitem "is_near_contour" begin
    using GaussianSpectralDensities.Boundaries: BoundaryResult, is_near_contour

    @testset "$F" for F in [Float32, Float64]
        # Contour with a few known points
        pts = Complex{F}[1, im, -1, -im, 1]
        shifts = zeros(F, 5)
        contour = BoundaryResult(pts, shifts, true, Complex{F}(1))

        # Point exactly on a contour point: distance = 0 < any positive threshold
        @test is_near_contour(Complex{F}(1), contour, F(0.1))

        # Point very close to a contour point
        @test is_near_contour(Complex{F}(1.05), contour, F(0.1))

        # Point far from all contour points
        @test !is_near_contour(Complex{F}(5.0 + 5.0im), contour, F(0.1))

        # Large threshold makes everything near
        @test is_near_contour(Complex{F}(2.0), contour, F(2.0))

        # Empty contour points
        empty_contour = BoundaryResult(Complex{F}[], F[], true, Complex{F}(0))
        @test !is_near_contour(Complex{F}(0), empty_contour, F(1.0))
    end
end

@testitem "is_near_contour threshold boundary" begin
    using GaussianSpectralDensities.Boundaries: BoundaryResult, is_near_contour

    @testset "$F" for F in [Float32, Float64]
        pts = Complex{F}[0]
        shifts = zeros(F, 1)
        contour = BoundaryResult(pts, shifts, true, Complex{F}(0))

        # Distance is exactly 0.5; threshold is 1.0 → near
        @test is_near_contour(Complex{F}(0.5), contour, F(1.0))

        # Distance is 0.5; threshold is 0.25 → not near
        @test !is_near_contour(Complex{F}(0.5), contour, F(0.25))
    end
end

@testitem "is_near_any_contour" begin
    using GaussianSpectralDensities.Boundaries: BoundaryResult, is_near_any_contour

    @testset "$F" for F in [Float32, Float64]
        pts1 = Complex{F}[1, im, -1, -im, 1]
        contour1 = BoundaryResult(pts1, zeros(F, 5), true, Complex{F}(1))

        pts2 = Complex{F}[5, 5 + im, 5 + 2im]
        contour2 = BoundaryResult(pts2, zeros(F, 3), false, Complex{F}(5))

        pts3 = Complex{F}[10, 10 + im, 10 + 2im, 10]
        contour3 = BoundaryResult(pts3, zeros(F, 4), true, Complex{F}(10))

        contours = BoundaryResult{F}[contour1, contour2, contour3]
        threshold = F(0.5)

        # Near contour1
        @test is_near_any_contour(Complex{F}(1.1), contours, threshold)

        # Near contour3
        @test is_near_any_contour(Complex{F}(10.1), contours, threshold)

        # Not near any closed contour (contour2 is open, so skipped)
        @test !is_near_any_contour(Complex{F}(5.1), contours, threshold)

        # Far from everything
        @test !is_near_any_contour(Complex{F}(50.0), contours, threshold)

        # Empty contours
        @test !is_near_any_contour(Complex{F}(1.0), BoundaryResult{F}[], threshold)
    end
end

@testitem "extract_boundaries single component" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: BoundaryTrace, extract_boundaries
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    traces = extract_boundaries(model)
    @test traces isa Vector{BoundaryTrace{Float64}}
    @test length(traces) == 1

    result = traces[1].result
    @test result.closed
    @test length(result.points) > 10

    # All shift values should be near zero
    @test all(s -> abs(s) < 1e-3, result.shift_values)

    # Each boundary point should lie approximately on the ellipse
    for z in result.points
        ellipse_val = (real(z) / 1.25)^2 + (imag(z) / 0.75)^2
        @test abs(ellipse_val - 1.0) < 0.1
    end
end

@testitem "extract_boundaries max_components limit" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundaries
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    traces = extract_boundaries(model; max_components=1)
    @test length(traces) <= 1
end

@testitem "extract_boundaries argument validation" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundaries
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    @test_throws ArgumentError extract_boundaries(model; step_size=0.0)
    @test_throws ArgumentError extract_boundaries(model; step_size=-0.1)
    @test_throws ArgumentError extract_boundaries(model; tol=0.0)
    @test_throws ArgumentError extract_boundaries(model; tol=-1e-6)
    @test_throws ArgumentError extract_boundaries(model; max_components=0)
    @test_throws ArgumentError extract_boundaries(model; max_components=-1)
end

@testitem "extract_boundaries keyword forwarding" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundaries
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # max_steps=2 should produce an open contour, which stops the loop
    traces = extract_boundaries(model; max_steps=2)
    @test length(traces) == 1
    @test !traces[1].result.closed
end

@testitem "extract_boundaries stops on open contour" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundaries
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Force open contour via very few max_steps; loop should stop after first trace
    traces = extract_boundaries(model; max_steps=3, max_components=10)
    @test length(traces) == 1
    @test !traces[1].result.closed
end
