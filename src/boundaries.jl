module Boundaries

using ..Solutions: issuccess
using ..Spreads: Spreads
using ..Surrogates: SurrogateModel, evaluate

public BoundaryResult, BoundaryTrace, extract_boundary

# TODO:
# - Fix Float64 -> F
# - Use tolerance from Defaults
# - Simplify radial probing using model sample points
# - Derive and implement analytic gradient (implicit differentiation)
# - Simplify interfaces. Do we need a solver type?

"""
    BoundaryResult(points, shift_values, closed, seed)

Container for traced boundary points and their shift values.

  - `points` are ordered along the contour.
  - `shift_values` aligns with `points` and should be near zero.
  - `closed` marks whether the contour loop closed.
  - `seed` stores the (refined) starting point.
"""
struct BoundaryResult
    points::Vector{ComplexF64}
    shift_values::Vector{Float64}
    closed::Bool
    seed::ComplexF64
end

"""
    BoundaryTrace(result, n_solver_calls, n_corrector_failures, n_gradient_failures)

Diagnostic wrapper for a boundary extraction run, including solver counts and failures.
"""
struct BoundaryTrace
    result::BoundaryResult
    n_solver_calls::Int
    n_corrector_failures::Int
    n_gradient_failures::Int
end

"""
    solve_shift(model, z) -> (success, shift)

Evaluate the surrogate model at `z` and return whether the solve succeeded along with the
shift value (or `NaN` on failure).
"""
function solve_shift(model::SurrogateModel, z::ComplexF64)
    sol = evaluate(model, z; refine=false)
    if issuccess(sol)
        return true, Float64(Spreads.shift(sol))
    end
    return false, NaN
end

"""
    compute_gradient_with(model, z, fd_step, solve_shift_fn) -> grad | nothing

Centered finite-difference gradient helper that delegates shift evaluations to the supplied
`solve_shift_fn`.
"""
function compute_gradient_with(
    model::SurrogateModel, z::ComplexF64, fd_step::Float64, solve_shift_fn
)
    success_xp, shift_xp = solve_shift_fn(model, z + fd_step)
    success_xm, shift_xm = solve_shift_fn(model, z - fd_step)
    success_yp, shift_yp = solve_shift_fn(model, z + im * fd_step)
    success_ym, shift_ym = solve_shift_fn(model, z - im * fd_step)
    if !(success_xp && success_xm && success_yp && success_ym)
        return nothing
    end

    ds_dx = (shift_xp - shift_xm) / (2 * fd_step)
    ds_dy = (shift_yp - shift_ym) / (2 * fd_step)
    grad = complex(ds_dx, ds_dy)
    if abs(grad) < 1e-10
        return nothing
    end
    return grad
end

"""
    compute_gradient(model, z, fd_step) -> grad | nothing

Compute the shift gradient at `z` via centered finite differences.
"""
function compute_gradient(model::SurrogateModel, z::ComplexF64, fd_step::Float64)
    return compute_gradient_with(model, z, fd_step, solve_shift)
end

"""
    corrector_step_with(model, z; tol, fd_step, max_iters, solve_shift_fn)

Newton corrector that projects `z` onto the shift=0 contour using the supplied
`solve_shift_fn` for shift evaluations.
"""
function corrector_step_with(
    model::SurrogateModel,
    z::ComplexF64;
    tol::Float64=1e-6,
    fd_step::Float64=0.01,
    max_iters::Int=10,
    solve_shift_fn=solve_shift,
)
    z_current = ComplexF64(z)
    for _ in 1:max_iters
        success, shift_val = solve_shift_fn(model, z_current)
        success || return nothing
        if abs(shift_val) < tol
            return z_current, shift_val
        end
        grad = compute_gradient_with(model, z_current, fd_step, solve_shift_fn)
        grad === nothing && return nothing
        z_current -= shift_val * grad / abs2(grad)
    end
    return nothing
end

"""
    corrector_step(model, z; tol, fd_step, max_iters)

Newton corrector that projects `z` onto the shift=0 contour.
"""
function corrector_step(
    model::SurrogateModel,
    z::ComplexF64;
    tol::Float64=1e-6,
    fd_step::Float64=0.01,
    max_iters::Int=10,
)
    return corrector_step_with(
        model, z; tol=tol, fd_step=fd_step, max_iters=max_iters, solve_shift_fn=solve_shift
    )
end

"""
    find_seed_with(model; center, n_directions, max_radius, radial_step, tol,
        fd_step, max_corrector_iters, solve_shift_fn)

Search radially from `center` for a sign change in the shift and refine it with a
corrector.
"""
function find_seed_with(
    model::SurrogateModel;
    center::ComplexF64=complex(0.0),
    n_directions::Int=8,
    max_radius::Float64=3.0,
    radial_step::Float64=0.1,
    tol::Float64=1e-6,
    fd_step::Float64=0.01,
    max_corrector_iters::Int=10,
    solve_shift_fn=solve_shift,
)
    # Evaluate the center point and ensure we're inside or on the boundary.
    center_z = ComplexF64(center)
    success, shift_center = solve_shift_fn(model, center_z)
    success || return nothing
    if shift_center < 0
        return nothing
    elseif abs(shift_center) < tol
        return center_z
    end

    # TODO: consider factoring out the radial scan and bisection into a helper.
    for k in 0:(n_directions - 1)
        # Probe outward along a radial ray.
        theta = 2 * pi * k / n_directions
        direction = cis(theta)
        z_inside = center_z
        s_inside = shift_center
        found = false
        z_outside = center_z
        s_outside = shift_center

        for r in radial_step:radial_step:max_radius
            z_trial = center_z + r * direction
            success, s_trial = solve_shift_fn(model, z_trial)
            success || break
            if s_trial < 0
                z_outside = z_trial
                s_outside = s_trial
                found = true
                break
            end
            z_inside = z_trial
            s_inside = s_trial
        end

        found || continue

        # Bisect to bracket the zero crossing.
        left_z = z_inside
        left_s = s_inside
        right_z = z_outside
        right_s = s_outside
        for _ in 1:40
            mid_z = (left_z + right_z) / 2
            success, mid_s = solve_shift_fn(model, mid_z)
            success || break
            if abs(mid_s) < tol
                left_z = mid_z
                left_s = mid_s
                right_z = mid_z
                right_s = mid_s
                break
            elseif mid_s > 0
                left_z = mid_z
                left_s = mid_s
            else
                right_z = mid_z
                right_s = mid_s
            end
        end

        # Snap the midpoint to the contour via Newton correction.
        candidate = (left_z + right_z) / 2
        refined = corrector_step_with(
            model,
            candidate;
            tol=tol,
            fd_step=fd_step,
            max_iters=max_corrector_iters,
            solve_shift_fn=solve_shift_fn,
        )
        refined === nothing && continue
        return refined[1]
    end

    return nothing
end

"""
    find_seed(model; center, n_directions, max_radius, radial_step, tol,
        fd_step, max_corrector_iters)

Find a boundary seed using `find_seed_with` and the default shift solver.
"""
function find_seed(
    model::SurrogateModel;
    center::ComplexF64=complex(0.0),
    n_directions::Int=8,
    max_radius::Float64=3.0,
    radial_step::Float64=0.1,
    tol::Float64=1e-6,
    fd_step::Float64=0.01,
    max_corrector_iters::Int=10,
)
    return find_seed_with(
        model;
        center=center,
        n_directions=n_directions,
        max_radius=max_radius,
        radial_step=radial_step,
        tol=tol,
        fd_step=fd_step,
        max_corrector_iters=max_corrector_iters,
        solve_shift_fn=solve_shift,
    )
end

"""
    tangent_from_gradient(grad, z) -> direction

Return the counter-clockwise tangent direction inferred from the shift gradient.
"""
function tangent_from_gradient(grad::ComplexF64, z::ComplexF64)
    direction = cis(angle(grad) + pi / 2)
    if imag(conj(z) * direction) < 0
        direction = -direction
    end
    return direction
end

"""
    extract_boundary(model; seed, step_size, max_steps, tol, fd_step,
        max_corrector_iters, max_corrector_failures, max_gradient_failures,
        min_steps_before_close) -> BoundaryTrace

Trace the shift=0 contour using a predictor-corrector loop and return diagnostics.
"""
function extract_boundary(
    model::SurrogateModel;
    seed::Union{ComplexF64,Nothing}=nothing,
    step_size::Float64=0.05,
    max_steps::Int=500,
    tol::Float64=1e-6,
    fd_step::Float64=0.01,
    max_corrector_iters::Int=10,
    max_corrector_failures::Int=5,
    max_gradient_failures::Int=10,
    min_steps_before_close::Int=20,
)
    if step_size <= 0
        throw(ArgumentError("step_size must be positive"))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive"))
    end
    if fd_step <= 0
        throw(ArgumentError("fd_step must be positive"))
    end

    # Initialize diagnostics and counters.
    n_solver_calls = Ref(0)
    n_corrector_failures = 0
    n_gradient_failures = 0
    consecutive_corrector_failures = 0
    consecutive_gradient_failures = 0

    function solve_shift_counted(model, z)
        n_solver_calls[] += 1
        return solve_shift(model, z)
    end

    function corrector_step_counted(
        z; tol_override=tol, max_iters_override=max_corrector_iters
    )
        return corrector_step_with(
            model,
            z;
            tol=tol_override,
            fd_step=fd_step,
            max_iters=max_iters_override,
            solve_shift_fn=solve_shift_counted,
        )
    end

    function compute_gradient_counted(z)
        return compute_gradient_with(model, z, fd_step, solve_shift_counted)
    end

    # TODO: consider factoring out the seed resolution and validation block.
    seed_z = if seed === nothing
        find_seed_with(
            model;
            tol=tol,
            fd_step=fd_step,
            max_corrector_iters=max_corrector_iters,
            solve_shift_fn=solve_shift_counted,
        )
    else
        ComplexF64(seed)
    end

    seed_z === nothing && return BoundaryTrace(
        BoundaryResult(ComplexF64[], Float64[], false, complex(0.0)),
        n_solver_calls[],
        n_corrector_failures,
        n_gradient_failures,
    )

    # Reject seeds that are clearly outside the support.
    seed_success, seed_shift = solve_shift_counted(model, seed_z)
    seed_success || return BoundaryTrace(
        BoundaryResult(ComplexF64[], Float64[], false, ComplexF64(seed_z)),
        n_solver_calls[],
        n_corrector_failures,
        n_gradient_failures,
    )

    if seed_shift < 0 && abs(seed_shift) > tol
        return BoundaryTrace(
            BoundaryResult(ComplexF64[], Float64[], false, ComplexF64(seed_z)),
            n_solver_calls[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    # Refine the seed to land on the contour.
    refined = corrector_step_counted(seed_z)
    if refined === nothing
        return BoundaryTrace(
            BoundaryResult(ComplexF64[], Float64[], false, ComplexF64(seed_z)),
            n_solver_calls[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    seed_z, seed_shift = refined
    points = ComplexF64[seed_z]
    shift_values = Float64[seed_shift]

    # Compute initial tangent direction from the seed gradient.
    grad = compute_gradient_counted(seed_z)
    if grad === nothing
        n_gradient_failures += 1
        return BoundaryTrace(
            BoundaryResult(points, shift_values, false, seed_z),
            n_solver_calls[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    direction = tangent_from_gradient(grad, seed_z)
    initial_direction = direction

    # TODO: consider factoring out the predictor-corrector step into a helper.
    closed = false
    current_z = seed_z
    for step_index in 1:max_steps
        # Predictor step along the tangent.
        z_pred = current_z + step_size * direction
        corrected = corrector_step_counted(z_pred)
        if corrected === nothing
            n_corrector_failures += 1
            consecutive_corrector_failures += 1
            if consecutive_corrector_failures > max_corrector_failures
                break
            end
            continue
        end

        consecutive_corrector_failures = 0
        current_z, current_shift = corrected
        push!(points, current_z)
        push!(shift_values, current_shift)

        # Update the tangent using the local gradient.
        grad = compute_gradient_counted(current_z)
        if grad === nothing
            n_gradient_failures += 1
            consecutive_gradient_failures += 1
            if consecutive_gradient_failures > max_gradient_failures
                break
            end
            continue
        end

        consecutive_gradient_failures = 0
        new_direction = tangent_from_gradient(grad, current_z)
        if real(direction * conj(new_direction)) < 0
            new_direction = -new_direction
        end
        direction = new_direction

        # TODO: consider factoring out the closure check into a helper.
        if step_index > min_steps_before_close
            close_enough = abs(current_z - points[1]) < step_size
            heading_back = real(direction * conj(initial_direction)) > 0
            if close_enough && heading_back
                final_refine = corrector_step_counted(
                    points[1]; tol_override=tol, max_iters_override=5
                )
                if final_refine !== nothing
                    final_z, final_shift = final_refine
                    points[1] = final_z
                    shift_values[1] = final_shift
                    push!(points, final_z)
                    push!(shift_values, final_shift)
                    closed = true
                    break
                end
            end
        end
    end

    result = BoundaryResult(points, shift_values, closed, seed_z)
    return BoundaryTrace(
        result, n_solver_calls[], n_corrector_failures, n_gradient_failures
    )
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

    trace = BoundaryTrace(result, 150, 2, 1)
    @test trace.result === result
    @test trace.n_solver_calls == 150
    @test trace.n_corrector_failures == 2
    @test trace.n_gradient_failures == 1
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
    success, s = solve_shift(model, complex(0.0))
    @test success
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
    success, s = solve_shift(model, complex(2.0))
    @test success
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
    success, s = solve_shift(model, complex(1.25))
    @test success
    @test abs(s) < 1e-4

    # Top of ellipse: (0, 0.75)
    success, s = solve_shift(model, complex(0.0, 0.75))
    @test success
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
    success, s = solve_shift(model, complex(3.0))
    @test !success
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
    grad = compute_gradient(model, complex(1.25), 0.01)
    @test grad !== nothing
    @test real(grad) < 0  # shift decreases outward, so gradient points outward (negative)

    # At top of ellipse (0, 0.75), gradient should point outward (positive imaginary direction)
    grad = compute_gradient(model, complex(0.0, 0.75), 0.01)
    @test grad !== nothing
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
    grad = compute_gradient(model, complex(0.5, 0.3), 0.01)
    @test grad !== nothing
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
    grad = compute_gradient(model, complex(3.0), 0.01)
    @test grad === nothing
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
    result = corrector_step(model, complex(1.2); tol=1e-6, fd_step=0.01)
    @test result !== nothing
    z_corrected, s_corrected = result
    @test abs(s_corrected) < 1e-6
    # The corrected point should be near the ellipse boundary
    @test abs(real(z_corrected)) > 1.0  # still near the right side

    # Start near the boundary (slightly outside)
    result = corrector_step(model, complex(1.3); tol=1e-6, fd_step=0.01)
    @test result !== nothing
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
    result = corrector_step(model, complex(3.0); tol=1e-12, fd_step=0.01, max_iters=1)
    @test result === nothing
end

@testitem "find_seed from inside support" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: find_seed, solve_shift
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Default center is the origin, which is inside the support
    seed = find_seed(model)
    @test seed !== nothing
    # The seed should be approximately on the boundary (shift ≈ 0)
    success, s = solve_shift(model, seed)
    @test success
    @test abs(s) < 1e-4
end

@testitem "find_seed from outside support" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: find_seed
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Center outside the support — should return nothing
    seed = find_seed(model; center=complex(5.0))
    @test seed === nothing
end

@testitem "find_seed no sign change" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: find_seed
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Search radius too small to find a sign change
    seed = find_seed(
        model; center=complex(0.0), max_radius=0.1, radial_step=0.1, n_directions=4
    )
    @test seed === nothing
end

@testitem "extract_boundary with explicit seed" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: BoundaryTrace, extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Provide a point on the known ellipse as seed (right endpoint)
    trace = extract_boundary(model; seed=complex(1.25))
    @test trace isa BoundaryTrace
    result = trace.result
    @test result.closed
    @test length(result.points) > 10

    # All shift values should be near zero
    @test all(s -> abs(s) < 1e-3, result.shift_values)

    # The seed field should record the (refined) seed
    @test abs(result.seed - complex(1.25)) < 0.1

    # Each boundary point should lie approximately on the ellipse: (x/1.25)² + (y/0.75)² ≈ 1
    for z in result.points
        ellipse_val = (real(z) / 1.25)^2 + (imag(z) / 0.75)^2
        @test abs(ellipse_val - 1.0) < 0.1
    end
end

@testitem "extract_boundary with automatic seed" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: BoundaryTrace, extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # No seed provided — should find one automatically
    trace = extract_boundary(model)
    @test trace isa BoundaryTrace
    result = trace.result
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

@testitem "extract_boundary argument validation" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    @test_throws ArgumentError extract_boundary(model; step_size=0.0)
    @test_throws ArgumentError extract_boundary(model; step_size=-0.1)
    @test_throws ArgumentError extract_boundary(model; tol=0.0)
    @test_throws ArgumentError extract_boundary(model; tol=-1e-6)
    @test_throws ArgumentError extract_boundary(model; fd_step=0.0)
    @test_throws ArgumentError extract_boundary(model; fd_step=-0.01)
end

@testitem "extract_boundary max steps" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    trace = extract_boundary(model; seed=complex(1.25), max_steps=2)
    @test !trace.result.closed
end

@testitem "BoundaryTrace diagnostics" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: BoundaryTrace, extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    trace = extract_boundary(model)

    @test trace.n_solver_calls > 0
    @test trace.n_corrector_failures >= 0
    @test trace.n_gradient_failures >= 0
end

@testitem "boundary geometry validation" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    trace = extract_boundary(model)
    result = trace.result
    @test result.closed
    pts = result.points

    # All points lie approximately on the ellipse
    @testset "point $i on ellipse" for i in eachindex(pts)
        z = pts[i]
        ellipse_val = (real(z) / 1.25)^2 + (imag(z) / 0.75)^2
        @test abs(ellipse_val - 1.0) < 0.1
    end

    # Boundary encloses the origin: compute winding number via sum of angle increments
    # For a closed contour enclosing the origin, winding number = ±1
    let winding_angle = 0.0
        for i in 1:(length(pts) - 1)
            dtheta = angle(pts[i + 1]) - angle(pts[i])
            # Wrap to [-π, π]
            dtheta = mod(dtheta + π, 2π) - π
            winding_angle += dtheta
        end
        winding_number = round(Int, winding_angle / (2π))
        @test abs(winding_number) == 1

        # Counter-clockwise traversal: winding number should be +1
        @test winding_number == 1
    end
end

@testitem "extract_boundary no seed found" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Boundaries: extract_boundary
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel

    using LinearAlgebra: I

    # Use the same ensemble but start seed finding from far outside
    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Provide a seed far outside the support — corrector refinement should fail
    # and extract_boundary should return empty result with closed=false
    trace = extract_boundary(model; seed=complex(10.0))
    result = trace.result
    @test !result.closed
end
