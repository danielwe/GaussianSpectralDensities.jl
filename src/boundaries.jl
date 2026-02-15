module Boundaries

using ..Defaults: Defaults
using ..Solutions: issuccess
using ..Spreads: Spreads
using ..Surrogates: SurrogateModel, evaluate

public BoundaryResult, BoundaryTrace, extract_boundary

# TODO:
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
struct BoundaryResult{F<:AbstractFloat}
    points::Vector{Complex{F}}
    shift_values::Vector{F}
    closed::Bool
    seed::Complex{F}
end

"""
    BoundaryTrace(result, n_solver_calls, n_corrector_failures, n_gradient_failures)

Diagnostic wrapper for a boundary extraction run, including solver counts and failures.
"""
struct BoundaryTrace{F<:AbstractFloat}
    result::BoundaryResult{F}
    n_solver_calls::Int
    n_corrector_failures::Int
    n_gradient_failures::Int
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
    compute_gradient(model, z, fd_step[, counter]) -> grad | nothing

Compute the shift gradient at `z` via centered finite differences.
"""
function compute_gradient(
    model::SurrogateModel{F},
    z::Complex{F},
    fd_step::F,
    counter::Union{Ref{Int},Nothing}=nothing,
) where {F}
    shift_xp = solve_shift(model, z + fd_step, counter)
    shift_xm = solve_shift(model, z - fd_step, counter)
    shift_yp = solve_shift(model, z + im * fd_step, counter)
    shift_ym = solve_shift(model, z - im * fd_step, counter)
    if !all(isfinite, (shift_xp, shift_xm, shift_yp, shift_ym))
        return nothing
    end

    ds_dx = (shift_xp - shift_xm) / (2 * fd_step)
    ds_dy = (shift_yp - shift_ym) / (2 * fd_step)
    grad = complex(ds_dx, ds_dy)
    if abs(grad) < Defaults.atol(F)
        return nothing
    end
    return grad
end

"""
    corrector_step(model, z; tol, fd_step, max_iters, counter)

Project `z` onto the shift=0 contour via gradient correction.
"""
function corrector_step(
    model::SurrogateModel{F},
    z::Complex{F};
    tol::F=Defaults.atol(F),
    fd_step::F=F(0.01),
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
        grad = compute_gradient(model, z_current, fd_step, counter)
        isnothing(grad) && return nothing
        z_current -= shift_val * grad / abs2(grad)
    end
    return nothing
end

"""
    find_seed(model; tol, fd_step, max_corrector_iters, counter)

Find a boundary seed by locating the surrogate sample point whose shift is closest to zero
and refining it with a corrector step.
"""
function find_seed(
    model::SurrogateModel{F};
    tol::F=Defaults.atol(F),
    fd_step::F=F(0.01),
    max_corrector_iters::Int=10,
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
        refined = corrector_step(
            model,
            z;
            tol=tol,
            fd_step=fd_step,
            max_iters=max_corrector_iters,
            counter=counter,
        )
        isnothing(refined) && continue
        return refined[1]
    end

    return nothing
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

"""
    extract_boundary(model; seed, step_size, max_steps, tol, fd_step,
        max_corrector_iters, max_corrector_failures, max_gradient_failures,
        min_steps_before_close) -> BoundaryTrace

Trace the shift=0 contour using a predictor-corrector loop and return diagnostics.
"""
function extract_boundary(
    model::SurrogateModel{F};
    seed::Union{Complex,Nothing}=nothing,
    step_size::Real=F(0.05),
    max_steps::Int=500,
    tol::Real=Defaults.atol(F),
    fd_step::Real=F(0.01),
    max_corrector_iters::Int=10,
    max_corrector_failures::Int=5,
    max_gradient_failures::Int=10,
    min_steps_before_close::Int=20,
) where {F}
    step_size = F(step_size)
    tol = F(tol)
    fd_step = F(fd_step)
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
    counter = Ref(0)
    n_corrector_failures = 0
    n_gradient_failures = 0
    consecutive_corrector_failures = 0
    consecutive_gradient_failures = 0

    # TODO: consider factoring out the seed resolution and validation block.
    seed_z = if isnothing(seed)
        find_seed(
            model;
            tol=tol,
            fd_step=fd_step,
            max_corrector_iters=max_corrector_iters,
            counter=counter,
        )
    else
        Complex{F}(seed)
    end

    if isnothing(seed_z)
        return BoundaryTrace(
            BoundaryResult(Complex{F}[], F[], false, complex(zero(F))),
            counter[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    # Reject seeds that are clearly outside the support.
    seed_shift = solve_shift(model, seed_z, counter)
    isfinite(seed_shift) || return BoundaryTrace(
        BoundaryResult(Complex{F}[], F[], false, Complex{F}(seed_z)),
        counter[],
        n_corrector_failures,
        n_gradient_failures,
    )

    if (seed_shift < 0) && (abs(seed_shift) > tol)
        return BoundaryTrace(
            BoundaryResult(Complex{F}[], F[], false, Complex{F}(seed_z)),
            counter[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    # Refine the seed to land on the contour.
    refined = corrector_step(
        model,
        seed_z;
        tol=tol,
        fd_step=fd_step,
        max_iters=max_corrector_iters,
        counter=counter,
    )
    if isnothing(refined)
        return BoundaryTrace(
            BoundaryResult(Complex{F}[], F[], false, Complex{F}(seed_z)),
            counter[],
            n_corrector_failures,
            n_gradient_failures,
        )
    end

    seed_z, seed_shift = refined
    points = Complex{F}[seed_z]
    shift_values = F[seed_shift]

    # Compute initial tangent direction from the seed gradient.
    grad = compute_gradient(model, seed_z, fd_step, counter)
    if isnothing(grad)
        n_gradient_failures += 1
        return BoundaryTrace(
            BoundaryResult(points, shift_values, false, seed_z),
            counter[],
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
        corrected = corrector_step(
            model,
            z_pred;
            tol=tol,
            fd_step=fd_step,
            max_iters=max_corrector_iters,
            counter=counter,
        )
        if isnothing(corrected)
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
        grad = compute_gradient(model, current_z, fd_step, counter)
        if isnothing(grad)
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
                final_refine = corrector_step(
                    model, points[1]; tol=tol, fd_step=fd_step, max_iters=5, counter=counter
                )
                if !isnothing(final_refine)
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
    return BoundaryTrace(result, counter[], n_corrector_failures, n_gradient_failures)
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
    grad = compute_gradient(model, complex(1.25), 0.01)
    @test !isnothing(grad)
    @test real(grad) < 0  # shift decreases outward, so gradient points outward (negative)

    # At top of ellipse (0, 0.75), gradient should point outward (positive imaginary direction)
    grad = compute_gradient(model, complex(0.0, 0.75), 0.01)
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
    grad = compute_gradient(model, complex(0.5, 0.3), 0.01)
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
    grad = compute_gradient(model, complex(3.0), 0.01)
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
    result = corrector_step(model, complex(1.2); fd_step=0.01)
    @test !isnothing(result)
    z_corrected, s_corrected = result
    @test abs(s_corrected) < 1e-6
    # The corrected point should be near the ellipse boundary
    @test abs(real(z_corrected)) > 1.0  # still near the right side

    # Start near the boundary (slightly outside)
    result = corrector_step(model, complex(1.3); fd_step=0.01)
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
    result = corrector_step(model, complex(3.0); tol=1e-12, fd_step=0.01, max_iters=1)
    @test isnothing(result)
end

@testitem "find_seed from sample points" begin
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

    seed = find_seed(model)
    @test !isnothing(seed)
    # The seed should be approximately on the boundary (shift ≈ 0)
    s = solve_shift(model, seed)
    @test isfinite(s)
    @test abs(s) < 1e-4
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
