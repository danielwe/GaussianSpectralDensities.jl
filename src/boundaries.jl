module Boundaries

using ..Defaults: Defaults
using ..Newton: increment
using ..Solutions: issuccess
using ..Spreads: Spreads
using ..Surrogates: SurrogateModel, evaluate

using Statistics: mean

public BoundaryResult, BoundaryTrace, extract_boundary

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
    @test trace.n_failures >= 0
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
    eq = ResolventEquation(ensemble; shiftmin=-0.5)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    model = SurrogateModel(eq, constraint, DirectSolve(), 96)

    # Provide a seed far outside the support — solver should fail with shiftmin violation if
    # nothing else, and extract_boundary should return empty result with closed=false
    trace = extract_boundary(model; seed=complex(10.0))
    result = trace.result
    @test !result.closed
end
