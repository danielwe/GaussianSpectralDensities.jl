# GaussianSpectralDensities.jl Agent Guide

## Package Purpose

Computes spectral densities of non-Hermitian random matrices using the method of hermitization. Handles matrices of the form `M + L·J·R` where `J` is drawn from a **Girko S-ensembles**, which is our term for a generalized Girko distribution with diagonal symmetry parameter `S`. The well-known elliptic Girko ensemble with scalar symmetry parameter `τ` is recovered by setting `S = √(τ) I`.

The core task is solving self-consistent nonlinear equations for the averaged doubled-up resolvent, from which spectral density is derived. Compared to previous work where `J` was drawn from the Ginibre ensemble, the nonzero symmetry parameter introduces an additional self-consistency equation for the off-diagonal blocks of the doubled-up self-energy.

## Commands

```bash
# Run all tests
julia --startup-file=no --project=@. -e 'using Pkg; Pkg.test()'

# Run specific testitems filtered by name and/or file (requires TestEnv installed in the base environment)
julia --startup-file=no --check-bounds=yes --warn-overwrite=yes --depwarn=yes --project=@. -e 'cd(joinpath(dirname(Base.active_project()), "test")); using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti -> (ti.name == "testitem name") && (basename(ti.filename) =="source_file.jl")'

# Run tests with coverage (requires CoverageTools installed in the @nvim-coverage shared environment)
# 1. start with a clean slate; run:
rm -f src/*.cov
# 2. run any of the test commands above with the additional argument `--code-coverage=@src`
# 3. run the following:
julia --startup-file=no --project=@nvim-coverage -e 'using CoverageTools; LCOV.writefile("lcov.info", process_folder("src"))'

# Format code (requires JuliaFormatter v2.2.0 or newer installed as a CLI app)
jlfmt --history-file=no -- --prioritize-config-file --inplace src/source_file.jl
```

## Codebase State

This code is being revived from a bitrotted rewrite. Many files are commented out in src/GaussianSpectralDensities.jl—**ignore files whose includes are commented out**. Focus only on active modules.

Active modules (in dependency order):

1. `ArrayTools` — Utilities for generic array programming
1. `MathTools` — Utilities for real/complex numbers and generic linear algebra
1. `Ensembles` — `MatrixEnsemble` type for `M + L·J·R` ensembles
1. `Spreads` — `Spread` type (shift vs gap parameterization)
1. `Constraints` — `AbstractConstraint` and `PTrace` types for problem constraints
1. `Problems` — `ResolventEquation` (shared context), `PointProblem`, init types
1. `Workspaces` — Mutable workspace with cached intermediate computations
1. `Solutions` — `ResolventParams` (data bundle) and `PointSolution` (result at a point)
1. `Newton` — Generic Newton solver with finite-difference Jacobian
1. `EquationSolvers` — `eqsolve!` implementations for problem types
1. `Algorithms` — Solution strategies: `DirectSolve`, `SpreadSchedule`, `SymmetrySchedule`
1. `Solvers` — User-facing `Solver` and `Solution` types with CommonSolve.jl interface

## Architecture Patterns

### Submodule structure

Each file defines a `module` followed by `@testitem` blocks (TestItems.jl). Tests are colocated with code but placed after the module's closing `end`.

This is a more explicitly modular structure than is typical in Julia, but in this project, we prefer tight encapsulation of each type tree and a limited but well-defined public interface to go with it, keeping the top-level module scope free of private helper functions. This also means we don't need to mark private helper functions with an underscore prefix---we just don't mark them public.

### Real/float type parameterization

Most types are parameterized by `F<:AbstractFloat`, though some have the more general allowance `F<:Real`. Internal types (used behind the scenes) typically support multiple constructor patterns for type adaptation:

- Type-inferring: `Foo(args...)` infers `F` from the first argument
- Type-enforcing: `Foo{F}(args...)` converts to specified precision; typically an inner constructor enforcing invariants
- Copy-converting: `Foo{F}(foo::Foo)` for precision conversion; also often an inner constructor that converts field values directly, bypassing any assertions and calculations in `Foo{F}(args...)` since these should have applied when constructing the original `foo`

User-facing types (like those in `Solvers`) may have simpler constructor patterns since precision conversion is less important at the API boundary.

### Array type flexibility

Code supports `Array`, `SArray` (StaticArrays), and `SparseMatrixCSC`. The `ArrayTools` and `MathTools` modules provide unified operations across these types.

### Workspace pattern

`Workspace` is a mutable cache for intermediate matrices, constructed from a `ResolventEquation`. It tracks what's "valid" via the `Valid` enum to avoid redundant computation. Functions like `P!`, `Q!`, `ptrace!` update and return cached values. These functions take `(ws, eq, z, shape, spread)` where `z` is the point in the complex plane.

### Algorithm/State pattern

Algorithms implement an iterator-like interface via `step!`:

- `step!(ws, problem, alg, nothing)` → `(solution, state)` (first step)
- `step!(ws, problem, alg, state)` → `(solution, state)` or `nothing` (subsequent steps)

## Naming Conventions

**Prefer descriptive ASCII names over single-letter or Greek symbols.** This codebase is moving away from math-notation variable names. Examples of the new style:

- `sscale` not `μ`
- `shift` not `γ²`
- `baregap` not `δ₀`
- `ptrace` not `k`
- `greens` not `E`

Functions ending in `!` mutate their first argument (Julia convention). Functions ending in `!!` may or may not mutate depending on array type (e.g., StaticArrays can't be mutated), and they always return the result such that mutating and non-mutating methods support the same usage patterns (e.g., you can always write `z = foo!!(x, y)` and obtain the result in `z`; however, if the call hit a mutating method, it may also be that `z === x`).

## Key Types

| Type                   | Purpose                                               |
| ---------------------- | ----------------------------------------------------- |
| `MatrixEnsemble{F}`    | Ensemble `M + L·J·R` with symmetry `S`                |
| `ResolventEquation{F}` | Shared context for resolvent equations                |
| `PointProblem{F}`      | Bundles equation, constraint, `z`, and init           |
| `ResolventParams{F}`   | Data bundle: `shape, shift, gap, ptrace, greens`      |
| `PointSolution{F}`     | Solution at a point with success flag                 |
| `Workspace{F}`         | Cached intermediate computations                      |
| `Spread{F}`            | Value + type (`:shift` or `:gap`)                     |
| `Solver{F}`            | User-facing pointwise solver with iteration interface |
| `Solution{F}`          | Final pointwise result with trace history             |

## Testing

### Writing tests

Tests use TestItems.jl. Each `@testitem` block is independent with no global state leakage.

**What to test** depends on module level. Internal modules need exhaustive coverage: test across `Float32` and `Float64`, array types (`Array`, `SArray`, `SparseMatrixCSC`), round-trip type conversions, and error conditions with `@test_throws`. User-facing modules like `Solvers` focus on API behavior rather than exhaustive support for type conversions.

**Setup code** can leverage verified internal APIs. Use type-enforcing constructors (e.g., `MatrixEnsemble{F}(...)`) to get correctly-typed objects without manual float-specific setup—the constructor handles conversion. This works because constructor correctness is tested separately.

**Test style:** wrap `for` loops in `@testset "value = $v"` to include loop values in failure output (not needed for setup-only loops). Test the API directly—don't define functions inside `@testitem` blocks. Prefer bare comparisons (`@test x < y`, not `@test foo(x, y)`) for clearer failure messages. Use exact equality with roundoff-resistant values when possible (powers of 2 are the most robust); define test-specific `isapprox` for internal types only when necessary.

### Running tests

**After making changes, do not immediately run tests.** Unless explicitly requested, stop and notify the user that the code is ready for testing. This gives the user a chance to review changes and catch misunderstandings before they trigger a lengthy edit-test-debug cycle.

**When running tests, work incrementally:**

1. Run one testitem or file at a time
2. For changes touching multiple files, follow the inclusion order in src/GaussianSpectralDensities.jl—run tests from each affected file and fix all failures before moving to the next
3. Run test/qa.jl and fix any failures
4. Only then run the full test suite to verify nothing else broke

## Commented Code

There's substantial commented code from the old implementation. Don't worry about cleaning it up unless specifically asked—focus on making active code work and be well-tested.
