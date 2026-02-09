# GaussianSpectralDensities.jl Agent Guide

## Package Purpose

Computes spectral densities of non-Hermitian random matrices using the method of hermitization. Handles matrices of the form `M + L·J·R` where `J` is drawn from a **Girko S-ensembles**, which is our term for a generalized Girko distribution with diagonal symmetry parameter `S`. The well-known elliptic Girko ensemble with scalar symmetry parameter `τ` is recovered by setting `S = √(τ) I`.

The core task is solving self-consistent nonlinear equations for the averaged doubled-up resolvent, from which spectral density is derived. Compared to previous work where `J` was drawn from the Ginibre ensemble, the nonzero symmetry parameter introduces an additional self-consistency equation for the off-diagonal blocks of the doubled-up self-energy.

## Commands

```bash
# Run all tests
julia --startup-file=no --project=@. -e 'using Pkg; Pkg.test()'

# Run specific testitems filtered by name and/or file
julia --startup-file=no --check-bounds=yes --warn-overwrite=yes --depwarn=yes --project=@. -e 'cd(joinpath(dirname(Base.active_project()), "test")); using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti -> (ti.name == "testitem name") && (basename(ti.filename) == "file.jl")'

# Run tests with coverage
# 1. Cleanup:
rm -f src/*.cov
# 2. Run: execute test command with extra argument `--code-coverage=@src`
# 3. Process:
julia --startup-file=no --project=@nvim-coverage -e 'using CoverageTools; LCOV.writefile("lcov.info", process_folder("src"))'

# Format code
jlfmt --history-file=no -- --prioritize-config-file --inplace src/source_file.jl
```

## Codebase State

This code is being revived from a bitrotted rewrite. Many files are commented out in src/GaussianSpectralDensities.jl—**ignore files whose includes are commented out**. Focus only on active modules (in dependency order):

1. `ArrayTools` — Generic array utilities
1. `MathTools` — Linear algebra operations
1. `Ensembles` — `MatrixEnsemble{F}` for `M + L·J·R` ensembles
1. `Spreads` — `Spread{F}` (shift vs gap parameterization)
1. `Problems` — `ResolventEquation{F}` (ensemble & auxiliary params), constraint types, `PointProblem{F}`
1. `Workspaces` — `Workspace{F}` (mutable cache for intermediate computations)
1. `Solutions` — `ResolventParams{F}` (resolvent unknowns), `PointSolution{F}`
1. `Newton` — Generic Newton solver with finite-difference Jacobian
1. `EquationSolvers` — `eqsolve!` per constraint type
1. `Algorithms` — Solution strategies: `DirectSolve`, `SpreadAnnealing`, `SymmetryAnnealing`
1. `Solvers` — User-facing `Solver` and `Solution` types with CommonSolve.jl interface

## Architecture Patterns

### Submodule structure

Each file defines a module followed by `@testitem` blocks (TestItems.jl). Tests are colocated with code but placed after the module's `end`. We prefer tight encapsulation with a limited, well-defined public interface, keeping the top-level scope free of private helpers. **Don't mark private helpers with underscore prefixes—simply don't export them.**

### Real/float type parameterization

Most types are parameterized by `F<:AbstractFloat`. Internal types support three constructor patterns: type-inferring `Foo(args...)`, type-enforcing `Foo{F}(args...)`, and copy-converting `Foo{F}(foo::Foo)`. **Internal types should have `convert` and `adaptrealtype` methods** (using copy-converting constructors for precision conversion). User-facing types may have simpler constructor patterns.

### Array type flexibility

Code supports `Array`, `SArray` (StaticArrays), and `SparseMatrixCSC`. The `ArrayTools` and `MathTools` modules provide unified operations across these types.

### Workspace pattern

`Workspace` is a mutable cache for intermediate matrices, constructed from a `ResolventEquation`. It tracks what's "valid" via the `Valid` enum to avoid redundant computation. Functions like `P!`, `Q!`, `ptrace!` update and return cached values. These functions take `(ws, eq, z, shape, spread)` where `z` is the point in the complex plane.

### Algorithm/State pattern

Algorithms implement an iterator-like interface via `step!`:

- `step!(ws, problem, alg, nothing)` → `(solution, state)` (first step)
- `step!(ws, problem, alg, state)` → `(solution, state)` or `nothing` (subsequent steps)

## Naming Conventions

Prefer descriptive ASCII names over single-letter or Greek symbols. Examples: `sscale` not `μ`, `shift` not `γ²`, `shape` not `q`.

Functions ending in `!` mutate their first argument (Julia convention). Functions ending in `!!` signal polymorphic mutation behavior: they mutate when the array type allows, but always return the result to allow generic use.

## Testing

### Writing tests

Tests use TestItems.jl. Each `@testitem` block is independent with no access to global state.

**What to test** depends on module level. Internal modules need exhaustive coverage: test across `Float32` and `Float64`, array types (`Array`, `SArray`, `SparseMatrixCSC`), round-trip type conversions, and error conditions with `@test_throws`. User-facing modules like `Solvers` focus on API behavior rather than exhaustive support for type conversions.

**Setup code** can leverage verified internal APIs. Use type-enforcing constructors (e.g., `MatrixEnsemble{F}(...)`) to get correctly-typed objects without manual float-specific setup—the constructor handles conversion. This works because constructor correctness is tested separately.

**Test style:** wrap `for` loops in `@testset "value = $v"` to include loop values in failure output (not needed for setup-only loops). Test the API directly—don't define functions inside `@testitem` blocks. Prefer bare comparisons (`@test x < y`, not `@test foo(x, y)`) for clearer failure messages. Use exact equality with roundoff-resistant values when possible (powers of 2 are the most robust); define test-specific `isapprox` for internal types only when necessary.

### Running tests

**After making changes, do not immediately run tests.** Unless explicitly requested, stop and notify the user that the code is ready for testing.

**When running tests, work incrementally:** Run affected testitems/files (following the inclusion order in src/GaussianSpectralDensities.jl), then test/qa.jl, then the full suite. Every test in one file should pass before moving onto the next.

## Commented Code

There's substantial commented code from the old implementation. Don't worry about cleaning it up unless specifically asked—focus on making active code work and be well-tested.
