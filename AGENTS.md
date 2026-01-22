# GaussianSpectralDensities.jl Agent Guide

## Package Purpose

Computes spectral densities of non-Hermitian random matrices using the averaged resolvent method. Handles **Girko T-ensembles**: matrices of the form `M + L·J·R` where `J` is drawn from a generalized Girko distribution with diagonal symmetry parameter `T`.

The core task is solving self-consistent nonlinear equations for the averaged resolvent, from which spectral density is derived.

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a specific testitem (requires TestEnv installed in the base environment)
julia --startup-file=no --check-bounds=yes --warn-overwrite=yes --depwarn=yes --project -e 'using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti->ti.name=="stepping"'

# Format code (requires JuliaFormatter v2.2.0 or newer installed as a CLI app)
jlfmt --inplace src/
```

## Codebase State

This code is being revived from a bitrotted rewrite. Many files are commented out in src/GaussianSpectralDensities.jl—**ignore files whose includes are commented out**. Focus only on active modules.

Active modules (in dependency order):

1. `MathHelpers` — Linear algebra utilities
2. `Ensembles` — `MatrixEnsemble` type for `M + L·J·R` ensembles
3. `Regularizers` — `Regularizer` type (shift vs gap parameterization)
4. `Objectives` — (minimal, being developed)
5. `Problems` — `LocalResolventEquation`, `LocalRegularizerProblem`, `LocalTraceProblem`
6. `Workspaces` — Mutable workspace with cached intermediate computations
7. `Solutions` — `LocalResolventSolution` result type
8. `Newton` — Generic Newton solver with finite-difference Jacobian
9. `EquationSolvers` — `eqsolve!` implementations for problem types
10. `Algorithms` — Solution strategies: `DirectSolve`, `RegularizerSchedule`, `SymmetrySchedule`

## Architecture Patterns

### Submodule structure

Each file defines a `module` and ends with `@testitem` blocks (TestItems.jl). Tests are colocated with code.

### Float type parameterization

All numeric types are parameterized by `F<:AbstractFloat`. Constructors support:

- Type-inferring: `Foo(args...)` infers `F` from the first argument
- Type-enforcing: `Foo{F}(args...)` converts to specified precision
- Copy-converting: `Foo{F}(foo::Foo)` for precision conversion

### Array type flexibility

Code supports `Array`, `SArray` (StaticArrays), and `SparseMatrixCSC`. The `MathHelpers` module provides unified operations across these types.

### Workspace pattern

`Workspace` is a mutable cache for intermediate matrices. It tracks what's "clean" via an enum to avoid redundant computation. Functions like `P!`, `Q!`, `k!` update and return cached values.

### Algorithm/State pattern

Algorithms implement an iterator-like interface via `step!`:

- `step!(ws, problem, alg)` → `(solution, state)` (first step)
- `step!(ws, problem, alg, state)` → `(solution, state)` or `nothing` (subsequent steps)

## Naming Conventions

**Prefer descriptive ASCII names over single-letter or Greek symbols.** This codebase is moving away from math-notation variable names. Examples of the new style:

- `qscale` not `μ`
- `shift` not `γ²`
- `baregap` not `δ₀`

Functions ending in `!` mutate their first argument (Julia convention). Functions ending in `!!` may or may not mutate depending on array type (e.g., StaticArrays can't be mutated), and they always return the result such that mutating and non-mutating methods can be used identically (e.g., it's always `z = foo!!(x, y)`, whereby you end up with `z === x` if this hit a mutating method).

## Key Types

| Type                         | Purpose                                |
| ---------------------------- | -------------------------------------- |
| `MatrixEnsemble{F}`          | Ensemble `M + L·J·R` with symmetry `T` |
| `LocalResolventEquation{F}`  | Equation specification at point `z`    |
| `LocalRegularizerProblem{F}` | Find `q` given regularizer             |
| `LocalTraceProblem{F}`       | Find `q, γ²` given trace target `k`    |
| `LocalResolventSolution{F}`  | Solution with `q, shift, gap, k, E`    |
| `Workspace{F}`               | Cached intermediate computations       |
| `Regularizer{F}`             | Value + type (`:shift` or `:gap`)      |

## Testing

Tests use TestItems.jl for isolation. Each `@testitem` block is independent—no global state leakage.

Common test patterns:

- Test across `Float32` and `Float64`
- Test across array types: `Array`, `SArray`, `SparseMatrixCSC`
- Test type conversions round-trip correctly
- Test error conditions with `@test_throws`

In higher-level behavior tests, use type-enforcing constructors (e.g., `MatrixEnsemble{F}(...)`) to obtain correctly-typed objects. Intermediate variables used to assemble these don't need float-specific versions—the constructor handles conversion. This keeps setup code simple since constructor correctness is verified separately.

Never use a plain `for` loop when the loop body contains a `@test` (or one of its siblings, like `@test_throws`). Such loops should always use the test set form: `@testset "$label" for ...`. This ensures that failing tests are printed with sufficient context to know in which iteration they failed.

## Commented Code

There's substantial commented code from the old implementation. Don't worry about cleaning it up unless specifically asked—focus on making active code work and be well-tested.
