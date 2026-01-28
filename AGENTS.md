# GaussianSpectralDensities.jl Agent Guide

## Package Purpose

Computes spectral densities of non-Hermitian random matrices using the averaged resolvent method. Handles **Girko S-ensembles**: matrices of the form `M + L·J·R` where `J` is drawn from a generalized Girko distribution with diagonal symmetry parameter `S`.

The core task is solving self-consistent nonlinear equations for the averaged resolvent, from which spectral density is derived.

## Commands

```bash
# Run all tests
julia --startup-file=no --project=@. -e 'using Pkg; Pkg.test()'

# Run specific testitems filtered by name and/or file (requires TestEnv installed in the base environment)
julia --startup-file=no --check-bounds=yes --warn-overwrite=yes --depwarn=yes --project=@. -e 'cd(joinpath(dirname(Base.active_project()), "test")); using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti->(ti.name=="testitem name" && endswith(ti.filename, "source_file.jl"))'

# Format code (requires JuliaFormatter v2.2.0 or newer installed as a CLI app)
jlfmt --inplace src/
```

After making an edit, **do not immediately run tests or launch a test--fix--test cycle**. Always stop and confirm that this is desired. In most cases the user will prefer to look over the edits and make their own adjustments before moving onto testing.

## Codebase State

This code is being revived from a bitrotted rewrite. Many files are commented out in src/GaussianSpectralDensities.jl—**ignore files whose includes are commented out**. Focus only on active modules.

Active modules (in dependency order):

1. `ArrayTools` — Utilities for generic array programming
1. `MathTools` — Utilities for real/complex numbers and generic linear algebra
1. `Ensembles` — `MatrixEnsemble` type for `M + L·J·R` ensembles
1. `Regularizers` — `Regularizer` type (shift vs gap parameterization)
1. `Problems` — `ResolventEquation`, `RegularizerProblem`, `PTraceProblem`
1. `Workspaces` — Mutable workspace with cached intermediate computations
1. `Solutions` — `ResolventSolution` result type
1. `Newton` — Generic Newton solver with finite-difference Jacobian
1. `EquationSolvers` — `eqsolve!` implementations for problem types
1. `Algorithms` — Solution strategies: `DirectSolve`, `RegularizerSchedule`, `SymmetrySchedule`

## Architecture Patterns

### Submodule structure

Each file defines a `module` and ends with `@testitem` blocks (TestItems.jl). Tests are colocated with code.

This is a more explicitly modular structure than is typical in Julia, but in this project, we prefer tight encapsulation of each type tree and a limited but well-defined public interface to go with it, keeping the top-level module scope free of private helper functions. This also means we don't need to mark private helper functions with an underscore prefix---we just don't mark them public.

### Real/float type parameterization

Most numeric types are parameterized by `F<:AbstractFloat`, though some have the more general allowance `F<:Real`. Constructors support:

- Type-inferring: `Foo(args...)` infers `F` from the first argument
- Type-enforcing: `Foo{F}(args...)` converts to specified precision; typically an inner constructor enforcing invariants
- Copy-converting: `Foo{F}(foo::Foo)` for precision conversion; also often an inner constructor to bypass parts of the type-enforcing constructor that are redundant when working with an existing instance

### Array type flexibility

Code supports `Array`, `SArray` (StaticArrays), and `SparseMatrixCSC`. The `ArrayTools` and `MathTools` modules provide unified operations across these types.

### Workspace pattern

`Workspace` is a mutable cache for intermediate matrices. It tracks what's "clean" via an enum to avoid redundant computation. Functions like `P!`, `Q!`, `ptrace!` update and return cached values.

### Algorithm/State pattern

Algorithms implement an iterator-like interface via `step!`:

- `step!(ws, problem, alg)` → `(solution, state)` (first step)
- `step!(ws, problem, alg, state)` → `(solution, state)` or `nothing` (subsequent steps)

## Naming Conventions

**Prefer descriptive ASCII names over single-letter or Greek symbols.** This codebase is moving away from math-notation variable names. Examples of the new style:

- `sscale` not `μ`
- `shift` not `γ²`
- `baregap` not `δ₀`
- `ptrace` not `k`
- `greens` not `E`

Functions ending in `!` mutate their first argument (Julia convention). Functions ending in `!!` may or may not mutate depending on array type (e.g., StaticArrays can't be mutated), and they always return the result such that mutating and non-mutating methods can be used identically (e.g., it's always `z = foo!!(x, y)`, whereby you always end up with the desired result in `z`, but `z === x` if and only if the call hits a mutating method).

## Key Types

| Type                    | Purpose                                       |
| ----------------------- | --------------------------------------------- |
| `MatrixEnsemble{F}`     | Ensemble `M + L·J·R` with symmetry `S`        |
| `ResolventEquation{F}`  | Equation specification at point `z`           |
| `RegularizerProblem{F}` | Find `q` given regularizer                    |
| `PTraceProblem{F}`      | Find `q, shift` given trace target `ptrace`   |
| `ResolventSolution{F}`  | Solution with `q, shift, gap, ptrace, greens` |
| `Workspace{F}`          | Cached intermediate computations              |
| `Regularizer{F}`        | Value + type (`:shift` or `:gap`)             |

## Testing

Tests use TestItems.jl for isolation. Each `@testitem` block is independent—no global state leakage.

Common test patterns:

- Test across `Float32` and `Float64`
- Test across array types: `Array`, `SArray`, `SparseMatrixCSC`
- Test type conversions round-trip correctly
- Test error conditions with `@test_throws`

In higher-level behavior tests, use type-enforcing constructors (e.g., `MatrixEnsemble{F}(...)`) to obtain correctly-typed objects. Intermediate variables used to assemble these don't need float-specific versions—the constructor handles conversion. This keeps setup code simple since constructor correctness is verified separately.

Do not use a plain `for` loop when the loop body contains a `@test` (or one of its siblings, like `@test_throws`). Such loops should always use the test set form: `@testset "$label" for ...`. This ensures that failing tests are printed with sufficient context to know in which iteration they failed.

## Commented Code

There's substantial commented code from the old implementation. Don't worry about cleaning it up unless specifically asked—focus on making active code work and be well-tested.
