# GaussianSpectralDensities.jl Agent Guide

## Package Purpose

Computes spectral densities of non-Hermitian random matrices using the method of hermitization. Handles matrices of the form `M + L·J·R` where `J` is drawn from a **Girko S-ensemble**, which is our term for a generalized Girko ensemble with diagonal symmetry parameter `S`. The well-known elliptic Girko ensemble with scalar symmetry parameter `τ` is recovered by setting `S = √(τ) I`.

The core task is to solve self-consistent equations for the averaged doubled-up resolvent, from which the spectral density and boundary are derived. In prior work, `J` was drawn from the Ginibre ensemble, and one had to solve a single self-consistency equation for the real-valued parameter `spread`. Here, the nonzero symmetry parameter `S` introduces an additional self-consistency equation for off-diagonal blocks, along with a complex-valued self-consistent parameter `shape`.

## Commands

```bash
# Run all tests
julia --startup-file=no --project=@. -e 'using Pkg; Pkg.test()'

# Run specific testitems filtered by name and/or file
julia --startup-file=no --check-bounds=yes --warn-overwrite=yes --depwarn=yes --project=@. -e 'cd(joinpath(dirname(Base.active_project()), "test")); using TestEnv; TestEnv.activate(); using TestItemRunner; @run_package_tests filter=ti -> (ti.name == "testitem name") && (basename(ti.filename) == "file.jl")'

# Run tests with coverage
# 1. Cleanup:
rm -f src/*.cov
# 2. Run any test command with the extra argument `--code-coverage=@src`
# 3. Process:
julia --startup-file=no --project=@nvim-coverage -e 'using CoverageTools; LCOV.writefile("lcov.info", process_folder("src"))'

# Format code
jlfmt --history-file=no -- --prioritize-config-file --inplace src/source_file.jl
```

## Architecture Patterns

### Submodule structure

Each file defines one submodule followed by `@testitem` blocks _after_ the module `end`. This encapsulation keeps internal APIs well-defined and avoids polluting the top-level namespace with private helpers. **Don't mark private helpers with underscore prefixes—simply don't make them public.**

### Real/float type parameterization

Most types are parameterized by `F<:AbstractFloat`. Internal types support three constructor patterns: type-inferring `Foo(args...)`, type-enforcing `Foo{F}(args...)`, and copy-converting `Foo{F}(foo::Foo)`. **Internal types should have `convert` and `adaptrealtype` methods** wrapping the copy-converting constructor. User-facing types may have simpler constructor patterns.

### Array type flexibility

Code supports `Array`, `SArray` (StaticArrays), and `SparseMatrixCSC`. The `ArrayTools` and `MathTools` modules provide unified operations across these types.

## Naming Conventions

Prefer descriptive ASCII names over single-character symbols. Examples: `shift` not `γ²`, `shape` not `q`.

Functions ending in `!` mutate their first argument (Julia convention). Functions ending in `!!` have polymorphic mutation behavior: they mutate when the array type allows, but always return the result to allow generic use.

## Testing

### Writing tests

Tests use TestItems.jl. Each `@testitem` block is an independent module with no access to global state and must therefore include all necessary imports, both from this package and dependencies.

**What to test:** internal modules need exhaustive coverage: test across `Float32` and `Float64`, array types (`Array`, `SArray`, `SparseMatrixCSC`), round-trip type conversions, and error conditions with `@test_throws`. User-facing modules like `SpectralModels` focus on testing API behavior rather than construction and conversion.

**Test setup code** should make full use of upstream submodule APIs to minimize boilerplate. For example, higher-level type-enforcing constructors (e.g., `MatrixEnsemble{F}(...)`) circumvent the need to create float type-specific instances of the lower-level components (e.g., `M,L,R,D`).

**Test style:** wrap `for` loops in `@testset "value = $v"` to include loop values in failure output (not needed for setup-only loops). Test the API directly—don't define functions inside `@testitem` blocks. Prefer bare comparisons (`@test x < y`, not `@test foo(x, y)`) for clearer failure messages. Prefer exact equality tests even for floats when able to choose roundoff-resistant values for both inputs and outputs (powers of 2 are the most robust).

### Running tests

**After making changes, do not immediately run tests.** Unless explicitly requested, stop and notify the user that the code is ready for testing.

**When running tests, work incrementally:** Run affected testitems/files (following the inclusion order in src/GaussianSpectralDensities.jl), then test/qa.jl, then the full suite. Every test in one file should pass before moving onto the next.

## Commented Code

There is commented-out code and stale notebooks in `demo/` and `demo.new/` from earlier iterations. Don't worry about cleaning it up unless specifically asked.
