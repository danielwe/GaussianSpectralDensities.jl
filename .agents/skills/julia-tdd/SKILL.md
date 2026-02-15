---
name: julia-tdd
description: "Structured Test-Driven Development workflow for Julia. Use when writing new Julia modules: plan first, write comprehensive tests, review thoroughly, then implement code."
---

# Julia TDD Workflow

A disciplined four-phase workflow for developing Julia modules with test-driven development (TDD):

1. **Plan** (smart mode) → write detailed specification to file
2. **Test** (smart mode handoff) → write comprehensive test suite based on plan
3. **Review** (deep mode handoff) → thoroughly validate test coverage and correctness
4. **Implement** (deep mode handoff) → write production code, do not modify tests

This workflow ensures tests are independent of implementation details and catches design issues early.

## Phase 1: Planning

**Mode**: Smart (you decide when to trigger)

1. Discuss module architecture, public API, key types, and algorithms in conversation
2. Write detailed specification to `<module>_plan.md` including:
   - Purpose and scope
   - Module location and dependencies
   - All struct/type definitions with field types and semantics
   - Constructor signatures and behavior
   - Public API functions (signatures, behavior, error conditions)
   - Internal implementation strategy
   - Known limitations and future work
3. Save plan file to root of workspace (e.g., `rbf_plan.md`)

**Checklist:**

- [ ] Module location and dependencies listed
- [ ] All public types defined with full field documentation
- [ ] Constructor contracts clear (what they build, invariants they establish)
- [ ] All public functions have signatures and documented behavior
- [ ] Edge cases and error conditions mentioned
- [ ] Internal design explained (not line-by-line, but strategic choices)

## Phase 2: Test Writing

**Mode**: Smart (handoff from root thread)

1. Accept the plan file as input
2. Read the plan thoroughly
3. Write comprehensive test suite using `@testitem` blocks in `test/<module>.jl`
4. Follow Julia testing patterns:
   - Test across `Float32` and `Float64`
   - Test with different array types (`Array`, `SArray`, `SparseMatrixCSC`) if module supports them
   - Test constructors (type-inference and type-enforcing patterns)
   - Test all public API functions with normal inputs, edge cases, and error conditions
   - Test invariants (e.g., round-trip conversions, monotonic operations)
   - Use descriptive test names that read like documentation
   - **Each test is independent**: does not rely on other tests running first, has its own setup
5. Do NOT write any implementation code
6. Stop and notify user; do not run tests yet

**Physics/Domain Specificity** (if module depends on external solvers or domain concepts):

- Choose problem setups carefully to produce known success/failure patterns
- Use minimal reproducible examples (e.g., identity matrices, simple scalar parameters)
- Separate "happy path" tests from error-handling and edge-case tests
- Avoid shared state between tests; each should verify behavior independently

**Checklist:**

- [ ] All public API functions tested
- [ ] Edge cases covered (empty inputs, single elements, large counts)
- [ ] Error conditions tested with `@test_throws`
- [ ] Float type compatibility tested (`Float32`, `Float64`)
- [ ] Array type compatibility tested (if applicable)
- [ ] Round-trip properties tested (e.g., x → y → x)
- [ ] Invariants documented and tested
- [ ] Test names descriptive and readable
- [ ] No implementation code written

## Phase 3: Test Review

**Mode**: Deep (handoff from test writing thread)

1. Accept module file with tests as input
2. Carefully read all tests against the plan
3. Verify against the plan:
   - All functions from plan are tested
   - Edge cases are thorough (empty, single, many, boundary values)
   - Error conditions match documented behavior
   - Tests are independent and do not depend on test order
   - Test names accurately describe what is tested
   - Type coverage is complete (Float32, Float64, array types)
   - Invariants are correctly expressed (e.g., f(f⁻¹(x)) == x)
   - No test reads or modifies state from other tests
   - If the module has domain-specific invariants (e.g., "no poisoning" for adaptive algorithms, "monotonic enrichment" for interpolants), verify they are tested and correctly expressed
4. Suggest improvements to test coverage or clarity
5. **Do not modify tests yet**
6. Report findings; wait for approval before making any changes

**Checklist:**

- [ ] All functions in plan have corresponding tests
- [ ] Test names match test content
- [ ] Edge case logic is sound (boundary detection, special values)
- [ ] Error tests correctly match documented exceptions
- [ ] Type conversions tested with proper assertions
- [ ] Invariant tests correctly encode the mathematical/logical property
- [ ] No test interdependencies
- [ ] Tests are deterministic (no randomness without seed)

## Phase 4: Implementation

**Mode**: Deep (handoff from root thread or after test approval)

1. Accept plan and module file with approved tests as inputs
2. **Do not modify tests**
3. Implement the module in `src/<module>.jl` following the plan
4. As you write, verify:
   - Type signatures match plan
   - Constructor behavior matches documented contracts
   - All public functions have correct signatures and return types
   - Code follows AGENTS.md conventions (naming, mutating functions, type patterns)
5. Run tests incrementally:
   - After implementing each type or group of functions, run affected tests
   - Fix implementation to pass tests; do not modify tests
   - **Stop and Discuss Rule:** If a test failure appears to be caused by an issue **within the test itself** (missing import, typo, incorrect assumption, unsuitable tolerance), **stop immediately and report it to the user**. Do not attempt workarounds.
   - **Hard Stop Rule (No Exceptions):** On the first occurrence of a suspected test issue, halt all further command execution and code changes. Respond with: "Test appears flawed: <reason>. Halting per workflow." Do not retry tests or continue debugging until the user confirms how to proceed.
   - **Prohibition of Workarounds:** Never attempt to fix test issues by monkey-patching, using `eval` to inject code, or modifying unrelated production code to bypass a test flaw.
   - **Threshold for Halting:** After one failed attempt to pass a test that indicates a missing dependency or structural flaw in the test suite, halt and request test correction.
6. Stop when all tests pass

**Checklist:**

- [ ] All types defined with correct fields and type parameters
- [ ] All constructors implemented per contract
- [ ] All public functions implemented
- [ ] Code follows AGENTS.md style (naming, conventions)
- [ ] All tests passing
- [ ] No test modifications made during implementation
- [ ] Build/format checks pass (if applicable)
- [ ] If a test fails due to a suspected bug in the test code (not the implementation), stopped and discussed rather than modifying the test or hacking a workaround
- [ ] All necessary types (like `UniformScaling`) are properly imported in test blocks if they are used

## Handoff Boundaries

**Phase 1 → 2**: After `<module>_plan.md` is written and committed.

- Root thread hands off to **new smart thread** with plan file attached
- Goal: "Write comprehensive test suite for [module] per plan in [file]"

**Phase 2 → 3**: After test suite is written (but not run).

- Test thread hands off to **new deep thread** with plan and module file with tests attached
- Goal: "Review test coverage and correctness against plan; suggest improvements"

**Phase 3 → 4** (after user approval): Root thread hands off to **new deep thread** with plan, module file with approved tests, and project context.

- Goal: "Implement [module] to pass all tests; do not modify tests"

## Example Handoff Commands

### Root → Test Writing

```
Handoff goal: Write comprehensive test suite for RBFInterpolant per plan in rbf_plan.md.
Include tests for constructors, evaluation, push!, append!, float type compatibility,
and all edge cases mentioned in the plan. Use @testitem blocks in test/rbfinterpolants.jl.
Do not write implementation code. Stop and notify when tests are ready.
```

### Test → Review

```
Handoff goal: Thoroughly review test coverage for RBFInterpolant against rbf_plan.md.
Verify all functions are tested with appropriate edge cases, error conditions, and type
coverage. Check test independence and invariant correctness. Report findings.
```

### Root → Implementation

```
Handoff goal: Implement RBFInterpolant module in src/rbfinterpolants.jl per rbf_plan.md
to pass all tests in test/rbfinterpolants.jl. Do not modify the tests. Run tests
incrementally. Stop when all tests pass.
```

## AGENTS.md Compliance

When implementing, follow the project's AGENTS.md conventions:

- Type parameterization by `F<:AbstractFloat`
- Constructor patterns: type-inference, type-enforcing, copy-converting
- `convert` and `adaptrealtype` for precision conversion
- Array type flexibility (Array, SArray, SparseMatrixCSC)
- Workspace pattern for mutable caches
- `!` suffix for in-place mutation, `!!` for polymorphic mutation
- `@testitem` for all tests; colocate tests with code after module `end`
- Descriptive ASCII names over Greek symbols
