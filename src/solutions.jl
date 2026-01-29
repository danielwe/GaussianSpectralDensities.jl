module Solutions

using ..Problems: AbstractResolventProblem
using ..Workspaces: Workspace, gap!, greens!, ptrace!, shift!

using ..ArrayTools: ArrayTools

public PointSolution

"""
    PointSolution(problem::AbstractResolventProblem, z, q, reg; workspace=Workspace(problem.eq))
    PointSolution{F}(problem, z, q, reg, workspace)
    PointSolution{F}(s::PointSolution) where {F}

Represent a solution of an averaged resolvent equation at a point in the complex plane

This struct holds only point-specific data suitable for storage in a StructArray trace.
The shared context (ensemble, shiftmin, normalizeQ) is stored separately in the problem/solver.

# Fields

  - `z`: point in the complex plane
  - `q`: value of the parameter ``q``
  - `shift`: regularizer shift ``γ²``
  - `gap`: regularized gap ``δ² = eigmin(QQ†) + γ²``
  - `ptrace`: normalized trace of P
  - `greens`: Green's function value
  - `sscale`: symmetry scale factor used for this solution
  - `success`: whether the solution is valid
"""
struct PointSolution{F<:AbstractFloat}
    z::Complex{F}
    q::Complex{F}
    shift::F
    gap::F
    ptrace::F
    greens::Complex{F}
    sscale::F
    success::Bool

    # Undocumented constructor for StructArrays compatibility. Does not assert invariants.
    function PointSolution{F}(z, q, shift, gap, ptrace, greens, sscale, success) where {F}
        return new{F}(z, q, shift, gap, ptrace, greens, sscale, success)
    end

    function PointSolution{F}(problem, z, q, reg, workspace) where {F}
        (; eq) = problem
        shift = shift!(workspace, eq, z, q, reg)
        gap = gap!(workspace, eq, z, q, reg)
        ptrace = ptrace!(workspace, eq, z, q, reg)
        greens = greens!(workspace, eq, z, q, reg)
        success = all(isfinite, (shift, gap, ptrace, greens)) && (shift >= eq.shiftmin)
        return PointSolution{F}(z, q, shift, gap, ptrace, greens, eq.sscale, success)
    end

    function PointSolution{F}(s::PointSolution) where {F}
        return new{F}(s.z, s.q, s.shift, s.gap, s.ptrace, s.greens, s.sscale, s.success)
    end
end

function PointSolution(
    problem::AbstractResolventProblem{F}, z, q, reg; workspace=Workspace(problem.eq)
) where {F}
    return PointSolution{F}(problem, z, q, reg, workspace)
end

## Conversions
## ===========
function Base.convert(::Type{PointSolution{F}}, s::PointSolution) where {F}
    return (s isa PointSolution{F}) ? s : PointSolution{F}(s)
end

function ArrayTools.adaptrealtype(::Type{F}, s::PointSolution) where {F}
    return convert(PointSolution{F}, s)
end

end  # module Solutions

@testitem "PointSolution" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: PTraceProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: PointSolution
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    sscale64 = 2 / 3
    eq64 = ResolventEquation(ensemble64; sscale=sscale64)
    ws64 = Workspace(eq64)
    z64 = 0.5 + 0.5im
    reg64 = Regularizer(0.5, :gap)
    ptrace64 = 1.0
    q64 = complex(eps(), eps())
    problem64 = PTraceProblem(eq64, ptrace64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        sscaleF = F(sscale64)
        eqF = ResolventEquation(ensembleF; sscale=sscaleF)
        wsF = Workspace(eqF)
        zF = Complex{F}(z64)
        regF = Regularizer(F(reg64.value), reg64.type)
        ptraceF = F(ptrace64)
        qF = Complex{F}(q64)
        problemF = PTraceProblem(eqF, ptraceF)

        solution64 = PointSolution(problem64, z64, q64, reg64; workspace=ws64)

        # Traits
        @test solution64 isa PointSolution{Float64}
        @test solution64.success
        @test solution64.z == z64
        @test solution64.sscale == sscale64
        @test solution64.q == q64

        # Problem-based constructor
        solutionF = PointSolution(problemF, zF, qF, regF; workspace=wsF)
        @test solutionF isa PointSolution{F}
        @test solutionF.z == zF
        @test solutionF.sscale == sscaleF

        # Declaring failure via NaN
        @test !PointSolution(problemF, zF, oftype(qF, NaN), regF; workspace=wsF).success

        # Declaring failure via shiftmin
        eq_impossibleF = ResolventEquation(ensembleF; shiftmin=Inf, sscale=sscaleF)
        problem_impossibleF = PTraceProblem(eq_impossibleF, ptraceF)
        solution_impossibleF = PointSolution(problem_impossibleF, zF, qF, regF)
        @test !solution_impossibleF.success

        # Secret raw value constructor
        solutionF_raw = PointSolution{F}(
            zF,
            qF,
            solutionF.shift,
            solutionF.gap,
            solutionF.ptrace,
            solutionF.greens,
            sscaleF,
            true,
        )
        @test solutionF_raw isa PointSolution{F}
        @test solutionF_raw === solutionF

        # Floattype conversions
        solutionF_from64 = PointSolution{F}(solution64)
        solution64_fromF = PointSolution{Float64}(solutionF)
        @test solutionF_from64 isa PointSolution{F}
        @test solution64_fromF isa PointSolution{Float64}
        @test convert(PointSolution{F}, solution64) == solutionF_from64
        @test convert(PointSolution{Float64}, solutionF) == solution64_fromF
        @test adaptrealtype(F, solution64) == solutionF_from64
        @test adaptrealtype(Float64, solutionF) == solution64_fromF
    end
end
