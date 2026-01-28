module Solutions

using ..Problems: ResolventEquation
using ..Workspaces: Workspace, gap!, greens!, ptrace!, shift!

using ..ArrayTools: ArrayTools

public ResolventSolution

"""
    ResolventSolution(eq::ResolventEquation q, reg; workspace=Workspace(eq.ensemble))
    ResolventSolution{F}(eq::ResolventEquation q, reg, workspace::Workspace{F})

Represent a solution of an averaged resolvent equation at a point in the complex plane

The constructor does not verify that the given values solve for a particular objective. Use
the function `eqsolve!` for that. It does, however, calculate self-consistent values for all
the derived quantities contained in the struct.

# Arguments

  - `eq`: the resolvent equation
  - `q`: value of the parameter ``q``
  - `reg` value of the regularizer
  - `workspace`: workspace used to compute derived quantities
"""
struct ResolventSolution{F,E<:ResolventEquation{F}}
    eq::E
    q::Complex{F}
    shift::F
    gap::F
    ptrace::F
    greens::Complex{F}
    success::Bool

    function common(::Type{F}, eq, q, shift, gap, trace, greens, success) where {F}
        eqF = convert(ResolventEquation{F}, eq)
        return new{F,typeof(eqF)}(eqF, q, shift, gap, trace, greens, success)
    end

    function ResolventSolution{F}(eq, q, reg, workspace) where {F}
        shift = shift!(workspace, eq, q, reg)
        gap = gap!(workspace, eq, q, reg)
        ptrace = ptrace!(workspace, eq, q, reg)
        greens = greens!(workspace, eq, q, reg)
        success = all(isfinite, (shift, gap, ptrace, greens)) && (shift >= eq.shiftmin)
        return common(F, eq, q, shift, gap, ptrace, greens, success)
    end

    function ResolventSolution{F}(s::ResolventSolution) where {F}
        return common(F, s.eq, s.q, s.shift, s.gap, s.ptrace, s.greens, s.success)
    end
end

function ResolventSolution(
    eq::ResolventEquation{F}, q, reg; workspace=Workspace(eq.ensemble)
) where {F}
    return ResolventSolution{F}(eq, q, reg, workspace)
end

## Conversions
## ===========
function Base.convert(::Type{P}, s::ResolventSolution) where {F,P<:ResolventSolution{F}}
    return (s isa P) ? s : ResolventSolution{F}(s)
end

function ArrayTools.adaptrealtype(::Type{F}, eq::ResolventSolution) where {F}
    return convert(ResolventSolution{F}, eq)
end

end  # module Solutions

@testitem "Solutions" begin
    using GaussianSpectralDensities.ArrayTools: adaptrealtype
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: ResolventSolution
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    ws64 = Workspace(ensemble64)
    z64 = 0.0im
    sscale = 2 // 3
    eq64 = ResolventEquation(ensemble64, z64; sscale)
    reg64 = Regularizer(0.5, :gap)
    q64 = complex(eps(), eps())

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I / 2)
        wsF = Workspace(ensembleF)
        zF = F(z64)
        eqF = ResolventEquation(ensembleF, zF; sscale)
        regF = Regularizer(F(reg64.value), reg64.type)
        qF = Complex{F}(q64)

        solution64 = ResolventSolution(eq64, q64, reg64)

        # Traits
        @test solution64 isa ResolventSolution{Float64}
        @test solution64.success

        # Floattype inferring constructor
        solutionF = ResolventSolution(eqF, qF, regF)
        @test solutionF isa ResolventSolution{F}

        # Declaring failure
        @test !ResolventSolution(eqF, oftype(qF, NaN), regF).success

        eq_impossibleF = ResolventEquation(ensembleF, zF; sscale, shiftmin=Inf)
        solution_impossibleF = ResolventSolution(eq_impossibleF, qF, regF)
        @test !solution_impossibleF.success

        # Floattype enforcing constructor
        solutionF_64 = ResolventSolution{Float64}(eqF, qF, regF, wsF)
        solution64_F = ResolventSolution{F}(eq64, q64, reg64, ws64)
        @test solutionF_64 isa ResolventSolution{Float64}
        @test solution64_F isa ResolventSolution{F}

        # Declaring failure
        @test !ResolventSolution{Float64}(eqF, oftype(qF, NaN), regF, wsF).success
        @test !ResolventSolution{F}(eq64, oftype(q64, NaN), reg64, ws64).success

        # Floattype conversions
        @test ResolventSolution{F}(solution64) === solutionF
        @test ResolventSolution{Float64}(solutionF) === solutionF_64
        @test convert(ResolventSolution{F}, solution64) === solutionF
        @test convert(ResolventSolution{Float64}, solutionF) === solutionF_64
        @test adaptrealtype(F, solution64) === solutionF
        @test adaptrealtype(Float64, solutionF) === solutionF_64
    end
end
