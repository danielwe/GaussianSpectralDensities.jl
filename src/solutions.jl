module Solutions

using ..Problems: AbstractLocalResolventProblem
using ..Regularizers: Regularizer

public LocalResolventSolution

"""
    LocalResolventSolution(
        problem::AbstractLocalResolventProblem, q, shift, gap, k, E, success
    )
    LocalResolventSolution{F}(
        problem::AbstractLocalResolventProblem, q, shift, gap, k, E, success
    )

Represent a solution of an averaged resolvent equation at a point in the complex plane

The constructor does not verify that the parameters form a consistent solution of the given
problem. Use the function `eqsolve!` to obtain a consistent `LocalResolventSolution`.

# Arguments

  - `problem`: the averaged resolvent equation problem
  - `q`: value of the self-consistently determined parameter ``q``
  - `shift`: value of the regularizer shift ``γ²``
  - `gap`: value of the regularized gap `eigmin(QQ†) + γ²`
  - `k`: value of the normalized trace ``\\tr[(QQ^† + γ²I)⁻¹] / N``.
  - `E`: value of the normalized trace of the resolvent ``\\tr[Q^†(QQ^† + γ²I)⁻¹] / N``
  - `success`: whether the solution was successful
"""
struct LocalResolventSolution{F,P<:AbstractLocalResolventProblem{F}}
    problem::P
    q::Complex{F}
    reg::Regularizer{F}
    shift::F
    gap::F
    k::F
    E::Complex{F}
    success::Bool

    function LocalResolventSolution{F}(
        problem::P, q, reg, shift, gap, k, E, success
    ) where {F,P}
        problemF = convert(AbstractLocalResolventProblem{F}, problem)
        return new{F,typeof(problemF)}(problemF, q, reg, shift, gap, k, E, success)
    end
end

function LocalResolventSolution(
    problem::AbstractLocalResolventProblem{F}, q, reg, shift, gap, k, E, success
) where {F}
    return LocalResolventSolution{F}(problem, q, reg, shift, gap, k, E, success)
end

function LocalResolventSolution{F}(s::LocalResolventSolution) where {F}
    sF = LocalResolventSolution{F}(
        s.problem, s.q, s.reg, s.shift, s.gap, s.k, s.E, s.success
    )
    return sF
end

function Base.convert(
    ::Type{P}, s::LocalResolventSolution
) where {F,P<:LocalResolventSolution{F}}
    return (s isa P) ? s : LocalResolventSolution{F}(s)
end

end  # module Solutions

# TODO: consider adding accessor methods instead of letting consumers rely on propery access

@testitem "Solutions" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: LocalResolventSolution

    using LinearAlgebra: I

    ensemble64, z64 = MatrixEnsemble(I, I, I, I), 0.0im
    eq64 = LocalResolventEquation(ensemble64, z64)
    reg64 = Regularizer(0.5, :gap)
    k64, q0_64, qscale64, E64 = 1.0, complex(eps(), eps()), 1.0, 0.5 - im
    init64 = (; reg=reg64, q=q0_64)

    @testset "$F" for F in [Float32, Float64]
        ensembleF = MatrixEnsemble{F}(I, I, I, I)
        zF = F(z64)
        eqF = LocalResolventEquation(ensembleF, zF)
        regF = Regularizer(F(reg64.value), reg64.type)
        kF, qscaleF = F.((k64, qscale64))
        q0_F, EF = Complex{F}.((q0_64, E64))
        initF = (; reg=regF, q=q0_F)

        @testset "$(nameof(P))" for (P, param64, paramF) in [
            (LocalRegularizerProblem, reg64, regF), (LocalTraceProblem, k64, kF)
        ]
            problem64 = P(eq64, param64, init64, qscale64)

            solution64 = LocalResolventSolution(
                problem64, q0_64, reg64, reg64.value, reg64.value, k64, E64, true
            )

            # Traits
            @test solution64 isa LocalResolventSolution{Float64}

            problemF = P(eqF, paramF, initF, qscaleF)

            solutionF = LocalResolventSolution(
                problemF, q0_F, regF, regF.value, regF.value, kF, EF, true
            )

            # Floattype inferring constructor
            @test solutionF isa LocalResolventSolution{F}

            # Floattype enforcing constructor
            @test LocalResolventSolution{F}(
                problem64, q0_64, reg64, reg64.value, reg64.value, k64, E64, true
            ) === solutionF
            @test LocalResolventSolution{Float64}(
                problemF, q0_F, regF, regF.value, regF.value, kF, EF, true
            ) === solution64

            # Floattype conversions
            @test convert(LocalResolventSolution{F}, solution64) === solutionF
            @test convert(LocalResolventSolution{Float64}, solutionF) === solution64
            @test LocalResolventSolution{F}(solution64) === solutionF
            @test LocalResolventSolution{Float64}(solutionF) === solution64
        end
    end
end
