module Solutions

using TestItems: @testitem

using ..Problems: AbstractLocalResolventProblem
using ..Regularizers: Regularizer

using ..Public: @public

@public LocalResolventSolution

"""
    LocalResolventSolution(problem::AbstractLocalResolventProblem, q, shift, gap, k, E)
    LocalResolventSolution{F}(problem::AbstractLocalResolventProblem, q, shift, gap, k, E)

Represent a solution of an averaged resolvent equation at a point in the complex plane

The constructor does not verify that the parameters form a consistent solution of the given
problem. Use the function `eqsolve!` to obtain a consistent `LocalResolventSolution`.

# Arguments

- `problem`: the averaged resolvent equation problem
- `success`: whether the solution was successful
- `q`: value of the self-consistently determined parameter ``q``
- `shift`: value of the regularizer shift ``γ²``
- `gap`: value of the regularized gap `eigmin(QQ†) + γ²`
- `k`: value of the normalized trace ``\\tr[(QQ^† + γ²I)⁻¹] / N``.
- `E`: value of the normalized trace of the resolvent ``\\tr[Q^†(QQ^† + γ²I)⁻¹] / N``
"""
struct LocalResolventSolution{F,P<:AbstractLocalResolventProblem{F}}
    problem::P
    success::Bool
    q::Complex{F}
    reg::Regularizer{F}
    shift::F
    gap::F
    k::F
    E::Complex{F}

    function LocalResolventSolution{F}(
        problem::P, success, q, reg, shift, gap, k, E
    ) where {F,P}
        problemF = convert(AbstractLocalResolventProblem{F}, problem)
        return new{F,typeof(problemF)}(problemF, success, q, reg, shift, gap, k, E)
    end
end

function LocalResolventSolution(
    problem::AbstractLocalResolventProblem{F}, success, q, reg, shift, gap, k, E
) where {F}
    return LocalResolventSolution{F}(problem, success, q, reg, shift, gap, k, E)
end

function LocalResolventSolution{F}(s::LocalResolventSolution) where {F}
    sF = LocalResolventSolution{F}(s.problem, s.success, s.q, s.reg, s.shift, s.gap, s.k, s.E)
    return sF
end

function Base.convert(
    ::Type{P}, s::LocalResolventSolution
) where {F,P<:LocalResolventSolution{F}}
    return (s isa P) ? s : LocalResolventSolution{F}(s)
end

@testitem "Solutions" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems: LocalResolventEquation,
        LocalRegularizerProblem, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: LocalResolventSolution

    using LinearAlgebra: I

    ensemble64, z64, normalizeQ = MatrixEnsemble(I, I, I, I), 0.0im, false
    eq64 = LocalResolventEquation(ensemble64, z64, normalizeQ)
    reg_value64, reg_type = 0.5, :gap
    reg64 = Regularizer(reg_value64, reg_type)
    k64, q64, qscale64, E64 = 1.0, complex(eps(), eps()), 1.0, 0.5 - im

    @testset "$(nameof(P))" for (P, params64) in [
        (LocalRegularizerProblem, ()),
        (LocalTraceProblem, (k64,))
    ]
        problem64 = P(eq64, params64..., reg64, q64, qscale64)

        solution64 = LocalResolventSolution(
            problem64, true, q64, reg64, reg_value64, reg_value64, k64, E64
        )

        # Traits
        @test solution64 isa LocalResolventSolution{Float64}

        @testset "$F" for F in [Float32, Float64]
            ensembleF = MatrixEnsemble{F}(I, I, I, I)
            zF = F(z64)
            eqF = LocalResolventEquation(ensembleF, zF, normalizeQ)
            reg_valueF = F(reg_value64)
            regF = Regularizer(reg_valueF, reg_type)
            kF, qscaleF = F.((k64, qscale64))
            qF, EF = Complex{F}.((q64, E64))
            paramsF = F.(params64)
            problemF = P(eqF, paramsF..., regF, qF, qscaleF)

            solutionF = LocalResolventSolution(
                problemF, true, qF, regF, reg_valueF, reg_valueF, kF, EF
            )

            # Floattype inferring constructor
            @test solutionF isa LocalResolventSolution{F}

            # Floattype enforcing constructor
            @test LocalResolventSolution{F}(
                problem64, true, q64, reg64, reg_value64, reg_value64, k64, E64
            ) === solutionF
            @test LocalResolventSolution{Float64}(
                problemF, true, qF, regF, reg_valueF, reg_valueF, kF, EF
            ) === solution64

            # Floattype conversions
            @test convert(LocalResolventSolution{F}, solution64) === solutionF
            @test convert(LocalResolventSolution{Float64}, solutionF) === solution64
            @test LocalResolventSolution{F}(solution64) === solutionF
            @test LocalResolventSolution{Float64}(solutionF) === solution64
        end
    end
end

end  # module Solutions
