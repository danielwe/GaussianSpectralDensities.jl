module EquationSolvers

using LinearAlgebra: LAPACKException, SingularException
using StaticArrays: SVector

using ..Newton: NewtonException, newton
using ..Problems: LocalRegularizerProblem, LocalTraceProblem
using ..Regularizers: Regularizer
using ..Solutions: LocalResolventSolution
using ..Workspaces: E!, Workspace, gap!, k!, q_rhs!, shift!

public eqsolve!

const ConvergenceException = Union{NewtonException,SingularException,LAPACKException}

"""
    eqsolve!(ws::Workspace{F}, problem::LocalRegularizerProblem{F}) where {F}
    eqsolve!(ws::Workspace{F}, problem::LocalTraceProblem{F}) where {F}

Solve an averaged resolvent equation at a point in the complex plane

Returns a `LocalResolventSolution`.
"""
function eqsolve! end

function eqsolve!(ws::Workspace{F}, problem::LocalRegularizerProblem{F}) where {F}
    fq = let ws = ws, qscale = problem.qscale, eq = problem.eq, reg = problem.reg
        function fq(qv)
            # avoid capturing q and reg from the outer function
            local q = complex(qv...)
            dq = qscale * q_rhs!(ws, eq, qscale * q, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
    q0 = problem.init.q
    q0v = SVector(real(q0), imag(q0))
    local qv, niters, success
    try
        qv, niters = newton(fq, q0v)
        success = true
    catch e
        (e isa ConvergenceException) || rethrow()
        qv, niters = q0v, -1
        success = false
    end
    # TODO: use logging instead
    # print_niters(niters, verbosity)
    q, reg = complex(qv...), problem.reg
    k = k!(ws, problem.eq, problem.qscale * q, reg)
    return make_resolvent_solution(ws, problem, success, q, reg, k)
end

function eqsolve!(ws::Workspace{F}, problem::LocalTraceProblem{F}) where {F}
    reg0, q0 = problem.init.reg, problem.init.q
    fqreg =
        let ws = ws,
            type = reg0.type,
            qscale = problem.qscale,
            eq = problem.eq,
            k = problem.k

            function fqreg(qregv)
                # avoid capturing q and reg from the outer function
                local q = complex(qregv[1], qregv[2])
                local reg = Regularizer(qregv[3], type)
                q_scaled = qscale * q
                dq = qscale * q_rhs!(ws, eq, q_scaled, reg) - q
                dk = k!(ws, eq, q_scaled, reg) - k
                return SVector(real(dq), imag(dq), dk)
            end
        end
    qreg0v = SVector(real(q0), imag(q0), reg0.value)
    local qregv, niters, success
    try
        qregv, niters = newton(fqreg, qreg0v)
        success = true
    catch e
        (e isa ConvergenceException) || rethrow()
        qregv, niters = qreg0v, -1
        success = false
    end
    # TODO: use logging instead
    # print_niters(niters, verbosity)
    q, reg = complex(qregv[1], qregv[2]), Regularizer(qregv[3], reg0.type)
    return make_resolvent_solution(ws, problem, success, q, reg, problem.k)
end

function make_resolvent_solution(ws, problem, success, q, reg, k)
    q_scaled = problem.qscale * q
    shift = shift!(ws, problem.eq, q_scaled, reg)
    gap = gap!(ws, problem.eq, q_scaled, reg)
    E = E!(ws, problem.eq, q_scaled, reg)
    return LocalResolventSolution(problem, q, reg, shift, gap, k, E, success)
end

## Attic
## =====
# function qj!(
#     problem::LocalSpectralProblem{NormalizeQ, <:Any, <:Fixγ²}, q, ws
# ) where {NormalizeQ}
#     T = problem.ensemble.T
#     P = P!(problem, q, ws)
#     PQ = PQ!(problem, q, ws)
#     QᴴPQ = QᴴPQ!(problem, q, ws)
#     df_dq = trTAᴴTB(T, PQ, PQ')  # Tr[(TQᴴP)²]
#     df_dq⁺ = trTAᴴTB(T, QᴴPQ, P)  # Tr[TQᴴPQTP]
#     if NormalizeQ
#         QP = QP!(problem, q, ws)
#         QPQ = QPQ!(problem, q, ws)
#         df_dq += trTAᴴTB(T, QPQ, P)  # Tr[TQᴴPQᴴTP]
#         df_dq⁺ += trTAᴴTB(T, PQ, QP)  # Tr[TQᴴPTQP]
#         df_dq /= 2
#         df_dq⁺ /= 2
#     end
#     df_dq⁺ -= trT²A(T, P)  # Tr[T²P]
#     μ²N⁻¹ = problem.μ^2 / problem.ensemble.N
#     df_dq = muladd(df_dq, μ²N⁻¹, -one(q))
#     df_dq⁺ *= μ²N⁻¹
#     return df_dq, df_dq⁺
# end

end  # module ResolventEquationSolvers

@testitem "eqsolve!" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.EquationSolvers: eqsolve!
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: E!, Workspace, gap!, k!, q_rhs!, shift!

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    Adata = [1 0.1im; 0 -1]
    Ddata = [0.75, 0.25cispi(1 // 3)]
    z, qscale = complex(0.8), 0.5
    k, q0 = 0.01, Defaults.q0
    reg_value = 0.5

    @testset "$label" for (label, A, D) in [
        ("Array", Adata, Diagonal(Ddata)),
        ("SArray", SMatrix{2,2}(Adata), Diagonal(SVector{2}(Ddata))),
        ("SparseMatrixCSC", sparse(Adata), Diagonal(Ddata)),
    ]
        @testset "$F" for F in [Float32, Float64]
            @testset "$signature" for (signature, ensembleF) in [
                ("(I, I, I, I)", MatrixEnsemble{F}(I, I, I, I)),
                ("(A, I, I, I)", MatrixEnsemble{F}(A, I, I, I)),
                ("(I, A, I, I)", MatrixEnsemble{F}(I, A, I, I)),
                ("(I, I, A, I)", MatrixEnsemble{F}(I, I, A, I)),
                ("(I, I, I, D)", MatrixEnsemble{F}(I, I, I, D)),
            ]
                wsF = Workspace(ensembleF)
                eqF = LocalResolventEquation(ensembleF, z)

                @testset "Regularizer type = $type" for type in [:shift, :gap]
                    reg = Regularizer(reg_value, type)

                    init = (; reg, q=q0)

                    @testset "$(nameof(typeof(problemF)))" for (problemF, param) in [
                        (LocalRegularizerProblem(eqF, reg, init, qscale), :reg)
                        (LocalTraceProblem(eqF, k, init, qscale), :k)
                    ]
                        solF = eqsolve!(wsF, problemF)

                        @test solF.success
                        @test solF.problem === problemF
                        @test getproperty(solF, param) === getproperty(problemF, param)

                        pqscale = problemF.qscale
                        sol_q_scaled = pqscale * solF.q
                        q = pqscale * q_rhs!(wsF, problemF.eq, sol_q_scaled, solF.reg)
                        shift = shift!(wsF, problemF.eq, sol_q_scaled, solF.reg)
                        gap = gap!(wsF, problemF.eq, sol_q_scaled, solF.reg)
                        k = k!(wsF, problemF.eq, sol_q_scaled, solF.reg)
                        E = E!(wsF, problemF.eq, sol_q_scaled, solF.reg)

                        tol = 10eps(F)^(3 // 4)
                        @test solF.q ≈ q rtol = tol atol = tol
                        @test solF.shift ≈ shift rtol = tol atol = tol
                        @test solF.gap ≈ gap rtol = tol atol = tol
                        @test solF.k ≈ k rtol = tol atol = tol
                        @test solF.E ≈ E rtol = tol atol = tol
                    end
                end
            end
        end
    end
end

@testitem "eqsolve! errors" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.EquationSolvers: eqsolve!
    using GaussianSpectralDensities.Problems:
        LocalRegularizerProblem, LocalResolventEquation, LocalTraceProblem
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    # engineer a singular equation
    k, z, q0 = 1.0, 0.1, 0.1
    ensemble = MatrixEnsemble(0I, I, I, 0.5I)
    ws = Workspace(ensemble)
    eq = LocalResolventEquation(ensemble, z)
    zeroshift = Regularizer(0.0, :shift)
    zerogap = Regularizer(0.0, :gap)

    @testset "$(nameof(typeof(problem)))" for (problem, reg) in [
        (LocalRegularizerProblem(eq, zeroshift, (; q=q0)), zeroshift),
        (LocalTraceProblem(eq, k, (; reg=zerogap, q=q0)), zerogap),
    ]
        sol = eqsolve!(ws, problem)

        @test !sol.success
        @test sol.problem === problem
        @test sol.q === problem.init.q
        @test sol.reg === reg
    end
end
