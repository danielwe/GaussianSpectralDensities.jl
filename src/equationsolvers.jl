module EquationSolvers

using LinearAlgebra: LAPACKException, SingularException
using StaticArrays: SVector

using ..Newton: NewtonException, newton
using ..Problems: PTraceProblem, RegularizerProblem
using ..Regularizers: Regularizer
using ..Solutions: PointSolution
using ..Workspaces: Workspace, ptrace!, qFP!

public eqsolve!

const ConvergenceException = Union{NewtonException,SingularException,LAPACKException}

"""
    eqsolve!(ws::Workspace{F}, problem::RegularizerProblem{F}) where {F}
    eqsolve!(ws::Workspace{F}, problem::PTraceProblem{F}) where {F}

Solve an averaged resolvent equation at a point in the complex plane

Returns a `PointSolution`.
"""
function eqsolve! end

function eqsolve!(ws::Workspace{F}, problem::RegularizerProblem{F}) where {F}
    (; z, sscale, reg) = problem
    fq = let ws = ws, z = z, sscale = sscale, reg = reg
        function fq(qv)
            # avoid capturing q from the outer function
            local q = complex(qv...)
            qscaled = sscale * q
            dq = sscale * qFP!(ws, z, qscaled, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
    q0 = problem.init.q
    q0v = SVector(real(q0), imag(q0))
    # NOTE: niters is unused at the moment, but will be used for logging, so don't remove
    qv, niters = try
        qv, niters = newton(fq, q0v)
    catch e
        (e isa ConvergenceException) || rethrow()
        toNaN.(q0v), -1
    end
    # FIXME: use logging instead
    # print_niters(niters, verbosity)
    q = complex(qv...)
    return PointSolution(problem, q, reg; workspace=ws)
end

function eqsolve!(ws::Workspace{F}, problem::PTraceProblem{F}) where {F}
    (; z, sscale, ptrace) = problem
    reg0 = problem.init.reg
    fqreg = let ws = ws, z = z, sscale = sscale, reg_type = reg0.type, ptrace = ptrace
        function fqreg(qregv)
            # avoid capturing reg and q from the outer function
            local reg, q = Regularizer(qregv[1], reg_type), complex(qregv[2], qregv[3])
            qscaled = sscale * q
            dptrace = ptrace!(ws, z, qscaled, reg) - ptrace
            dq = sscale * qFP!(ws, z, qscaled, reg) - q
            return SVector(dptrace, real(dq), imag(dq))
        end
    end
    q0 = problem.init.q
    qreg0v = SVector(reg0.value, real(q0), imag(q0))
    # NOTE: niters is unused at the moment, but will be used for logging, so don't remove
    qregv, niters = try
        newton(fqreg, qreg0v)
    catch e
        (e isa ConvergenceException) || rethrow()
        toNaN.(qreg0v), -1
    end
    # FIXME: use logging instead
    # print_niters(niters, verbosity)
    reg, q = Regularizer(qregv[1], reg0.type), complex(qregv[2], qregv[3])
    return PointSolution(problem, q, reg; workspace=ws)
end

toNaN(::F) where {F<:AbstractFloat} = F(NaN)

end  # module EquationSolvers

@testitem "eqsolve!" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.EquationSolvers: eqsolve!
    using GaussianSpectralDensities.Problems:
        PTraceProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace, qFP!

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    Adata = [1 0.1im; 0 -1]
    Ddata = [0.75, 0.25cispi(1 // 3)]
    z, sscale = complex(0.8), 2 // 3
    ptrace, q0 = 0.01, Defaults.q0
    reg_value = 0.5

    @testset "$label" for (label, A, D) in [
        ("Array", Adata, Diagonal(Ddata)),
        ("SArray", SMatrix{2,2}(Adata), Diagonal(SVector{2}(Ddata))),
        ("SparseMatrixCSC", sparse(Adata), Diagonal(Ddata)),
    ]
        @testset "$F" for F in [Float32, Float64]
            @testset "$signature" for (signature, ensembleF) in [
                ("(I, I, I, I / 2)", MatrixEnsemble{F}(I, I, I, I / 2)),
                ("(A, I, I, I / 2)", MatrixEnsemble{F}(A, I, I, I / 2)),
                ("(I, A, I, I / 2)", MatrixEnsemble{F}(I, A, I, I / 2)),
                ("(I, I, A, I / 2)", MatrixEnsemble{F}(I, I, A, I / 2)),
                ("(I, I, I, D)", MatrixEnsemble{F}(I, I, I, D)),
            ]
                eqF = ResolventEquation(ensembleF)
                wsF = Workspace(eqF)

                @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
                    reg = Regularizer(reg_value, reg_type)

                    init = (; reg, q=q0)

                    @testset "$(nameof(P))" for (P, obj) in [
                        (RegularizerProblem, reg), (PTraceProblem, ptrace)
                    ]
                        problemF = P(eqF, z, obj, init; sscale)
                        solF = eqsolve!(wsF, problemF)

                        @test solF.success
                        @test solF.z == problemF.z
                        @test solF.sscale == problemF.sscale

                        solF_reg = Regularizer(getproperty(solF, reg_type), reg_type)

                        tol = 10eps(F)^(3 // 4)
                        if problemF isa RegularizerProblem
                            @test solF_reg.value === F(reg.value)
                        else
                            @test solF.ptrace ≈ problemF.ptrace rtol=tol atol=tol
                        end
                        qFP = sscale * qFP!(wsF, z, sscale * solF.q, solF_reg)
                        @test solF.q ≈ qFP rtol=tol atol=tol
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
        PTraceProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    # engineer a singular equation
    ptrace, z, q0 = 1.0, 0.1, 0.1
    ensemble = MatrixEnsemble(0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)
    zeroshift = Regularizer(0.0, :shift)
    zerogap = Regularizer(0.0, :gap)

    @testset "$(nameof(typeof(problem)))" for (problem, reg) in [
        (RegularizerProblem(eq, z, zeroshift, (; q=q0)), zeroshift),
        (PTraceProblem(eq, z, ptrace, (; reg=zerogap, q=q0)), zerogap),
    ]
        sol = eqsolve!(ws, problem)

        @test sol.z == problem.z
        @test !sol.success
    end
end
