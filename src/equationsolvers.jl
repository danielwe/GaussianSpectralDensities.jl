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
    eqsolve!(ws::Workspace{F}, problem::RegularizerProblem{F}, z, init) where {F}
    eqsolve!(ws::Workspace{F}, problem::PTraceProblem{F}, z, init) where {F}

Solve an averaged resolvent equation at a point in the complex plane

# Arguments

  - `ws`: workspace
  - `problem`: problem specification with equation and objective
  - `z`: complex point in the plane
  - `init`: initialization NamedTuple with fields `:q` (for both problem types) and `:reg`
    (for PTraceProblem)

Returns a `PointSolution`.
"""
function eqsolve! end

function eqsolve!(ws::Workspace{F}, problem::RegularizerProblem{F}, z, init) where {F}
    zF = Complex{F}(z)
    q0 = Complex{F}(init.q)
    q0v = SVector(real(q0), imag(q0))

    fq = let ws = ws, eq=problem.eq, z = zF, reg = problem.reg
        function fq(qv)
            # avoid capturing q from the outer function
            local q = complex(qv...)
            dq = qFP!(ws, eq, z, q, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
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
    return PointSolution(problem, zF, q, problem.reg; workspace=ws)
end

function eqsolve!(ws::Workspace{F}, problem::PTraceProblem{F}, z, init) where {F}
    zF = Complex{F}(z)
    reg0, q0 = Regularizer{F}(init.reg), Complex{F}(init.q)
    qreg0v = SVector(reg0.value, real(q0), imag(q0))

    fqreg =
        let ws = ws, eq=problem.eq, z = zF, reg_type = reg0.type, ptrace = problem.ptrace
            function fqreg(qregv)
                # avoid capturing reg and q from the outer function
                local reg, q = Regularizer(qregv[1], reg_type), complex(qregv[2], qregv[3])
                dptrace = ptrace!(ws, eq, z, q, reg) - ptrace
                dq = qFP!(ws, eq, z, q, reg) - q
                return SVector(dptrace, real(dq), imag(dq))
            end
        end
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
    return PointSolution(problem, zF, q, reg; workspace=ws)
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
    ptrace = 0.01
    q0 = Defaults.q0
    reg_value = 0.5

    @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
        reg = Regularizer(reg_value, reg_type)
        init = (; reg, q=q0)

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
                    eqF = ResolventEquation(ensembleF; sscale)
                    wsF = Workspace(eqF)

                    @testset "$(nameof(P))" for (P, obj) in [
                        (RegularizerProblem, reg), (PTraceProblem, ptrace)
                    ]
                        problemF = P(eqF, obj)
                        solF = eqsolve!(wsF, problemF, z, init)

                        @test solF.success
                        @test solF.z == Complex{F}(z)
                        @test solF.sscale == eqF.sscale

                        solF_reg = Regularizer(getproperty(solF, reg_type), reg_type)

                        tol = 10eps(F)^(3 // 4)
                        if problemF isa RegularizerProblem
                            @test solF_reg.value ≈ reg.value rtol=tol atol=tol
                        else
                            @test solF.ptrace ≈ ptrace rtol=tol atol=tol
                        end
                        qFPF = qFP!(wsF, problemF.eq, solF.z, solF.q, solF_reg)
                        @test solF.q ≈ qFPF rtol=tol atol=tol
                    end
                end
            end
        end
    end
end

@testitem "eqsolve! failure" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.EquationSolvers: eqsolve!
    using GaussianSpectralDensities.Problems:
        PTraceProblem, RegularizerProblem, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    # engineer a singular equation
    z = complex(0.1)
    q0 = complex(0.1)
    ptrace = 1.0
    ensemble = MatrixEnsemble(0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)
    zeroshift = Regularizer(0.0, :shift)
    zerogap = Regularizer(0.0, :gap)
    init = (; reg=zerogap, q=q0)

    @testset "$(nameof(P))" for (P, obj) in
                                [(RegularizerProblem, zeroshift), (PTraceProblem, ptrace)]
        problem = P(eq, obj)
        sol = eqsolve!(ws, problem, z, init)

        @test sol.z == z
        @test !sol.success
    end
end
