module EquationSolvers

using LinearAlgebra: LAPACKException, SingularException
using StaticArrays: SVector

using ..Newton: NewtonException, newton
using ..Problems: PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
using ..Regularizers: Regularizer
using ..Solutions: PointSolution
using ..Workspaces: Workspace, ptrace!, qFP!

public eqsolve!

const ConvergenceException = Union{NewtonException,SingularException,LAPACKException}

"""
    eqsolve!(ws::Workspace{F}, problem::PointProblem{F}) where {F}

Solve an averaged resolvent equation at a point in the complex plane.
"""
function eqsolve!(ws::Workspace{F}, problem::PointProblem{F}) where {F}
    return eqsolve_inner!(ws::Workspace{F}, problem.equation, problem.constraint, problem.z)
end

function eqsolve_inner!(
    ws::Workspace{F},
    eq::ResolventEquation{F},
    constraint::RegularizerConstraint{F},
    z::Complex{F},
) where {F}
    reg = constraint.val
    fq = let ws = ws, eq = eq, reg = reg, z = z
        function fq(qv)
            local q = complex(qv...)
            dq = qFP!(ws, eq, z, q, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
    qv0 = SVector(real(constraint.q0), imag(constraint.q0))
    qv = eqsolve_common!(fq, qv0)
    q = complex(qv...)
    return PointSolution(eq, z, q, reg; workspace=ws)
end

function eqsolve_inner!(
    ws::Workspace{F},
    eq::ResolventEquation{F},
    constraint::PTraceConstraint{F},
    z::Complex{F},
) where {F}
    (; reg0) = constraint
    fqreg = let ws = ws, eq = eq, ptrace = constraint.val, z = z, reg_type = reg0.type
        function fqreg(qregv)
            local q, reg = complex(qregv[1], qregv[2]), Regularizer(qregv[3], reg_type)
            dq = qFP!(ws, eq, z, q, reg) - q
            dptrace = ptrace!(ws, eq, z, q, reg) - ptrace
            return SVector(real(dq), imag(dq), dptrace)
        end
    end
    qregv0 = SVector(real(constraint.q0), imag(constraint.q0), reg0.val)
    qregv = eqsolve_common!(fqreg, qregv0)
    q, reg = complex(qregv[1], qregv[2]), Regularizer(qregv[3], reg0.type)
    return PointSolution(eq, z, q, reg; workspace=ws)
end

function eqsolve_common!(f, v0)
    # NOTE: niters is unused at the moment, but will be used for logging, so don't remove
    v, niters = try
        newton(f, v0)
    catch e
        (e isa ConvergenceException) || rethrow()
        oftype.(v0, NaN), -1
    end
    # FIXME: use logging instead
    # print_niters(niters, verbosity)
    return v
end

end  # module EquationSolvers

@testitem "eqsolve!" begin
    using GaussianSpectralDensities.Defaults: Defaults
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.EquationSolvers: eqsolve!
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    Adata = [1 0.1im; 0 -1]
    Ddata = [0.75, 0.25cispi(1 // 3)]
    z = complex(0.25)
    (; q0) = Defaults
    reg_val = 0.5
    sscale = 2 // 3

    @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
        regc_kwargs = (; reg_type => reg_val, :q0 => q0)
        regc = RegularizerConstraint(; regc_kwargs...)
        ptracec_kwargs = (; Symbol(reg_type, :0) => reg_val, :q0 => q0)
        ptracec = PTraceConstraint(0.01; ptracec_kwargs...)

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

                    @testset "$(nameof(typeof(constraint)))" for constraint in
                                                                 [regc, ptracec]
                        problemF = PointProblem(eqF, constraint, z)
                        solF = eqsolve!(wsF, problemF)
                        @test solves(solF, problemF)
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
        PTraceConstraint, PointProblem, RegularizerConstraint, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    # engineer a singular equation
    z = complex(0.1)
    q0 = complex(0.1)
    ensemble = MatrixEnsemble(0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)

    regc = RegularizerConstraint(; shift=0.0, q0)
    ptracec = PTraceConstraint(1.0; q0, gap0=0.0)

    @testset "$(nameof(typeof(constraint)))" for constraint in [regc, ptracec]
        problem = PointProblem(eq, constraint, z)
        sol = eqsolve!(ws, problem)

        @test sol.equation == eq
        @test sol.z == z
        @test !issuccess(sol)
    end
end
