module EquationSolvers

using LinearAlgebra: LAPACKException, SingularException
using StaticArrays: SVector

using ..Constraints: PTrace
using ..Newton: NewtonException, newton
using ..Problems: PTraceInit, PointProblem, RegularizerInit, ResolventEquation
using ..Regularizers: Regularizer, regularizertype
using ..Solutions: PointSolution
using ..Workspaces: Workspace, ptrace!, qFP!

public eqsolve!

const ConvergenceException = Union{NewtonException,SingularException,LAPACKException}

"""
    eqsolve!(ws::Workspace{F}, problem::PointProblem{F}) where {F}

Solve an averaged resolvent equation at a point in the complex plane.
"""
function eqsolve!(ws::Workspace{F}, problem::PointProblem{F}) where {F}
    return eqsolve_inner!(
        ws::Workspace{F}, problem.equation, problem.constraint, problem.z, problem.init
    )
end

function eqsolve_inner!(
    ws::Workspace{F},
    eq::ResolventEquation{F},
    constraint::Regularizer{F},
    z::Complex{F},
    init::RegularizerInit{F},
) where {F}
    fq = let ws = ws, eq = eq, reg = constraint, z = z
        function fq(qv)
            # avoid capturing q from the outer function
            local q = complex(qv...)
            dq = qFP!(ws, eq, z, q, reg) - q
            return SVector(real(dq), imag(dq))
        end
    end
    qv0 = SVector(real(init.q), imag(init.q))
    qv = eqsolve_common!(fq, qv0)
    q = complex(qv...)
    return PointSolution(eq, z, q, constraint; workspace=ws)
end

function eqsolve_inner!(
    ws::Workspace{F},
    eq::ResolventEquation{F},
    constraint::PTrace{F},
    z::Complex{F},
    init::PTraceInit{F},
) where {F}
    reg_type = regularizertype(init.reg)
    fqreg = let ws = ws, eq = eq, ptrace = constraint.value, z = z, reg_type = reg_type
        function fqreg(qregv)
            # avoid capturing reg and q from the outer function
            local reg, q = Regularizer(qregv[1], reg_type), complex(qregv[2], qregv[3])
            dptrace = ptrace!(ws, eq, z, q, reg) - ptrace
            dq = qFP!(ws, eq, z, q, reg) - q
            return SVector(dptrace, real(dq), imag(dq))
        end
    end
    qregv0 = SVector(init.reg.value, real(init.q), imag(init.q))
    qregv = eqsolve_common!(fqreg, qregv0)
    reg, q = Regularizer(qregv[1], reg_type), complex(qregv[2], qregv[3])
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
        PTrace, PTraceInit, PointProblem, RegularizerInit, ResolventEquation
    using GaussianSpectralDensities.Regularizers: Regularizer
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace, qFP!

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    Adata = [1 0.1im; 0 -1]
    Ddata = [0.75, 0.25cispi(1 // 3)]
    z = complex(0.25)
    (; q0) = Defaults
    reg_value = 0.5
    ptrace = PTrace(0.01)
    sscale = 2 // 3

    @testset "Regularizer type = $reg_type" for reg_type in [:shift, :gap]
        reg = Regularizer(reg_value, reg_type)
        reg_init = RegularizerInit(q0)
        ptrace_init = PTraceInit(q0, reg)

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

                    @testset "$(nameof(typeof(constraint)))" for (constraint, init) in [
                        (reg, reg_init), (ptrace, ptrace_init)
                    ]
                        problemF = PointProblem(eqF, constraint, z, init)
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
        PTrace, PTraceInit, PointProblem, RegularizerInit, ResolventEquation
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
    init = PTraceInit(q0, zerogap)

    @testset "$(nameof(typeof(constraint)))" for constraint in [zeroshift, PTrace(ptrace)]
        initF = constraint isa Regularizer ? RegularizerInit(q0) : init
        problem = PointProblem(eq, constraint, z, initF)
        sol = eqsolve!(ws, problem)

        @test sol.equation == eq
        @test sol.z == z
        @test !sol.success
    end
end
