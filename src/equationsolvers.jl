module EquationSolvers

using LinearAlgebra: LAPACKException, SingularException
using StaticArrays: SVector

using ..Newton: NewtonException, newton
using ..Problems: PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
using ..Spreads: Spread
using ..Solutions: PointSolution
using ..Workspaces: Workspace, ptrace!, shapeFP!

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
    constraint::SpreadConstraint{F},
    z::Complex{F},
) where {F}
    spread = constraint.val
    fshape = let ws = ws, eq = eq, spread = spread, z = z
        function fshape(shapev)
            local shape = complex(shapev...)
            dshape = shapeFP!(ws, eq, z, shape, spread) - shape
            return SVector(real(dshape), imag(dshape))
        end
    end
    shapev0 = SVector(real(constraint.shape0), imag(constraint.shape0))
    shapev = eqsolve_common!(fshape, shapev0)
    shape = complex(shapev...)
    # FIXME: if the solution didn't converge (niters < 0) we need to make sure the solution
    # has success = false. This is supposed to happen automatically due to the consistency
    # check in the PointSolution constructor, but maybe we should let this function enforce
    # it in case we're right on the tolerance margin and the stars somehow aligned such that
    # newton declared failure but the result is good enough for PointSolution. Or is it
    # actually more consistent to let the constructor be the sole arbiter of success? Think.
    return PointSolution(eq, z, shape, spread; workspace=ws)
end

function eqsolve_inner!(
    ws::Workspace{F},
    eq::ResolventEquation{F},
    constraint::PTraceConstraint{F},
    z::Complex{F},
) where {F}
    (; spread0) = constraint
    fshapespread =
        let ws = ws, eq = eq, ptrace = constraint.val, z = z, spread_type = spread0.type
            function fshapespread(shapespreadv)
                local shape = complex(shapespreadv[1], shapespreadv[2])
                local spread = Spread(shapespreadv[3], spread_type)
                dshape = shapeFP!(ws, eq, z, shape, spread) - shape
                dptrace = ptrace!(ws, eq, z, shape, spread) - ptrace
                return SVector(real(dshape), imag(dshape), dptrace)
            end
        end
    shapespreadv0 = SVector(real(constraint.shape0), imag(constraint.shape0), spread0.val)
    shapespreadv = eqsolve_common!(fshapespread, shapespreadv0)
    shape = complex(shapespreadv[1], shapespreadv[2])
    spread = Spread(shapespreadv[3], spread0.type)
    # FIXME: see above.
    return PointSolution(eq, z, shape, spread; workspace=ws)
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
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: solves
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: Diagonal, I
    using SparseArrays: sparse
    using StaticArrays: SMatrix, SVector

    Adata = [1 0.1im; 0 -1]
    Ddata = [0.75, 0.25cispi(1 // 3)]
    z = complex(0.25)
    (; shape0) = Defaults
    spread = 0.5
    sscale = 2 // 3

    @testset "Spread type = $spread_type" for spread_type in [:shift, :gap]
        spreadc = SpreadConstraint(; spread_type => spread, shape0)
        ptracec = PTraceConstraint(0.01; shape0, Symbol(spread_type, "0") => spread)

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
                                                                 [spreadc, ptracec]
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
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    # engineer a singular equation
    z = complex(0.1)
    shape0 = complex(0.1)
    ensemble = MatrixEnsemble(0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    ws = Workspace(eq)

    spreadc = SpreadConstraint(; shift=0.0, shape0)
    ptracec = PTraceConstraint(1.0; shape0, gap0=0.0)

    @testset "$(nameof(typeof(constraint)))" for constraint in [spreadc, ptracec]
        problem = PointProblem(eq, constraint, z)
        sol = eqsolve!(ws, problem)

        @test sol.equation == eq
        @test sol.z == z
        @test !issuccess(sol)
    end
end
