module Surrogates

using ..Algorithms: AbstractAlgorithm
using ..Problems:
    AbstractConstraint,
    PTraceConstraint,
    PointProblem,
    ResolventEquation,
    SpreadConstraint,
    update
using ..RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!
using ..Solutions: PointSolution, issuccess, shape, spreadfrom
using ..Solvers: pointsolve
using ..Spreads: Spread
using ..Workspaces: Workspace

using StaticArrays: SVector

public SurrogateModel, evaluate, interpolate, pack, unpack_args, unpack_constraint

struct SurrogateModel{
    F,
    E<:ResolventEquation{F},
    C<:AbstractConstraint{F},
    A<:AbstractAlgorithm,
    R<:RBFInterpolant{F},
    W<:Workspace{F},
}
    equation::E
    constraint::C
    algorithm::A
    rbf::R
    workspace::W
end

function pack(ps::PointSolution{F}, ::SpreadConstraint{F}) where {F}
    shape_val = shape(ps)
    return SVector{2,F}(real(shape_val), imag(shape_val))
end

function pack(ps::PointSolution{F}, constraint::PTraceConstraint{F}) where {F}
    shape_val = shape(ps)
    spread_val = spreadfrom(ps, constraint.spread0.type).val
    return SVector{3,F}(real(shape_val), imag(shape_val), spread_val)
end

function unpack_args(y::SVector{2,F}, constraint::SpreadConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    return shape_val, constraint.val
end

function unpack_args(y::SVector{3,F}, constraint::PTraceConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    spread_val = Spread(y[3], constraint.spread0.type)
    return shape_val, spread_val
end

function unpack_constraint(y::SVector{2,F}, constraint::SpreadConstraint{F}) where {F}
    return update(constraint; shape0=complex(y[1], y[2]))
end

function unpack_constraint(y::SVector{3,F}, constraint::PTraceConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    spread_val = Spread(y[3], constraint.spread0.type)
    return update(constraint; shape0=shape_val, spread0=spread_val)
end

outputdim(::SpreadConstraint) = 2
outputdim(::PTraceConstraint) = 3

zpoint(z, ::Type{F}) where {F} = SVector{2,F}(F(real(z)), F(imag(z)))

haspoint(rbf::RBFInterpolant{F}, z::SVector{2,F}) where {F} = any(xi -> xi == z, rbf.x)

function SurrogateModel(
    equation::ResolventEquation{F},
    constraint::AbstractConstraint{F},
    algorithm::AbstractAlgorithm,
    zs::AbstractVector{<:Complex},
) where {F}
    workspace = Workspace(equation)
    xs = SVector{2,F}[]
    ys = SVector{outputdim(constraint),F}[]

    constraint_guess = constraint
    for z in zs
        zF = Complex{F}(z)
        problem = PointProblem(equation, constraint_guess, zF)
        solution = pointsolve(problem, algorithm)
        if issuccess(solution)
            y = pack(solution, constraint)
            Base.push!(xs, zpoint(zF, F))
            Base.push!(ys, y)
            constraint_guess = unpack_constraint(y, constraint)
        end
    end

    rbf = RBFInterpolant(ThinPlateSpline(), xs, ys)
    return SurrogateModel(equation, constraint, algorithm, rbf, workspace)
end

interpolate(model::SurrogateModel{F}, z) where {F} = model.rbf(zpoint(z, F))

function evaluate(model::SurrogateModel{F}, z) where {F}
    zvec = zpoint(z, F)
    y = model.rbf(zvec)
    shape_val, spread_val = unpack_args(y, model.constraint)
    zF = Complex{F}(z)
    solution = PointSolution(
        model.equation, zF, shape_val, spread_val; workspace=model.workspace
    )
    if issuccess(solution)
        if !haspoint(model.rbf, zvec)
            y_new = pack(solution, model.constraint)
            push!(model.rbf, zvec, y_new)
        end
        return solution
    end

    constraint_guess = unpack_constraint(y, model.constraint)
    problem = PointProblem(model.equation, constraint_guess, zF)
    solution = pointsolve(problem, model.algorithm)
    if issuccess(solution)
        y_new = pack(solution, model.constraint)
        z_new = zpoint(zF, F)
        if !haspoint(model.rbf, z_new)
            push!(model.rbf, z_new, y_new)
        end
    end
    return solution
end

end  # module Surrogates

@testitem "pack/unpack round-trip SpreadConstraint" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint, update
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess, shape, spreadfrom
    using GaussianSpectralDensities.Spreads: Spread
    using GaussianSpectralDensities.Surrogates: pack, unpack_args, unpack_constraint
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I
    using StaticArrays: SVector

    ensemble64 = MatrixEnsemble(I, I, I, I / 2)
    eq64 = ResolventEquation(ensemble64)
    ws64 = Workspace(eq64)

    spread64 = Spread(0.5, :gap)
    z64 = 0.25 + 0.25im
    shape64 = -0.5 - 0.5im

    constraint64 = SpreadConstraint(; gap=0.5, shape0=complex(0.25))

    ps64 = PointSolution(eq64, z64, shape64, spread64; workspace=ws64)
    @test issuccess(ps64)

    @testset "$F" for F in [Float32, Float64]
        eqF = ResolventEquation{F}(eq64)
        wsF = Workspace(eqF)
        constraintF = SpreadConstraint{F}(constraint64)
        spreadF = Spread{F}(spread64)
        psF = PointSolution(eqF, z64, shape64, spreadF; workspace=wsF)
        @test issuccess(psF)

        y = pack(psF, constraintF)
        @test y isa SVector{2,F}
        @test y[1] == real(shape(psF))
        @test y[2] == imag(shape(psF))

        shape_rt, spread_rt = unpack_args(y, constraintF)
        @test shape_rt == shape(psF)
        @test spread_rt == constraintF.val

        constraint_rt = unpack_constraint(y, constraintF)
        @test constraint_rt isa SpreadConstraint{F}
        @test constraint_rt.val == constraintF.val
        @test constraint_rt.shape0 == shape_rt
    end
end

@testitem "pack/unpack round-trip PTraceConstraint" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess, shape, spreadfrom
    using GaussianSpectralDensities.Solvers: pointsolve
    using GaussianSpectralDensities.Surrogates: pack, unpack_args, unpack_constraint

    using LinearAlgebra: I
    using StaticArrays: SVector

    ensemble64 = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq64 = ResolventEquation(ensemble64)
    z64 = 0.5 + 0.5im
    constraint64 = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))

    @testset "$F" for F in [Float32, Float64]
        eqF = ResolventEquation{F}(eq64)
        constraintF = PTraceConstraint{F}(constraint64)
        problemF = PointProblem(eqF, constraintF, z64)
        psF = pointsolve(problemF, DirectSolve())
        @test issuccess(psF)

        y = pack(psF, constraintF)
        @test y isa SVector{3,F}
        @test y[1] == real(shape(psF))
        @test y[2] == imag(shape(psF))
        spread_val = spreadfrom(psF, constraintF.spread0.type).val
        @test y[3] == spread_val

        shape_rt, spread_rt = unpack_args(y, constraintF)
        @test shape_rt == shape(psF)
        @test spread_rt.val == spread_val
        @test spread_rt.type == constraintF.spread0.type

        constraint_rt = unpack_constraint(y, constraintF)
        @test constraint_rt isa PTraceConstraint{F}
        @test constraint_rt.val == constraintF.val
        @test constraint_rt.shape0 == shape_rt
        @test constraint_rt.spread0.val == spread_val
        @test constraint_rt.spread0.type == constraintF.spread0.type
    end
end

@testitem "interpolate returns correct SVector dimension" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Surrogates: SurrogateModel, interpolate

    using LinearAlgebra: I
    using StaticArrays: SVector

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    alg = DirectSolve()

    zs = ComplexF64[0.25 + 0.25im, 0.5 + 0.25im, 0.25 + 0.5im, 0.5 + 0.5im]

    @testset "SpreadConstraint" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        model = SurrogateModel(eq, constraint, alg, zs)
        y = interpolate(model, 0.375 + 0.375im)
        @test y isa SVector{2,Float64}
    end

    @testset "PTraceConstraint" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        model = SurrogateModel(eq, constraint, alg, zs)
        y = interpolate(model, 0.375 + 0.375im)
        @test y isa SVector{3,Float64}
    end
end

@testitem "evaluate fast path" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess
    using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluate

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    alg = DirectSolve()

    @testset "SpreadConstraint" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        zs = ComplexF64[
            0.2 + 0.2im,
            0.4 + 0.2im,
            0.2 + 0.4im,
            0.4 + 0.4im,
            0.3 + 0.3im,
            0.3 + 0.2im,
            0.2 + 0.3im,
            0.4 + 0.3im,
            0.3 + 0.4im,
        ]
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.3 + 0.3im)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before
    end

    @testset "PTraceConstraint" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs = ComplexF64[
            0.2 + 0.2im,
            0.4 + 0.2im,
            0.2 + 0.4im,
            0.4 + 0.4im,
            0.3 + 0.3im,
            0.3 + 0.2im,
            0.2 + 0.3im,
            0.4 + 0.3im,
            0.3 + 0.4im,
        ]
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.3 + 0.3im)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before
    end
end

@testitem "evaluate slow path / adaptive refinement" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess
    using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluate

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    alg = DirectSolve()

    @testset "SpreadConstraint" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        zs = ComplexF64[0.1 + 0.1im, 0.9 + 0.1im, 0.1 + 0.9im]
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.5 + 0.5im)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before + 1
    end

    @testset "PTraceConstraint" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs = ComplexF64[0.1 + 0.1im, 0.9 + 0.1im, 0.1 + 0.9im]
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.5 + 0.5im)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before + 1
    end
end

@testitem "no poisoning" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluate

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    alg = DirectSolve()

    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    zs = ComplexF64[0.1 + 0.1im, 0.3 + 0.1im, 0.1 + 0.3im]
    model = SurrogateModel(eq, constraint, alg, zs)
    n_before = length(model.rbf.x)

    z_far = 5.0 + 5.0im
    ps = evaluate(model, z_far)
    @test !issuccess(ps)
    @test length(model.rbf.x) == n_before
end

@testitem "SurrogateModel constructor" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Surrogates: SurrogateModel
    using GaussianSpectralDensities.Workspaces: Workspace

    using LinearAlgebra: I

    alg = DirectSolve()

    @testset "all points solvable" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
        eq = ResolventEquation(ensemble)
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs = ComplexF64[0.25 + 0.25im, 0.5 + 0.25im, 0.25 + 0.5im, 0.5 + 0.5im]

        model = SurrogateModel(eq, constraint, alg, zs)
        @test model.equation === eq
        @test model.constraint === constraint
        @test model.algorithm === alg
        @test model.workspace isa Workspace{Float64}
        @test length(model.rbf.x) == length(zs)
    end

    @testset "excludes failed solves" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
        eq = ResolventEquation(ensemble; shiftmin=0.0)
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs_ok = ComplexF64[0.1 + 0.1im, 0.3 + 0.1im, 0.1 + 0.3im]
        zs_bad = ComplexF64[5.0 + 5.0im, 6.0 + 6.0im]
        zs = vcat(zs_ok, zs_bad)

        model = SurrogateModel(eq, constraint, alg, zs)
        @test length(model.rbf.x) < length(zs)
        @test length(model.rbf.x) == length(zs_ok)
    end

    @testset "SpreadConstraint" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
        eq = ResolventEquation(ensemble)
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        zs = ComplexF64[0.25 + 0.25im, 0.5 + 0.25im, 0.25 + 0.5im, 0.5 + 0.5im]

        model = SurrogateModel(eq, constraint, alg, zs)
        @test length(model.rbf.x) == length(zs)
    end
end
