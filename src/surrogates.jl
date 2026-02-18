module Surrogates

using ..Algorithms: AbstractAlgorithm, DirectSolve
using ..Batches: BatchSolution
using ..Problems:
    AbstractConstraint,
    PTraceConstraint,
    PointProblem,
    ResolventEquation,
    SpreadConstraint,
    spreadtype,
    update
using ..RBFInterpolants: RBFInterpolant, RBFInterpolants, ThinPlateSpline
using ..Samples: sample_eigenvalues
using ..Solutions: PointSolution, SolutionStatus, issuccess, shape, spreadfrom
using ..Solvers: pointsolve
using ..Spreads: Spread
using ..Workspaces: Workspace

using StaticArrays: @SVector, SVector

public SurrogateModel,
    evaluate, evaluatebatch, interpolate, pack, unpack_args, unpack_constraint

struct SurrogateModel{
    F,
    N,
    E<:ResolventEquation{F},
    C<:AbstractConstraint{F},
    A<:AbstractAlgorithm,
    R<:RBFInterpolant{F,N},
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
    spread_val = spreadfrom(ps, spreadtype(constraint)).val
    return SVector{3,F}(real(shape_val), imag(shape_val), spread_val)
end

function unpack_args(y::SVector{2,F}, constraint::SpreadConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    return shape_val, constraint.val
end

function unpack_args(y::SVector{3,F}, constraint::PTraceConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    spread_val = Spread(y[3], spreadtype(constraint))
    return shape_val, spread_val
end

function unpack_constraint(y::SVector{2,F}, constraint::SpreadConstraint{F}) where {F}
    return update(constraint; shape0=complex(y[1], y[2]))
end

function unpack_constraint(y::SVector{3,F}, constraint::PTraceConstraint{F}) where {F}
    shape_val = complex(y[1], y[2])
    spread_val = Spread(y[3], spreadtype(constraint))
    return update(constraint; shape0=shape_val, spread0=spread_val)
end

outputdim(::SpreadConstraint) = 2
outputdim(::PTraceConstraint) = 3

zpoint(::Type{F}, z) where {F<:Real} = SVector{2,F}(real(z), imag(z))

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
            push!(xs, zpoint(F, zF))
            push!(ys, y)
            constraint_guess = unpack_constraint(y, constraint)
        end
    end

    rbf = RBFInterpolant(ThinPlateSpline(), xs, ys)
    return SurrogateModel(equation, constraint, algorithm, rbf, workspace)
end

function SurrogateModel(
    equation::ResolventEquation{F},
    constraint::AbstractConstraint{F},
    algorithm::AbstractAlgorithm,
    N::Integer=64,
) where {F}
    zs = sample_eigenvalues(equation.ensemble, N)
    return SurrogateModel(equation, constraint, algorithm, zs)
end

function SurrogateModel(model::SurrogateModel{F}, zs::AbstractVector{<:Complex}) where {F}
    (; equation, constraint, algorithm, workspace) = model
    sols = filter(issuccess, evaluatebatch(model, zs; refine=false))
    xs = map(sol -> zpoint(F, sol.z), sols)
    ys = map(sol -> pack(sol, constraint), sols)
    rbf = RBFInterpolant(ThinPlateSpline(), xs, ys)
    return SurrogateModel(equation, constraint, algorithm, rbf, workspace)
end

interpolate(model::SurrogateModel{F}, z) where {F} = model.rbf(zpoint(F, z))

function evaluate_one(model::SurrogateModel{F}, z) where {F}
    zF = Complex{F}(z)
    zv = zpoint(F, zF)
    model_y = model.rbf(zv)
    shape, spread = unpack_args(model_y, model.constraint)
    model_sol = PointSolution(model.equation, zF, shape, spread; workspace=model.workspace)
    issuccess(model_sol) && return model_sol, false
    (model_sol.status == SolutionStatus.shift_below_min) &&
        return nan_solution(model, zF), false

    constraint_guess = unpack_constraint(model_y, model.constraint)
    problem = PointProblem(model.equation, constraint_guess, zF)
    sol = if model_sol.status < SolutionStatus.not_intheballpark
        pointsolve(problem, DirectSolve())
    else
        pointsolve(problem, model.algorithm)
    end
    issuccess(sol) || return nan_solution(model, zF), false
    return sol, !haspoint(model.rbf, zv)
end

function evaluate(model::SurrogateModel{F}, z; refine::Bool=false) where {F}
    sol, refine_candidate = evaluate_one(model, z)
    if refine && refine_candidate
        RBFInterpolants.push!(model.rbf, zpoint(F, z), pack(sol, model.constraint))
    end
    return sol
end

function evaluatebatch(model::SurrogateModel{F,N}, zs; refine::Bool=false) where {F,N}
    refine_xs = SVector{2,F}[]
    refine_ys = SVector{N,F}[]

    sols = BatchSolution{F}(model.equation, zs)
    for i in eachindex(zs, sols)
        z = zs[i]
        sol, refine_candidate = evaluate_one(model, z)
        sols[i] = sol
        if refine && refine_candidate
            push!(refine_xs, zpoint(F, z))
            push!(refine_ys, pack(sol, model.constraint))
        end
    end

    if !isempty(refine_xs)
        RBFInterpolants.append!(model.rbf, refine_xs, refine_ys)
    end
    return sols
end

function nan_solution(model::SurrogateModel{F,N}, z::Complex{F}) where {F,N}
    nan_y = @SVector fill(F(NaN), N)
    nan_shape, nan_spread = unpack_args(nan_y, model.constraint)
    sol = PointSolution(model.equation, z, nan_shape, nan_spread; workspace=model.workspace)
    return sol
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
        PTraceConstraint, PointProblem, ResolventEquation, spreadtype
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess, shape, spreadfrom
    using GaussianSpectralDensities.Solvers: pointsolve
    using GaussianSpectralDensities.Spreads: Spread
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
        spread_val = spreadfrom(psF, spreadtype(constraintF)).val
        @test y[3] == spread_val

        shape_rt, spread_rt = unpack_args(y, constraintF)
        @test shape_rt == shape(psF)
        @test spread_rt == Spread(spread_val, spreadtype(constraintF))

        constraint_rt = unpack_constraint(y, constraintF)
        @test constraint_rt isa PTraceConstraint{F}
        @test constraint_rt.val == constraintF.val
        @test constraint_rt.shape0 == shape_rt
        @test constraint_rt.spread0 == Spread(spread_val, spreadtype(constraintF))
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
    using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluate, evaluatebatch

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    alg = DirectSolve()

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

    @testset "SpreadConstraint" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.3 + 0.3im; refine=true)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before

        sols = evaluatebatch(model, [0.3 + 0.3im, 0.35 + 0.35im]; refine=true)
        @test all(issuccess(sols))
        @test length(model.rbf.x) == n_before
    end

    @testset "PTraceConstraint" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        ps = evaluate(model, 0.3 + 0.3im; refine=true)
        @test ps isa PointSolution{Float64}
        @test issuccess(ps)
        @test length(model.rbf.x) == n_before

        sols = evaluatebatch(model, [0.3 + 0.3im, 0.35 + 0.35im]; refine=true)
        @test all(issuccess(sols))
        @test length(model.rbf.x) == n_before
    end

    @testset "empty input" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        sols = evaluatebatch(model, ComplexF64[]; refine=true)
        @test isempty(sols)
        @test length(model.rbf.x) == n_before
    end
end

@testitem "evaluate slow path / adaptive refinement" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation, SpreadConstraint
    using GaussianSpectralDensities.Solutions: PointSolution, issuccess, shape
    using GaussianSpectralDensities.Surrogates:
        SurrogateModel, evaluate, evaluatebatch, interpolate, unpack_args

    using LinearAlgebra: Diagonal, I
    using StaticArrays: @SVector

    ensemble = MatrixEnsemble(Diagonal(@SVector[0.75, -0.75]), I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    alg = DirectSolve()
    zs = ComplexF64[0.1 + 0.1im, 0.2 + 0.1im, 0.1 + 0.2im]
    candidates = ComplexF64[0.6 + 0.6im, 0.7 + 0.2im, 0.2 + 0.7im]

    @testset "SpreadConstraint" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))

        found = false
        for z_new in candidates
            model = SurrogateModel(eq, constraint, alg, zs)
            y_guess = interpolate(model, z_new)
            shape_val, spread_val = unpack_args(y_guess, constraint)
            ps_guess = PointSolution(
                eq, z_new, shape_val, spread_val; workspace=model.workspace
            )
            if !issuccess(ps_guess)
                n_before = length(model.rbf.x)
                ps = evaluate(model, z_new; refine=true)
                @test ps isa PointSolution{Float64}
                @test issuccess(ps)
                @test length(model.rbf.x) == n_before + 1
                found = true
                break
            end
        end

        @test found
    end

    @testset "PTraceConstraint" begin
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))

        found = false
        for z_new in candidates
            model = SurrogateModel(eq, constraint, alg, zs)
            y_guess = interpolate(model, z_new)
            shape_val, spread_val = unpack_args(y_guess, constraint)
            ps_guess = PointSolution(
                eq, z_new, shape_val, spread_val; workspace=model.workspace
            )
            if !issuccess(ps_guess)
                n_before = length(model.rbf.x)
                ps = evaluate(model, z_new; refine=true)
                @test ps isa PointSolution{Float64}
                @test issuccess(ps)
                @test length(model.rbf.x) == n_before + 1
                found = true
                break
            end
        end

        @test found
    end

    @testset "evaluatebatch" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))

        model_single = SurrogateModel(eq, constraint, alg, zs)
        sols_single = [evaluate(model_single, z) for z in candidates]

        model_batch = SurrogateModel(eq, constraint, alg, zs)
        sols_batch = evaluatebatch(model_batch, candidates)

        @test length(sols_batch) == length(candidates)
        for (s1, s2) in zip(sols_single, sols_batch)
            @test s1.status == s2.status
            if issuccess(s1) && issuccess(s2)
                @test shape(s1) == shape(s2)
            end
        end
    end

    @testset "evaluatebatch refinement" begin
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))

        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)
        sols = evaluatebatch(model, candidates; refine=true)
        n_refined = count(issuccess(sols))
        @test n_refined > 1
        @test length(model.rbf.x) == n_before + n_refined
    end
end

@testitem "no poisoning" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Problems:
        PTraceConstraint, PointProblem, ResolventEquation
    using GaussianSpectralDensities.Solutions: issuccess
    using GaussianSpectralDensities.Surrogates: SurrogateModel, evaluate, evaluatebatch

    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
    eq = ResolventEquation(ensemble; shiftmin=0.0)
    alg = DirectSolve()

    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    zs = ComplexF64[0.1 + 0.1im, 0.3 + 0.1im, 0.1 + 0.3im]

    @testset "evaluate" begin
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        z_far = 5.0 + 5.0im
        ps = evaluate(model, z_far; refine=true)
        @test !issuccess(ps)
        @test length(model.rbf.x) == n_before
    end

    @testset "evaluatebatch" begin
        model = SurrogateModel(eq, constraint, alg, zs)
        n_before = length(model.rbf.x)

        zs_far = ComplexF64[5.0 + 5.0im, 6.0 + 6.0im]
        sols = evaluatebatch(model, zs_far; refine=true)
        @test !any(issuccess(sols))
        @test length(model.rbf.x) == n_before
    end
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

    @testset "constructor with eigvals" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
        eq = ResolventEquation(ensemble)
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))

        model = SurrogateModel(eq, constraint, alg, 16)
        @test model.equation === eq
        @test model.constraint === constraint
        @test model.algorithm === alg
        @test model.workspace isa Workspace{Float64}
        @test length(model.rbf.x) == 16
    end

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

    @testset "resampling round-trip" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
        eq = ResolventEquation(ensemble)
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs = ComplexF64[0.25 + 0.25im, 0.5 + 0.25im, 0.25 + 0.5im, 0.5 + 0.5im]

        model = SurrogateModel(eq, constraint, alg, zs)
        model2 = SurrogateModel(model, zs)
        @test model2.equation === model.equation
        @test model2.constraint === model.constraint
        @test model2.algorithm === model.algorithm
        @test model2.workspace === model.workspace
        @test model2.rbf !== model.rbf
        @test length(model2.rbf.x) == length(zs)
    end

    @testset "resampling round-trip SpreadConstraint" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
        eq = ResolventEquation(ensemble)
        constraint = SpreadConstraint(; gap=0.5, shape0=complex(0.25))
        zs = ComplexF64[0.25 + 0.25im, 0.5 + 0.25im, 0.25 + 0.5im, 0.5 + 0.5im]

        model = SurrogateModel(eq, constraint, alg, zs)
        model2 = SurrogateModel(model, zs)
        @test model2.rbf !== model.rbf
        @test length(model2.rbf.x) == length(zs)
    end

    @testset "resampling excludes failed" begin
        ensemble = MatrixEnsemble(0.0I, I, I, 0.0I)
        eq = ResolventEquation(ensemble; shiftmin=0.0)
        constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
        zs_ok = ComplexF64[0.1 + 0.1im, 0.3 + 0.1im, 0.1 + 0.3im]
        zs_bad = ComplexF64[5.0 + 5.0im, 6.0 + 6.0im]

        model = SurrogateModel(eq, constraint, alg, zs_ok)
        model2 = SurrogateModel(model, vcat(zs_ok, zs_bad))
        @test length(model2.rbf.x) == length(zs_ok)
    end
end
