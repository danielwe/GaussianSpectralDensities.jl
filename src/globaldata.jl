module GlobalData

using AutoHashEquals: @auto_hash_equals
using Meshes: Meshes, SimpleMesh, centroid, coords, elements, indices, topology, vertices
using StructArrays: StructVector

using ..Problems: AbstractConstraint, PointProblem, ResolventEquation
using ..Spreads: Spread, Spreads
using ..Solutions: PointSolution, ResolventParams, Solutions

using ..ArrayTools: adaptrealtype

public ElementGradient, GlobalProblem, GlobalSolution, element_gradients, zval, zvals

## Complex plane mapping
## =====================

"""
    zval(mesh, i::Int) -> Complex

Return the complex z-value for vertex `i` of `mesh`, where `z = x + iy`.
"""
function zval(mesh, i::Int)
    c = coords(vertices(mesh)[i])
    return Complex(c.x.val, c.y.val)
end

"""
    zvals(mesh) -> Vector{Complex}

Return all vertex z-values for `mesh`.
"""
zvals(mesh) = [zval(mesh, i) for i in eachindex(vertices(mesh))]

abstract type AbstractGlobalMesh{T} <: AbstractVector{T} end

Base.size(d::AbstractGlobalMesh) = (Meshes.nvertices(d.mesh),)
Base.IndexStyle(::Type{<:AbstractGlobalMesh}) = IndexLinear()

## GlobalProblem
## =============

@auto_hash_equals typearg=true struct GlobalProblem{
    F,E<:ResolventEquation{F},C<:AbstractConstraint{F},M<:SimpleMesh
} <: AbstractGlobalMesh{PointProblem{F,E,C}}
    equation::E
    constraint::C
    mesh::M

    function GlobalProblem{F}(equation, constraint, mesh::SimpleMesh) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        constraintF = adaptrealtype(F, constraint)
        return new{F,typeof(eqF),typeof(constraintF),typeof(mesh)}(eqF, constraintF, mesh)
    end
end

function GlobalProblem(equation::ResolventEquation{F}, constraint, mesh) where {F}
    return GlobalProblem{F}(equation, constraint, mesh)
end

function Base.getindex(prob::GlobalProblem, i::Int)
    return PointProblem(prob.equation, prob.constraint, zval(prob.mesh, i))
end

## GlobalSolution
## ==============

@auto_hash_equals typearg=true struct GlobalSolution{
    F,E<:ResolventEquation{F},M<:SimpleMesh,P<:StructVector{ResolventParams{F}}
} <: AbstractGlobalMesh{PointSolution{F,E}}
    equation::E
    mesh::M
    params::P
    success::BitVector

    function GlobalSolution{F}(equation, mesh::SimpleMesh) where {F}
        eqF = convert(ResolventEquation{F}, equation)
        n = Meshes.nvertices(mesh)
        params = StructVector{ResolventParams{F}}(undef, n)
        success = falses(n)
        return new{F,typeof(eqF),typeof(mesh),typeof(params)}(eqF, mesh, params, success)
    end
end

function GlobalSolution(equation::ResolventEquation{F}, mesh::SimpleMesh) where {F}
    return GlobalSolution{F}(equation, mesh)
end

function Base.getindex(sol::GlobalSolution{F,E}, i::Int) where {F,E}
    (; equation, mesh, params, success) = sol
    return PointSolution{F,E}(equation, zval(mesh, i), params[i], success[i])
end

function Base.setindex!(sol::GlobalSolution, ps::PointSolution, i::Int)
    if ps.equation !== sol.equation
        msg = "Cannot assign to GlobalSolution using PointSolution for different equation"
        throw(ArgumentError(msg))
    end
    zi = zval(sol.mesh, i)
    if ps.z != zi
        msg = "PointSolution z=$(ps.z) does not match vertex $i with z=$zi"
        throw(ArgumentError(msg))
    end
    sol.params[i] = ps.params
    sol.success[i] = ps.success
    return sol
end

## Element gradients
## =================

"""
    ElementGradient{F,V}

Gradient of a field evaluated at an element centroid.

# Fields

  - `centroid::Complex{F}`: z-location of element centroid
  - `dfdx::V`: derivative with respect to Re(z)
  - `dfdy::V`: derivative with respect to Im(z)

`V` matches the field type: `F` for real fields, `Complex{F}` for complex fields.
"""
@auto_hash_equals struct ElementGradient{F<:AbstractFloat,V}
    centroid::Complex{F}
    dfdx::V
    dfdy::V
end

"""
    element_gradients(sol::GlobalSolution, field::Symbol) -> Vector{ElementGradient}

Compute element-wise gradients of a ResolventParams field over the solution mesh.

Uses linear FEM basis functions, so gradient is constant over each triangular element
and evaluated at the centroid.

`field` must be one of: `:q`, `:shift`, `:gap`, `:ptrace`, `:greens`.
"""
function element_gradients(sol::GlobalSolution, field::Symbol)
    mesh = sol.mesh
    fvals = getproperty(sol.params, field)
    verts = vertices(mesh)
    connec = collect(elements(topology(mesh)))

    return map(enumerate(connec)) do (i, conn)
        vidx = indices(conn)

        x = ntuple(j -> coords(verts[vidx[j]]).x.val, 3)
        y = ntuple(j -> coords(verts[vidx[j]]).y.val, 3)
        fv = ntuple(j -> fvals[vidx[j]], 3)

        A2 = (x[2] - x[1]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[1])
        dfdx = (fv[1] * (y[2] - y[3]) + fv[2] * (y[3] - y[1]) + fv[3] * (y[1] - y[2])) / A2
        dfdy = (fv[1] * (x[3] - x[2]) + fv[2] * (x[1] - x[3]) + fv[3] * (x[2] - x[1])) / A2

        c = centroid(mesh[i])
        z_centroid = Complex(coords(c).x.val, coords(c).y.val)

        return ElementGradient(z_centroid, dfdx, dfdy)
    end
end

## Accessors for GlobalSolution
## ----------------------------
## We had to choose between having an abstract solution type to avoid duplicated accessor
## implementations, or subtyping AbstractArray to get full-featured iteration and indexing
## (complete with axes, eachindex, et cetera) for free. We chose the latter.
"""
    issuccess(sol::GlobalSolution) -> BitVector

Return whether the solution is finite, self-consistent, and satisfies shiftmin.
"""
Solutions.issuccess(sol::GlobalSolution) = sol.success

"""
    q(sol::GlobalSolution{F}) -> Vector{Complex{F}}

Return the self-consistency parameter `q of the solution.
"""
Solutions.q(sol::GlobalSolution) = sol.params.q

"""
    ptrace(sol::GlobalSolution{F}) -> Vector{F}

Return the trace parameter `ptrace` from the solution.
"""
Solutions.ptrace(sol::GlobalSolution) = sol.params.ptrace

"""
    greens(sol::GlobalSolution{F}) -> Vector{Complex{F}}

Return the value of the Green's function (resolvent trace) from the solution.
"""
Solutions.greens(sol::GlobalSolution) = sol.params.greens

"""
    shift(sol::GlobalSolution{F}) -> Vector{F}

Return the spectral shift from the solution.
"""
Spreads.shift(sol::GlobalSolution) = sol.params.shift

"""
    gap(sol::GlobalSolution{F}) -> Vector{F}

Return the spectral gap from the solution.
"""
Spreads.gap(sol::GlobalSolution) = sol.params.gap

"""
    spreadfrom(sol::GlobalSolution{F}, type) -> Vector{Spread{F}}

Create a spread parameter of the given `type` from the solution.
"""
function Solutions.spreadfrom(sol::GlobalSolution, type)
    return map(spread -> Spread(spread, type), getproperty(sol.params, Symbol(type)))
end

end  # module GlobalData

@testitem "zval" begin
    using GaussianSpectralDensities.GlobalData: zval, zvals
    using Meshes: Box, Point, RegularDiscretization, discretize, nvertices, simplexify

    disc = discretize(Box(Point(0.0, 0.0), Point(1.0, 1.0)), RegularDiscretization(2, 2))
    mesh = simplexify(disc)

    @test zval(mesh, 1) == 0.0 + 0.0im
    @test length(zvals(mesh)) == nvertices(mesh)

    zs = zvals(mesh)
    @test all(z -> 0 <= real(z) <= 1 && 0 <= imag(z) <= 1, zs)
end

@testitem "GlobalProblem" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.GlobalData: GlobalProblem, zval
    using GaussianSpectralDensities.Problems:
        PointProblem, ResolventEquation, SpreadConstraint
    using LinearAlgebra: I
    using Meshes: Box, Point, RegularDiscretization, discretize, nvertices, simplexify

    disc = discretize(Box(Point(0.0, 0.0), Point(1.0, 1.0)), RegularDiscretization(2, 2))
    mesh = simplexify(disc)
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))
    constraint = SpreadConstraint(; gap=0.5, q0=complex(0.25))

    prob = GlobalProblem(eq, constraint, mesh)

    @test prob isa GlobalProblem{Float64}
    @test IndexStyle(typeof(prob)) == IndexLinear()
    @test length(prob) == nvertices(mesh)

    pp = prob[1]
    @test pp isa PointProblem{Float64}
    @test pp.z == zval(mesh, 1)
    @test pp.equation == eq
    @test pp.constraint == constraint
end

@testitem "GlobalSolution" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.GlobalData: GlobalSolution, zval
    using GaussianSpectralDensities.Problems: ResolventEquation
    using GaussianSpectralDensities.Spreads: Spread, gap, shift
    using GaussianSpectralDensities.Solutions:
        PointSolution, ResolventParams, greens, issuccess, ptrace, q, spreadfrom
    using LinearAlgebra: I
    using Meshes: Box, Point, RegularDiscretization, discretize, nvertices, simplexify

    disc = discretize(Box(Point(0.0, 0.0), Point(1.0, 1.0)), RegularDiscretization(2, 2))
    mesh = simplexify(disc)
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    sol = GlobalSolution(eq, mesh)

    @test sol isa GlobalSolution{Float64}
    @test IndexStyle(typeof(sol)) == IndexLinear()
    @test length(sol) == nvertices(mesh)
    @test !any(issuccess(sol))

    params = ResolventParams(0.5 + 0.5im, 0.25, 0.5, 1.0, -0.5 + 2.0im)
    sol.params[1] = params
    sol.success[1] = true

    @test sol.params[1] == params
    @test issuccess(sol)[1]

    # setindex! with PointSolution
    z1 = zval(mesh, 1)
    ps = PointSolution(eq, z1, 1.0, Spread(; shift=1.0))
    sol[1] = ps
    @test sol.params[1] == ps.params
    @test sol.success[1] == ps.success

    # getindex
    @test sol[1] == ps

    # setindex! with mismatched eq or z throws ArgumentError
    eq2 = ResolventEquation(eq.ensemble; shiftmin=0)
    z2 = zval(mesh, 2)
    ps_wrongA = PointSolution(eq2, z1, 1.0, Spread(; shift=1.0))
    ps_wrongB = PointSolution(eq, z2, 1.0, Spread(; shift=1.0))
    @test_throws ArgumentError sol[1] = ps_wrongA
    @test_throws ArgumentError sol[1] = ps_wrongB

    # accessors
    sol.params.q .= ComplexF64[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    sol.params.ptrace .= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sol.params.greens .= ComplexF64[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    sol.params.shift .= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sol.params.gap .= [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    @test q(sol) == sol.params.q
    @test ptrace(sol) == sol.params.ptrace
    @test greens(sol) == sol.params.greens
    @test shift(sol) == sol.params.shift
    @test gap(sol) == sol.params.gap
    spread_shift = spreadfrom(sol, :shift)
    spread_gap = spreadfrom(sol, :gap)
    @test length(spread_shift) == length(sol)
    @test length(spread_gap) == length(sol)
    @test all(r -> r isa Spread, spread_shift)
    @test all(r -> r isa Spread, spread_gap)
    @test all(r -> Symbol(r.type) == :shift, spread_shift)
    @test all(r -> Symbol(r.type) == :gap, spread_gap)
    @test [r.val for r in spread_shift] == sol.params.shift
    @test [r.val for r in spread_gap] == sol.params.gap
end

@testitem "element_gradients" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.GlobalData:
        ElementGradient, GlobalSolution, element_gradients
    using GaussianSpectralDensities.Problems: ResolventEquation
    using LinearAlgebra: I
    using Meshes: Box, Point, RegularDiscretization, discretize, nelements, simplexify

    disc = discretize(Box(Point(0.0, 0.0), Point(1.0, 1.0)), RegularDiscretization(2, 2))
    mesh = simplexify(disc)
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    sol = GlobalSolution(eq, mesh)
    shift_vals = 1:length(sol)
    sol.params.shift .= shift_vals
    sol.params.greens .= complex.(shift_vals, shift_vals .^ 2)

    grads_real = element_gradients(sol, :shift)
    @test length(grads_real) == nelements(mesh)
    @test grads_real[1] isa ElementGradient{Float64,Float64}
    @test all(g -> 0 <= real(g.centroid) <= 1 && 0 <= imag(g.centroid) <= 1, grads_real)

    grads_complex = element_gradients(sol, :greens)
    @test length(grads_complex) == nelements(mesh)
    @test grads_complex[1] isa ElementGradient{Float64,ComplexF64}
end
