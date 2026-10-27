module RBFInterpolants

using LinearAlgebra: Symmetric, norm
using LogExpFunctions: xlogx
using StaticArrays: SMatrix, SVector

public RBFInterpolant, ThinPlateSpline, append!, push!

struct ThinPlateSpline end

(::ThinPlateSpline)(r::T) where {T<:Real} = r * xlogx(r)

mutable struct RBFInterpolant{F<:AbstractFloat,N,Phi,P<:SMatrix{N,3,F}}
    const phi::Phi
    const x::Vector{SVector{2,F}}
    const y::Vector{SVector{N,F}}
    const coeffs::Vector{SVector{N,F}}
    poly::P
    const nn_dists::Vector{F}
    const npoints::Int

    function RBFInterpolant(
        phi::Phi,
        x::AbstractVector{SVector{2,F}},
        y::AbstractVector{SVector{N,F}},
        pairwise::Matrix{F},
        nn_dists::AbstractVector{F},
        npoints::Integer,
    ) where {F<:AbstractFloat,N,Phi}
        if !(npoints in (0, length(x)))
            throw(ArgumentError("npoints must be 0 or equal to length(x)"))
        end
        ax = axes(x, 1)
        if (axes(y, 1) != ax) || (axes(nn_dists, 1) != ax)
            # pairwise size is checked in rbf_coefficients
            throw(ArgumentError("x/y/nn_dists axes mismatch"))
        end
        if axes(pairwise) != (ax, ax)
            throw(ArgumentError("pairwise size/index mismatch with x"))
        end
        pairwise_pruned = prune_nonfinite!(x, y, nn_dists, pairwise)
        coeffs, poly = rbf_coefficients(phi, x, y, pairwise_pruned)
        return new{F,N,Phi,typeof(poly)}(phi, x, y, coeffs, poly, nn_dists, npoints)
    end
end

function RBFInterpolant(
    phi::Phi, x::AbstractVector{SVector{2,F}}, y; npoints::Integer=0
) where {F<:AbstractFloat,Phi}
    if !(0 <= npoints <= length(x))
        # 0 means don't prune
        throw(ArgumentError("npoints must satisfy 0 <= npoints <= length(x)"))
    end
    pairwise = pairwise_distances(x)
    keep = prune_to_count(pairwise, npoints)
    x_pruned = x[keep]
    y_pruned = (y isa AbstractVector) ? y[keep] : map(y, x_pruned)
    pairwise_pruned = pairwise[keep, keep]
    nn_dists_pruned = map(axes(pairwise_pruned, 2)) do j
        return minimum(pairwise_pruned[i, j] for i in axes(pairwise_pruned, 1) if i != j)
    end
    return RBFInterpolant(
        phi, x_pruned, y_pruned, pairwise_pruned, nn_dists_pruned, npoints
    )
end

function (rbf::RBFInterpolant{F})(pt::SVector{2,F}) where {F}
    intercept = rbf.poly * SVector(oneunit(F), pt...)
    return intercept + sum(eachindex(rbf.x, rbf.coeffs)) do index
        return rbf.coeffs[index] * rbf.phi(norm(pt - rbf.x[index]))
    end
end

function push!(rbf::RBFInterpolant{F}, x_new::SVector{2,F}, y_new) where {F}
    (y_new isa SVector) && return append!(rbf, [x_new], [y_new])
    return append!(rbf, [x_new], y_new)
end

function append!(
    rbf::RBFInterpolant{F}, xs_new::AbstractVector{SVector{2,F}}, ys_new
) where {F}
    if ys_new isa AbstractVector
        (length(xs_new) == length(ys_new)) || throw(ArgumentError("x/y length mismatch"))
    end
    lasti_old = lastindex(rbf.x)
    Base.append!(rbf.x, xs_new)
    pairwise_full = pairwise_distances(rbf.x)
    keep = prune_to_count(pairwise, rbf.npoints)
    keep_spliti = searchsortedlast(keep, lasti_old)

    keepat!(rbf.x, keep)

    keep_old = @view keep[begin:keep_spliti]
    keep_new = [keep[i] - lasti_old for i in (keep_spliti + 1):lastindex(keep)]
    ys_new_v = if ys_new isa AbstractVector
        ys_new[keep_new]
    else
        map(ys_new, @view xs_new[keep_new])
    end

    keepat!(rbf.y, keep_old)
    Base.append!(rbf.y, ys_new_v)

    pairwise = pairwise_full[keep, keep]

    resize!(rbf.nn_dists, length(keep))
    map!(rbf.nn_dists, pairs(eachcol(pairwise))) do j, col
        return minimum(p for (i, p) in pairs(col) if i != j)
    end
    map!(rbf.nn_dists, axes(pairwise_pruned, 2)) do j
        return minimum(pairwise_pruned[i, j] for i in axes(pairwise_pruned, 1) if i != j)
    end

    pairwise_pruned = prune_nonfinite!(rbf.x, rbf.y, rbf.nn_dists, pairwise)
    nkept = length(eachindex(rbf.x, rbf.y, rbf.nn_dists))
    resize!(rbf.coeffs, nkept)
    rbf.poly = rbf_coefficients!(rbf.coeffs, rbf.phi, rbf.x, rbf.y, pairwise_pruned)
    return rbf
end

function prune_nonfinite!(xs, ys, nn_dists, pairwise)
    drop = findall(yi -> !all(isfinite, yi), ys)
    isempty(drop) && return pairwise
    keep = setdiff(axes(pairwise, 1), drop)
    pairwise_pruned = pairwise[keep, keep]
    deleteat!(xs, drop)
    deleteat!(ys, drop)
    resize!(nn_dists, length(keep))
    map!(nn_dists, axes(pairwise_pruned, 2)) do j
        return minimum(pairwise_pruned[i, j] for i in axes(pairwise_pruned, 1) if i != j)
    end
    return pairwise_pruned
end

function pairwise_distances(x::AbstractVector{SVector{2,F}}) where {F<:AbstractFloat}
    Base.require_one_based_indexing(x)
    n = length(x)
    pairwise = Matrix{F}(undef, n, n)
    for j in 1:n
        pairwise[j, j] = zero(F)
        xj = x[j]
        for i in (j + 1):n
            d = norm(xj - x[i])
            pairwise[i, j] = d
            pairwise[j, i] = d
        end
    end
    return pairwise
end

function prune_to_count(pairwise::Matrix{F}, npoints::Int) where {F<:AbstractFloat}
    m, n = size(pairwise)
    (m == n) || throw(ArgumentError("pairwise must be square"))
    (npoints in (0, n)) && return 1:n
    (n > 64) && return approximate_prune_to_count(pairwise, npoints)
    return exact_prune_to_count(pairwise, npoints)
end

function exact_prune_to_count(pairwise::Matrix{F}, npoints::Int) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    thresholds = F[zero(F)]
    for j in 1:(n - 1)
        for i in (j + 1):n
            Base.push!(thresholds, pairwise[j, i])
        end
    end
    sort!(unique!(thresholds))

    best = collect(1:npoints)
    low, high = 1, length(thresholds)
    while low <= high
        mid = (low + high) >>> 1
        feasible, candidate = clique_of_size(pairwise, thresholds[mid], npoints)
        if feasible
            best = candidate
            low = mid + 1
        else
            high = mid - 1
        end
    end
    return sort!(best)
end

function clique_of_size(
    pairwise::Matrix{F}, threshold::F, npoints::Int
) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    compat = [BitSet() for _ in 1:n]
    for j in 1:(n - 1)
        for i in (j + 1):n
            if pairwise[j, i] >= threshold
                Base.push!(compat[i], j)
                Base.push!(compat[j], i)
            end
        end
    end

    solution = Ref(Int[])
    function search!(r::Vector{Int}, p::BitSet, x_blocked::BitSet)
        if length(r) == npoints
            solution[] = copy(r)
            return true
        end
        if length(r) + length(p) < npoints
            return false
        end
        for v in collect(p)
            Base.push!(r, v)
            if search!(r, intersect(p, compat[v]), intersect(x_blocked, compat[v]))
                return true
            end
            pop!(r)
            delete!(p, v)
            Base.push!(x_blocked, v)
        end
        return false
    end

    feasible = search!(Int[], BitSet(1:n), BitSet())
    return feasible, solution[]
end

function approximate_prune_to_count(
    pairwise::Matrix{F}, npoints::Int
) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    base = farthest_first_with_seed_pair(pairwise, npoints)
    best_score = minimum_pairwise_distance(pairwise, base)
    best = base

    trial_count = min(32, max(8, n ÷ 8))
    for _ in 1:trial_count
        seed = rand(1:n)
        candidate = farthest_first_with_seed(pairwise, npoints, seed)
        candidate_score = minimum_pairwise_distance(pairwise, candidate)
        if candidate_score > best_score
            best = candidate
            best_score = candidate_score
        end
    end
    return sort!(best)
end

function farthest_first_with_seed_pair(
    pairwise::Matrix{F}, npoints::Int
) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    if npoints == 1
        return [1]
    end

    seed_i, seed_j = 1, 2
    best_dist = pairwise[1, 2]
    for j in 1:(n - 1)
        for i in (j + 1):n
            if pairwise[j, i] > best_dist
                best_dist = pairwise[j, i]
                seed_i, seed_j = j, i
            end
        end
    end
    return farthest_first_with_initial(pairwise, npoints, [seed_i, seed_j])
end

function farthest_first_with_seed(
    pairwise::Matrix{F}, npoints::Int, seed::Int
) where {F<:AbstractFloat}
    return farthest_first_with_initial(pairwise, npoints, [seed])
end

function farthest_first_with_initial(
    pairwise::Matrix{F}, npoints::Int, initial::Vector{Int}
) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    selected = copy(initial)
    in_selected = falses(n)
    min_to_selected = fill(F(Inf), n)

    for idx in selected
        in_selected[idx] = true
    end

    for idx in selected
        for v in 1:n
            min_to_selected[v] = min(min_to_selected[v], pairwise[idx, v])
        end
    end

    while length(selected) < npoints
        best_v = 0
        best_score = -typemax(F)
        for v in 1:n
            in_selected[v] && continue
            score = min_to_selected[v]
            if score > best_score
                best_score = score
                best_v = v
            end
        end
        Base.push!(selected, best_v)
        in_selected[best_v] = true
        for v in 1:n
            min_to_selected[v] = min(min_to_selected[v], pairwise[best_v, v])
        end
    end
    return selected
end

function minimum_pairwise_distance(
    pairwise::Matrix{F}, selected::Vector{Int}
) where {F<:AbstractFloat}
    k = length(selected)
    k <= 1 && return typemax(F)
    best = typemax(F)
    for j in 1:(k - 1)
        sj = selected[j]
        for i in (j + 1):k
            best = min(best, pairwise[sj, selected[i]])
        end
    end
    return best
end

function rbf_coefficients(
    phi::Phi,
    x::AbstractVector{SVector{2,F}},
    y::AbstractVector{SVector{N,F}},
    pairwise::Matrix{F},
) where {F<:AbstractFloat,N,Phi}
    coeffs = similar(y)
    poly = rbf_coefficients!(coeffs, phi, x, y, pairwise)
    return coeffs, poly
end

function rbf_coefficients!(
    coeffs::AbstractVector{SVector{N,F}},
    phi::Phi,
    x::AbstractVector{SVector{2,F}},
    y::AbstractVector{SVector{N,F}},
    pairwise::Matrix{F},
) where {F<:AbstractFloat,N,Phi}
    Base.require_one_based_indexing(x, y)
    n = length(x)
    (n == length(y)) || throw(ArgumentError("x/y length mismatch"))
    (n == length(coeffs)) || throw(ArgumentError("coeffs length mismatch"))
    (size(pairwise) == (n, n)) || throw(ArgumentError("pairwise size mismatch"))

    system = Matrix{F}(undef, n + 3, n + 3)
    for j in 1:n
        system[j, j] = zero(F)
        for i in (j + 1):n
            d = pairwise[j, i]
            system[i, j] = phi(d)
            system[j, i] = system[i, j]
        end
    end

    for j in 1:n
        system[(n + 1):end, j] = SVector(one(F), x[j]...)
    end

    system[1:n, (n + 1):end] = system[(n + 1):end, 1:n]'
    system[(n + 1):end, (n + 1):end] .= zero(F)

    rhs = Matrix{F}(undef, N, n + 3)
    for j in 1:n
        rhs[:, j] = y[j]
    end
    rhs[:, (n + 1):end] .= zero(F)

    solution = rhs / Symmetric(system)

    for j in 1:n
        coeffs[j] = solution[:, j]
    end

    return SMatrix{N,3,F}(solution[:, (n + 1):end])
end

end  # module RBFInterpolants

@testitem "ThinPlateSpline" begin
    using GaussianSpectralDensities.RBFInterpolants: ThinPlateSpline

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        @test phi(zero(F)) === zero(F)
        @test phi(one(F)) === zero(F)
        @test phi(F(2)) === F(4) * log(F(2))
        @test phi(F(0.5)) === F(0.25) * log(F(0.5))
    end
end

@testitem "RBFInterpolant constructor and evaluation" begin
    using GaussianSpectralDensities: Defaults
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x = SVector{2,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
        y = SVector{1,F}[(1,), (2,), (4,), (8,)]
        y_lookup = Dict(xi => yi for (xi, yi) in zip(x, y))

        rbfv = RBFInterpolant(phi, x, y)
        @test rbfv isa RBFInterpolant{F,1,ThinPlateSpline}
        @test rbfv.npoints == 0

        rbff = RBFInterpolant(phi, x, pt -> y_lookup[pt])
        @test rbff isa RBFInterpolant{F,1,ThinPlateSpline}
        @test rbff.npoints == 0

        # Exact interpolation at sample points
        @testset "point $i" for i in eachindex(x)
            yv, yf = rbfv(x[i]), rbff(x[i])
            @test yv ≈ y[i] atol = Defaults.atol(F)
            @test yf ≈ y[i] atol = Defaults.atol(F)
            @test yv == yf
        end
    end
end

@testitem "RBFInterpolant vector-valued output" begin
    using GaussianSpectralDensities: Defaults
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x = SVector{2,F}[(0, 0), (1, 0), (0, 1), (1, 1)]
        y = SVector{3,F}[(1, 2, 4), (2, 4, 8), (4, 8, 16), (8, 16, 32)]
        y_lookup = Dict(xi => yi for (xi, yi) in zip(x, y))

        rbfv = RBFInterpolant(phi, x, y)
        @test rbfv isa RBFInterpolant{F,3,ThinPlateSpline}

        rbff = RBFInterpolant(phi, x, pt -> y_lookup[pt])
        @test rbff isa RBFInterpolant{F,3,ThinPlateSpline}

        @testset "point $i" for i in eachindex(x)
            yv, yf = rbfv(x[i]), rbff(x[i])
            @test yv ≈ y[i] atol = Defaults.atol(F)
            @test yf ≈ y[i] atol = Defaults.atol(F)
            @test yv == yf
        end
    end
end

@testitem "RBFInterpolant constructor accepts function and only evaluates kept points" begin
    using GaussianSpectralDensities: Defaults
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (0.1, 0), (1, 0), (0, 1), (1, 1)]
    call_count = Ref(0)
    function yfun(pt)
        call_count[] += 1
        return SVector{1,Float64}(pt[1] + 2pt[2])
    end

    rbf = RBFInterpolant(phi, x, yfun; npoints=3)
    @test call_count[] == 3
    @test length(rbf.x) == 3
    @test length(rbf.y) == 3

    for i in eachindex(rbf.x)
        @test rbf.y[i] == SVector{1,Float64}(rbf.x[i][1] + 2rbf.x[i][2])
        @test rbf(rbf.x[i]) ≈ rbf.y[i] atol = Defaults.atol(Float64)
    end
end

@testitem "RBFInterpolant constructor function output must be SVector" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]

    @test_throws MethodError RBFInterpolant(phi, x, pt -> (pt[1],); npoints=2)
end

@testitem "push! and append! produce same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = SVector{2,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{2,F}[(1, 2), (4, 8), (16, 32)]
        y_init_lookup = Dict(xi => yi for (xi, yi) in zip(x_init, y_init))

        x_new = SVector{2,F}(0.5, 0.5)
        y_new = SVector{2,F}(3, 6)
        y_new_lookup = Dict(xi => yi for (xi, yi) in zip(x_new, y_new))

        x_all = vcat(x_init, [x_new])
        y_all = vcat(y_init, [y_new])
        y_all_lookup = Dict(xi => yi for (xi, yi) in zip(x_all, y_all))

        # Build from scratch with all points
        rbf_v = RBFInterpolant(phi, copy(x_all), copy(y_all); npoints=3)
        rbf_f = RBFInterpolant(phi, copy(x_all), pt -> y_all_lookup[pt]; npoints=3)

        # Build incrementally phase 1
        rbf_vv = RBFInterpolant(phi, copy(x_init), copy(y_init); npoints=3)
        rbf_vf = RBFInterpolant(phi, copy(x_init), copy(y_init); npoints=3)
        rbf_fv = RBFInterpolant(phi, copy(x_init), pt -> y_init_lookup[pt]; npoints=3)
        rbf_ff = RBFInterpolant(phi, copy(x_init), pt -> y_init_lookup[pt]; npoints=3)

        @testset "$fname" for (fname, f!, container) in
                              [("push!", push!, identity), ("append!", append!, Base.vect)]
            # Build incrementally phase 2
            f!(rbf_vv, container(x_new), container(y_new))
            f!(rbf_vf, container(x_new), y_new_lookup)
            f!(rbf_fv, container(x_new), container(y_new))
            f!(rbf_ff, container(x_new), y_new_lookup)
            @test rbf_vv.npoints == 3
            @test rbf_vf.npoints == 3
            @test rbf_fv.npoints == 3
            @test rbf_ff.npoints == 3

            @testset "$variant" for (variant, rbf) in [
                ("vv", rbf_vv), ("vf", rbf_vf), ("fv", rbf_fv), ("ff", rbf_ff), ("f", rbf_f)
            ]
                # Coefficients should match
                @test rbf.coeffs ≈ rbf_v.coeffs
                @test rbf.poly ≈ rbf_v.poly
                @test rbf.nn_dists ≈ rbf_v.nn_dists

                # Evaluation should match at a test point
                pt = SVector{2,F}(F(0.25), F(0.75))
                @test rbf(pt) ≈ rbf_v(pt)
            end
        end
    end
end

@testitem "constructor pruning uses npoints" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (0.1, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,), (8,)]

    rbf = RBFInterpolant(phi, x, y; npoints=3)

    @test rbf.npoints == 3
    @test length(rbf.x) == 3
    @test length(unique(rbf.x)) == 3
end

@testitem "push! and append! reprune using stored npoints" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (10, 0), (0, 10)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), y; npoints=2)
    @test rbf.npoints == 2

    push!(rbf, SVector{2,Float64}(0.3, 0.0), SVector{1,Float64}(99))

    @test rbf.npoints == 2
    @test length(rbf.x) == 2

    append!(
        rbf, SVector{2,Float64}[(10.2, 0.0), (30.0, 30.0)], SVector{1,Float64}[(98,), (16,)]
    )

    @test rbf.npoints == 2
    @test length(rbf.x) == 2
end

@testitem "large-n pruning uses approximate solver and fixed count" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[]
    y = SVector{1,Float64}[]
    for j in 0:8
        for i in 0:8
            Base.push!(x, SVector{2,Float64}(i / 8, j / 8))
            Base.push!(y, SVector{1,Float64}(i + j))
        end
    end

    rbf = RBFInterpolant(phi, x, y; npoints=24)
    @test length(x) == 81
    @test rbf.npoints == 24
    @test length(rbf.x) == 24
end

@testitem "interpolation convergence" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using LinearAlgebra: norm
    using StaticArrays: SVector
    using Statistics: mean

    phi = ThinPlateSpline()
    F = Float64

    # Known smooth function: f(x, y) = cos(pi*x/2) * cos(pi*y/2)
    # on [-1, 1] x [-1, 1], evaluated on a dense grid
    testfun(v) = SVector(cos(F(pi) * v[1] / 2) * cos(F(pi) * v[2] / 2))

    mean_errors = F[]
    max_errors = F[]
    eval_n = 9
    eval_points = SVector{2,F}[]
    for i in 0:(eval_n - 1), j in 0:(eval_n - 1)
        xi = F(-1) + F(2) * i / (eval_n - 1)
        yj = F(-1) + F(2) * j / (eval_n - 1)
        push!(eval_points, SVector{2,F}(xi, yj))
    end

    for n in [4, 8, 16]
        xs = SVector{2,F}[]
        for i in 0:(n - 1), j in 0:(n - 1)
            xi = F(-1) + F(2) * i / (n - 1)
            yj = F(-1) + F(2) * j / (n - 1)
            push!(xs, SVector{2,F}(xi, yj))
        end
        rbf = RBFInterpolant(phi, xs, testfun)
        errors = [norm(rbf(pt) - testfun(pt)) for pt in eval_points]
        push!(mean_errors, mean(errors))
        push!(max_errors, maximum(errors))
    end

    # Error should decrease as sample density increases
    @test mean_errors[2] < mean_errors[1]
    @test mean_errors[3] < mean_errors[2]
    @test max_errors[2] < max_errors[1]
    @test max_errors[3] < max_errors[2]
    # Finest grid should be quite accurate on average
    @test mean_errors[3] < 1e-3
end
