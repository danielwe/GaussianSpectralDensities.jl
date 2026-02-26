module RBFInterpolants

using LinearAlgebra: Symmetric, norm
using LogExpFunctions: xlogx
using Random: default_rng, seed!
using StaticArrays: SMatrix, SVector
using Statistics: median

public RBFInterpolant, ThinPlateSpline, append!, push!

struct ThinPlateSpline end

(::ThinPlateSpline)(r::T) where {T<:Real} = r * xlogx(r)

mutable struct RBFInterpolant{F<:AbstractFloat,N,Phi,P<:SMatrix{N,3,F}}
    phi::Phi
    x::Vector{SVector{2,F}}
    y::Vector{SVector{N,F}}
    coeffs::Vector{SVector{N,F}}
    poly::P
    nn_dists::Vector{F}
    dmin::F
end

function RBFInterpolant(
    phi::Phi,
    x::AbstractVector{SVector{2,F}},
    y::AbstractVector{SVector{N,F}};
    dmin_coeff::Real=F(1 // 2),
) where {F<:AbstractFloat,N,Phi}
    dmin_coeff >= 0 || throw(ArgumentError("dmin_coeff must be nonnegative"))
    xv, yv = convert(Vector{SVector{2,F}}, x), convert(Vector{SVector{N,F}}, y)
    dmin_coeffF = F(dmin_coeff)
    pairwise, nn_dists = pairwise_distances(xv)
    dmin = compute_dmin(nn_dists, dmin_coeffF)
    keep = keep_indices(pairwise, dmin)
    x_pruned = xv[keep]
    y_pruned = yv[keep]
    nn_dists_pruned = nn_dists[keep]
    pairwise_pruned = pairwise[keep, keep]
    coeffs, poly = rbf_coefficients(phi, x_pruned, y_pruned, pairwise_pruned)
    return RBFInterpolant{F,N,Phi,typeof(poly)}(
        phi, x_pruned, y_pruned, coeffs, poly, nn_dists_pruned, dmin
    )
end

function (rbf::RBFInterpolant{F})(pt::SVector{2,F}) where {F}
    intercept = rbf.poly * SVector(oneunit(F), pt...)
    return intercept + sum(eachindex(rbf.x, rbf.coeffs)) do index
        return rbf.coeffs[index] * rbf.phi(norm(pt - rbf.x[index]))
    end
end

function push!(
    rbf::RBFInterpolant{F,N}, x_new::SVector{2,F}, y_new::SVector{N,F}
) where {F,N}
    Base.push!(rbf.x, x_new)
    Base.push!(rbf.y, y_new)
    reprune!(rbf)
    return rbf
end

function append!(
    rbf::RBFInterpolant{F,N},
    xs_new::AbstractVector{SVector{2,F}},
    ys_new::AbstractVector{SVector{N,F}},
) where {F,N}
    length(xs_new) == length(ys_new) || throw(ArgumentError("x/y length mismatch"))

    Base.append!(rbf.x, xs_new)
    Base.append!(rbf.y, ys_new)
    reprune!(rbf)
    return rbf
end

function pairwise_distances(x::Vector{SVector{2,F}}) where {F<:AbstractFloat}
    n = length(x)
    pairwise = Matrix{F}(undef, n, n)
    nn_dists = fill(F(Inf), n)
    for j in 1:n
        pairwise[j, j] = zero(F)
        xj = x[j]
        for i in (j + 1):n
            d = norm(xj - x[i])
            pairwise[i, j] = d
            pairwise[j, i] = d
            nn_dists[i] = min(nn_dists[i], d)
            nn_dists[j] = min(nn_dists[j], d)
        end
    end
    return pairwise, nn_dists
end

function compute_dmin(nn_dists::Vector{F}, dmin_coeff::F) where {F<:AbstractFloat}
    if length(nn_dists) <= 1
        return zero(F)
    end
    return dmin_coeff * median(nn_dists)
end

function keep_indices(pairwise::Matrix{F}, dmin::F) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    if n <= 1
        return collect(1:n)
    end

    if n > 64
        return approximate_keep_indices(pairwise, dmin)
    end

    return exact_keep_indices(pairwise, dmin)
end

function exact_keep_indices(pairwise::Matrix{F}, dmin::F) where {F<:AbstractFloat}
    n = size(pairwise, 1)

    compat = [BitSet() for _ in 1:n]
    for j in 1:(n - 1)
        for i in (j + 1):n
            if pairwise[j, i] >= dmin
                Base.push!(compat[i], j)
                Base.push!(compat[j], i)
            end
        end
    end

    best = Ref(Int[])
    function bron_kerbosch!(r::Vector{Int}, p::BitSet, x_blocked::BitSet)
        if length(r) + length(p) <= length(best[])
            return nothing
        end
        if isempty(p) && isempty(x_blocked)
            if length(r) > length(best[])
                best[] = copy(r)
            end
            return nothing
        end
        for v in collect(p)
            Base.push!(r, v)
            bron_kerbosch!(r, intersect(p, compat[v]), intersect(x_blocked, compat[v]))
            pop!(r)
            delete!(p, v)
            Base.push!(x_blocked, v)
        end
    end

    bron_kerbosch!(Int[], BitSet(1:n), BitSet())
    return sort!(best[])
end

function approximate_keep_indices(pairwise::Matrix{F}, dmin::F) where {F<:AbstractFloat}
    n = size(pairwise, 1)
    conflicts = [BitSet() for _ in 1:n]
    for j in 1:(n - 1)
        for i in (j + 1):n
            if pairwise[j, i] < dmin
                Base.push!(conflicts[i], j)
                Base.push!(conflicts[j], i)
            end
        end
    end

    static_degrees = [length(conflicts[i]) for i in 1:n]
    best = deterministic_greedy_independent_set(conflicts, static_degrees)

    # GRASP-style randomized passes for larger instances where exact search is too costly.
    trial_count = min(32, max(8, n ÷ 8))
    rng = seed!(default_rng(), 0)
    for _ in 1:trial_count
        candidate = randomized_greedy_independent_set(conflicts, static_degrees, rng)
        if length(candidate) > length(best)
            best = candidate
        end
    end
    return best
end

function deterministic_greedy_independent_set(
    conflicts::Vector{BitSet}, static_degrees::Vector{Int}
)
    order = sortperm(eachindex(conflicts); by=i -> (static_degrees[i], i))
    chosen = BitSet()
    kept = Int[]
    for v in order
        has_intersection(chosen, conflicts[v]) && continue
        Base.push!(chosen, v)
        Base.push!(kept, v)
    end
    return kept
end

function randomized_greedy_independent_set(
    conflicts::Vector{BitSet}, static_degrees::Vector{Int}, rng
)
    n = length(conflicts)
    remaining = BitSet(1:n)
    chosen = BitSet()
    kept = Int[]
    rcl = Int[]

    while !isempty(remaining)
        min_degree = typemax(Int)
        for v in remaining
            degree = intersection_count(conflicts[v], remaining)
            if degree < min_degree
                min_degree = degree
            end
        end

        empty!(rcl)
        for v in remaining
            degree = intersection_count(conflicts[v], remaining)
            degree <= min_degree + 1 || continue
            Base.push!(rcl, v)
        end

        sort!(rcl; by=i -> (static_degrees[i], i))
        take = max(1, length(rcl) ÷ 3)
        choice = rcl[rand(rng, 1:take)]

        Base.push!(chosen, choice)
        Base.push!(kept, choice)
        delete!(remaining, choice)
        for neigh in conflicts[choice]
            delete!(remaining, neigh)
        end
    end

    return sort!(kept)
end

function has_intersection(a::BitSet, b::BitSet)
    if length(a) <= length(b)
        for v in a
            if v in b
                return true
            end
        end
    else
        for v in b
            if v in a
                return true
            end
        end
    end
    return false
end

function intersection_count(a::BitSet, b::BitSet)
    count = 0
    if length(a) <= length(b)
        for v in a
            count += v in b
        end
    else
        for v in b
            count += v in a
        end
    end
    return count
end

function reprune!(rbf::RBFInterpolant{F,N}) where {F,N}
    pairwise, nn_dists = pairwise_distances(rbf.x)
    keep = keep_indices(pairwise, rbf.dmin)
    rbf.x = rbf.x[keep]
    rbf.y = rbf.y[keep]
    rbf.nn_dists = nn_dists[keep]
    pairwise_pruned = pairwise[keep, keep]
    rbf.coeffs, rbf.poly = rbf_coefficients(rbf.phi, rbf.x, rbf.y, pairwise_pruned)
    return rbf
end

function rbf_coefficients(
    phi::Phi, x::Vector{SVector{2,F}}, y::Vector{SVector{N,F}}, pairwise::Matrix{F}
) where {F<:AbstractFloat,N,Phi}
    n = length(x)
    n == length(y) || throw(ArgumentError("x/y length mismatch"))
    size(pairwise) == (n, n) || throw(ArgumentError("pairwise size mismatch"))

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

    coeffs = similar(y)
    for j in 1:n
        coeffs[j] = solution[:, j]
    end

    poly_coeffs = SMatrix{N,3,F}(solution[:, (n + 1):end])
    return coeffs, poly_coeffs
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

        rbf = RBFInterpolant(phi, x, y)
        @test rbf isa RBFInterpolant{F,1,ThinPlateSpline}
        @test rbf.dmin == F(1 // 2)

        # Exact interpolation at sample points
        @testset "point $i" for i in eachindex(x)
            @test rbf(x[i]) ≈ y[i] atol = Defaults.atol(F)
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

        rbf = RBFInterpolant(phi, x, y)
        @test rbf isa RBFInterpolant{F,3,ThinPlateSpline}

        @testset "point $i" for i in eachindex(x)
            @test rbf(x[i]) ≈ y[i] atol = Defaults.atol(F)
        end
    end
end

@testitem "push! produces same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = SVector{2,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{2,F}[(1, 2), (4, 8), (16, 32)]

        x_new = SVector{2,F}(F(0.5), F(0.5))
        y_new = SVector{2,F}(3, 6)

        # Build incrementally
        rbf_push = RBFInterpolant(phi, copy(x_init), copy(y_init); dmin_coeff=zero(F))
        push!(rbf_push, x_new, y_new)
        @test rbf_push.dmin === zero(F)

        # Build from scratch with all points
        x_all = vcat(x_init, [x_new])
        y_all = vcat(y_init, [y_new])
        rbf_full = RBFInterpolant(phi, x_all, y_all; dmin_coeff=zero(F))

        # Coefficients should match
        @test rbf_push.coeffs ≈ rbf_full.coeffs
        @test rbf_push.poly ≈ rbf_full.poly
        @test rbf_push.nn_dists ≈ rbf_full.nn_dists

        # Evaluation should match at a test point
        pt = SVector{2,F}(F(0.25), F(0.75))
        @test rbf_push(pt) ≈ rbf_full(pt)
    end
end

@testitem "append! produces same result as rebuild" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()

    @testset "$F" for F in [Float32, Float64]
        x_init = SVector{2,F}[(0, 0), (1, 0), (0, 1)]
        y_init = SVector{1,F}[(1,), (2,), (4,)]

        xs_new = SVector{2,F}[(0.5, 0.5), (1, 1)]
        ys_new = SVector{1,F}[(8,), (16,)]

        # Build incrementally
        rbf_append = RBFInterpolant(phi, copy(x_init), copy(y_init); dmin_coeff=zero(F))
        append!(rbf_append, xs_new, ys_new)
        @test rbf_append.dmin === zero(F)

        # Build from scratch with all points
        x_all = vcat(x_init, xs_new)
        y_all = vcat(y_init, ys_new)
        rbf_full = RBFInterpolant(phi, x_all, y_all; dmin_coeff=zero(F))

        @test rbf_append.coeffs ≈ rbf_full.coeffs
        @test rbf_append.poly ≈ rbf_full.poly
        @test rbf_append.nn_dists ≈ rbf_full.nn_dists

        pt = SVector{2,F}(F(0.25), F(0.75))
        @test rbf_append(pt) ≈ rbf_full(pt)
    end
end

@testitem "constructor pruning uses dmin_coeff and stores dmin" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (0.1, 0), (1, 0), (0, 1)]
    y = SVector{1,Float64}[(1,), (2,), (4,), (8,)]

    rbf = RBFInterpolant(phi, x, y; dmin_coeff=1)

    @test rbf.dmin === 0.5
    @test length(rbf.x) == 3
    @test length(unique(rbf.x)) == 3
end

@testitem "push! reprunes using stored dmin" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline, push!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (10, 0), (0, 10)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y); dmin_coeff=1 // 10)
    dmin = rbf.dmin
    @test dmin === 1.0

    push!(rbf, SVector{2,Float64}(0.3, 0.0), SVector{1,Float64}(99))

    @test rbf.dmin === dmin
    @test length(rbf.x) == 3
    @test all(xi -> xi != SVector{2,Float64}(0.3, 0.0), rbf.x)
end

@testitem "append! reprunes using stored dmin" begin
    using GaussianSpectralDensities.RBFInterpolants:
        RBFInterpolant, ThinPlateSpline, append!

    using StaticArrays: SVector

    phi = ThinPlateSpline()
    x = SVector{2,Float64}[(0, 0), (10, 0), (0, 10)]
    y = SVector{1,Float64}[(1,), (2,), (4,)]

    rbf = RBFInterpolant(phi, copy(x), copy(y); dmin_coeff=1 // 10)
    dmin = rbf.dmin
    @test dmin === 1.0

    append!(
        rbf,
        SVector{2,Float64}[(0.3, 0.0), (10.2, 0.0), (30.0, 30.0)],
        SVector{1,Float64}[(99,), (98,), (16,)],
    )

    @test rbf.dmin === dmin
    @test length(rbf.x) == 4
    @test all(xi -> xi != SVector{2,Float64}(0.3, 0.0), rbf.x)
    @test all(xi -> xi != SVector{2,Float64}(10.2, 0.0), rbf.x)
    @test SVector{2,Float64}(30.0, 30.0) in rbf.x
end

@testitem "large-n pruning uses approximate solver and enforces spacing" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using LinearAlgebra: norm
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

    # On this grid, nearest-neighbor distances are 1/8. Using dmin_coeff > 1
    # forces dmin above grid spacing, so pruning is required.
    rbf = RBFInterpolant(phi, x, y; dmin_coeff=6 // 5)
    @test length(x) == 81
    @test rbf.dmin > 1 / 8
    @test length(rbf.x) < length(x)

    for j in 1:length(rbf.x)
        for i in 1:(j - 1)
            @test norm(rbf.x[j] - rbf.x[i]) >= rbf.dmin
        end
    end
end

@testitem "interpolation convergence" begin
    using GaussianSpectralDensities.RBFInterpolants: RBFInterpolant, ThinPlateSpline

    using StaticArrays: SVector
    using Statistics: mean

    phi = ThinPlateSpline()
    F = Float64

    # Known smooth function: f(x, y) = cos(pi*x/2) * cos(pi*y/2)
    # on [-1, 1] x [-1, 1], evaluated on a dense grid
    testfun(x, y) = cos(F(pi) * x / 2) * cos(F(pi) * y / 2)

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
        ys = SVector{1,F}[]
        for i in 0:(n - 1), j in 0:(n - 1)
            xi = F(-1) + F(2) * i / (n - 1)
            yj = F(-1) + F(2) * j / (n - 1)
            push!(xs, SVector{2,F}(xi, yj))
            push!(ys, SVector{1,F}(testfun(xi, yj)))
        end
        rbf = RBFInterpolant(phi, xs, ys)
        errors = F[]
        for pt in eval_points
            exact_val = testfun(pt...)
            err = abs(rbf(pt)[1] - exact_val)
            push!(errors, err)
        end
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
