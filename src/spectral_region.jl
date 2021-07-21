using .LazySets: LazySet, VPolygon, Bloating, overapproximate, radius

function spectral_region(evals::Vector{Complex{R}}; margin=default(:margin)) where {R}
    evalsxy = reinterpret(R, reshape(evals, 1, :))
    region = VPolygon(evalsxy)
    r = radius(region)
    region = Bloating(region, R(2 * margin) * r)
    region = overapproximate(region, R(0.01) * r)
    return region
end

function spectral_region(A::AbstractMatrix; kwargs...)
    return spectral_region(eigvals(A, sortby=nothing); kwargs...)
end

function spectral_region!(A::AbstractMatrix; kwargs...)
    return spectral_region(eigvals!(A, sortby=nothing); kwargs...)
end

spectral_region(args...; kwargs...) = spectral_region!(randsample(args...)[1]; kwargs...)

function automesh(region::LazySet, args...)
    bounds = overapproximate(region)
    xcenter, ycenter = bounds.center
    xradius, yradius = bounds.radius
    xmin, xmax = xcenter - xradius, xcenter + xradius
    ymin, ymax = ycenter - yradius, ycenter + yradius
    xs, ys = automesh(xmin, xmax, ymin, ymax, args...)
    zs = [[x, y] in region ? complex(x, y) : missing for y in ys, x in xs]
    return xs, ys, zs
end
