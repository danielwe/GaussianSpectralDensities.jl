using .LazySets: LazySet, VPolygon, Bloating, overapproximate, radius

function spectral_region(evals::Vector{Complex{R}}) where {R}
    evalsxy = reinterpret(R, reshape(evals, 1, :))
    region = VPolygon(evalsxy)
    region = Bloating(region, R(0.125) * radius(region))
    region = overapproximate(region, R(0.01) * radius(region))
    return region
end

spectral_region(args...) = spectral_region(eigvals(sample(args...)[1]))

function autogrid(region::LazySet, n::Integer=64^2)
    bounds = overapproximate(region)
    xcenter, ycenter = bounds.center
    xradius, yradius = bounds.radius
    xmin, xmax = xcenter - xradius, xcenter + xradius
    ymin, ymax = ycenter - yradius, ycenter + yradius
    xs = range(xmin, xmax; length=2 * round(Int, sqrt(n * xradius / yradius) / 2) + 1)
    ys = range(ymin, ymax; length=2 * round(Int, sqrt(n * yradius / xradius) / 2) + 1)
    zs = [[x, y] in region ? complex(x, y) : missing for y in ys, x in xs]
    return xs, ys, zs
end

autogrid(x, args...) = autogrid(spectral_region(x), args...)

function autogrid(x, ensemble::AbstractMatrixEnsemble, args...)
    return autogrid(spectral_region(x, ensemble), args...)
end

function autogrid(x, y, ensemble::AbstractMatrixEnsemble, args...)
    return autogrid(spectral_region(x, y, ensemble), args...)
end
