using .LazySets: LazySet, VPolygon, Bloating, overapproximate, radius

function spectral_region(evals)
    ftype = real(eltype(evals))
    evals2d = reinterpret(ftype, reshape(evals, 1, :))
    region = VPolygon(evals2d)
    region = Bloating(region, ftype(0.125) * radius(region))
    region = overapproximate(region, ftype(0.01) * radius(region))
    return region
end

function spectral_region(ensemble::AbstractMatrixEnsemble)
    return spectral_region(eigvals(sample(ensemble)[1]))
end

function autogrid(region::LazySet, n=64^2)
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

function autogrid(x, args...)
    return autogrid(spectral_region(x), args...)
end
