function autogrid(xmin::Real, xmax::Real, ymin::Real, ymax::Real, n::Integer=64^2)
    ratio = (xmax - xmin) / (ymax - ymin)
    nx = sqrt(n * ratio) / 2
    xs = LinRange(xmin, xmax, 2 * round(Int, nx) + 1)
    ys = LinRange(ymin, ymax, 2 * round(Int, nx / ratio) + 1)
    return xs, ys
end

function autogrid(evals::Vector{Complex{R}}, args...) where {R}
    xmin, xmax = extrema(real, evals)
    ymin, ymax = extrema(imag, evals)
    margin = R(0.05) * max(xmax - xmin, ymax - ymin)
    xmin, xmax = xmin - margin, xmax + margin
    ymin, ymax = ymin - margin, ymax + margin
    return autogrid(xmin, xmax, ymin, ymax, args...)
end

autogrid(A::AbstractMatrix, args...) = autogrid(eigvals(A; sortby=nothing), args...)
autogrid!(A::AbstractMatrix, args...) = autogrid(eigvals!(A; sortby=nothing), args...)

function autogrid(ensemble::AbstractMatrixEnsemble, args...)
    return autogrid!(randsample(ensemble), args...)
end

function autogrid(rng_or_type, ensemble::AbstractMatrixEnsemble, args...)
    return autogrid!(randsample(rng_or_type, ensemble), args...)
end

function autogrid(rng, type, ensemble::AbstractMatrixEnsemble, args...)
    return autogrid!(randnsample(rng, type, ensemble), args...)
end
