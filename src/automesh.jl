function automesh(xmin::Real, xmax::Real, ymin::Real, ymax::Real, n::Integer=64^2)
    ratio = (xmax - xmin) / (ymax - ymin)
    nx = sqrt(n * ratio) / 2
    xs = LinRange(xmin, xmax, 2 * round(Int, nx) + 1)
    ys = LinRange(ymin, ymax, 2 * round(Int, nx / ratio) + 1)
    return xs, ys
end

function automesh(evals::Vector{Complex{R}}, args...; margin=default(:margin)) where {R}
    xmin, xmax = extrema(real, evals)
    ymin, ymax = extrema(imag, evals)
    absmargin = R(margin) * max(xmax - xmin, ymax - ymin)
    xmin, xmax = xmin - absmargin, xmax + absmargin
    ymin, ymax = ymin - absmargin, ymax + absmargin
    return automesh(xmin, xmax, ymin, ymax, args...)
end

function automesh(A::AbstractMatrix, args...; kwargs...)
    return automesh(eigvals(A; sortby=nothing), args...; kwargs...)
end

function automesh!(A::AbstractMatrix, args...; kwargs...)
    return automesh(eigvals!(A; sortby=nothing), args...; kwargs...)
end

function automesh(ensemble::AbstractMatrixEnsemble, args...)
    return automesh!(randsample(ensemble)[1], args...)
end

function automesh(rng_or_type, ensemble::AbstractMatrixEnsemble, args...)
    return automesh!(randsample(rng_or_type, ensemble)[1], args...)
end

function automesh(rng, type, ensemble::AbstractMatrixEnsemble, args...)
    return automesh!(randsample(rng, type, ensemble)[1], args...)
end
