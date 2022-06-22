function spectral_density(solution::GlobalSpectralSolution{R}, xs, ys) where {R}
    if !all(solution.k .≈ one(R))
        throw(
            DomainError(
                solution, "can only compute density for solution with objective Fixk(1.0)"
            )
        )
    end
    rx = (
        (solution.r[2:end - 1, 3:end] .- solution.r[2:end - 1, 1:end - 2])
        ./ permutedims(xs[3:end] .- xs[1:end - 2])
    )
    ry = (
        (solution.r[3:end, 2:end - 1] .- solution.r[1:end - 2, 2:end - 1])
        ./ (ys[3:end] .- ys[1:end - 2])
    )
    ρ = real.(rx .+ im .* ry) ./ R(2pi)
    for j in 1 : (length(xs) - 2), i in 1 : (length(ys) - 2)
        if ρ[i, j] !== missing
            if solution.γ²[i + 1, j + 1] === missing
                ρ[i, j] = missing
            elseif solution.γ²[i + 1, j + 1] < 0
                ρ[i, j] = 0
            end
        end
    end
    return ρ, xs[2:end - 1], ys[2:end - 1]
end
