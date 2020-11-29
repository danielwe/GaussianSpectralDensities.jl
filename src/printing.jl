print_hline() = println("-------------------------------------------------------------------")
print_title() = println("    μ       γ²         δ²                  q                k")
print_mline() = println("-------  ----------  ---------  ------------------------  ---------")

function print_z(z, verbosity)
    if verbosity > 0
        print_z(z)
    end
    return nothing
end

function print_header(verbosity)
    if verbosity > 1
        print_hline()
        print_title()
        print_mline()
    end
    return nothing
end

function print_foot_divider(verbosity)
    if verbosity > 2
        print_mline()
    end
    return nothing
end

function print_footer(solution, verbosity)
    print_solution(solution, verbosity + 1)
    if verbosity > 1
        print_hline()
    end
    return nothing
end

function print_solution(solution, verbosity)
    if verbosity > 2
        print_solution(solution)
    end
    return nothing
end

function print_solution(solution)
    println(
        sprint_μ(solution.μ),
        "  ",
        sprint_γ²(solution.γ²),
        "  ",
        sprint_δ²(solution.δ²),
        "  ",
        sprint_q(solution.q),
        "  ",
        sprint_k(solution.k),
    )
    return nothing
end

function print_niters(niters, verbosity)
    if verbosity > 3
        @printf("Iterations: %3i\n", niters)
    end
    return nothing
end

function print_trace(solution, verbosity)
    print_header(verbosity)
    for sol in solution.trace
        print_solution(sol, verbosity)
    end
    print_footer(solution, verbosity)
    return nothing
end

sprint_z(z) = @sprintf("%6.3f + i%6.3f", real(z), imag(z))
sprint_μ(μ) = @sprintf("%7.3f", μ)
sprint_γ²(γ²) = @sprintf("%10.3e", γ²)
sprint_δ²(δ²) = @sprintf("%9.3e", δ²)
sprint_q(q) = @sprintf("%10.3e + i%10.3e", real(q), imag(q))
sprint_k(k) = @sprintf("%9.3e", k)
sprint_z(::Missing) = " missing        "
sprint_μ(::Missing) = "missing"
sprint_γ²(::Missing) = " missing  "
sprint_δ²(::Missing) = "missing  "
sprint_q(::Missing) = " missing                "
sprint_k(::Missing) = "missing  "

print_z(z) = println(" z = ", sprint_z(z))
print_μ(μ) = println(" μ =  ", sprint_μ(μ))
print_γ²(γ²) = println("γ² = ", sprint_γ²(γ²))
print_δ²(δ²) = println("δ² =  ", sprint_δ²(δ²))
print_q(q) = println(" q = ", sprint_q(q))
print_k(k) = println(" k =  ", sprint_k(k))

function print_local_info(solver, z)
    println("Exception at  z = ", sprint_z(z))
    println("Objective:   ", solver.objective)
    println("Algorithm:   ", solver.algorithm)
    return nothing
end

function print_local_info(solver, z, q₀, δ₀²)
    print_local_info(solver, z)
    println("Starting from q = ", sprint_q(q₀), ", δ² = ", sprint_δ²(δ₀²))
    return nothing
end
