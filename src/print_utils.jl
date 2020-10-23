function print_hline()
    println("-------------------------------------------------------")
end

function print_header()
    println("   γ^2        δ^2               q                k")
    print_hline()
end

print_footer() = print_hline()

function print_row(s::AbstractSolution)
    @printf(
        "%9.2e  %9.2e  %9.2e + i%9.2e  %9.2e\n",
        s.γsq,
        s.δsq,
        real(s.q),
        imag(s.q),
        s.k,
    )
end

print_z(z) = @printf("  z = %6.3f + i%6.3f\n", real(z), imag(z))
print_gsq(gsq) = @printf("g^2 = %9.2e\n", gsq)
print_dsq(dsq) = @printf("d^2 = %9.2e\n", dsq)
print_q(q) = @printf("  q = %9.2e + i%9.2e\n", real(q), imag(q))
print_k(k) = @printf("  k = %9.2e\n", k)
