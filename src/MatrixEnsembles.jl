struct MatrixEnsemble{
    U<:AbstractMatrix{ComplexF64},
    V<:Union{AbstractMatrix{ComplexF64},UniformScaling{ComplexF64}},
}
    N::Int
    M::U
    L::V
    R::V
    T::Vector{Float64}
    LiRi::V
    LiMRi::U

    function MatrixEnsemble(M::U, L::V, R::V, T) where {U,V}
        N = size(M, 1)
        if size(M, 2) != N
            throw(DimensionMismatch("`M` is not square"))
        end
        if !(V <: UniformScaling) && !(size(L) == (N, N) && size(R) == (N, N))
            throw(DimensionMismatch("the sizes of `L` and `R` do not match `M`"))
        end
        if length(T) != N
            throw(DimensionMismatch("the length of `T` does not match the size of `M`"))
        end
        return new{U,V}(N, M, L, R, T, (L \ I) / R, (L \ M) / R)
    end
end

function bare_resolvent_inv!(ensemble::MatrixEnsemble, G0_inv, z)
    for j in 1:ensemble.N, i in 1:ensemble.N
        G0_inv[i, j] = muladd(z, ensemble.LiRi[i, j], -ensemble.LiMRi[i, j])
    end
    return nothing
end
