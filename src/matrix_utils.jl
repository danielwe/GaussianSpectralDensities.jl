const BiSymTri = Union{Bidiagonal, SymTridiagonal}
const RealSymmetric{T<:Real, A} = Union{Symmetric{T, A}, Hermitian{T, A}}
const SelfAdjoint{T<:Real, A} = Union{RealSymmetric{T, A}, Hermitian{Complex{T}, A}}

@inline function zA⁻B!(C::Diagonal, z, A::UniformScaling, B::Diagonal)
    C.diag .= muladd.(z, A.λ, .-B.diag)
    return nothing
end

@inline function zA⁻B!(C::Diagonal, z, A::Diagonal, B::Diagonal)
    C.diag .= muladd.(z, A.diag, .-B.diag)
    return nothing
end

@inline function zA⁻B!(C::BST, z, A::UniformScaling, B::BST) where {BST<:BiSymTri}
    C.dv .= muladd.(z, A.λ, .-B.dv)
    C.ev .= .-B.ev
    return nothing
end

@inline function zA⁻B!(C::BST, z, A::Diagonal, B::BST) where {BST<:BiSymTri}
    C.dv .= muladd.(z, A.diag, .-B.dv)
    C.ev .= .-B.ev
    return nothing
end

@inline function zA⁻B!(C::BST, z, A::BST, B::BST) where {BST<:BiSymTri}
    C.dv .= muladd.(z, A.dv, .-B.dv)
    C.ev .= muladd.(z, A.ev, .-B.ev)
    return nothing
end

@inline function zA⁻B!(C::Tridiagonal, z, A::UniformScaling, B::Tridiagonal)
    C.d .= muladd.(z, A.λ, .-B.d)
    C.dl .= .-B.dl
    C.du .= .-B.du
    return nothing
end

@inline function zA⁻B!(C::Tridiagonal, z, A::Diagonal, B::Tridiagonal)
    C.d .= muladd.(z, A.diag, .-B.d)
    C.dl .= .-B.dl
    C.du .= .-B.du
    return nothing
end

@inline function zA⁻B!(C::Tridiagonal, z, A::Bidiagonal, B::Tridiagonal)
    C.d .= muladd.(z, A.dv, .-B.d)
    if A.uplo == 'U'
        C.dl .= .-B.dl
        C.du .= muladd.(z, A.ev, .-B.du)
    else
        C.dl .= muladd.(z, A.ev, .-B.dl)
        C.du .= .-B.du
    end
    return nothing
end

@inline function zA⁻B!(C::Tridiagonal, z, A::SymTridiagonal, B::Tridiagonal)
    C.d .= muladd.(z, A.dv, .-B.d)
    C.dl .= muladd.(z, A.ev, .-B.dl)
    C.du .= muladd.(z, A.ev, .-B.du)
    return nothing
end

@inline function zA⁻B!(C::Tridiagonal, z, A::Tridiagonal, B::Tridiagonal)
    C.d .= muladd.(z, A.d, .-B.d)
    C.dl .= muladd.(z, A.dl, .-B.dl)
    C.du .= muladd(z, A.du, .-B.du)
    return nothing
end

@inline function zA⁻B!(C::Matrix, z, A::Union{UniformScaling, Diagonal}, B::Matrix)
    N = size(C, 1)
    @boundscheck begin
        @assert size(C, 2) == N && size(B) == (N, N)
        if A isa Diagonal
            @assert length(A.diag) == N
        end
    end
    @inbounds for j in 1:N
        for i in 1 : (j - 1)
            C[i, j] = -B[i, j]
        end
        C[j, j] = muladd(z, A[j, j], -B[j, j])
        for i in (j + 1) : N
            C[i, j] = -B[i, j]
        end
    end
    return nothing
end

function zA⁻B!(C::Matrix, z, A, B::Matrix)
    C .= muladd.(z, A, .-B)
    return nothing
end

Base.@propagate_inbounds copy_diag!(d, A::Diagonal) = copyto!(d, A.diag)
Base.@propagate_inbounds copy_diag!(d, A::BiSymTri) = copyto!(d, A.dv)
Base.@propagate_inbounds copy_diag!(d, A::Tridiagonal) = copyto!(d, A.d)

@inline function copy_diag!(d, A)
    N = length(d)
    @boundscheck (@assert size(A) == (N, N))
    @inbounds for i in 1:N
        d[i] = A[i, i]
    end
    return d
end

Base.@propagate_inbounds function set_diag_muladd!(A::Diagonal, λ, v1, v2)
    A.diag .= muladd.(λ, v1, v2)
    return nothing
end

Base.@propagate_inbounds function set_diag_muladd!(A::BiSymTri, λ, v1, v2)
    A.dv .= muladd.(λ, v1, v2)
    return nothing
end

Base.@propagate_inbounds function set_diag_muladd!(A::Tridiagonal, λ, v1, v2)
    A.d .= muladd.(λ, v1, v2)
    return nothing
end

@inline function set_diag_muladd!(A, λ, v1, v2)
    N = length(v1)
    @boundscheck (@assert size(A) == (N, N) && length(v2) == N)
    @inbounds for i in 1:N
        A[i, i] = muladd(λ, v1[i], v2[i])
    end
    return nothing
end

@inline function add_to_diag!(A, λ)
    N = size(A, 1)
    @boundscheck (@assert size(A, 2) == N)
    @inbounds for i in 1:N
        A[i, i] += λ
    end
    return nothing
end

# eigmin! differs from LinearAlgebra.eigmin by modifying `A`
eigmin!(A) = eigvals!(A, 1:1)[1]
# Use custom trace and dot to allow CUDA specialization without type piracy
_tr(A) = tr(A)
_dot(a, b) = dot(a, b)

@inline function trTAᴴB(T, A, B)
    N, M = size(A)
    @boundscheck (@assert size(A) == size(B) && length(T) == M)
    out = zero(promote_type(eltype(T), eltype(A), eltype(B)))
    @inbounds for j in 1:M, i in 1:N
        out += T[j] * conj(A[i, j]) * B[i, j]
    end
    return out
end

@inline function trTAᴴTB(T, A::SelfAdjoint, B::SelfAdjoint)
    N = length(T)
    @boundscheck (@assert size(A) == size(B) == (N, N))
    out = zero(promote_type(eltype(T), real(eltype(A)), real(eltype(B))))
    @inbounds for j in 1:N
        for i in 1 : (j - 1)
            out += 2 * T[i] * T[j] * real(conj(A[i, j]) * B[i, j])
        end
        out += T[j]^2 * real(A[j, j]) * real(B[j, j])
    end
    return out
end

@inline function trTAᴴTB(T, A, B)
    N = length(T)
    @boundscheck (@assert size(A) == size(B) == (N, N))
    out = zero(promote_type(eltype(T), eltype(A), eltype(B)))
    @inbounds for j in 1:N, i in 1:N
        out += T[i] * T[j] * conj(A[i, j]) * B[i, j]
    end
    return out
end

@inline function trT²A(T, A::SelfAdjoint)
    N = length(T)
    @boundscheck (@assert size(A) == (N, N))
    out = zero(promote_type(eltype(T), real(eltype(A))))
    @inbounds for i in 1:N
        out += T[i]^2 * real(A[i, i])
    end
    return out
end
