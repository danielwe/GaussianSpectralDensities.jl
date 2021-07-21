using .CUDA
using .CUDA: @allowscalar, unsafe_free!
using .CUDA.APIUtils
using .CUDA.CUSOLVER: dense_handle
using .CUDA.CUSOLVER: cusolverDnSsyevdx, cusolverDnSsyevdx_bufferSize
using .CUDA.CUSOLVER: cusolverDnDsyevdx, cusolverDnDsyevdx_bufferSize
using .CUDA.CUSOLVER: cusolverDnCheevdx, cusolverDnCheevdx_bufferSize
using .CUDA.CUSOLVER: cusolverDnZheevdx, cusolverDnZheevdx_bufferSize
using .CUDA.CUSOLVER: cusolverEigRange_t
using .CUDA.CUSOLVER: CUSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_V
using LinearAlgebra: checksquare, BlasReal

CUDA.allowscalar(false)

const GeneralCuMatrix{T} = Union{
    CuMatrix{T}, Adjoint{T, <:CuMatrix}, Transpose{T, <:CuMatrix}
}
const AnyCuMatrix = Union{GeneralCuMatrix, SelfAdjoint{<:Real, <:CuMatrix}}

struct CuMatrixEnsemble{R, U, V, W} <: AbstractMatrixEnsemble{R, U, V, W}
    M::U
    L::V
    R::W
    L⁻¹R⁻¹::CuMatrix{Complex{R}}
    L⁻¹MR⁻¹::CuMatrix{Complex{R}}
    T::CuVector{R}

    function CuMatrixEnsemble(M, L, R, T::AbstractVector)
        T = CUDA.adapt(CuArray, T)
        N = length(T)
        for (A, c) in ((M, 'M'), (L, 'L'), (R, 'R'))
            if !(A isa UniformScaling) && (size(A) != (N, N))
                throw(DimensionMismatch("the size of $c does not match the length of T"))
            end
        end
        Lf, Rf = _factorize(L), _factorize(_complex(R))
        L⁻¹R⁻¹ = _dense_cumatrix(A⁻¹BC⁻¹(Lf, I, Rf), N)
        L⁻¹MR⁻¹ = _dense_cumatrix(A⁻¹BC⁻¹(Lf, M, Rf), N)
        return new{real(eltype(M)), typeof(M), typeof(L), typeof(R)}(
            M, L, R, L⁻¹R⁻¹, L⁻¹MR⁻¹, T
        )
    end
end

_factorize(A::Union{CuMatrix, CUDA.WrappedArray{<:Any, 2, CuArray, CuArray}}) = A

_dense_cumatrix(A::CuMatrix, ::Any) = A
_dense_cumatrix(A::UniformScaling, N) = CuMatrix(A(N))
_dense_cumatrix(A, ::Any) = CuMatrix(CUDA.adapt(Array, A))  # Dense without device indexing

function CuMatrixEnsemble(M, L, R, τ::Real)
    return CuMatrixEnsemble(M, L, R, CUDA.fill(sqrt(τ), size(M, 1)))
end

function CuMatrixEnsemble(ensemble::MatrixEnsemble)
    return CuMatrixEnsemble(ensemble.M, ensemble.L, ensemble.R, ensemble.T)
end

struct CuWorkspace{R, U, V, W} <: AbstractWorkspace{R}
    status::WorkspaceStatus{R, CuMatrixEnsemble{R, U, V, W}}
    G₀⁻¹_diag::CuVector{Complex{R}}
    Q::CuMatrix{Complex{R}}
    QQᴴ::Hermitian{Complex{R},CuMatrix{Complex{R}}}
    P⁻¹f::CuMatrix{Complex{R}}
    P::Hermitian{Complex{R},CuMatrix{Complex{R}}}
    PQ::CuMatrix{Complex{R}}
    QᴴPQ::Hermitian{Complex{R},CuMatrix{Complex{R}}}
    QP::CuMatrix{Complex{R}}
    QPQ::CuMatrix{Complex{R}}  # Also used for PL⁻¹R⁻¹
    s²min::Base.RefValue{R}

    function CuWorkspace(ensemble::CuMatrixEnsemble{R, U, V, W}) where {R, U, V, W}
        N = length(ensemble.T)
        G₀⁻¹_diag = CuVector{Complex{R}}(undef, N)
        Q = similar(ensemble.L⁻¹MR⁻¹)
        PQ = CuMatrix{Complex{R}}(undef, N, N)
        P = Hermitian(similar(PQ))
        return new{R, U, V, W}(
            WorkspaceStatus(ensemble),
            G₀⁻¹_diag,
            Q,
            Hermitian(similar(PQ)),
            similar(PQ),
            P,
            PQ,
            Hermitian(similar(PQ)),
            similar(PQ),
            similar(PQ),
            Ref(R(NaN)),
        )
    end
end

workspace(ensemble::CuMatrixEnsemble) = CuWorkspace(ensemble)

NBLOCKS() = 64
NTHREADS() = 32
NBLOCKS_LINEAR() = 1
NTHREADS_LINEAR() = 64

@inline function zA⁻B!(C::CuMatrix, z, A::CuMatrix, B::CuMatrix)
    function zA⁻B_kernel!(C, z, A, B)
        @inbounds for j in blockIdx().x:gridDim().x:size(C, 2)
            for i in threadIdx().x:blockDim().x:size(C, 1)
                C[i, j] = muladd(z, A[i, j], -B[i, j])
            end
        end
        return nothing
    end

    @boundscheck (@assert size(A) == size(B) == size(C))
    @cuda blocks=NBLOCKS() threads=NTHREADS() zA⁻B_kernel!(C, z, A, B)
    return nothing
end

@inline function copy_diag!(d::CuVector, A::CuMatrix)
    function copy_diag_kernel!(d, A)
        gi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        @inbounds for i in gi:stride:length(d)
            d[i] = A[i, i]
        end
        return nothing
    end

    @boundscheck begin
        N = length(d)
        @assert size(A) == (N, N)
    end
    @cuda blocks=NBLOCKS_LINEAR() threads=NTHREADS_LINEAR() copy_diag_kernel!(d, A)
    return nothing
end

@inline function set_diag_muladd!(A::CuMatrix, λ, v1::CuVector, v2::CuVector)
    function set_diag_muladd_kernel!(A, λ, v1, v2)
        gi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        @inbounds for i in gi:stride:length(v1)
            A[i, i] = muladd(λ, v1[i], v2[i])
        end
        return nothing
    end

    @boundscheck begin
        N = length(v1)
        @assert length(v2) == N && size(A) == (N, N)
    end
    @cuda blocks=NBLOCKS_LINEAR() threads=NTHREADS_LINEAR() set_diag_muladd_kernel!(
        A, λ, v1, v2
    )
    return nothing
end

@inline function add_to_diag!(A::CuMatrix, λ)
    function add_to_diag_kernel!(A, λ)
        gi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        @inbounds for i in gi:stride:size(A, 1)
            A[i, i] += λ
        end
        return nothing
    end

    @boundscheck (@assert size(A, 1) == size(A, 2))
    @cuda blocks=NBLOCKS_LINEAR() threads=NTHREADS_LINEAR() add_to_diag_kernel!(A, λ)
    return nothing
end

@inline function trTAᴴB(
    T::CuVector{R}, A::GeneralCuMatrix{U}, B::SelfAdjoint{<:Real, <:CuMatrix}
) where {R, U}
    function trTAᴴB_kernel!(out::CuDeviceVector{S}, T, A, uplo, B) where {S}
        ti, bi = threadIdx().x, blockIdx().x
        nt, nb = blockDim().x, gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        if uplo === 'U'
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[j] * conj(A[i, j]) * B[i, j]
                    elseif i == j
                        shared[ti] += T[j] * conj(A[i, j]) * real(B[i, j])
                    else
                        shared[ti] += T[j] * conj(A[i, j]) * conj(B[j, i])
                    end
                end
            end
        else
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[j] * conj(A[i, j]) * conj(B[j, i])
                    elseif i == j
                        shared[ti] += T[j] * conj(A[i, j]) * real(B[i, j])
                    else
                        shared[ti] += T[j] * conj(A[i, j]) * B[i, j]
                    end
                end
            end
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    @boundscheck begin
        N = length(T)
        @assert size(A) == size(B) == (N, N)
    end
    blocks, threads = NBLOCKS(), NTHREADS()
    S = promote_type(R, U, eltype(B))
    out = CuVector{S}(undef, blocks)
    shmem = threads * sizeof(S)
    @cuda blocks=blocks threads=threads shmem=shmem trTAᴴB_kernel!(
        out, T, A, B.uplo, B.data
    )
    s = _sum(out)
    unsafe_free!(out)
    return s
end

@inline function trTAᴴTB(T::CuVector{R}, A::AnyCuMatrix, B::AnyCuMatrix) where {R}
    function trTAᴴTB_kernel!(out::CuDeviceVector{S}, T, A, B) where {S}
        ti, bi = threadIdx().x, blockIdx().x
        nt, nb = blockDim().x, gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        @inbounds for j in bi:nb:length(T), i in ti:nt:length(T)
            shared[ti] += T[i] * T[j] * conj(A[i, j]) * B[i, j]
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    function trTAᴴTB_ASA_kernel!(out::CuDeviceVector{S}, T, Auplo, A, B) where {S}
        ti, bi = threadIdx().x, blockIdx().x
        nt, nb = blockDim().x, gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        if Auplo === 'U'
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * B[i, j]
                    elseif i == j
                        shared[ti] += T[i] * T[j] * real(A[i, j]) * B[i, j]
                    else
                        shared[ti] += T[i] * T[j] * A[j, i] * B[i, j]
                    end
                end
            end
        else
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[i] * T[j] * A[j, i] * B[i, j]
                    elseif i == j
                        shared[ti] += T[i] * T[j] * real(A[i, j]) * B[i, j]
                    else
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * B[i, j]
                    end
                end
            end
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    function trTAᴴTB_BSA_kernel!(out::CuDeviceVector{S}, T, A, Buplo, B) where {S}
        ti, bi = threadIdx().x, blockIdx().x
        nt, nb = blockDim().x, gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        if Buplo === 'U'
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * B[i, j]
                    elseif i == j
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * real(B[i, j])
                    else
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * conj(B[j, i])
                    end
                end
            end
        else
            @inbounds for j in bi:nb:length(T)
                for i in ti:nt:length(T)
                    if i < j
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * conj(B[j, i])
                    elseif i == j
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * real(B[i, j])
                    else
                        shared[ti] += T[i] * T[j] * conj(A[i, j]) * B[i, j]
                    end
                end
            end
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    function trTAᴴTB_ABSA_kernel!(out::CuDeviceVector{S}, T, Auplo, A, Buplo, B) where {S}
        ti, bi = threadIdx().x, blockIdx().x
        nt, nb = blockDim().x, gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        if Auplo === 'U'
            if Buplo === 'U'
                @inbounds for j in bi:nb:length(T)
                    for i in ti:nt:length(T)
                        if i < j
                            shared[ti] += T[i] * T[j] * real(conj(A[i, j]) * B[i, j])
                        elseif i == j
                            shared[ti] += T[i] * T[j] * real(A[i, j]) * real(B[i, j])
                        else
                            shared[ti] += T[i] * T[j] * real(A[j, i] * conj(B[j, i]))
                        end
                    end
                end
            else
                @inbounds for j in bi:nb:length(T)
                    for i in ti:nt:length(T)
                        if i < j
                            shared[ti] += T[i] * T[j] * real(conj(A[i, j]) * conj(B[j, i]))
                        elseif i == j
                            shared[ti] += T[i] * T[j] * real(A[i, j]) * real(B[i, j])
                        else
                            shared[ti] += T[i] * T[j] * real(A[j, i] * B[i, j])
                        end
                    end
                end
            end
        else
            if Buplo === 'U'
                @inbounds for j in bi:nb:length(T)
                    for i in ti:nt:length(T)
                        if i < j
                            shared[ti] += T[i] * T[j] * real(A[j, i] * B[i, j])
                        elseif i == j
                            shared[ti] += T[i] * T[j] * real(A[i, j]) * real(B[i, j])
                        else
                            shared[ti] += T[i] * T[j] * real(conj(A[i, j]) * conj(B[j, i]))
                        end
                    end
                end
            else
                @inbounds for j in bi:nb:length(T)
                    for i in ti:nt:length(T)
                        if i < j
                            shared[ti] += T[i] * T[j] * real(A[j, i] * conj(B[j, i]))
                        elseif i == j
                            shared[ti] += T[i] * T[j] * real(A[i, j]) * real(B[i, j])
                        else
                            shared[ti] += T[i] * T[j] * real(conj(A[i, j]) * B[i, j])
                        end
                    end
                end
            end
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    @boundscheck begin
        N = length(T)
        @assert size(A) == size(B) == (N, N)
    end
    blocks, threads = NBLOCKS(), NTHREADS()
    if (A isa SelfAdjoint) && (B isa SelfAdjoint)
        S = promote_type(R, real(eltype(A)), real(eltype(B)))
    else
        S = promote_type(R, eltype(A), eltype(B))
    end
    out = CuVector{S}(undef, blocks)
    shmem = threads * sizeof(S)
    if (A isa SelfAdjoint) && (B isa SelfAdjoint)
        @cuda blocks=blocks threads=threads shmem=shmem trTAᴴTB_ABSA_kernel!(
            out, T, A.uplo, A.data, B.uplo, B.data
        )
    elseif A isa SelfAdjoint
        @cuda blocks=blocks threads=threads shmem=shmem trTAᴴTB_ASA_kernel!(
            out, T, A.uplo, A.data, B
        )
    elseif B isa SelfAdjoint
        @cuda blocks=blocks threads=threads shmem=shmem trTAᴴTB_BSA_kernel!(
            out, T, A, B.uplo, B.data
        )
    else
        @cuda blocks=blocks threads=threads shmem=shmem trTAᴴTB_kernel!(out, T, A, B)
    end
    s = _sum(out)
    unsafe_free!(out)
    return s
end

@inline function trT²A(T::CuVector{R}, A::SelfAdjoint{<:Real, <:CuMatrix}) where {R}
    function trT²A_kernel!(out::CuDeviceVector{S}, T, A) where {S}
        ti, bi, nt = threadIdx().x, blockIdx().x, blockDim().x
        gi = (bi - 1) * nt + ti
        stride = nt * gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        @inbounds for i in gi:stride:length(T)
            shared[ti] += T[i]^2 * real(A[i, i])
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    @boundscheck begin
        N = length(T)
        @assert size(A) == (N, N)
    end
    blocks, threads = NBLOCKS_LINEAR(), NTHREADS_LINEAR()
    S = promote_type(R, real(eltype(A)))
    out = CuVector{S}(undef, blocks)
    shmem = threads * sizeof(S)
    @cuda blocks=blocks threads=threads shmem=shmem trT²A_kernel!(out, T, A.data)
    s = _sum(out)
    unsafe_free!(out)
    return s
end

@inline function _tr(A::AnyCuMatrix)
    function _tr_kernel!(out::CuDeviceVector{S}, A) where {S}
        ti, bi, nt = threadIdx().x, blockIdx().x, blockDim().x
        gi = (bi - 1) * nt + ti
        stride = nt * gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        @inbounds for i in gi:stride:size(A, 1)
            shared[ti] += A[i, i]
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    function _tr_SA_kernel!(out::CuDeviceVector{S}, A) where {S}
        ti, bi, nt = threadIdx().x, blockIdx().x, blockDim().x
        gi = (bi - 1) * nt + ti
        stride = nt * gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        @inbounds for i in gi:stride:size(A, 1)
            shared[ti] += real(A[i, i])
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    if A isa SelfAdjoint  # Square check redundant for SelfAdjoint
        T = real(eltype(A))
    else
        @boundscheck (@assert size(A, 1) == size(A, 2))
        T = eltype(A)
    end
    blocks, threads = NBLOCKS_LINEAR(), NTHREADS_LINEAR()
    shmem = threads * sizeof(T)
    out = CuVector{T}(undef, blocks)
    if A isa SelfAdjoint
        @cuda blocks=blocks threads=threads shmem=shmem _tr_SA_kernel!(out, A.data)
    else
        @cuda blocks=blocks threads=threads shmem=shmem _tr_kernel!(out, A)
    end
    s = _sum(out)
    unsafe_free!(out)
    return s
end

@inline function _dot(a::CuVector{T}, b::CuVector{U}) where {T, U}
    function _dot_kernel!(out::CuDeviceVector{S}, a, b) where {S}
        ti, bi, nt = threadIdx().x, blockIdx().x, blockDim().x
        gi = (bi - 1) * nt + ti
        stride = nt * gridDim().x
        shared = @cuDynamicSharedMem(S, nt)
        shared[ti] = 0
        @inbounds for i in gi:stride:length(a)
            shared[ti] += conj(a[i]) * b[i]
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[bi] = shared[1]
        end
        return nothing
    end

    @boundscheck (@assert length(a) == length(b))
    blocks, threads = NBLOCKS_LINEAR(), NTHREADS_LINEAR()
    S = promote_type(T, U)
    out = CuVector{S}(undef, blocks)
    shmem = threads * sizeof(S)
    @cuda blocks=blocks threads=threads shmem=shmem _dot_kernel!(out, a, b)
    s = _sum(out)
    unsafe_free!(out)
    return s
end

function _sum(a::CuArray{T}) where {T}
    function _sum_kernel!(out::CuDeviceArray{T}, a) where {T}
        ti, nt = threadIdx().x, blockDim().x
        shared = @cuDynamicSharedMem(T, nt)
        shared[ti] = 0
        @inbounds for i in ti:nt:length(a)
            shared[ti] += a[i]
        end
        reduce_sum(shared, ti, nt)
        if ti == 1
            out[1] = shared[1]
        end
        return nothing
    end

    N = length(a)
    if N == 1
        return @allowscalar a[1]
    end
    threads, next = 1, 2
    while next <= N  # Find largest 2^k <= N
        threads = next
        next <<= 1
    end
    out = CuVector{T}(undef, 1)
    shmem = threads * sizeof(T)
    @cuda threads=threads shmem=shmem _sum_kernel!(out, a)
    synchronize(CuDefaultStream())
    s = @allowscalar out[1]
    unsafe_free!(out)
    return s
end

function reduce_sum(arr::CuDeviceVector, ti, n)
    while n > 1
        n >>= 1
        sync_threads()
        if ti <= n
            arr[ti] += arr[ti + n]
        end
    end
    return nothing
end

# A pull request or two to CUDA.jl is in order
for (jname, bname, fname, elty, relty) in ((:syevdx!, :cusolverDnSsyevdx_bufferSize, :cusolverDnSsyevdx, :Float32, :Float32),
                                           (:syevdx!, :cusolverDnDsyevdx_bufferSize, :cusolverDnDsyevdx, :Float64, :Float64),
                                           (:heevdx!, :cusolverDnCheevdx_bufferSize, :cusolverDnCheevdx, :ComplexF32, :Float32),
                                           (:heevdx!, :cusolverDnZheevdx_bufferSize, :cusolverDnZheevdx, :ComplexF64, :Float64))
    @eval begin
        function $jname(jobz::Char,
                        range::Char,
                        uplo::Char,
                        A::StridedCuMatrix{$elty},
                        vl::AbstractFloat,
                        vu::AbstractFloat,
                        il::Integer,
                        iu::Integer)
            n       = checksquare(A)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            lda     = max(1, stride(A, 2))
            h_meig  = Ref{Cint}()
            W       = CuArray{$relty}(undef, n)

            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dense_handle(), jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig,
                       W, out)
                return out[]
            end

            devinfo = CuArray{Cint}(undef, 1)
            with_workspace($elty, bufferSize) do buffer
                $fname(dense_handle(), jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig,
                       W, buffer, length(buffer), devinfo)
            end

            info = @allowscalar devinfo[1]
            unsafe_free!(devinfo)
            if info < 0
                throw(ArgumentError("The $(info)th parameter is wrong"))
            end

            if jobz == 'N'
                return W[1:h_meig[]], A[:, 1:0]
            else
                return W[1:h_meig[]], A[:, 1:h_meig[]]
            end
        end
    end
end

function Base.convert(::Type{cusolverEigRange_t}, range::Char)
    if range == 'A'
        return CUSOLVER_EIG_RANGE_ALL
    end
    if range == 'V'
        return CUSOLVER_EIG_RANGE_V
    end
    if range == 'I'
        return CUSOLVER_EIG_RANGE_I
    end
    throw(ArgumentError("unkonwn eigenvalue solver range $range"))
end

function LinearAlgebra.eigvals!(
    A::RealSymmetric{T, <:CuMatrix}, irange::UnitRange
) where {T<:BlasReal}
    return syevdx!('N', 'I', A.uplo, A.data, T(0.0), T(0.0), irange.start, irange.stop)[1]
end

function LinearAlgebra.eigvals!(
    A::Hermitian{Complex{T}, <:CuMatrix}, irange::UnitRange
) where {T<:BlasReal}
    return heevdx!('N', 'I', A.uplo, A.data, T(0.0), T(0.0), irange.start, irange.stop)[1]
end

function eigmin!(A::SelfAdjoint{<:BlasReal, <:CuMatrix})
    ev = eigvals!(A, 1:1)
    return @allowscalar ev[1]
end

function LinearAlgebra.mul!(
    C::CuMatrix{T}, A::CuMatrix{T}, B::RealSymmetric{T, <:CuMatrix}
) where {T<:BlasReal}
    return CUBLAS.symm!('R', B.uplo, T(1.0), B.data, A, T(0.0), C)
end

function LinearAlgebra.mul!(
    C::CuMatrix{T}, A::RealSymmetric{T, <:CuMatrix}, B::CuMatrix{T}
) where {T<:BlasReal}
    return CUBLAS.symm!('L', A.uplo, T(1.0), A.data, B, T(0.0), C)
end

function LinearAlgebra.mul!(
    C::CuMatrix{Complex{T}}, A::CuMatrix{Complex{T}}, B::Hermitian{Complex{T}, <:CuMatrix}
) where {T<:BlasReal}
    return CUBLAS.hemm!('R', B.uplo, T(1.0), B.data, A, T(0.0), C)
end

function LinearAlgebra.mul!(
    C::CuMatrix{Complex{T}}, A::Hermitian{Complex{T}, <:CuMatrix}, B::CuMatrix{Complex{T}}
) where {T<:BlasReal}
    return CUBLAS.hemm!('L', A.uplo, T(1.0), A.data, B, T(0.0), C)
end
