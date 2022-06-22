using CUDA

function unified_cuarray(::Type{T}, dims::NTuple{N}) where {T, N}
    buf = Mem.alloc(Mem.Unified, prod(dims) * sizeof(T))
    array = unsafe_wrap(CuArray{T, N}, convert(CuPtr{T}, buf), dims)
    finalizer(_ -> Mem.free(buf), array)
    return array
end

device!(0)
A = CUDA.rand(5, 5)
device!(1)
U = unified_cuarray(Float32, (5, 5))
device!(0)
copyto!(U, A)
