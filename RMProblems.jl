module RMProblems

export RMProblem, example_problems

using LinearAlgebra
using SparseArrays

struct RMProblem{
    U<:AbstractMatrix{ComplexF64},
    V<:Union{AbstractMatrix{ComplexF64},UniformScaling{ComplexF64}},
}
    M::U
    L::V
    R::V
    T::Vector{Float64}
    LiRi::V
    LiMRi::U

    function RMProblem(M::U, L::V, R::V, T) where {U,V}
        N = size(M, 1)
        if size(M, 2) != N
            throw(DimensionMismatch("`M` is not square"))
        end
        if !(V <: UniformScaling) && (
            size(L, 1) != N || size(L, 2) != N || size(R, 1) != N || size(R, 2) != 1
        )
            throw(DimensionMismatch("the sizes of `L` and `R` do not match `M`"))
        end
        if length(T) != N 
            throw(DimensionMismatch("the length of `T` does not match the problem size"))
        end
        return new{U,V}(
            M, L, R, T, (L \ I) / R, (L \ M) / R
        )
    end
end

const example_problems = [
    (; N, σ, τ) -> RMProblem(
        Diagonal(complex(-1.0).^(1:N)),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, σ, τ) -> RMProblem(
        Diagonal(exp(im * 2pi / 3).^(1:N)),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, σ, τ) -> RMProblem(
        Diagonal(exp(im * pi / 4) * ((1.0im).^(1:N))),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, σ, w, τ) -> RMProblem(
        Bidiagonal(
            complex(-1.0).^(1:N),
            complex(w) * vcat(
                fill(1.0, rem(N - 1, 2)), repeat([0.0; 1.0], div(N - 1, 2)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, σ, w, τ) -> RMProblem(
        Bidiagonal(
            exp(im * 2pi / 3).^(1:N),
            complex(w) * vcat(
                fill(1.0, rem(N - 1, 3)),
                repeat([0.0; 1.0; 1.0], div(N - 1, 3)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, σ, w, τ) -> RMProblem(
        Bidiagonal(
            exp(im * pi / 4) * ((1.0im).^(1:N)),
            complex(w) * vcat(
                fill(1.0, rem(N - 1, 4)),
                repeat([0.0; 1.0; 1.0; 1.0], div(N - 1, 4)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
        fill(sqrt(τ), N),
    );
    (; N, w, τ) -> RMProblem(
        spdiagm(1 => fill(complex(w), N - 1)),
        complex(1.0) * I,
        complex(1.0) * I,
        fill(sqrt(τ), N),
    );
    (; N, w, τ) -> RMProblem(
        spdiagm(1 => fill(complex(w), N - 1), -(N - 1) => [complex(w)]),
        complex(1.0) * I,
        complex(1.0) * I,
        fill(sqrt(τ), N),
    );
]

end
