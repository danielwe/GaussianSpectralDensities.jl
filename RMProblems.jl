module RMProblems

export RMProblem, example_problems

using LinearAlgebra

struct RMProblem{
    U<:AbstractMatrix{ComplexF64},
    V<:Union{AbstractMatrix{ComplexF64},UniformScaling{ComplexF64}},
}
    M::U
    L::V
    R::V
    LiRi::V
    LiMRi::U
    RMProblem(M::U, L::V, R::V) where {U,V} = new{U,V}(
        M, L, R, (L \ I) / R, (L \ M) / R
    )
end

example_problems = [
    (; N, σ) -> RMProblem(
        Diagonal(complex(-1.0).^(1:N)),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ) -> RMProblem(
        Diagonal(exp(im * 2pi / 3).^(1:N)),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ) -> RMProblem(
        Diagonal(exp(im * pi / 4) * ((1.0im).^(1:N))),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ, u) -> RMProblem(
        Bidiagonal(
            complex(-1.0).^(1:N),
            complex(u) * vcat(
                repeat([1.0], rem(N - 1, 2)), repeat([0.0; 1.0], div(N - 1, 2)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ, u) -> RMProblem(
        Bidiagonal(
            exp(im * 2pi / 3).^(1:N),
            complex(u) * vcat(
                repeat([1.0], rem(N - 1, 3)),
                repeat([0.0; 1.0; 1.0], div(N - 1, 3)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ, u) -> RMProblem(
        Bidiagonal(
            exp(im * pi / 4) * ((1.0im).^(1:N)),
            complex(u) * vcat(
                repeat([1.0], rem(N - 1, 4)),
                repeat([0.0; 1.0; 1.0; 1.0], div(N - 1, 4)),
            ),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
    );
]

end
