module RMProblems

export RMProblem, rm_problems

using LinearAlgebra

struct RMProblem{
        T<:AbstractArray{Complex{Float64},2},
        U<:Union{AbstractArray{Complex{Float64},2},UniformScaling{Complex{Float64}}},
    }
    M::T
    L::U
    R::U
    LiRi::U
    LiMRi::T
    RMProblem(M::T, L::U, R::U) where {T,U} = new{T,U}(
        M, L, R, (L \ I) / R, (L \ M) / R
    )
end

rm_problems = [
    (; N, σ) -> RMProblem(
        Diagonal(complex(-1.0) .^ (1:N)),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ, u) -> RMProblem(
        Bidiagonal(
            complex(-1.0) .^ (1:N),
            complex(u) * vcat(repeat([1.0; 0.0], div(N, 2) - 1), 1.0),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ) -> RMProblem(
        Diagonal(exp(im * pi / 4) * ((1.0im) .^ (1:N))),
        complex(1.0) * I,
        complex(σ) * I,
    );
    (; N, σ, u) -> RMProblem(
        Bidiagonal(
            exp(im * pi / 4) * ((1.0im) .^ (1:N)),
            complex(u) * vcat(repeat([1.0; 0.0], div(N, 2) - 1), 1.0),
            :U,
        ),
        complex(1.0) * I,
        complex(σ) * I,
    );
]

end
