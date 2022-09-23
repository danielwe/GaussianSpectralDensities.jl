"""
    LocalResolventSolution(problem::AbstractLocalResolventProblem, q, γ², δ², k, E)
    LocalResolventSolution{F}(problem, q, γ², δ², k, E)

Represent a solution of an averaged resolvent equation at a point in the complex plane

The constructor does not verify that the parameters form a consistent solution of the given
problem. Use the function `eqsolve!` to obtain a consistent `LocalResolventSolution`.

# Arguments

- `problem`: the averaged resolvent equation problem
- `q`: value of the self-consistently determined parameter ``q``
- `γ²`: value of the regularizing parameter ``γ²``
- `δ²`: value of the regularizing parameter ``δ²``
- `k`: value of the normalized trace ``\\tr[(QQ^† + γ²I)⁻¹] / N``.
- `E`: value of the normalized trace of the resolvent ``\\tr[Q^†(QQ^† + γ²I)⁻¹] / N``
"""
struct LocalResolventSolution{F,P<:AbstractLocalResolventProblem{F}}
    problem::P
    q::Complex{F}
    regularizer::Regularizer{F}
    γ²::F
    δ²::F
    k::F
    E::Complex{F}

    function LocalResolventSolution{F}(problem, q, regularizer, γ², δ², k, E) where {F}
        problem = adapt(F, problem)
        return new{F,typeof(problem)}(problem, q, regularizer, γ², δ², k, E)
    end
end

function LocalResolventSolution(
    problem::AbstractLocalResolventProblem{F}, q, regularizer, γ², δ², k, E
) where {F}
    return LocalResolventSolution{F}(problem, q, regularizer, γ², δ², k, E)
end

function Base.convert(
    ::Type{P}, s::Q
) where {F,P<:LocalResolventSolution{F},Q<:LocalResolventSolution}
    (s isa P) && return s
    return LocalResolventSolution{F}(s.problem, s.q, s.regularizer, s.γ², s.δ², s.k, s.E)
end
