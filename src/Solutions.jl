abstract type AbstractSolution end

struct IntermediateSolution <: AbstractSolution
    γsq::Float64
    δsq::Float64
    q::ComplexF64
    k::Float64
end

function IntermediateSolution()
    return IntermediateSolution(NaN, NaN, complex(NaN, NaN), NaN)
end

const SolutionTrace = Vector{IntermediateSolution}

struct Solution <: AbstractSolution
    γsq::Float64
    δsq::Float64
    q::ComplexF64
    k::Float64
    trace::SolutionTrace
end

function Solution(trace::SolutionTrace)
    return Solution(NaN, NaN, complex(NaN, NaN), NaN, trace)
end
