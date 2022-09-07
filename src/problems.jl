"""
    ResolventProblem(
        ensemble::AbstractMatrixEnsemble{F},
        z::Union{Number,AbstractMatrix},
        objective=TraceObjective(one(F));
        normalizeQ=false,
    ) where {F}

Specification for the resolvent averaging problem at a single point in the complex plane

# Arguments

- `ensemble`: random matrix ensemble over which to average the resolvent
- `z`: coordinates in the complex plane at which to calculate the averaged resolvent. This
    can be either a single complex number or a matrix representing a rectangular grid in the
    complex plane. Use `missing` to exclude grid points from calculation.
- `objective`: objective to solve for. The default trace-based
    objective obtains the true averaged resolvent, but other objectives, including
    regularizer-based ones, are relevant for certain purposes.
- `normalizeQ`: whether to replace ``QQ^†`` with ``(QQ^† + Q^†Q) / 2`` in the
    self-consistent equations for the averaged resolvent. This mimicks a scenario where
    ``QQ^† = Q^†Q``, i.e., where ``Q`` is normal. This is a mathematically unjustified trick
    but seems to be a quick way around order of limits issues in certain experiments.
    However, in most cases it gives answers that are subtly but unambiguously wrong.
"""
struct ResolventProblem{
    # TODO: what's a better name than ResolventProblem?
    F,
    E<:AbstractMatrixEnsemble{F},
    Z<:Union{Complex{F},Matrix{Union{Missing,Complex{F}}}},
    O<:AbstractObjective{F},
}
    ensemble::E
    z::Z
    objective::O
    normalizeQ::Bool
    # TODO: would a list of points, i.e., unstructured meshes, be better than a rectangular
    # grid represented by a matrix? Big change, and it seems harder to approximate
    # derivatives and hence the spectral density, but finite element packages might have
    # built-in solutions for that

    function ResolventProblem(
        ensemble::E, z, objective=TraceObjective(one(F)); normalizeQ=false
    ) where {F,E<:AbstractMatrixEnsemble{F}}
        if z isa Number
            z = convert(Complex{F}, z)
        elseif z isa AbstractMatrix
            z = convert(Matrix{Union{Missing,Complex{F}}}, z)
        end
        objective = adapt(F, objective)
        return new{F,E,typeof(z),typeof(objective)}(
            ensemble, z, objective, normalizeQ
        )
    end
end

#= WIP automesh is not ready for this yet, and this would depend on LazySets.jl

ResolventProblem(objective, ensemble; kwargs...)
    return SpectralProblem(objective, automesh(ensemble), ensemble; kwargs...)
end
=#
