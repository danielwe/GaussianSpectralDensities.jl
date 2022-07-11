using Aqua
using GaussianSpectralDensities
using Test

@testset "Aqua" begin
    Aqua.test_all(GaussianSpectralDensities)
end

include("tests.jl")
