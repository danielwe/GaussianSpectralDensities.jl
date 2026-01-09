using TestItemRunner

@run_package_tests

@testitem "Aqua.jl" begin
    using Aqua
    test_all(
        GaussianSpectralDensities;
        # XXX: Temporary until the rest of the code is revived
        stale_deps=false,
    )
end
