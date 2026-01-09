using TestItemRunner

@run_package_tests

@testitem "Aqua.jl" begin
    using Aqua: test_all
    test_all(
        GaussianSpectralDensities;
        # XXX: Temporary until the rest of the code is revived
        stale_deps=false,
    )
end

@testitem "ExplicitImports.jl" begin
    # TODO: add other checks from ExplicitImports.jl
    using ExplicitImports: check_no_implicit_imports

    # ExplicitImports.jl chokes on EnumX modules
    const enumx_modules = (
        GaussianSpectralDensities.RegularizerType,
        GaussianSpectralDensities.Workspaces.Clean,
    )

    @test nothing === check_no_implicit_imports(
        GaussianSpectralDensities; allow_unanalyzable=enumx_modules,
    )
end
