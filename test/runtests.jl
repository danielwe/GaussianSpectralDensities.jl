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
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports,
        check_all_explicit_imports_via_owners, check_all_explicit_imports_are_public

    # ExplicitImports.jl chokes on EnumX modules
    const enumx_modules = (
        GaussianSpectralDensities.Regularizers.RegularizerType,
        GaussianSpectralDensities.Workspaces.Clean,
    )

    @test nothing === check_no_implicit_imports(
        GaussianSpectralDensities; allow_unanalyzable=enumx_modules,
    )
    @test nothing === check_no_stale_explicit_imports(
        GaussianSpectralDensities; allow_unanalyzable=enumx_modules,
    )
    @test nothing === check_all_explicit_imports_via_owners(
        GaussianSpectralDensities;
        require_submodule_import=true,
        allow_internal_imports=false,
    )
    @test nothing === check_all_explicit_imports_are_public(
        GaussianSpectralDensities; allow_internal_imports=false,
    )
end
