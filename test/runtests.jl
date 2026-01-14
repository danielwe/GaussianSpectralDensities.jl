using TestItemRunner

@run_package_tests verbose = true

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
        check_all_explicit_imports_via_owners, check_all_explicit_imports_are_public,
        check_all_qualified_accesses_via_owners, check_all_qualified_accesses_are_public,
        check_no_self_qualified_accesses

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
    @test nothing === check_all_qualified_accesses_via_owners(
        GaussianSpectralDensities;
        require_submodule_access=true,
        allow_internal_accesses=false,
    )
    @test nothing === check_no_self_qualified_accesses(GaussianSpectralDensities)
    if (VERSION >= v"1.11.0-DEV.469")
        # Old Julia only has export, not public, making these tests way too strict
        @test nothing === check_all_explicit_imports_are_public(
            GaussianSpectralDensities; allow_internal_imports=false,
        )
        @test nothing === check_all_qualified_accesses_are_public(
            GaussianSpectralDensities;
            ignore=(
                :T,        # EnumX.jl; XXX: remove when EnumX.jl#17 is merged
                :CHOLMOD,  # SparseArrays.jl
                :LM,       # ArnoldiMethod.jl
                :potrf!,   # LinearAlgebra.LAPACK
                :potri!,   # LinearAlgebra.LAPACK
            ),
            allow_internal_accesses=false,
        )
    end
end
