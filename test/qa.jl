@testitem "Aqua.jl" begin
    using Aqua: test_all
    test_all(GaussianSpectralDensities)
end

@testitem "ExplicitImports.jl" begin
    using ExplicitImports:
        test_all_explicit_imports_are_public,
        test_all_explicit_imports_via_owners,
        test_all_qualified_accesses_are_public,
        test_all_qualified_accesses_via_owners,
        test_no_implicit_imports,
        test_no_self_qualified_accesses,
        test_no_stale_explicit_imports

    # ExplicitImports.jl chokes on EnumX modules
    const enumx_modules = (
        GaussianSpectralDensities.Regularizers.RegularizerType,
        GaussianSpectralDensities.Workspaces.Valid,
    )

    test_no_implicit_imports(GaussianSpectralDensities; allow_unanalyzable=enumx_modules)
    test_no_stale_explicit_imports(
        GaussianSpectralDensities; allow_unanalyzable=enumx_modules
    )
    test_all_explicit_imports_via_owners(
        GaussianSpectralDensities;
        require_submodule_import=true,
        allow_internal_imports=false,
    )
    test_all_explicit_imports_are_public(
        GaussianSpectralDensities; allow_internal_imports=false
    )
    test_all_qualified_accesses_via_owners(
        GaussianSpectralDensities;
        require_submodule_access=true,
        allow_internal_accesses=false,
    )
    test_all_qualified_accesses_are_public(
        GaussianSpectralDensities;
        ignore=(
            :SizeUnknown,        # Base
            :throw_boundserror,  # Base
            :CHOLMOD,            # SparseArrays.jl
            :LM,                 # ArnoldiMethod.jl
            :sym_uplo,           # LinearAlgebra
            :potrf!,             # LinearAlgebra.LAPACK
            :potri!,             # LinearAlgebra.LAPACK
            :init,               # CommonSolve.jl
            :step!,              # CommonSolve.jl
            :solve!,             # CommonSolve.jl
        ),
        allow_internal_accesses=false,
    )
    test_no_self_qualified_accesses(GaussianSpectralDensities)
end
