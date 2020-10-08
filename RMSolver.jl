module RMSolver

export RMWorkspace, rm_solvers

using LinearAlgebra
using NLsolve
using Roots
using Printf

using RMUtils
using RMProblems

struct RMWorkspace{
        T<:AbstractArray{Complex{Float64},2},
        U<:AbstractArray{Complex{Float64},2},
    }
    Mz::T
    Qz::T
    QzQzg::U
    Pz::U
    other::U
end

function RMWorkspace(problem::RMProblem)
    M = problem.M
    MMa = Array(M * M')
    return RMWorkspace(similar(M), similar(M), similar(MMa), similar(MMa), similar(MMa))
end

function Mz!(z, workspace, problem)
    copyto!(workspace.Mz, z * problem.LiRi)
    workspace.Mz .-= problem.LiMRi
end

function Qzs!(q, gsq, workspace; delta)
    copyto!(workspace.Qz, workspace.Mz)
    for k in 1:size(workspace.Qz)[1]
        workspace.Qz[k, k] -= q
    end
        
    mul!(workspace.QzQzg, workspace.Qz, workspace.Qz')
    if delta
        gsq -= eigmin(Hermitian(workspace.QzQzg))
    end
    for k in 1:size(workspace.QzQzg)[1]
        workspace.QzQzg[k, k] += gsq
    end
    return gsq
end

function fj!(F, J, qv, τ, gsq, workspace; delta)
    N = size(workspace.Mz)[1]
    τNinv = τ / N
    q = complex(qv...)

    Qzs!(q, gsq, workspace, delta=delta)
    QzQzg_f = cholesky!(Hermitian(workspace.QzQzg), check=false)

    copyto!(workspace.Pz, workspace.Qz')
    rdiv!(workspace.Pz, QzQzg_f)

    if !(F == nothing)
        Fc = τNinv * tr(workspace.Pz) - q
        F[1] = real(Fc)
        F[2] = imag(Fc)
    end

    if !(J == nothing)
        # other is (Pz Qz - I)_QzQzg
        mul!(workspace.other, workspace.Pz, workspace.Qz)
        for k in 1:N
            workspace.other[k, k] -= 1.0
        end
        rdiv!(workspace.other, QzQzg_f)
        B = τNinv * tr(workspace.other)
        
        # other is Pz^2
        mul!(workspace.other, workspace.Pz, workspace.Pz)
        A = τNinv * tr(workspace.other) - 1.0

        dFc_dqr = A + B
        dFc_dqi_i = A - B
        J[1, 1] = real(dFc_dqr)
        J[1, 2] = -imag(dFc_dqi_i)
        J[2, 1] = imag(dFc_dqr)
        J[2, 2] = real(dFc_dqi_i)
    end
end

function q!(q0, τ, gsq, workspace; delta)
    qv = nlsolve(
        # only_fj!((F, J, qv) -> fj!(F, J, qv, τ, gsq, workspace, delta=delta)),
        (F, qv) -> fj!(F, nothing, qv, τ, gsq, workspace, delta=delta),
        [real(q0), imag(q0)],
        method=:anderson,
        m=1,
    ).zero
    return complex(qv...)
end

function K!(q, gsq, workspace; delta)
    Qzs!(q, gsq, workspace, delta=delta)
    return K!(workspace)
end

function K!(workspace)
    copyto!(workspace.other, I)
    QzQzg_f = cholesky!(Hermitian(workspace.QzQzg), check=false)
    ldiv!(QzQzg_f, workspace.other)
    return real(tr(workspace.other)) / size(workspace.other)[1]
    # ηz = eigvals!(Hermitian(workspace.QzQzg))
    # return sum(inv.(ηz)) / length(ηz)
end

function q_gsq!(
        z,
        τ,
        dgsq_sequence,
        workspace,
        problem;
        root_algorithm=FalsePosition(:10),
        verbose=false,
    )
    q_sol, gsq_sol = complex(NaN, NaN), NaN
    # gsq_sequence = similar(dgsq_sequence)

    Mz!(z, workspace, problem)
    dgsq = dgsq_sequence[1]
    
    qold = τ * conj(tr(workspace.Mz)) / (dgsq * size(workspace.Mz)[1])
    qold = q!(qold, τ, dgsq, workspace, delta=true)
    # gsq_sequence[1] = gsq = Qzs!(qold, dgsq, workspace, delta=true)

    if verbose 
        q_sequence = similar(dgsq_sequence, Complex{eltype(dgsq_sequence)})
        gsq_sequence = similar(dgsq_sequence)
        K_sequence = similar(dgsq_sequence)
        q_sequence[1] = qold
        gsq_sequence[1] = gsq = Qzs!(qold, dgsq, workspace, delta=true)
        K_sequence[1] = K = K!(workspace)
        last = 1
        println("--------------------------------------------")
        println("   g^2               q                K")
        println("--------------------------------------------")
        @printf("%9.2e  %9.2e + i%9.2e  %9.2e\n", gsq, real(qold), imag(qold), K)
    end

    for (k, dgsq) in enumerate(dgsq_sequence[2:end])
        q = q!(qold, τ, dgsq, workspace, delta=true)
        gsq = Qzs!(q, dgsq, workspace, delta=true)
        # gsq_sequence[k + 1] = gsq
        K = K!(workspace)

        if verbose
            q_sequence[k + 1] = q
            gsq_sequence[k + 1] = gsq
            K_sequence[k + 1] = K
            last = k + 1
            @printf("%9.2e  %9.2e + i%9.2e  %9.2e\n", gsq, real(q), imag(q), K)
        end

        if gsq <= -1.0
            break
        end

        if K >= 1.0
            if verbose
                println("--------------------------------------------")
            end

            # gsq_sol = find_zero(
            #     gsq_ -> K!(
            #         q!(qold, τ, gsq_, workspace, delta=false),
            #         gsq_,
            #         workspace,
            #         delta=false,
            #     ) - 1.0,
            #     (gsq_sequence[k], gsq),
            #     root_algorithm,
            #     verbose=verbose,
            # )
            # q_sol = q!(qold, τ, gsq_sol, workspace, delta=false)

            dgsq_sol = find_zero(
                dgsq_ -> K!(
                    q!(qold, τ, dgsq_, workspace, delta=true),
                    dgsq_,
                    workspace,
                    delta=true,
                ) - 1.0,
                (dgsq_sequence[k], dgsq),
                root_algorithm,
                verbose=verbose,
            )
            q_sol = q!(qold, τ, dgsq_sol, workspace, delta=true)
            gsq_sol = Qzs!(q_sol, dgsq_sol, workspace, delta=true)
            
            break
        end
        qold = q
    end

    if verbose
        println("--------------------------------------------")
        return (
            q_sol,
            gsq_sol,
            gsq_sequence[1:last],
            q_sequence[1:last],
            K_sequence[1:last],
        )
    else
        return q_sol, gsq_sol
    end
end

function q_K!(
        z,
        τ_sequence,
        workspace,
        problem;
        root_algorithm=FalsePosition(:10),
        verbose=false,
    )
    Mz!(z, workspace, problem)

    τ = τ_sequence[1]
    q = q!(0.0, τ, 0.0, workspace, delta=false)

    if verbose 
        T = real(eltype(τ_sequence))
        q_sequence = similar(τ_sequence, Complex{T})
        K_sequence = similar(τ_sequence, T)
        q_sequence[1] = q
        K_sequence[1] = K = K!(q, 0.0, workspace, delta=false)
        println("---------------------------------------------------------")
        println("          τ                       q                K")
        println("---------------------------------------------------------")
        @printf(
            "%9.2e + i%9.2e  %9.2e + i%9.2e  %9.2e\n",
            real(τ),
            imag(τ),
            real(q),
            imag(q),
            K,
        )
    end

    for (k, τ) in enumerate(τ_sequence[2:end])
        q = q!(q, τ, 0.0, workspace, delta=false)
        K = K!(q, 0.0, workspace, delta=false)

        if verbose
            q_sequence[k + 1] = q
            K_sequence[k + 1] = K
            @printf(
                "%9.2e + i%9.2e  %9.2e + i%9.2e  %9.2e\n",
                real(τ),
                imag(τ),
                real(q),
                imag(q),
                K,
            )
        end
    end

    if verbose
        println("--------------------------------------------")
        return q, K, τ_sequence, q_sequence, K_sequence
    else
        return q, K
    end
end

function rm_solvers(; problem, τ, dτ, b, n)
    workspace = RMWorkspace(problem)
    
    dgsq_sequence = b.^(n:-1:-n)
    τabs = abs(τ)
    τ_sequence = phase(τ) .* (0.0:dτ:τabs)
    if abs(τ_sequence[end]) < τabs
        τ_sequence = [τ_sequence; τ]
    end

    function gsq_solver(z; verbose=false)
        @printf("  z = %6.3f + i%6.3f\n", real(z), imag(z))
        out = q_gsq!(z, τ, dgsq_sequence, workspace, problem, verbose=verbose)
        @printf("  q = %9.2e + i%9.2e\n", real(out[1]), imag(out[1]))
        @printf("g^2 = %9.2e\n", out[2])
        return out
    end

    function K_solver(z; verbose=false)
        @printf("  z = %6.3f + i%6.3f\n", real(z), imag(z))
        out = q_K!(z, τ_sequence, workspace, problem, verbose=verbose)
        @printf("  q = %9.2e + i%9.2e\n", real(out[1]), imag(out[1]))
        @printf("  K = %9.2e\n", out[2])
        return out
    end
    return gsq_solver, K_solver
end

end
