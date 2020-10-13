module RMSolver

export rm_solvers

using LinearAlgebra
using NLsolve
using KrylovKit
using Printf

using RMUtils
using RMProblems

const I_ = Int
const R_ = Float64
const C_ = Complex{R_}

abstract type AbstractRMSolution end

struct RMSolution{U<:Union{Nothing,Vector{<:AbstractRMSolution}}} <: AbstractRMSolution
    γsq::R_
    δsq::R_
    q::C_
    k::R_
    trace::U
end

const RMTrace = Vector{RMSolution{Nothing}}

function RMSolution(::Nothing)
    RMSolution(NaN, NaN, complex(NaN, NaN), NaN, nothing)
end

function RMSolution(trace::RMTrace)
    RMSolution(NaN, NaN, complex(NaN, NaN), NaN, trace)
end

struct RMWorkspace{U<:AbstractMatrix{C_},P<:RMProblem{U}}
    N::I_
    problem::P
    G0i::U
    Q::U
    P::Hermitian{C_,Matrix{C_}}
    QhP::Matrix{C_}
    QhPQ_I::Hermitian{C_,Matrix{C_}}
    γsq::Ref{R_}
end

function RMWorkspace(problem::RMProblem{U,V}) where {U,V}
    Q = similar(problem.LiMRi)
    N = size(Q, 1)
    QhP = Matrix{C_}(undef, N, N)
    return RMWorkspace{U,RMProblem{U,V}}(
        N,
        problem,
        Q,
        similar(Q),
        Hermitian(similar(QhP)),
        QhP,
        Hermitian(similar(QhP)),
        Ref(NaN),
    )
end

function safe_eigmin(P::Hermitian)
    result = eigsolve(P, 1, :SR, ishermitian=true)
    if result[3].converged > 0
        return result[1][1]
    end
    return eigmin(P)
end

function safe_eigmax(P::Hermitian)
    result = eigsolve(P, 1, :LR, ishermitian=true)
    if result[3].converged > 0
        return result[1][1]
    end
    return eigmax(P)
end

function G0i!(ws, z)
    for j in 1:ws.N, i in 1:ws.N
        ws.G0i[i, j] = ws.Q[i, j] = muladd(
            z, ws.problem.LiRi[i, j], -ws.problem.LiMRi[i, j],
        )
    end
    return nothing
end

function Q_P!(ws, T, rsq, q, ::Val{δ}) where δ
    for i in 1:ws.N
        ws.Q[i, i] = muladd(-q, T[i], ws.G0i[i, i])
    end
    mul!(ws.P.data, ws.Q, ws.Q')
    if δ
        ws.γsq[] = rsq -= safe_eigmin(ws.P)
    end
    for i in diagind(ws.P)
        ws.P[i] += rsq
    end
    LAPACK.potrf!('U', ws.P.data)
    LAPACK.potri!('U', ws.P.data)
    return nothing
end

function QhPQ!(ws)
    mul!(ws.QhP, ws.Q', ws.P)
    mul!(ws.QhPQ_I.data, ws.QhP, ws.Q)
    for i in diagind(ws.QhPQ_I)
        ws.QhPQ_I[i] -= 1.0
    end
    return nothing
end

function qresidue!(Fv, Jv, ws, T, rsq, qv, ::Val{δ}) where δ
    qc = complex(qv[1], qv[2])
    Q_P!(ws, T, rsq, qc, Val(δ))
    if Jv === nothing
        qresidue!(Fv, ws, T, qc)
    else
        qresidue!(Fv, Jv, ws, T, qc)
    end
    return nothing
end

function qresidue!(Fv, ws, T, qc)
    Fc = complex(0.0)
    for j in 1:ws.N  # optimized Tr(TQhP)
        t = T[j]
        for i in 1:ws.N
            Fc += t * conj(ws.Q[i, j]) * ws.P[i, j]
        end
    end
    Fc /= ws.N
    Fc -= qc
    Fv[1] = real(Fc)
    Fv[2] = imag(Fc)
    return nothing
end

function qresidue!(Fv, Jv, ws, T, qc)
    QhPQ!(ws)
    Fc = complex(0.0)
    dFc_dq = complex(0.0)
    dFc_dqconj = 0.0
    for j in 1:ws.N  # optimized(?) Tr((TQhP)^2) and Tr(T(QhPQ - I)TP)
        t = T[j]
        Fc += t * ws.QhP[j, j]
        for i in 1:ws.N
            tt = t * T[i]
            # [j, i] does not respect memory order. Would copying help?
            dFc_dq += tt * ws.QhP[i, j] * ws.QhP[j, i]
            dFc_dqconj += tt * real(ws.QhPQ_I[i, j] * ws.P[j, i])
        end
    end
    Fc /= ws.N
    Fc -= qc
    dFc_dq /= ws.N
    dFc_dq -= 1.0
    dFc_dqconj /= ws.N
    Fv[1] = real(Fc)
    Fv[2] = imag(Fc)
    Jv[1, 1] = real(dFc_dq) + dFc_dqconj
    Jv[1, 2] = -imag(dFc_dq)
    Jv[2, 1] = imag(dFc_dq)
    Jv[2, 2] = real(dFc_dq) - dFc_dqconj
    return nothing
end

function q!(ws, T, rsq, q0; δ)
    qv = nlsolve(
        #only_fj!((Fv, Jv, qv) -> qresidue!(Fv, Jv, ws, T, rsq, qv, Val(δ))),
        #[real(q0); imag(q0)],
        #method=:newton,
        (Fv, qv) -> qresidue!(Fv, nothing, ws, T, rsq, qv, Val(δ)),
        [real(q0); imag(q0)],
        method=:anderson,
        m=2,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return complex(qv[1], qv[2])
end

function k(ws)
    return tr(ws.P) / ws.N
end

function rsq_qresidue!(F, ws, T, rsq_q, ::Val{δ}) where δ
    qresidue!((@view F[2:3]), nothing, ws, T, rsq_q[1], (@view rsq_q[2:3]), Val(δ))
    F[1] = k(ws) - 1.0
    return nothing
end

function rsq_q!(ws, T, rsq, q0; δ)
    rsq_q = nlsolve(
        (F, rsq_q) -> rsq_qresidue!(F, ws, T, rsq_q, Val(δ)),
        [rsq; real(q0); imag(q0)],
        method=:anderson,
        m=3,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return rsq_q[1], complex(rsq_q[2], rsq_q[3])
end

function δsq(ws)
    return 1.0 / safe_eigmax(ws.P)
end

function qstart(ws, T, γsq)
    q = complex(0.0)
    for i in 1:ws.N  # optimized Tr(TG0i')
        q += T[i] * conj(ws.G0i[i, i])
    end
    return q / (γsq * ws.N)
end

function find_gsq!(ws, T, δsqs, z; debug=false)
    if !isfinite(z)
        return RMSolution(RMTrace())
    end
    if debug
        trace = fill(RMSolution(nothing), length(δsqs))
        print_header()
    else
        trace = RMTrace()
    end
    solution = RMSolution(trace)
    G0i!(ws, z)
    qprev = qstart(ws, T, δsqs[1])
    γsqprev = δsqs[1]
    for (i, δsq_) in enumerate(δsqs)
        q_ = q!(ws, T, δsq_, qprev, δ=true)
        γsq_ = ws.γsq[]
        k_ = k(ws)
        if debug
            trace[i] = RMSolution(γsq_, δsq_, q_, k_, nothing)
            print_row(trace[i])
        end
        if γsq_ * δsq_ < -0.5
            break
        end
        if k_ >= 1.0
            γsq_, q_ = rsq_q!(ws, T, γsqprev, qprev, δ=false)
            δsq_ = δsq(ws)
            solution = RMSolution(γsq_, δsq_, q_, 1.0, trace)
            break
        end
        qprev = q_
        γsqprev = γsq_
    end
    if debug
        print_hline()
    end
    return solution
end

function find_k!(ws, T, γsqs, z; debug=false)
    if !isfinite(z)
        return RMSolution(RMTrace())
    end
    if debug
        trace = RMTrace(undef, length(γsqs))
        print_header()
    else
        trace = RMTrace()
    end
    G0i!(ws, z)
    qprev = qstart(ws, T, γsqs[1])
    for (i, γsq_) in enumerate(γsqs)
        q_ = q!(ws, T, γsq_, qprev, δ=false)
        k_ = k(ws)
        if debug
            δsq_ = δsq(ws)
            trace[i] = RMSolution(γsq_, δsq_, q_, k_, nothing)
            print_row(trace[i])
        end
        qprev = q_
    end
    q_ = q!(ws, T, 0.0, qprev, δ=false)
    k_ = k(ws)
    δsq_ = δsq(ws)
    if debug
        print_hline()
    end
    return RMSolution(0.0, δsq_, q_, k_, trace)
end

function print_hline()
    println("-------------------------------------------------------")
end

function print_header()
    print_hline()
    println("   γ^2        δ^2               q                k")
    print_hline()
end

function print_row(s::RMSolution)
    @printf(
        "%9.2e  %9.2e  %9.2e + i%9.2e  %9.2e\n",
        s.γsq,
        s.δsq,
        real(s.q),
        imag(s.q),
        s.k,
    )
end

print_z(z)      = @printf("  z = %6.3f + i%6.3f\n", real(z), imag(z))
print_gsq(gsq_) = @printf("g^2 = %9.2e\n", gsq_)
print_dsq(dsq_) = @printf("d^2 = %9.2e\n", dsq_)
print_q(q_)     = @printf("  q = %9.2e + i%9.2e\n", real(q_), imag(q_))
print_k(k_)     = @printf("  k = %9.2e\n", k_)

function rm_solvers(; problem, rbase, start, stop)
    ws = RMWorkspace(problem)
    rsqs = rbase .^ (start:-1:stop)
    function solve_gsq(T, z; debug=false)
        if debug
            print_z(z)
        end
        solution = find_gsq!(ws, T, rsqs, z, debug=debug)
        if debug
            print_q(solution.q)
            print_gsq(solution.γsq)
        end
        return solution
    end
    function solve_k(T, z; debug=false)
        if debug
            print_z(z)
        end
        solution = find_k!(ws, T, rsqs, z, debug=debug)
        if debug
            print_q(solution.q)
            print_k(solution.k)
        end
        return solution
    end
    return solve_gsq, solve_k
end

end
