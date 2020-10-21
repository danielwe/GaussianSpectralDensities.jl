module RMSolver

export rm_solvers

using Printf
using LinearAlgebra
using NLsolve
using KrylovKit

using RMUtils
using RMProblems

abstract type AbstractRMSolution end

struct RMSolution{U<:Union{Nothing,Vector{<:AbstractRMSolution}}} <: AbstractRMSolution
    γsq::Float64
    δsq::Float64
    q::ComplexF64
    k::Float64
    trace::U
end

const RMTrace = Vector{RMSolution{Nothing}}

function RMSolution(::Nothing)
    return RMSolution(NaN, NaN, complex(NaN, NaN), NaN, nothing)
end

function RMSolution(trace::RMTrace)
    return RMSolution(NaN, NaN, complex(NaN, NaN), NaN, trace)
end

struct RMWorkspace{U,P<:RMProblem{U}}
    N::Int
    Ni::Float64
    problem::P
    G0i::U
    Q::U
    P::Hermitian{ComplexF64,Matrix{ComplexF64}}
    QhP::Matrix{ComplexF64}
    QhPQ_I::Hermitian{ComplexF64,Matrix{ComplexF64}}
    v::Vector{ComplexF64}
    γsq::Ref{Float64}

    function RMWorkspace(problem::RMProblem{U,V}) where {U,V}
        Q = similar(problem.LiMRi)
        N = size(Q, 1)
        QhP = Matrix{ComplexF64}(undef, N, N)
        v = randn(ComplexF64, N)
        v ./= norm(v)
        return new{U,RMProblem{U,V}}(
            N,
            1.0 / N,
            problem,
            Q,
            similar(Q),
            Hermitian(similar(QhP)),
            QhP,
            Hermitian(similar(QhP)),
            v,
            Ref(NaN),
        )
    end
end

const COMPLEX_EIGENVALUE_WARN_MSG = (
    "complex eigenvalue appeared in hermitian eigensolver; "
    * "falling back to general solver and returning real part"
)

function safe_eigmin!(P::Hermitian, v)
    local vals, vecs, info
    try
        vals, vecs, info = eigsolve(P, v, 1, :SR)
    catch err
        if isa(err, ErrorException) && contains(err.msg, "hermitian")
            @warn COMPLEX_EIGENVALUE_WARN_MSG
            _, vecs, vals, info = schursolve(P, v, 1, :SR, Arnoldi())
            vals = real(vals)
        else
            rethrow()
        end
    end
    if info.converged > 0
        v .= vecs[1]
        return vals[1]
    end
    return eigmin(P)
end

function safe_eigmax!(P::Hermitian, v)
    local vals, vecs, info
    try
        vals, vecs, info = eigsolve(P, v, 1, :LR)
    catch err
        if isa(err, ErrorException) && contains(err.msg, "hermitian")
            @warn COMPLEX_EIGENVALUE_WARN_MSG
            _, vecs, vals, info = schursolve(P, v, 1, :LR, Arnoldi())
            vals = real(vals)
        else
            rethrow()
        end
    end
    if info.converged > 0
        v .= vecs[1]
        return vals[1]
    end
    return eigmax(P)
end

function G0i!(ws, z)
    @inbounds for j in 1:ws.N, i in 1:ws.N
        ws.G0i[i, j] = ws.Q[i, j] = muladd(
            z, ws.problem.LiRi[i, j], -ws.problem.LiMRi[i, j]
        )
    end
    return nothing
end

function _Q_P!(ws, rsq, q, ::Val{δ}) where δ
    @inbounds for i in 1:ws.N
        ws.Q[i, i] = muladd(-q, ws.problem.T[i], ws.G0i[i, i])
    end
    mul!(ws.P.data, ws.Q, ws.Q')
    if δ
        rsq -= safe_eigmin!(ws.P, ws.v)
    end
    ws.γsq[] = rsq
    @inbounds for i in diagind(ws.P)
        ws.P[i] += rsq
    end
    LAPACK.potrf!('U', ws.P.data)
    LAPACK.potri!('U', ws.P.data)
    return nothing
end

function _QhPQ!(ws)
    mul!(ws.QhP, ws.Q', ws.P)
    mul!(ws.QhPQ_I.data, ws.QhP, ws.Q)
    @inbounds for i in diagind(ws.QhPQ_I)
        ws.QhPQ_I[i] -= 1.0
    end
    return nothing
end

function _qresidue!(Fv, Jv, ws, rsq, qv, ::Val{δ}) where δ
    qc = complex(qv[1], qv[2])
    _Q_P!(ws, rsq, qc, Val(δ))
    if Jv === nothing
        _qresidue!(Fv, ws, qc)
    else
        _qresidue!(Fv, Jv, ws, qc)
    end
    return nothing
end

function _qresidue!(Fv, ws, qc)
    Fc = complex(0.0)
    @inbounds for j in 1:ws.N, i in 1:ws.N  # optimized Tr(TQhP)
        Fc += ws.problem.T[j] * conj(ws.Q[i, j]) * ws.P[i, j]
    end
    Fc = muladd(Fc, ws.Ni, -qc)
    Fv[1] = real(Fc)
    Fv[2] = imag(Fc)
    return nothing
end

function _qresidue!(Fv, Jv, ws, qc)
    _QhPQ!(ws)
    Fc = complex(0.0)
    dFc_dq = complex(0.0)
    dFc_dqconj = 0.0
    @inbounds for j in 1:ws.N  # optimized(?) Tr((TQhP)^2) and Tr(T(QhPQ - I)TP)
        Fc += ws.problem.T[j] * ws.QhP[j, j]
        for i in 1:ws.N
            # [j, i] does not respect memory order. Would copying help?
            τ = ws.problem.T[i] * ws.problem.T[j]
            dFc_dq += τ * ws.QhP[i, j] * ws.QhP[j, i]
            dFc_dqconj += τ * real(ws.QhPQ_I[i, j] * ws.P[j, i])
        end
    end
    Fc = muladd(Fc, ws.Ni, -qc)
    dFc_dq = muladd(dFc_dq, ws.Ni, -1.0)
    dFc_dqconj *= ws.Ni
    Fv[1] = real(Fc)
    Fv[2] = imag(Fc)
    Jv[1, 1] = real(dFc_dq) + dFc_dqconj
    Jv[1, 2] = -imag(dFc_dq)
    Jv[2, 1] = imag(dFc_dq)
    Jv[2, 2] = real(dFc_dq) - dFc_dqconj
    return nothing
end

function q!(ws, rsq, q0; δ)
    qv = nlsolve(
        #only_fj!((Fv, Jv, qv) -> _qresidue!(Fv, Jv, ws, rsq, qv, Val(δ))),
        #[real(q0); imag(q0)];
        #method=:newton,
        (Fv, qv) -> _qresidue!(Fv, nothing, ws, rsq, qv, Val(δ)),
        [real(q0); imag(q0)];
        method=:anderson,
        m=1,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return complex(qv[1], qv[2])
end

function k(ws)
    return tr(ws.P) * ws.Ni
end

function _rsq_qresidue!(F, ws, rsq_q, ::Val{δ}) where δ
    _qresidue!((@view F[2:3]), nothing, ws, rsq_q[1], (@view rsq_q[2:3]), Val(δ))
    F[1] = k(ws) - 1.0
    return nothing
end

function rsq_q!(ws, rsq, q0; δ)
    rsq_q = nlsolve(
        (F, rsq_q) -> _rsq_qresidue!(F, ws, rsq_q, Val(δ)),
        [rsq; real(q0); imag(q0)];
        method=:anderson,
        m=1,
        #show_trace=true,
        #extended_trace=true,
    ).zero
    return rsq_q[1], complex(rsq_q[2], rsq_q[3])
end

function δsq(ws)
    return 1.0 / safe_eigmax!(ws.P, ws.v)
end

function qstart(ws, γsq)
    q = complex(0.0)
    @inbounds for i in 1:ws.N  # optimized Tr(TG0i')
        q += ws.problem.T[i] * conj(ws.G0i[i, i])
    end
    return (q * ws.Ni) / γsq
end

function find_gsq!(ws, δsqs, z; debug=false)
    if !isfinite(z)
        return RMSolution(RMTrace())
    end
    local q_, k_, γsq_
    if debug
        trace = fill(RMSolution(nothing), length(δsqs))
        print_header()
    else
        trace = RMTrace()
    end
    solution = RMSolution(trace)
    G0i!(ws, z)
    qprev = qstart(ws, δsqs[1])
    γsqprev = δsqs[1]
    for (i, δsq_) in enumerate(δsqs)
        q_ = q!(ws, δsq_, qprev, δ=true)
        γsq_ = ws.γsq[]
        k_ = k(ws)
        @inbounds if debug
            trace[i] = RMSolution(γsq_, δsq_, q_, k_, nothing)
            print_row(trace[i])
        end
        if γsq_ * δsq_ < -0.5
            break
        end
        if k_ >= 1.0
            γsq_, q_ = rsq_q!(ws, γsqprev, qprev, δ=false)
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

function find_k!(ws, γsqs, z; debug=false)
    if !isfinite(z)
        return RMSolution(RMTrace())
    end
    local q_, k_, δsq_
    if debug
        trace = RMTrace(undef, length(γsqs))
        print_header()
    else
        trace = RMTrace()
    end
    G0i!(ws, z)
    qprev = qstart(ws, γsqs[1])
    for (i, γsq_) in enumerate(γsqs)
        q_ = q!(ws, γsq_, qprev; δ=false)
        k_ = k(ws)
        @inbounds if debug
            δsq_ = δsq(ws)
            trace[i] = RMSolution(γsq_, δsq_, q_, k_, nothing)
            print_row(trace[i])
        end
        qprev = q_
    end
    try
        q_ = q!(ws, 0.0, qprev; δ=false)
    catch err
        if isa(err, SingularException) || isa(err, LAPACKException)
            q_ = complex(NaN, NaN)
        else
            rethrow()
        end
    end
    if isfinite(q_)
        k_ = k(ws)
        δsq_ = δsq(ws)
    else
        k_ = Inf
        δsq_ = 0.0
    end
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

print_z(z) = @printf("  z = %6.3f + i%6.3f\n", real(z), imag(z))
print_gsq(gsq) = @printf("g^2 = %9.2e\n", gsq)
print_dsq(dsq) = @printf("d^2 = %9.2e\n", dsq)
print_q(q) = @printf("  q = %9.2e + i%9.2e\n", real(q), imag(q))
print_k(k) = @printf("  k = %9.2e\n", k)

function rm_solvers(; problem, rbase, start, stop)
    ws = RMWorkspace(problem)
    rsqs = rbase.^(start:-1:stop)
    solve_gsq = let ws=ws, δsqs=rsqs
        function(z; debug=false)
            print_z(z)
            solution = find_gsq!(ws, δsqs, z; debug=debug)
            if debug
                print_row(solution)
                print_hline()
            end
            return solution
        end
    end
    solve_k = let ws = ws, γsqs = rsqs
        function(z; debug=false)
            print_z(z)
            solution = find_k!(ws, γsqs, z; debug=debug)
            if debug
                print_row(solution)
                print_hline()
            end
            return solution
        end
    end
    return solve_gsq, solve_k
end

end
