module QSim_MT

using LinearAlgebra
using SparseArrays
using Kronecker
using Base.Threads

export statevector, u!, u2!, h!, x!, y!, z!, rx!, ry!, rz!,
       cnot!, crx!, cry!, crz!, swap!, mp, prstate, measure_x, measure_y, measure_z, density_matrix, expectation

# --- Constants (sparse) ---
const c1 = ComplexF64(1)
const im_F64 = ComplexF64(0, 1)
const I2 = sparse([1 0; 0 1])
const H = ComplexF64(1/√2) * sparse([1 1; 1 -1])
const X = sparse([0 1; 1 0])
const Y = sparse([0 -im_F64; im_F64 0])
const Z = sparse([1 0; 0 -1])
const P0 = sparse([1 0; 0 0])
const P1 = sparse([0 0; 0 1])

# --- Gate caches (sparse) ---
const RX_CACHE = Dict{Float64, SparseMatrixCSC{ComplexF64,Int}}()
const RY_CACHE = Dict{Float64, SparseMatrixCSC{ComplexF64,Int}}()
const RZ_CACHE = Dict{Float64, SparseMatrixCSC{ComplexF64,Int}}()

# --- Utility ---
function id(n::Int)
    n == 0 ? sparse([c1;;]) : spdiagm(0 => ones(ComplexF64, 1 << n))
end

function statevector(n::Int, m::Int)
    s = spzeros(ComplexF64, 1 << n)
    s[m+1] = c1
    s
end

function statevector_to_density_matrix(s::SparseVector{ComplexF64,Int})
    s * s'
end

function density_matrix(n::Int, m::Int)
    s = statevector(n, m)
    statevector_to_density_matrix(s)
end

nb(s::SparseVector{ComplexF64,Int}) = round(Int, log2(length(s)))
nb(rho::SparseMatrixCSC{ComplexF64,Int}) = round(Int, log2(size(rho, 1)))

# --- Multi-threaded explicit kron for two sparse matrices (fallback) ---
function threaded_kron(A::SparseMatrixCSC, B::SparseMatrixCSC)
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = mA*mB, nA*nB
    C = spzeros(eltype(A), mC, nC)
    @threads for ja in 1:nA
        for ia in 1:mA
            a = A[ia, ja]
            if a != 0
                for jb in 1:nB, ib in 1:mB
                    b = B[ib, jb]
                    if b != 0
                        C[(ia-1)*mB+ib, (ja-1)*nB+jb] = a*b
                    end
                end
            end
        end
    end
    C
end

# --- Efficient (lazy) kron using Kronecker.jl ---
function lazy_kron(A, B)
    KroneckerProduct(A, B)
end

# --- Gate application (use lazy kron for operators) ---
function u!(s::SparseVector{ComplexF64,Int}, U::AbstractMatrix)
    temp = U * s
    copyto!(s, temp)
    s
end

function u2!(s::SparseVector{ComplexF64,Int}, t::Int, U::AbstractMatrix)
    q = nb(s)
    l = t - 1
    r = q - t
    # Lazy kron for operator
    op = lazy_kron(id(r), lazy_kron(U, id(l)))
    temp = op * s
    copyto!(s, temp)
    s
end

function u!(rho::SparseMatrixCSC{ComplexF64,Int}, U::AbstractMatrix)
    temp = U * rho
    rho_new = temp * U'
    copyto!(rho, rho_new)
    rho
end

function u2!(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int, U::AbstractMatrix)
    q = round(Int, log2(size(rho,1)))
    l = t - 1
    r = q - t
    op = lazy_kron(id(r), lazy_kron(U, id(l)))
    temp = op * rho
    rho_new = temp * op'
    copyto!(rho, rho_new)
    rho
end

# --- Rotation gates (sparse, cached) ---
function rx_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RX_CACHE, theta_f64)
        return RX_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*X
    RX_CACHE[theta_f64] = gate
    gate
end

function ry_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RY_CACHE, theta_f64)
        return RY_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*Y
    RY_CACHE[theta_f64] = gate
    gate
end

function rz_gate(theta::Real)
    theta_f64 = Float64(theta)
    if haskey(RZ_CACHE, theta_f64)
        return RZ_CACHE[theta_f64]
    end
    c = ComplexF64(cos(theta_f64/2))
    s = ComplexF64(sin(theta_f64/2))
    gate = c*I2 - im_F64*s*Z
    RZ_CACHE[theta_f64] = gate
    gate
end

function rx!(s::SparseVector{ComplexF64,Int}, t::Int, theta::Real)
    u2!(s, t, rx_gate(theta))
end

function rx!(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int, theta::Real)
    u2!(rho, t, rx_gate(theta))
end

function ry!(s::SparseVector{ComplexF64,Int}, t::Int, theta::Real)
    u2!(s, t, ry_gate(theta))
end

function ry!(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int, theta::Real)
    u2!(rho, t, ry_gate(theta))
end

function rz!(s::SparseVector{ComplexF64,Int}, t::Int, theta::Real)
    u2!(s, t, rz_gate(theta))
end

function rz!(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int, theta::Real)
    u2!(rho, t, rz_gate(theta))
end

# --- Controlled gates (use lazy kron) ---
function controlled!(s::SparseVector{ComplexF64,Int}, c::Int, t::Int, V::AbstractMatrix)
    q = nb(s)
    a, b = min(c, t), max(c, t)
    left = id(q - b)
    right = id(a - 1)
    mid = (b - a - 1) > 0 ? id(b - a - 1) : sparse([c1;;])
    U2 = c < t ? lazy_kron(id(1), P0) + lazy_kron(V, P1) : lazy_kron(P0, id(1)) + lazy_kron(P1, V)
    op = lazy_kron(left, lazy_kron(U2, lazy_kron(mid, right)))
    temp = op * s
    copyto!(s, temp)
    s
end

function cnot!(s::SparseVector{ComplexF64,Int}, c::Int, t::Int)
    controlled!(s, c, t, X)
end

function controlled!(rho::SparseMatrixCSC{ComplexF64,Int}, c::Int, t::Int, V::AbstractMatrix)
    q = round(Int, log2(size(rho, 1)))
    a, b = min(c, t), max(c, t)
    left = id(q - b)
    right = id(a - 1)
    mid = (b - a - 1) > 0 ? id(b - a - 1) : sparse([c1;;])
    U2 = c < t ? lazy_kron(id(1), P0) + lazy_kron(V, P1) : lazy_kron(P0, id(1)) + lazy_kron(P1, V)
    op = lazy_kron(left, lazy_kron(U2, lazy_kron(mid, right)))
    temp = op * rho
    rho_new = temp * op'
    copyto!(rho, rho_new)
    rho
end

function cnot!(rho::SparseMatrixCSC{ComplexF64,Int}, c::Int, t::Int)
    controlled!(rho, c, t, X)
end

function crx!(s::SparseVector{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rx_gate(theta))
end

function crx!(rho::SparseMatrixCSC{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, rx_gate(theta))
end

function cry!(s::SparseVector{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, ry_gate(theta))
end

function cry!(rho::SparseMatrixCSC{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, ry_gate(theta))
end

function crz!(s::SparseVector{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rz_gate(theta))
end

function crz!(rho::SparseMatrixCSC{ComplexF64,Int}, c::Int, t::Int, theta::Real)
    controlled!(rho, c, t, rz_gate(theta))
end

function swap!(s::SparseVector{ComplexF64,Int}, q1::Int, q2::Int)
    q1 == q2 && return s
    cnot!(s, q1, q2)
    cnot!(s, q2, q1)
    cnot!(s, q1, q2)
    s
end

function swap!(rho::SparseMatrixCSC{ComplexF64,Int}, q1::Int, q2::Int)
    q1 == q2 && return rho
    cnot!(rho, q1, q2)
    cnot!(rho, q2, q1)
    cnot!(rho, q1, q2)
    rho
end

# --- Measurement (multi-threaded) ---
mp(s::SparseVector{ComplexF64,Int}) = abs2.(Array(s))
mp(rho::SparseMatrixCSC{ComplexF64,Int}) = real.(diag(rho))

function measure_z(s::SparseVector{ComplexF64,Int}, t::Int)
    n = nb(s)
    probs = mp(s)
    p0 = Threads.Atomic{Float64}(0.0)
    p1 = Threads.Atomic{Float64}(0.0)
    t_mask = 1 << (t-1)
    @threads for i in 0:(length(probs)-1)
        prob = probs[i+1]
        if prob > 1e-10
            if (i & t_mask) == 0
                atomic_add!(p0, prob)
            else
                atomic_add!(p1, prob)
            end
        end
    end
    (p0[], p1[])
end

function measure_x(s::SparseVector{ComplexF64,Int}, t::Int)
    s_copy = copy(s)
    h!(s_copy, t)
    measure_z(s_copy, t)
end

function measure_y(s::SparseVector{ComplexF64,Int}, t::Int)
    s_copy = copy(s)
    rx!(s_copy, t, π/2)
    measure_z(s_copy, t)
end

function measure_z(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int)
    q = round(Int, log2(size(rho, 1)))
    probs = real.(diag(rho))
    p0 = Threads.Atomic{Float64}(0.0)
    p1 = Threads.Atomic{Float64}(0.0)
    t_mask = 1 << (t - 1)
    @threads for i in 0:(length(probs) - 1)
        if probs[i + 1] > 1e-10
            if (i & t_mask) == 0
                atomic_add!(p0, probs[i + 1])
            else
                atomic_add!(p1, probs[i + 1])
            end
        end
    end
    (p0[], p1[])
end

function measure_x(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int)
    rho_copy = copy(rho)
    h!(rho_copy, t)
    measure_z(rho_copy, t)
end

function measure_y(rho::SparseMatrixCSC{ComplexF64,Int}, t::Int)
    rho_copy = copy(rho)
    rx!(rho_copy, t, π / 2)
    measure_z(rho_copy, t)
end

# --- Expectation ---
function expectation(x::SparseVector{ComplexF64,Int}, observable::Function, t::Int)
    x_conj = copy(x)'
    x_conj * observable(x, t)
end

function expectation(rho::SparseMatrixCSC{ComplexF64,Int}, observable::Function, t::Int)
    r = nb(rho)
    tr(rho * observable(id(r), t))
end

# --- Print state ---
function prstate(s::SparseVector{ComplexF64,Int})
    n = nb(s)
    for i in 0:(length(s)-1)
        a = s[i+1]
        abs(a) > 1e-10 && println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
    end
end

function prstate(x::SparseMatrixCSC{ComplexF64,Int})
    q = round(Int, log2(size(x, 1)))
    probs = mp(x)
    for i in 0:(size(x, 1) - 1)
        p = probs[i + 1]
        abs(p) > 1e-10 && println("|", lpad(string(i, base=2), q, '0'), "⟩ : ", p)
    end
    nothing
end

end
