module multithreaded_sim

using LinearAlgebra, SparseArrays, Base.Threads, Polyester

export sv, u!, h!, x!, y!, z!, rx!, ry!, rz!, 
       cnot!, crx!, cry!, crz!, swap!, mp, prstate

# Core Constants and Operators
const c1 = ComplexF16(1)
const im_F16 = ComplexF16(0, 1)
const I2 = sparse(ComplexF16[1 0; 0 1])

const H = ComplexF16(1/√2) * sparse(ComplexF16[1 1; 1 -1])
const X = sparse(ComplexF16[0 1; 1 0])
const Y = sparse(ComplexF16[0 -im_F16; im_F16 0])
const Z = sparse(ComplexF16[1 0; 0 -1])
const P0 = sparse(ComplexF16[1 0; 0 0])
const P1 = sparse(ComplexF16[0 0; 0 1])

# Thread-safe Sparse Matrix Operations
struct ThreadedSparseMatrix{Tv,Ti}
    mat::SparseMatrixCSC{Tv,Ti}
end

Base.size(tsm::ThreadedSparseMatrix) = size(tsm.mat)
Base.getindex(tsm::ThreadedSparseMatrix, I...) = tsm.mat[I...]

function Base.:*(A::ThreadedSparseMatrix{ComplexF16,Int}, b::SparseVector{ComplexF16,Int})
    result = spzeros(ComplexF16, size(A, 1))
    rows = rowvals(A.mat)
    vals = nonzeros(A.mat)
    
    @threads for col in 1:size(A.mat, 2)
        if b[col] != 0
            @inbounds for i in nzrange(A.mat, col)
                row = rows[i]
                result[row] += vals[i] * b[col]
            end
        end
    end
    result
end

# Parallelized Core Functions
function id(n::Int)
    n == 0 ? sparse([c1]) : spdiagm(0 => ones(ComplexF16, 1 << n))
end

function sv(n::Int, m::Int)
    r"
    Creates a sparse vector of length 2^n with the m-th element set to 1.
    "
    s = spzeros(ComplexF16, 1 << n)
    s[m+1] = c1
    s
end

nb(s::SparseVector{ComplexF16,Int}) = round(Int, log2(length(s)))

function u!(s::SparseVector{ComplexF16,Int}, U::SparseMatrixCSC{ComplexF16,Int})
    r"
    Applies the unitary matrix U to the quantum state s.
    "
    size(U, 2) == length(s) || error("Dimension mismatch")
    result = spzeros(ComplexF16, size(U, 1))
    rows = rowvals(U)
    vals = nonzeros(U)
    
    @batch per=core for col in 1:size(U, 2)
        if s[col] != 0
            @inbounds for i in nzrange(U, col)
                row = rows[i]
                result[row] += vals[i] * s[col]
            end
        end
    end
    s[:] = result
end

function u!(s::SparseVector{ComplexF16,Int}, t::Int, U::SparseMatrixCSC{ComplexF16,Int})
    q = nb(s)
    l = t - 1
    r = q - t
    s[:] = ThreadedSparseMatrix(kron(id(r), kron(U, id(l)))) * s
end

# Gate Definitions with Thread Optimizations
h!(s::SparseVector{ComplexF16,Int}, t::Int) = u!(s, t, H)
x!(s::SparseVector{ComplexF16,Int}, t::Int) = u!(s, t, X)
y!(s::SparseVector{ComplexF16,Int}, t::Int) = u!(s, t, Y)
z!(s::SparseVector{ComplexF16,Int}, t::Int) = u!(s, t, Z)

# Rotation Gates
function rx_gate(theta::Real)
    r" computes the matrix representation of the RX gate
    with rotation angle theta.

    Args:
        theta (Real): The rotation angle in radians.
        # complete this doc string
    
    "
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*X
end

function ry_gate(theta::Real)
    r"
    Computes the matrix representation of the RY gate
    with rotation angle theta.
    "
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Y
end

function rz_gate(theta::Real)
    r"
    Computes the matrix representation of the RZ gate
    with rotation angle theta.
    "
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Z
end

rx!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = u!(s, t, rx_gate(theta))
ry!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = u!(s, t, ry_gate(theta))
rz!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = u!(s, t, rz_gate(theta))

# Controlled Operations
function controlled!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, V::SparseMatrixCSC{ComplexF16,Int})
    r"
    Computes the controlled version of the gate U on the qubits whaere
    controlled and target qubits are c and t respectively."
    q = nb(s)
    a = min(c, t)
    b = max(c, t)
    left = id(q - b)
    right = id(a - 1)
    mid = (b - a - 1) > 0 ? id(b - a - 1) : sparse([c1])
    
    U2 = c < t ? kron(id(1), P0) + kron(V, P1) : kron(P0, id(1)) + kron(P1, V)
    U_full = ThreadedSparseMatrix(kron(left, kron(U2, kron(mid, right))))
    s[:] = U_full * s
end

cnot!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int) = controlled!(s, c, t, X)
crx!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = controlled!(s, c, t, rx_gate(theta))
cry!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = controlled!(s, c, t, ry_gate(theta))
crz!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = controlled!(s, c, t, rz_gate(theta))

# SWAP Implementation
function swap!(s::SparseVector{ComplexF16,Int}, q1::Int, q2::Int)
    r"
    Swaps the states of the qubits q1 and q2 in the quantum state s.
    "
    q1 == q2 && return
    cnot!(s, q1, q2)
    cnot!(s, q2, q1)
    cnot!(s, q1, q2)
end

# Measurement and Output
mp(s::SparseVector{ComplexF16,Int}) = abs2.(s)

function prstate(s::SparseVector{ComplexF16,Int})
    r"
    Prints the quantum state in the computational basis.
    "
    n = nb(s)
    vec = Array(s)
    println("Quantum State:")
    @threads for i in 0:(length(vec)-1)
        a = vec[i+1]
        abs(a) > 1e-10 && println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
    end
end

end # module