module QSim_MT

using LinearAlgebra, SparseArrays, Base.Threads, Polyester

export statevector, u!, h!, x!, y!, z!, rx!, ry!, rz!, 
       cnot!, crx!, cry!, crz!, swap!, mp, prstate,measure

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

# Threaded Sparse Matrix Type with Full Array Interface
struct ThreadedSparseMatrix{Tv,Ti<:Integer}
    mat::SparseMatrixCSC{Tv,Ti}
end

# Array interface implementations
Base.parent(tsm::ThreadedSparseMatrix) = tsm.mat
Base.size(tsm::ThreadedSparseMatrix) = size(tsm.mat)
Base.size(tsm::ThreadedSparseMatrix, dim::Integer) = size(tsm.mat, dim)
Base.ndims(tsm::ThreadedSparseMatrix) = 2
Base.axes(tsm::ThreadedSparseMatrix) = axes(tsm.mat)
Base.axes(tsm::ThreadedSparseMatrix, dim::Integer) = axes(tsm.mat, dim)

# Sparse matrix interface
SparseArrays.issparse(::ThreadedSparseMatrix) = true
SparseArrays.nnz(tsm::ThreadedSparseMatrix) = nnz(tsm.mat)
SparseArrays.nonzeros(tsm::ThreadedSparseMatrix) = nonzeros(tsm.mat)
SparseArrays.rowvals(tsm::ThreadedSparseMatrix) = rowvals(tsm.mat)
SparseArrays.getcolptr(tsm::ThreadedSparseMatrix) = getcolptr(tsm.mat)

# Threaded matrix-vector multiplication
function Base.:*(A::ThreadedSparseMatrix{ComplexF16,Int}, b::SparseVector{ComplexF16,Int})
    result = spzeros(ComplexF16, size(A, 1))
    rows = rowvals(A)
    vals = nonzeros(A)
    
    @threads for col in 1:size(A, 2)
        if b[col] != 0
            @inbounds for i in nzrange(A.mat, col)
                row = rows[i]
                result[row] += vals[i] * b[col]
            end
        end
    end
    result
end

# Core quantum operations (remainder of implementation)
function id(n::Int)
    n == 0 ? sparse([c1]) : spdiagm(0 => ones(ComplexF16, 1 << n))
end

function statevector(n::Int, m::Int)
    s = spzeros(ComplexF16, 1 << n)
    s[m+1] = c1
    s
end

nb(s::SparseVector{ComplexF16,Int}) = round(Int, log2(length(s)))


#single qubit gates

function u!(s::SparseVector{ComplexF16,Int}, U::SparseMatrixCSC{ComplexF16,Int})
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
    dropzeros!(s) #experimental may remove later
end

function u2!(s::SparseVector{ComplexF16,Int}, t::Int, U::SparseMatrixCSC{ComplexF16,Int})
    q = nb(s)
    l = t - 1
    r = q - t
    s[:] = ThreadedSparseMatrix(kron(id(r), kron(U, id(l)))) * s
    dropzeros!(s) #experimental may remove later
end


function h!(s::SparseVector{ComplexF16,Int}, t::Int)
    u2!(s, t, H)
end

function x!(s::SparseVector{ComplexF16,Int}, t::Int)
    u2!(s, t, X)
end

function y!(s::SparseVector{ComplexF16,Int}, t::Int)
    u2!(s, t, Y)
end

function z!(s::SparseVector{ComplexF16,Int}, t::Int)
    u2!(s, t, Z)
end

# Rotation gates
function rx_gate(theta::Real)
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*X
end

function ry_gate(theta::Real)
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Y
end

function rz_gate(theta::Real)
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Z
end

function rx!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real)
    u2!(s, t, rx_gate(theta))
end

function ry!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real)
    u2!(s, t, ry_gate(theta))
end

function rz!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real)
    u2!(s, t, rz_gate(theta))
end

#controlled gates

function controlled!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, V::SparseMatrixCSC{ComplexF16,Int})
    # Total number of qubits in state vector
    q = nb(s)

    # Identify lower and higher indices
    a = min(c, t)
    b = max(c, t)

    # Build identity operators for regions outside control/target pair
    left = id(q - b)              # Qubits more significant than b
    right = id(a - 1)             # Qubits less significant than a
    mid = (b - a - 1) > 0 ? id(b - a - 1) : sparse([c1]) # Intermediate

    # Construct two-qubit controlled operator
    U2 = if c < t
        kron(id(1), P0) + kron(V, P1)
    else
        kron(P0, id(1)) + kron(P1, V)
    end

    # Assemble full operator on q-qubit Hilbert space
    U_full = ThreadedSparseMatrix(kron(left, kron(U2, kron(mid, right))))

    # Apply operator to state vector
    s[:] = U_full * s

    # Remove near-zero entries from sparse vector
    dropzeros!(s)

    return s
end


function cnot!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int)
    controlled!(s, c, t, X)
end

function crx!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rx_gate(theta))
end

function cry!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, ry_gate(theta))
end

function crz!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)
    controlled!(s, c, t, rz_gate(theta))
end


function swap!(s::SparseVector{ComplexF16,Int}, q1::Int, q2::Int)
    q = nb(s)
    if q1 > q || q2 > q
        error("Qubit index out of range")
    end
    if q1 == q2
        return
    end
    cnot!(s, q1, q2)
    cnot!(s, q2, q1)
    cnot!(s, q1, q2)
end
# Measurement and Output
mp(s::SparseVector{ComplexF16,Int}) = abs2.(s)

function measure_z(s::SparseVector{ComplexF16,Int}, t::Int)
    r"""
    measure_y(s::SparseVector{ComplexF16,Int}, t::Int)

Measure the quantum state `s` in the Z basis.


# Arguments
- `s`: A `SparseVector{ComplexF16,Int}` representing the quantum state.
- `t`: An `Int` specifying the qubit index to be measured.

# Returns
- The measurement result in the Z basis as returned by `measure_z`.
"""
    n = nb(s)
    probs = mp(s)
    p0 = 0.0
    p1 = 0.0
    for i in 0:(length(probs)-1)
        # Since the gate application functions assume qubit 1 is LSB,
        # extract tth qubit bit by shifting (t-1) bits.
        bit = (i >> (t-1)) & 1
        if bit == 0
            p0 += probs[i+1]
        else
            p1 += probs[i+1]
        end
    end
    return (p0, p1)
end

# To measure in the X basis, first apply a Hadamard gate (rotates X⇔Z)
function measure_x(s::SparseVector{ComplexF16,Int}, t::Int)
    r"""
    measure_y(s::SparseVector{ComplexF16,Int}, t::Int)

Measure the quantum state `s` in the Y basis.


# Arguments
- `s`: A `SparseVector{ComplexF16,Int}` representing the quantum state.
- `t`: An `Int` specifying the qubit index to be measured.

# Returns
- The measurement result in the Z basis as returned by `measure_z`.
"""
    s_copy = copy(s)
    h!(s_copy, t)
    return measure_z(s_copy, t)
end

# To measure in the Y basis, apply a rotation around the X axis by π/2.
# This maps the eigenstates of Y into the Z basis.
function measure_y(s::SparseVector{ComplexF16,Int}, t::Int)
    r"""
    measure_y(s::SparseVector{ComplexF16,Int}, t::Int)

Measure the quantum state `s` in the Y basis.


# Arguments
- `s`: A `SparseVector{ComplexF16,Int}` representing the quantum state.
- `t`: An `Int` specifying the qubit index to be measured.

# Returns
- The measurement result in the Z basis as returned by `measure_z`.
"""
    s_copy = copy(s)
    rx!(s_copy, t, π/2)
    return measure_z(s_copy, t)
end



function prstate(s::SparseVector{ComplexF16,Int})
    r"""
    prstate(s::SparseVector{ComplexF16,Int})

Print the quantum state in a human-readable format.

For each basis state with amplitude above a small threshold, the function prints the basis state in binary (padded to the number of qubits) along with its amplitude.

# Arguments
- `s`: A `SparseVector{ComplexF16,Int}` representing the quantum state.

# Returns
- Nothing. The state is printed to standard output.
"""
    n = nb(s)
    vec = Array(s)
    println("Quantum State:")
    @threads for i in 0:(length(vec)-1)
        a = vec[i+1]
        abs(a) > 1e-10 && println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
    end
end

end # module

