module multithreaded_sim

using LinearAlgebra, SparseArrays, Base.Threads, Polyester

export c1, im_F16, I2, H, X, Y, Z, P0, P1,
       ThreadedSparseMatrix,
       h!, x!, y!, z!,
       rx!, ry!, rz!,
       crx!, cry!, crz!,measure,print_state

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
    r"""
    Multiplies a threaded sparse matrix A by a sparse vector b.
    
    Returns:
        A new sparse vector representing the product.
    """
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

# Identity and State Vector Constructors

identity_operator(n::Int) = n == 0 ? sparse([c1]) : spdiagm(0 => ones(ComplexF16, 1 << n))


function statevector(n::Int, m::Int)
    r"
    Returns a sparse vector representing the quantum state |m⟩ of n qubits.
    "
    s = spzeros(ComplexF16, 1 << n)
    s[m+1] = c1
    s
end


num_qubits(s::SparseVector{ComplexF16,Int}) = round(Int, log2(length(s)))

# Unitary Operations

function apply_unitary!(s::SparseVector{ComplexF16,Int}, U::SparseMatrixCSC{ComplexF16,Int})
    r"""
    apply_unitary!(s::SparseVector{ComplexF16,Int}, U::SparseMatrixCSC{ComplexF16,Int})

    Applies the unitary matrix `U` to the quantum state vector `s`.
    """
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


function apply_unitary_on_qubit!(s::SparseVector{ComplexF16,Int}, t::Int, U::SparseMatrixCSC{ComplexF16,Int})
    r"""
    apply_unitary_on_qubit!(s::SparseVector{ComplexF16,Int}, t::Int, U::SparseMatrixCSC{ComplexF16,Int})

    Applies the unitary matrix `U` on the qubit at position `t` in the quantum state `s`.
    """
    q = num_qubits(s)
    l = t - 1
    r = q - t
    s[:] = ThreadedSparseMatrix(kron(identity_operator(r), kron(U, identity_operator(l)))) * s
end

# Gate Definitions

h!(s::SparseVector{ComplexF16,Int}, t::Int) = apply_unitary_on_qubit!(s, t, H)


x!(s::SparseVector{ComplexF16,Int}, t::Int) = apply_unitary_on_qubit!(s, t, X)


y!(s::SparseVector{ComplexF16,Int}, t::Int) = apply_unitary_on_qubit!(s, t, Y)


z!(s::SparseVector{ComplexF16,Int}, t::Int) = apply_unitary_on_qubit!(s, t, Z)

# Rotation Gate Matrices and Operations

function rotation_x_gate(theta::Real)
    r"""
    rotation_x_gate(theta::Real) -> SparseMatrixCSC

    Computes the matrix representation of the RX rotation gate for angle `theta`.
    """
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*X
end

function rotation_y_gate(theta::Real)
    r"""
        rotation_y_gate(theta::Real) -> SparseMatrixCSC
    
    Computes the matrix representation of the RY rotation gate for angle `theta`.
    """
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Y
end


function rotation_z_gate(theta::Real)
    r"""
    rotation_z_gate(theta::Real) -> SparseMatrixCSC

    Computes the matrix representation of the RZ rotation gate for angle `theta`.
    """
    c = ComplexF16(cos(theta/2))
    s = ComplexF16(sin(theta/2))
    c*I2 - im_F16*s*Z
end


rx!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = apply_unitary_on_qubit!(s, t, rotation_x_gate(theta))

ry!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = apply_unitary_on_qubit!(s, t, rotation_y_gate(theta))


rz!(s::SparseVector{ComplexF16,Int}, t::Int, theta::Real) = apply_unitary_on_qubit!(s, t, rotation_z_gate(theta))

# Controlled Operations

function apply_controlled_gate!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, V::SparseMatrixCSC{ComplexF16,Int})
    r"""
    apply_controlled_gate!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, V::SparseMatrixCSC{ComplexF16,Int})

    Applies the controlled gate defined by unitary matrix `V` 
    with control qubit `c` and target qubit `t` on the state `s`.
    """
    q = num_qubits(s)
    a = min(c, t)
    b = max(c, t)
    left = identity_operator(q - b)
    right = identity_operator(a - 1)
    mid = (b - a - 1) > 0 ? identity_operator(b - a - 1) : sparse([c1])
    
    U2 = c < t ? kron(identity_operator(1), P0) + kron(V, P1) :
                 kron(P0, identity_operator(1)) + kron(P1, V)
    U_full = ThreadedSparseMatrix(kron(left, kron(U2, kron(mid, right))))
    s[:] = U_full * s
end

"""
    apply_cnot!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int)

Applies the CNOT gate with control qubit `c` and target qubit `t`
to the state `s`.
"""
cnot!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int) = apply_controlled_gate!(s, c, t, X)

"""
    apply_controlled_rotation_x!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)

Applies the controlled RX gate with control qubit `c` and target qubit `t` 
and rotation angle `theta` on state `s`.
"""
crx!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = apply_controlled_gate!(s, c, t, rotation_x_gate(theta))

"""
    apply_controlled_rotation_y!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)

Applies the controlled RY gate with control qubit `c` and target qubit `t` 
and rotation angle `theta` on state `s`.
"""
cry!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = apply_controlled_gate!(s, c, t, rotation_y_gate(theta))

"""
    apply_controlled_rotation_z!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real)

Applies the controlled RZ gate with control qubit `c` and target qubit `t` 
and rotation angle `theta` on state `s`.
"""
crz!(s::SparseVector{ComplexF16,Int}, c::Int, t::Int, theta::Real) = apply_controlled_gate!(s, c, t, rotation_z_gate(theta))

# SWAP Implementation

function swap!(s::SparseVector{ComplexF16,Int}, q1::Int, q2::Int)
    "
    apply_swap!(s::SparseVector{ComplexF16,Int}, q1::Int, q2::Int)

    Swaps the states of qubits `q1` and `q2` in the quantum state `s` using
    three consecutive CNOT gates.
    "
    q1 == q2 && return
    apply_cnot!(s, q1, q2)
    apply_cnot!(s, q2, q1)
    apply_cnot!(s, q1, q2)
end


measure(s::SparseVector{ComplexF16,Int}) = abs2.(s)


function print_state(s::SparseVector{ComplexF16,Int})
    
    r"
    print_state(s::SparseVector{ComplexF16,Int})

    Prints the quantum state represented by `s` in the computational basis.
    Only states with non-negligible amplitude are displayed.
    "
    vec = Array(s)
    println("Quantum State:")
    @threads for i in 0:(length(vec)-1)
        a = vec[i+1]
        if abs(a) > 1e-10
            println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
        end
    end
end

end # module