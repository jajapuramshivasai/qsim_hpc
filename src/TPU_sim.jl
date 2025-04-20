module QSim_TPU

using LinearAlgebra
using Reactant

export statevector, u!, u2!, h!, x!, y!, z!, rx!, ry!, rz!,
       cnot!, crx!, cry!, crz!, swap!, mp, prstate, measure_z, measure_x, measure_y

# Set Reactant backend to TPU
Reactant.set_default_backend("tpu")

# Core Constants and Operators (Dense)
const c1 = ComplexF32(1)
const im_F32 = ComplexF32(0, 1)
const I2 = Matrix{ComplexF32}(I, 2, 2)

const H = ComplexF32(1/√2) * Matrix{ComplexF32}([1 1; 1 -1])
const X = Matrix{ComplexF32}([0 1; 1 0])
const Y = Matrix{ComplexF32}([0 -im_F32; im_F32 0])
const Z = Matrix{ComplexF32}([1 0; 0 -1])
const P0 = Matrix{ComplexF32}([1 0; 0 0])
const P1 = Matrix{ComplexF32}([0 0; 0 1])

# Utility: number of qubits for a state vector
nb(s::AbstractVector) = round(Int, log2(length(s)))

# Dense identity matrix for n qubits
id(n::Int) = n == 0 ? Matrix{ComplexF32}(I, 1, 1) : Matrix{ComplexF32}(I, 2^n, 2^n)

# Statevector: |m⟩ in n qubits, as dense vector on Reactant device
function statevector(n::Int, m::Int)
    s = zeros(ComplexF32, 1 << n)
    s[m+1] = c1
    Reactant.ConcreteRArray(s)
end

# Move dense operator to Reactant device
to_device(A::AbstractArray) = Reactant.ConcreteRArray(A)

# Named function for matrix-vector multiplication (for Reactant compilation)
function matvec_mul(U, s)
    return U * s
end

# Matrix-vector multiplication (on device)
function u!(s, U)
    # Ensure both arrays are on device
    U_dev = isa(U, Reactant.ConcreteRArray) ? U : to_device(U)
    s_dev = isa(s, Reactant.ConcreteRArray) ? s : to_device(s)
    
    # Define and compile the function for this specific operation
    compiled_fn = @compile matvec_mul(U_dev, s_dev)
    
    # Apply the compiled function
    result = compiled_fn(U_dev, s_dev)
    
    # Update input
    if isa(s, Reactant.ConcreteRArray)
        s[:] = result
    else
        s[:] = Array(result)
    end
    
    s
end

# Apply single-qubit gate U to qubit t (1-based, LSB)
function u2!(s, t, U)
    q = nb(s)
    l = t - 1
    r = q - t
    
    # Build full operator: kron(id(r), kron(U, id(l)))
    op = kron(id(r), kron(U, id(l)))
    op_dev = to_device(op)
    
    # Apply the operator
    u!(s, op_dev)
    s
end

# Gate wrappers
function h!(s, t)
    u2!(s, t, H)
end

function x!(s, t)
    u2!(s, t, X)
end

function y!(s, t)
    u2!(s, t, Y)
end

function z!(s, t)
    u2!(s, t, Z)
end

# Rotation gates (dense)
function rx_gate(theta)
    c = ComplexF32(cos(theta/2))
    s = ComplexF32(sin(theta/2))
    c*I2 - im_F32*s*X
end

function ry_gate(theta)
    c = ComplexF32(cos(theta/2))
    s = ComplexF32(sin(theta/2))
    c*I2 - im_F32*s*Y
end

function rz_gate(theta)
    c = ComplexF32(cos(theta/2))
    s = ComplexF32(sin(theta/2))
    c*I2 - im_F32*s*Z
end

function rx!(s, t, theta)
    u2!(s, t, rx_gate(theta))
end

function ry!(s, t, theta)
    u2!(s, t, ry_gate(theta))
end

function rz!(s, t, theta)
    u2!(s, t, rz_gate(theta))
end

# Controlled gates (dense)
function controlled!(s, c, t, V)
    q = nb(s)
    a = min(c, t)
    b = max(c, t)
    
    left = id(q - b)
    right = id(a - 1)
    mid = (b - a - 1) > 0 ? id(b - a - 1) : Matrix{ComplexF32}(I, 1, 1)
    
    # Construct two-qubit controlled operator
    U2 = if c < t
        kron(id(1), P0) + kron(V, P1)
    else
        kron(P0, id(1)) + kron(P1, V)
    end
    
    # Assemble full operator
    op = kron(left, kron(U2, kron(mid, right)))
    op_dev = to_device(op)
    
    # Apply the operator
    u!(s, op_dev)
    s
end

function cnot!(s, c, t)
    controlled!(s, c, t, X)
end

function crx!(s, c, t, theta)
    controlled!(s, c, t, rx_gate(theta))
end

function cry!(s, c, t, theta)
    controlled!(s, c, t, ry_gate(theta))
end

function crz!(s, c, t, theta)
    controlled!(s, c, t, rz_gate(theta))
end

function swap!(s, q1, q2)
    if q1 == q2; return; end
    cnot!(s, q1, q2)
    cnot!(s, q2, q1)
    cnot!(s, q1, q2)
end

# Measurement and Output
mp(s) = abs2.(Array(s))

function measure_z(s, t)
    n = nb(s)
    probs = mp(s)
    p0 = 0.0
    p1 = 0.0
    for i in 0:(length(probs)-1)
        bit = (i >> (t-1)) & 1
        if bit == 0
            p0 += probs[i+1]
        else
            p1 += probs[i+1]
        end
    end
    (p0, p1)
end

function measure_x(s, t)
    s_copy = copy(Array(s))
    s_dev = Reactant.ConcreteRArray(s_copy)
    h!(s_dev, t)
    measure_z(s_dev, t)
end

function measure_y(s, t)
    s_copy = copy(Array(s))
    s_dev = Reactant.ConcreteRArray(s_copy)
    rx!(s_dev, t, π/2)
    measure_z(s_dev, t)
end

function prstate(s)
    n = nb(s)
    vec = Array(s)
    println("Quantum State:")
    for i in 0:(length(vec)-1)
        a = vec[i+1]
        abs(a) > 1e-10 && println("|", lpad(string(i, base=2), n, '0'), "⟩: ", a)
    end
end

end # module
