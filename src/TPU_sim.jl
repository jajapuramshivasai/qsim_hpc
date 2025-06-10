module QSimTPU

using Reactant, LinearAlgebra, Printf, Random, BenchmarkTools

export statevector_tpu, densitymatrix_tpu, u_tpu!, u2_tpu!, h_tpu!, x_tpu!, y_tpu!, z_tpu!,
       rx_tpu!, ry_tpu!, rz_tpu!, cnot_tpu!, crx_tpu!, cry_tpu!, crz_tpu!, swap_tpu!,
       measure_z_tpu, measure_x_tpu, measure_y_tpu, probabilities_tpu,
       bell_state_tpu, qft_tpu, prstate_tpu, benchmark_quantum_ops, controlled_tpu!

# ====== TPU-Optimized Core Operations ======

# Initialize TPU-optimized state vector
function statevector_tpu(n_qubits::Int, state_idx::Int=0)
    dim = 1 << n_qubits
    s = zeros(ComplexF32, dim)
    s[state_idx+1] = 1.0f0 + 0.0f0im
    return Reactant.ConcreteRArray(s)
end

# Initialize TPU-optimized density matrix
function densitymatrix_tpu(state::Reactant.ConcreteRArray)
    state_cpu = Array(state)
    dm = state_cpu * state_cpu'
    return Reactant.ConcreteRArray(dm)
end

function densitymatrix_tpu(n_qubits::Int, state_idx::Int=0)
    state = statevector_tpu(n_qubits, state_idx)
    return densitymatrix_tpu(state)
end

# TPU-optimized matrix operations
function tpu_matrix_vector_multiply(matrix::Reactant.ConcreteRArray, vector::Reactant.ConcreteRArray)
    return matrix * vector
end

function tpu_matrix_matrix_multiply(A::Reactant.ConcreteRArray, B::Reactant.ConcreteRArray)
    return A * B
end

function tpu_kronecker_product(A::Reactant.ConcreteRArray, B::Reactant.ConcreteRArray)
    m_A, n_A = size(A)
    m_B, n_B = size(B)

    result = zeros(ComplexF32, m_A * m_B, n_A * n_B)
    result_tpu = Reactant.ConcreteRArray(result)

    for i in 1:m_A, j in 1:n_A
        start_row = (i-1) * m_B + 1
        end_row = i * m_B
        start_col = (j-1) * n_B + 1
        end_col = j * n_B

        result_tpu[start_row:end_row, start_col:end_col] = A[i,j] * B
    end

    return result_tpu
end

# Helper function to determine number of qubits
function nb_tpu(state::Reactant.ConcreteRArray)
    s = Array(state)
    if ndims(s) == 1
        return Int(log2(length(s)))
    else
        return Int(log2(size(s, 1)))
    end
end

# ====== TPU-Optimized Basic Gates ======

function basic_gates_tpu()
    I2 = Reactant.ConcreteRArray(ComplexF32[1 0; 0 1])
    H = Reactant.ConcreteRArray(ComplexF32[1 1; 1 -1] / sqrt(2.0f0))
    X = Reactant.ConcreteRArray(ComplexF32[0 1; 1 0])
    Y = Reactant.ConcreteRArray(ComplexF32[0 -im; im 0])
    Z = Reactant.ConcreteRArray(ComplexF32[1 0; 0 -1])
    P0 = Reactant.ConcreteRArray(ComplexF32[1 0; 0 0])
    P1 = Reactant.ConcreteRArray(ComplexF32[0 0; 0 1])
    return (I2, H, X, Y, Z, P0, P1)
end

function identity_tpu(n::Int)
    if n == 0
        return Reactant.ConcreteRArray(reshape(ComplexF32[1], 1, 1))
    else
        dim = 1 << n
        id_matrix = Matrix{ComplexF32}(I, dim, dim)
        return Reactant.ConcreteRArray(id_matrix)
    end
end

# Rotation gate constructors
function rx_gate_tpu(θ::Real)
    c, s = cos(Float32(θ)/2), sin(Float32(θ)/2)
    return Reactant.ConcreteRArray(ComplexF32[c -im*s; -im*s c])
end

function ry_gate_tpu(θ::Real)
    c, s = cos(Float32(θ)/2), sin(Float32(θ)/2)
    return Reactant.ConcreteRArray(ComplexF32[c -s; s c])
end

function rz_gate_tpu(θ::Real)
    p1 = exp(-im*Float32(θ)/2)
    p2 = exp(im*Float32(θ)/2)
    return Reactant.ConcreteRArray(ComplexF32[p1 0; 0 p2])
end

# ====== TPU-Optimized Gate Operations ======

# Apply unitary to entire state (supports both state vectors and density matrices)
function u_tpu_impl(state::Reactant.ConcreteRArray, gate::Reactant.ConcreteRArray)
    s = Array(state)
    if ndims(s) == 1
        # State vector case
        return tpu_matrix_vector_multiply(gate, state)
    else
        # Density matrix case: U * ρ * U†
        temp = tpu_matrix_matrix_multiply(gate, state)
        # Convert adjoint to ConcreteRArray before multiplication
        return tpu_matrix_matrix_multiply(temp, Reactant.ConcreteRArray(Array(adjoint(gate))))
    end
end

function u_tpu!(state::Reactant.ConcreteRArray, gate::Reactant.ConcreteRArray)
    return u_tpu_impl(state, gate)
end

# Single qubit gate application
function u2_tpu_impl(state::Reactant.ConcreteRArray, target::Int, gate::Reactant.ConcreteRArray, n_qubits::Int)
    gates = basic_gates_tpu()
    I2 = gates[1]

    # Build full operator using kronecker products
    op = Reactant.ConcreteRArray(ComplexF32[1;;])

    for i in 1:n_qubits
        if i == target
            op = tpu_kronecker_product(op, gate)
        else
            op = tpu_kronecker_product(op, I2)
        end
    end

    return u_tpu!(state, op)
end

# TPU-optimized standard gates
function h_tpu!(state::Reactant.ConcreteRArray, target::Int)
    n_qubits = nb_tpu(state)
    gates = basic_gates_tpu()
    return u2_tpu_impl(state, target, gates[2], n_qubits)
end

function x_tpu!(state::Reactant.ConcreteRArray, target::Int)
    n_qubits = nb_tpu(state)
    gates = basic_gates_tpu()
    return u2_tpu_impl(state, target, gates[3], n_qubits)
end

function y_tpu!(state::Reactant.ConcreteRArray, target::Int)
    n_qubits = nb_tpu(state)
    gates = basic_gates_tpu()
    return u2_tpu_impl(state, target, gates[4], n_qubits)
end

function z_tpu!(state::Reactant.ConcreteRArray, target::Int)
    n_qubits = nb_tpu(state)
    gates = basic_gates_tpu()
    return u2_tpu_impl(state, target, gates[5], n_qubits)
end

# Rotation gates
function rx_tpu!(state::Reactant.ConcreteRArray, target::Int, θ::Real)
    n_qubits = nb_tpu(state)
    return u2_tpu_impl(state, target, rx_gate_tpu(θ), n_qubits)
end

function ry_tpu!(state::Reactant.ConcreteRArray, target::Int, θ::Real)
    n_qubits = nb_tpu(state)
    return u2_tpu_impl(state, target, ry_gate_tpu(θ), n_qubits)
end

function rz_tpu!(state::Reactant.ConcreteRArray, target::Int, θ::Real)
    n_qubits = nb_tpu(state)
    return u2_tpu_impl(state, target, rz_gate_tpu(θ), n_qubits)
end

# ====== TPU-Optimized Controlled Operations ======

function controlled_tpu!(state::Reactant.ConcreteRArray, control::Int, target::Int, gate::Reactant.ConcreteRArray)
    n_qubits = nb_tpu(state)
    gates = basic_gates_tpu()
    I2, _, _, _, _, P0, P1 = gates

    # Build projection operators
    proj0 = identity_tpu(0)
    proj1 = identity_tpu(0)

    for i in 1:n_qubits
        if i == control
            proj0 = tpu_kronecker_product(proj0, P0)
            proj1 = tpu_kronecker_product(proj1, P1)
        else
            proj0 = tpu_kronecker_product(proj0, I2)
            proj1 = tpu_kronecker_product(proj1, I2)
        end
    end

    # Build controlled operation
    controlled_op = identity_tpu(0)
    for i in 1:n_qubits
        if i == target
            controlled_op = tpu_kronecker_product(controlled_op, gate)
        else
            controlled_op = tpu_kronecker_product(controlled_op, I2)
        end
    end

    # Full operator: |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
    full_op = proj0 + tpu_matrix_matrix_multiply(proj1, controlled_op)

    return u_tpu!(state, full_op)
end

function cnot_tpu_impl(state::Reactant.ConcreteRArray, control::Int, target::Int, n_qubits::Int)
    # Efficient CNOT implementation for TPU
    dim = 1 << n_qubits
    cnot_matrix = zeros(ComplexF32, dim, dim)

    for i in 0:(dim-1)
        if (i >> (control-1)) & 1 == 1
            j = i ⊻ (1 << (target-1))
            cnot_matrix[j+1, i+1] = 1.0f0
        else
            cnot_matrix[i+1, i+1] = 1.0f0
        end
    end

    cnot_tpu = Reactant.ConcreteRArray(cnot_matrix)
    return u_tpu!(state, cnot_tpu)
end

function cnot_tpu!(state::Reactant.ConcreteRArray, control::Int, target::Int)
    n_qubits = nb_tpu(state)
    return cnot_tpu_impl(state, control, target, n_qubits)
end

# Controlled rotation gates
function crx_tpu!(state::Reactant.ConcreteRArray, control::Int, target::Int, θ::Real)
    return controlled_tpu!(state, control, target, rx_gate_tpu(θ))
end

function cry_tpu!(state::Reactant.ConcreteRArray, control::Int, target::Int, θ::Real)
    return controlled_tpu!(state, control, target, ry_gate_tpu(θ))
end

function crz_tpu!(state::Reactant.ConcreteRArray, control::Int, target::Int, θ::Real)
    return controlled_tpu!(state, control, target, rz_gate_tpu(θ))
end

function swap_tpu!(state::Reactant.ConcreteRArray, q1::Int, q2::Int)
    if q1 == q2
        return state
    end
    state = cnot_tpu!(state, q1, q2)
    state = cnot_tpu!(state, q2, q1)
    state = cnot_tpu!(state, q1, q2)
    return state
end

# ====== TPU-Optimized Measurement Functions ======

function probabilities_tpu(state::Reactant.ConcreteRArray)
    s = Array(state)
    if ndims(s) == 1
        # State vector case
        return abs2.(s)
    else
        # Density matrix case
        return real.(diag(s))
    end
end

function measure_z_tpu_impl(state::Reactant.ConcreteRArray, target::Int)
    probs = probabilities_tpu(state)
    n_qubits = nb_tpu(state)
    t_mask = 1 << (target-1)

    p0 = 0.0f0
    p1 = 0.0f0

    for i in 0:(length(probs)-1)
        if abs(probs[i+1]) > 1f-10
            if (i & t_mask) == 0
                p0 += probs[i+1]
            else
                p1 += probs[i+1]
            end
        end
    end

    return (p0, p1)
end

function measure_z_tpu(state::Reactant.ConcreteRArray, target::Int)
    return measure_z_tpu_impl(state, target)
end

function measure_x_tpu(state::Reactant.ConcreteRArray, target::Int)
    # Copy state and apply H gate to rotate X basis to Z basis
    state_copy = Reactant.ConcreteRArray(Array(state))
    state_copy = h_tpu!(state_copy, target)
    return measure_z_tpu(state_copy, target)
end

function measure_y_tpu(state::Reactant.ConcreteRArray, target::Int)
    # Copy state and apply rotation to measure in Y basis
    state_copy = Reactant.ConcreteRArray(Array(state))
    state_copy = rx_tpu!(state_copy, target, π/2)
    return measure_z_tpu(state_copy, target)
end

# ====== Enhanced State Display Function ======

function prstate_tpu(state::Reactant.ConcreteRArray; threshold::Float64=1e-6, as_binary::Bool=true)
    st = Array(state)

    if ndims(st) == 1
        # Pure state (state vector)
        n = Int(log2(length(st)))
        probs = abs2.(st)
        println("Pure Quantum State: $n qubits")
        println("----------------------")
        total = 0.0

        for i in 0:length(st)-1
            p = probs[i+1]
            if p > threshold
                total += p
                basis = as_binary ? lpad(string(i, base=2), n, '0') : string(i)
                amp = st[i+1]
                re, im = real(amp), imag(amp)
                amp_str = abs(im)<1e-10 ? @sprintf("%.6f", re) :
                          abs(re)<1e-10 ? @sprintf("%.6fi", im) :
                          @sprintf("%.6f %+.6fi", re, im)
                @printf("|%s⟩: %s (prob: %.6f)\n", basis, amp_str, p)
            end
        end

        println("----------------------")
        @printf("Total probability: %.6f\n", total)

    else
        # Mixed state (density matrix)
        n = Int(log2(size(st, 1)))
        ps = real.(diag(st))
        println("Density Matrix: $n qubits")
        println("----------------------")
        total = 0.0
        println("Computational Basis Probabilities:")

        for i in 0:length(ps)-1
            p = ps[i+1]
            if p > threshold
                total += p
                basis = as_binary ? lpad(string(i, base=2), n, '0') : string(i)
                @printf("|%s⟩: %.6f\n", basis, p)
            end
        end

        println("----------------------")
        @printf("Total probability: %.6f\n", total)

        # Calculate and display purity
        purity = real(tr(st * st))
        @printf("Purity: %.6f\n", purity)
    end

    return nothing
end
end