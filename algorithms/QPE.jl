using QSim_MT

"""
Quantum Phase Estimation Algorithm
"""
function quantum_phase_estimation(U::AbstractMatrix, ψ::AbstractVector, n::Int)
    # Number of qubits in the counting register
    counting_register = n

    # Create the quantum circuit
    circuit = Circuit(counting_register + 1)

    # Apply Hadamard gates to the counting register
    for i in 1:counting_register
        add_gate!(circuit, HGate(), i)
    end

    # Prepare the target register with the input state ψ
    add_state!(circuit, ψ, counting_register + 1)

    # Apply controlled-U^2^j gates
    for j in 0:(counting_register - 1)
        controlled_U = U^(2^j)
        add_gate!(circuit, ControlledGate(controlled_U), j + 1, counting_register + 1)
    end

    # Apply inverse Quantum Fourier Transform (QFT) to the counting register
    add_gate!(circuit, InverseQFTGate(counting_register), 1:counting_register)

    # Measure the counting register
    result = measure(circuit, 1:counting_register)

    return result
end

# Example usage
function main()
    # Define a unitary matrix U (e.g., a rotation gate)
    θ = π / 4
    U = [cos(θ) - im * sin(θ) 0; 0 cos(θ) + im * sin(θ)]

    # Define the input state ψ
    ψ = [1.0, 0.0]

    # Number of qubits in the counting register
    n = 3

    # Perform Quantum Phase Estimation
    phase = quantum_phase_estimation(U, ψ, n)
    println("Estimated phase: ", phase)
end

main()