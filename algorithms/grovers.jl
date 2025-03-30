include("../src/multithreaded_sim.jl")  
using .QSim_MT



function grover_diffusion_matrix(n_qubits::Int)
    N = 2^n_qubits  
    matrix = fill(2/N + 0im, N, N)  # Initialize with 2/N + 0im (complex)
    # Set diagonal elements to 2/N - 1
    for i in 1:N
        matrix[i, i] = 2/N - 1 + 0im
    end
    return matrix
end

function diffusion_operator!(statevector ::Vector{ComplexF64})
    n_qubits = Int(log2(length(statevector)))
    diffusion_matrix = grover_diffusion_matrix(n_qubits)
    u!(statevector, diffusion_matrix)
end


function grovers_search(Oracle ::Function, num_qubits::Int, num_iterations::Int)
    # Initialize the quantum state to |0⟩^n
    state = QSim_MT.initialize_state(num_qubits)
    # Apply Hadamard gate to all qubits to create a superposition
    for qubit in 1:num_qubits
        QSim_MT.apply_hadamard!(state, qubit)
    end
    # Grover's iterations
    for _ in 1:num_iterations
        # Oracle: Flip the amplitude of the target state
        QSim_MT.oracle!(state, target_state)
        # Diffusion operator: Reflect about the mean amplitude
        QSim_MT.diffusion_operator!(state)
    end
    # Measure the state
    return state
end




