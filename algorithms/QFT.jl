include("../src/multithreaded_sim.jl")  
using .QSim_MT


function QFT( n::Int, psi::SparseVector{ComplexF16,Int})    
    # Apply Hadamard gate to the first qubit
    h!(psi, 1)
    
    # Apply controlled phase gates
    for j in 2:n
        for k in 1:(j-1)
            crz!(psi, j, k, π/(2^(j-k)))
        end
    end
    
    # Apply Hadamard gate to the last qubit
    h!(psi, n)
    
    # Reverse the order of the qubits
    for i in 1:div(n,2)
        swap!(psi, i, n-i+1)
    end
    
    return psi
    
end


# # Example 
# n = 3
# psi = statevector(n, 0)
# QFT(n, psi)
# println("QFT Result: ", Array(psi))