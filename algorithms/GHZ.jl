include("../src/multithreaded_sim.jl")  
using .QSim_MT


function GHZ(num_qubits::Int)
  
    psi = statevector(num_qubits, 0)
    h!(psi, 1)
    for i in 2:num_qubits
        cnot!(psi, 1,i)
    end
    return psi
end
