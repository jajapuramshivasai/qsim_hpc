#!/usr/bin/env julia

# Include the QSim_MT module
include("multithreaded_sim.jl")

# Import the module
using .QSim_MT

# Print thread information
println("Running with ", Threads.nthreads(), " threads")

# Create a 2-qubit quantum state initialized to |00⟩
# The statevector function takes the number of qubits and the decimal representation of the initial state
state = statevector(2, 0)
println("Initial state:")
prstate(state)

# Apply a Hadamard gate to the first qubit
println("\nApplying Hadamard gate to qubit 1:")
h!(state, 1)
prstate(state)

# Apply a CNOT gate with control on first qubit and target on second qubit
println("\nApplying CNOT gate with control on qubit 1 and target on qubit 2:")
cnot!(state, 1, 2)
println("Final Bell state:")
prstate(state)

# Measure both qubits in the Z basis
println("\nMeasurement results:")
m1 = QSim_MT.measure_z(state, 1)
m2 = QSim_MT.measure_z(state, 2)
println("Qubit 1 measurement probabilities: |0⟩ = $(m1[1]), |1⟩ = $(m1[2])")
println("Qubit 2 measurement probabilities: |0⟩ = $(m2[1]), |1⟩ = $(m2[2])")

# Print correlation information
println("\nThis Bell state should show perfect correlation between the qubits.")


