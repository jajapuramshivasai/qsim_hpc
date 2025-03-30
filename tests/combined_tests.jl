# julia
using Test
using SparseArrays
include("../src/multithreaded_sim.jl")
using .multithreaded_sim

# For testing non‐exported helper functions like statevector
statevector = multithreaded_sim.statevector



# @testset "ThreadedSparseMatrix Multiplication" begin
#     A = ThreadedSparseMatrix(sparse(ComplexF16[1 0; 0 1]))
#     b = sparsevec([1,2], [c1, ComplexF16(0)], 2)
#     c = A * b
#     @test isapprox(norm(c - b), 0.0)
# end

# @testset "Single Qubit Gates" begin
#     # Test Hadamard (h!) gate: |0⟩ -> (|0⟩ + |1⟩)/√2
#     s = statevector(1, 0)
#     h!(s, 1)
#     m = measure(s)
#     @test isapprox(sum(m), 1.0; atol=1e-6)
#     @test isapprox(m[1], 0.5; atol=1e-6)
#     @test isapprox(m[2], 0.5; atol=1e-6)

#     # Test Pauli X: |0⟩ -> |1⟩
#     s = statevector(1, 0)
#     x!(s, 1)
#     m = measure(s)
#     @test isapprox(m[1], 0.0; atol=1e-6)
#     @test isapprox(m[2], 1.0; atol=1e-6)

#     # Test Pauli Y: |0⟩ -> i|1⟩ (magnitude 1 state)
#     s = statevector(1, 0)
#     y!(s, 1)
#     m = measure(s)
#     @test isapprox(m[1], 0.0; atol=1e-6)
#     @test isapprox(m[2], 1.0; atol=1e-6)

#     # Test Pauli Z: |0⟩ -> |0⟩ (global phase ignored)
#     s = statevector(1, 0)
#     z!(s, 1)
#     m = measure(s)
#     @test isapprox(m[1], 1.0; atol=1e-6)
#     @test isapprox(m[2], 0.0; atol=1e-6)
# end

# @testset "Rotation Gates" begin
#     # With zero rotation the state shouldn't change.
#     s = statevector(1, 0)
#     rx!(s, 1, 0.0)
#     m = measure(s)
#     @test isapprox(m[1], 1.0; atol=1e-6)

#     s = statevector(1, 0)
#     ry!(s, 1, 0.0)
#     m = measure(s)
#     @test isapprox(m[1], 1.0; atol=1e-6)

#     s = statevector(1, 0)
#     rz!(s, 1, 0.0)
#     m = measure(s)
#     @test isapprox(m[1], 1.0; atol=1e-6)
# end

# @testset "Controlled Rotation Gates" begin
#     # For two qubits, starting with |00⟩.
#     s = statevector(2, 0)
#     # Apply a controlled RX with control qubit 1 and target qubit 2.
#     crx!(s, 1, 2, π)
#     @test isapprox(sum(measure(s)), 1.0; atol=1e-6)

#     s = statevector(2, 0)
#     cry!(s, 1, 2, π)
#     @test isapprox(sum(measure(s)), 1.0; atol=1e-6)

#     s = statevector(2, 0)
#     crz!(s, 1, 2, π)
#     @test isapprox(sum(measure(s)), 1.0; atol=1e-6)
# end

@testset "Print State" begin
    s = statevector(1, 0)
    io = IOBuffer()
    redirect_stdout(io) do
        print_state(s)
    end
    output = String(take!(io))
    @test occursin("Quantum State:", output)
end