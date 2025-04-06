using Test
include("QFT.jl")

# Example test file for QFT.jl
@testset "Quantum Fourier Transform Tests" begin
    # Test 1: Check QFT on a simple input
    input = statevector(2,0)  # Example input state
    expected_output = [0.5, 0.5im, -0.5, -0.5im]  # Replace with the actual expected result

    @test QFT(2,input) ≈ expected_output

    # # Test 2: Check QFT on zero input
    # input = [0.0, 0.0, 0.0, 0.0]
    # expected_output = [0.0, 0.0, 0.0, 0.0]
    # @test qft(input) ≈ expected_output

    # # Test 3: Check QFT is unitary (if applicable)
    # input = [1.0, 0.0, 0.0, 0.0]
    # output = qft(input)
    # @test norm(output) ≈ norm(input)

    # Add more tests as needed based on the functionality of QFT.jl
end