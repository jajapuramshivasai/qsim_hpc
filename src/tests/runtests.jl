include("../basic_sim.jl") 

using .QSim_base
using Test

@testset "Source Code Tests" begin
    @testset "Algorithm Tests" begin
        # Add your algorithm test cases here
        @testset "StateVector Tests" begin
            sv = statevector(2,0)  # Initialize a 2-qubit state vector
            sv[1] = 1.0          # Set the first amplitude
            sv[2] = 0.0          # Set the second amplitude
            @test sv[1] == 1.0
            @test sv[2] == 0.0
        end

        @testset "DensityMatrix Tests" begin
            dm = density_matrix(2,0)  # Initialize a 2-qubit density matrix
            dm[1, 1] = 1.0         # Set the first diagonal element
            dm[2, 2] = 0.0         # Set the second diagonal element
            @test dm[1, 1] == 1.0
            @test dm[2, 2] == 0.0
        end

        @testset "Quantum Gates Tests" begin
            sv = statevector(2,0)  # Initialize a 2-qubit state vector
            h!(sv,1)  # Apply Hadamard gate to the first qubit
            @test isapprox(sv[1], 1 / sqrt(2))
            @test isapprox(sv[2], 1 / sqrt(2))



        end
    end
end