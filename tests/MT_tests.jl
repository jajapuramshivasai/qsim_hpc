using Test
using LinearAlgebra
using SparseArrays
include("../src/multithreaded_sim.jl")  
using .QSim_MT


@testset "QSim_MT Tests" begin
    @testset "State Initialization" begin
        @test QSim_MT.statevector(2, 0) == sparsevec([1], [ComplexF16(1)], 4)
        @test QSim_MT.statevector(3, 2) == sparsevec([3], [ComplexF16(1)], 8)
    end

    @testset "Single Qubit Gates" begin
        s = QSim_MT.statevector(1, 0)
        
        QSim_MT.h!(s, 1)
        @test isapprox(Array(s), [1/√2, 1/√2], atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        QSim_MT.x!(s, 1)
        @test Array(s) == [0, 1]
        
        s = QSim_MT.statevector(1, 0)
        QSim_MT.y!(s, 1)
        @test isapprox(Array(s), [0, im], atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        QSim_MT.z!(s, 1)
        @test Array(s) == [1, 0]
    end

    @testset "Rotation Gates" begin
        s = QSim_MT.statevector(1, 0)
        QSim_MT.rx!(s, 1, π)
        @test isapprox(Array(s), [0, -im], atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        QSim_MT.ry!(s, 1, π)
        @test isapprox(Array(s), [0, 1], atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        QSim_MT.rz!(s, 1, π)
        @test isapprox(Array(s), [-im, 0], atol=1e-3)
    end

    @testset "Controlled Gates" begin
        s = QSim_MT.statevector(2, 0)
        QSim_MT.cnot!(s, 1, 2)
        @test Array(s) == [1, 0, 0, 0]
        
        s = QSim_MT.statevector(2, 1)
        QSim_MT.cnot!(s, 1, 2)
        @test Array(s) == [0, 0, 0, 1]
    end

    @testset "SWAP Gate" begin
        s = QSim_MT.statevector(2, 1)
        QSim_MT.swap!(s, 1, 2)
        @test Array(s) == [0, 0, 1, 0]
    end

    @testset "Measurement" begin
        s = QSim_MT.statevector(1, 0)
        QSim_MT.h!(s, 1)
        p0, p1 = QSim_MT.measure_z(s, 1)
        @test isapprox(p0, 0.5, atol=1e-3)
        @test isapprox(p1, 0.5, atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        p0, p1 = QSim_MT.measure_x(s, 1)
        @test isapprox(p0, 0.5, atol=1e-3)
        @test isapprox(p1, 0.5, atol=1e-3)
        
        s = QSim_MT.statevector(1, 0)
        p0, p1 = QSim_MT.measure_y(s, 1)
        @test isapprox(p0, 0.5, atol=1e-3)
        @test isapprox(p1, 0.5, atol=1e-3)
    end

    @testset "Multi-qubit Operations" begin
        s = QSim_MT.statevector(3, 0)
        QSim_MT.h!(s, 1)
        QSim_MT.cnot!(s, 1, 2)
        QSim_MT.cnot!(s, 2, 3)
        @test isapprox(QSim_MT.mp(s), [0.5, 0, 0, 0, 0, 0, 0, 0.5], atol=1e-3)
    end
end
