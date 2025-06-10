using JuliaTest

@testset "Algorithm Tests" begin
    @testset "Test Algorithm 1" begin
        @test algorithm1(input1) == expected_output1
        @test algorithm1(input2) == expected_output2
    end

    @testset "Test Algorithm 2" begin
        @test algorithm2(input1) == expected_output1
        @test algorithm2(input2) == expected_output2
    end
end