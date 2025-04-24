include("../algorithms/GHZ.jl")
using BenchmarkTools

println("Number of threads: $(Threads.nthreads())")

GHZ(4)

# Benchmark GHZ generation for various n
for n in (4, 8, 12, 16,20,24,27,30)                                     #if possible add 32, 34 ,35
    println("\nBenchmarking GHZ generation for $n qubits:")
    # `@benchmark` must be at top‐level so `$n` interpolation works
    bm = @benchmark GHZ($n) samples=2
    display(bm)
end

