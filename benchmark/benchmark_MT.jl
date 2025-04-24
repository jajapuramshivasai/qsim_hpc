include("../algorithms/GHZ.jl")
using BenchmarkTools


GHZ(4)

# Benchmark GHZ generation for various n
for n in (4, 8, 12, 16,)
    println("\nBenchmarking GHZ generation for $n qubits:")
    # `@benchmark` must be at top‐level so `$n` interpolation works
    bm = @benchmark GHZ($n) samples=2
    display(bm)
end

