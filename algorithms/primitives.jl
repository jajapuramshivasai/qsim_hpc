module Primitives


# Include all algorithm files
include("grovers.jl")
include("QFT.jl")
include("GHZ.jl")

# Export all algorithms
export grovers , IQFT ,QFT, GHZ

end