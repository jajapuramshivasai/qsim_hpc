# Qsim_HPC - High Performance Quantum Circuit Simulator

![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2?logo=julia&logoColor=white)
![License](https://img.shields.io/badge/License-GNU%20GPL-blue)

A high-performance quantum circuit simulator implemented in Julia, featuring multi-threaded CPU execution and TPU acceleration capabilities. The simulator supports a variety of quantum algorithms and primitives with optimized implementations for different hardware backends.

## Features

- **Multi-backend support**:
  - Basic CPU simulator
  - Multi-threaded CPU simulator with Polyester.jl
  - TPU-accelerated simulator with Reactant.jl
- **Quantum primitives**:
  - Single-qubit gates (H, X, Y, Z, rotations)
  - Multi-qubit gates (CNOT, CRX, CRY, CRZ, SWAP)
  - Measurement in X, Y, Z bases
- **Pre-implemented algorithms**:
  - GHZ state generation
  - Grover's search algorithm
  - Quantum Fourier Transform (QFT)
- **Optimizations**:
  - Sparse state vector representation
  - Threaded matrix operations
  - Gate operation caching
  - TPU-specific optimizations

## Installation

1. Ensure you have Julia 1.6 or later installed
2. Clone this repository:
   ```bash
   git clone https://github.com/jajapuramshivasai/qsim_hpc.git
   cd qsim_hpc
   ```
3. Start Julia and instantiate the project:
   ```julia
   julia> ] activate .
   julia> instantiate
   ```

For TPU support, additional setup may be required for Reactant.jl.

## Usage

### Basic Simulation

```julia
using .QSim  # or QSim_MT for multi-threaded, QSim_TPU for TPU

# Create a 2-qubit state
ψ = statevector(2, 0)

# Apply gates
h!(ψ, 1)
cnot!(ψ, 1, 2)

# Measure
p0, p1 = measure_z(ψ, 1)
```

### Running Algorithms

```julia
using .Primitives

# Create GHZ state
ghz_state = GHZ(4)

# Run Grover's search
oracle(state) = ...  # Define your oracle function
result = grovers_search(oracle, 5, 3)  # 5 qubits, 3 iterations

# Quantum Fourier Transform
ψ = statevector(3, 0)
qft_state = QFT(3, ψ)
```

## Benchmarks

To run benchmarks:

```julia
include("benchmark/benchmark_MT.jl")
```

Example benchmark output for GHZ state generation across different qubit counts.

## Backends

1. **Basic Simulator** (`QSim_base`):
   - Simple dense matrix implementation
   - Good for small circuits (<20 qubits)

2. **Multi-threaded Simulator** (`QSim_MT`):
   - Sparse state vector representation
   - Threaded operations with Polyester.jl
   - Recommended for medium-sized circuits (20-30 qubits)

3. **TPU Simulator** (`QSim_TPU`):
   - Accelerated with Reactant.jl
   - Best for large circuits when TPU is available

## Testing

Run tests with:

```julia
include("tests/MT_tests.jl")
```

Tests cover:
- State initialization
- Gate operations
- Measurement
- Multi-qubit operations

## Directory Structure

```
jajapuramshivasai-qsim_hpc/
├── README.md               # This file
├── Project.toml            # Project dependencies
├── algorithms/             # Quantum algorithms
│   ├── GHZ.jl              # GHZ state generation
│   ├── grovers.jl          # Grover's algorithm
│   ├── primitives.jl       # Algorithm exports
│   ├── QFT.jl              # Quantum Fourier Transform
│   └── test.jl             # Algorithm tests
├── benchmark/              # Performance benchmarks
│   └── benchmark_MT.jl     
├── src/                    # Core simulator code
│   ├── basic_sim.jl        # Basic CPU simulator
│   ├── multithreaded_sim.jl # Threaded simulator
│   ├── qsim.jl             # Main module
│   └── TPU_sim.jl          # TPU simulator
└── tests/                  # Unit tests
    └── MT_tests.jl         # Multi-threaded tests
```

## Dependencies

- Julia 1.6+
- BenchmarkTools.jl (for benchmarking)
- Polyester.jl (for multi-threading)
- Reactant.jl (for TPU support)
- LinearAlgebra (standard library)
- SparseArrays (standard library)

## Contributing

Contributions are welcome! Please open an issue or pull request for any bug fixes or feature additions.

## License
[GNU-GPL](https://www.gnu.org/licenses/gpl-3.0.en.html)
