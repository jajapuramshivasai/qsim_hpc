#include <iostream>
#include "xla/client/xla_builder.h"
#include "xla/client/local_client.h"
#include "xla/client/client_library.h"
#include "xla/shape_util.h"
#include "xla/literal_util.h"

using namespace xla;

// Function to create a dense matrix literal
Literal CreateMatrixLiteral(const Shape& shape, const std::vector<float>& data) {
    Literal literal(shape);
    for (int i = 0; i < data.size(); ++i) {
        literal.Set<float>({i / shape.dimensions(1), i % shape.dimensions(1)}, data[i]);
    }
    return literal;
}

// Function to build matrix multiplication computation
XlaComputation BuildMatMulComputation(const Shape& shape) {
    XlaBuilder builder("matmul");
    auto lhs = Parameter(&builder, 0, shape, "lhs");
    auto rhs = Parameter(&builder, 1, shape, "rhs");
    auto result = Dot(lhs, rhs);
    return builder.Build().ValueOrDie();
}

// Custom Kronecker product implementation using XLA primitives
XlaComputation BuildKroneckerProduct(const Shape& shapeA, const Shape& shapeB) {
    XlaBuilder builder("kronecker");
    auto A = Parameter(&builder, 0, shapeA, "A");
    auto B = Parameter(&builder, 1, shapeB, "B");
    
    // Since XLA doesn't have built-in kron, we implement it manually
    // Kronecker product: A ⊗ B where each element A[i,j] multiplies entire matrix B
    int64 a_rows = shapeA.dimensions(0);
    int64 a_cols = shapeA.dimensions(1);
    int64 b_rows = shapeB.dimensions(0);
    int64 b_cols = shapeB.dimensions(1);
    
    // Create expanded dimensions for broadcasting
    Shape expanded_a_shape = ShapeUtil::MakeShape(F32, {a_rows, 1, a_cols, 1});
    Shape expanded_b_shape = ShapeUtil::MakeShape(F32, {1, b_rows, 1, b_cols});
    
    // Reshape A and B for broadcasting
    auto A_reshaped = Reshape(A, {a_rows, 1, a_cols, 1});
    auto B_reshaped = Reshape(B, {1, b_rows, 1, b_cols});
    
    // Broadcast and multiply
    auto kronecker_4d = Mul(A_reshaped, B_reshaped);
    
    // Reshape to final 2D result
    auto result = Reshape(kronecker_4d, {a_rows * b_rows, a_cols * b_cols});
    
    return builder.Build().ValueOrDie();
}

// Function to create complex matrix literal
Literal CreateComplexMatrixLiteral(const Shape& shape, const std::vector<std::complex<float>>& data) {
    Literal literal(shape);
    for (int i = 0; i < data.size(); ++i) {
        int row = i / shape.dimensions(1);
        int col = i % shape.dimensions(1);
        literal.Set<std::complex<float>>({row, col}, data[i]);
    }
    return literal;
}

// Function to build complex matrix multiplication
XlaComputation BuildComplexMatMulComputation(const Shape& shape) {
    XlaBuilder builder("complex_matmul");
    auto lhs = Parameter(&builder, 0, shape, "lhs");
    auto rhs = Parameter(&builder, 1, shape, "rhs");
    auto result = Dot(lhs, rhs);
    return builder.Build().ValueOrDie();
}

// Function to create sparse matrix representation using XLA
XlaComputation BuildSparseMatMulComputation(const Shape& dense_shape) {
    XlaBuilder builder("sparse_matmul");
    
    // For sparse matrices, we represent them as dense but with many zeros
    // XLA will optimize zero multiplications automatically
    auto sparse_lhs = Parameter(&builder, 0, dense_shape, "sparse_lhs");
    auto dense_rhs = Parameter(&builder, 1, dense_shape, "dense_rhs");
    
    // XLA automatically optimizes sparse patterns
    auto result = Dot(sparse_lhs, dense_rhs);
    
    return builder.Build().ValueOrDie();
}

int main() {
    // Setup XLA client
    auto client = ClientLibrary::LocalClientOrDie();

    std::cout << "=== XLA Matrix Operations Test ===" << std::endl;

    // Test 1: Dense Real Matrix Multiplication
    std::cout << "\n1. Dense Real Matrix Multiplication:" << std::endl;
    Shape matrix_shape = ShapeUtil::MakeShape(F32, {3, 3});
    
    std::vector<float> dataA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> dataB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    
    auto literalA = CreateMatrixLiteral(matrix_shape, dataA);
    auto literalB = CreateMatrixLiteral(matrix_shape, dataB);
    
    auto matmul_computation = BuildMatMulComputation(matrix_shape);
    
    // XLA handles multithreading internally
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = client->Execute(matmul_computation, {literalA, literalB}).ValueOrDie();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << result.Get<float>({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    // Test 2: Complex Matrix Multiplication
    std::cout << "\n2. Complex Matrix Multiplication:" << std::endl;
    Shape complex_shape = ShapeUtil::MakeShape(C64, {2, 2});
    
    std::vector<std::complex<float>> complex_dataA = {
        {1, 1}, {2, 2}, {3, 3}, {4, 4}
    };
    std::vector<std::complex<float>> complex_dataB = {
        {1, -1}, {2, -2}, {3, -3}, {4, -4}
    };
    
    auto complex_literalA = CreateComplexMatrixLiteral(complex_shape, complex_dataA);
    auto complex_literalB = CreateComplexMatrixLiteral(complex_shape, complex_dataB);
    
    auto complex_computation = BuildComplexMatMulComputation(complex_shape);
    
    start_time = std::chrono::high_resolution_clock::now();
    auto complex_result = client->Execute(complex_computation, {complex_literalA, complex_literalB}).ValueOrDie();
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "Complex result matrix:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto val = complex_result.Get<std::complex<float>>({i, j});
            std::cout << "(" << val.real() << "," << val.imag() << "i) ";
        }
        std::cout << std::endl;
    }

    // Test 3: Kronecker Product
    std::cout << "\n3. Kronecker Product:" << std::endl;
    Shape small_shape = ShapeUtil::MakeShape(F32, {2, 2});
    
    std::vector<float> kron_dataA = {1, 2, 3, 4};
    std::vector<float> kron_dataB = {0, 5, 6, 7};
    
    auto kron_literalA = CreateMatrixLiteral(small_shape, kron_dataA);
    auto kron_literalB = CreateMatrixLiteral(small_shape, kron_dataB);
    
    auto kronecker_computation = BuildKroneckerProduct(small_shape, small_shape);
    
    start_time = std::chrono::high_resolution_clock::now();
    auto kron_result = client->Execute(kronecker_computation, {kron_literalA, kron_literalB}).ValueOrDie();
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "Kronecker product result:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << kron_result.Get<float>({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    // Test 4: Sparse Matrix Multiplication
    std::cout << "\n4. Sparse Matrix Multiplication:" << std::endl;
    
    // Create a sparse matrix (mostly zeros)
    std::vector<float> sparse_data = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    std::vector<float> dense_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    auto sparse_literal = CreateMatrixLiteral(matrix_shape, sparse_data);
    auto dense_literal = CreateMatrixLiteral(matrix_shape, dense_data);
    
    auto sparse_computation = BuildSparseMatMulComputation(matrix_shape);
    
    start_time = std::chrono::high_resolution_clock::now();
    auto sparse_result = client->Execute(sparse_computation, {sparse_literal, dense_literal}).ValueOrDie();
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "Sparse multiplication result:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << sparse_result.Get<float>({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
