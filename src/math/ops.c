#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <complex.h>
#include <string.h>
#include <sys/time.h>

#define MAX_SIZE 1000
#define MAX_THREADS 8
#define SPARSE_THRESHOLD 0.1

// Structure for real matrices
typedef struct {
    double **data;
    int rows, cols;
    int is_sparse;
    int *row_ptr;  // For CSR format
    int *col_idx;
    double *values;
    int nnz;  // Number of non-zeros
} RealMatrix;

// Structure for complex matrices
typedef struct {
    double complex **data;
    int rows, cols;
    int is_sparse;
    int *row_ptr;
    int *col_idx;
    double complex *values;
    int nnz;
} ComplexMatrix;

// Thread data structure
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int num_threads;
    void *mat_a;
    void *mat_b;
    void *mat_c;
    int operation; // 0=mult, 1=kron
    int is_complex;
} ThreadData;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

// Utility functions
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Real matrix operations
RealMatrix* create_real_matrix(int rows, int cols, int sparse) {
    RealMatrix *mat = malloc(sizeof(RealMatrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->is_sparse = sparse;
    
    if (!sparse) {
        mat->data = malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            mat->data[i] = calloc(cols, sizeof(double));
        }
    } else {
        // Simplified sparse representation
        mat->nnz = 0;
        mat->row_ptr = malloc((rows + 1) * sizeof(int));
        mat->col_idx = malloc(rows * cols * sizeof(int));
        mat->values = malloc(rows * cols * sizeof(double));
    }
    return mat;
}

ComplexMatrix* create_complex_matrix(int rows, int cols, int sparse) {
    ComplexMatrix *mat = malloc(sizeof(ComplexMatrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->is_sparse = sparse;
    
    if (!sparse) {
        mat->data = malloc(rows * sizeof(double complex*));
        for (int i = 0; i < rows; i++) {
            mat->data[i] = calloc(cols, sizeof(double complex));
        }
    } else {
        mat->nnz = 0;
        mat->row_ptr = malloc((rows + 1) * sizeof(int));
        mat->col_idx = malloc(rows * cols * sizeof(int));
        mat->values = malloc(rows * cols * sizeof(double complex));
    }
    return mat;
}

void fill_real_matrix_random(RealMatrix *mat, double sparsity) {
    srand(time(NULL));
    
    if (!mat->is_sparse) {
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                if (!mat->is_sparse || (rand() / (double)RAND_MAX) > sparsity) {
                    mat->data[i][j] = (rand() / (double)RAND_MAX) * 10.0;
                }
            }
        }
    } else {
        // Fill sparse matrix in CSR format[8]
        int idx = 0;
        mat->row_ptr[0] = 0;
        
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                if ((rand() / (double)RAND_MAX) > sparsity) {
                    mat->col_idx[idx] = j;
                    mat->values[idx] = (rand() / (double)RAND_MAX) * 10.0;
                    idx++;
                }
            }
            mat->row_ptr[i + 1] = idx;
        }
        mat->nnz = idx;
    }
}

void fill_complex_matrix_random(ComplexMatrix *mat, double sparsity) {
    srand(time(NULL) + 1);
    
    if (!mat->is_sparse) {
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                if (!mat->is_sparse || (rand() / (double)RAND_MAX) > sparsity) {
                    double real_part = (rand() / (double)RAND_MAX) * 10.0;
                    double imag_part = (rand() / (double)RAND_MAX) * 10.0;
                    mat->data[i][j] = real_part + imag_part * I;
                }
            }
        }
    } else {
        int idx = 0;
        mat->row_ptr[0] = 0;
        
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                if ((rand() / (double)RAND_MAX) > sparsity) {
                    mat->col_idx[idx] = j;
                    double real_part = (rand() / (double)RAND_MAX) * 10.0;
                    double imag_part = (rand() / (double)RAND_MAX) * 10.0;
                    mat->values[idx] = real_part + imag_part * I;
                    idx++;
                }
            }
            mat->row_ptr[i + 1] = idx;
        }
        mat->nnz = idx;
    }
}

// Multithreaded matrix multiplication function[1][2]
void* matrix_mult_thread(void* arg) {
    ThreadData *data = (ThreadData*)arg;
    
    if (data->is_complex) {
        ComplexMatrix *a = (ComplexMatrix*)data->mat_a;
        ComplexMatrix *b = (ComplexMatrix*)data->mat_b;
        ComplexMatrix *c = (ComplexMatrix*)data->mat_c;
        
        if (!a->is_sparse && !b->is_sparse) {
            // Dense-dense multiplication
            for (int i = data->start_row; i < data->end_row; i++) {
                for (int j = 0; j < b->cols; j++) {
                    c->data[i][j] = 0.0 + 0.0*I;
                    for (int k = 0; k < a->cols; k++) {
                        c->data[i][j] += a->data[i][k] * b->data[k][j];
                    }
                }
            }
        }
    } else {
        RealMatrix *a = (RealMatrix*)data->mat_a;
        RealMatrix *b = (RealMatrix*)data->mat_b;
        RealMatrix *c = (RealMatrix*)data->mat_c;
        
        if (!a->is_sparse && !b->is_sparse) {
            // Dense-dense multiplication[5]
            for (int i = data->start_row; i < data->end_row; i++) {
                for (int j = 0; j < b->cols; j++) {
                    c->data[i][j] = 0.0;
                    for (int k = 0; k < a->cols; k++) {
                        c->data[i][j] += a->data[i][k] * b->data[k][j];
                    }
                }
            }
        } else if (a->is_sparse && !b->is_sparse) {
            // Sparse-dense multiplication[8]
            for (int i = data->start_row; i < data->end_row; i++) {
                for (int j = 0; j < b->cols; j++) {
                    c->data[i][j] = 0.0;
                    for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
                        int col = a->col_idx[k];
                        c->data[i][j] += a->values[k] * b->data[col][j];
                    }
                }
            }
        }
    }
    
    pthread_exit(NULL);
}

// Multithreaded Kronecker product function[3][7]
void* kronecker_thread(void* arg) {
    ThreadData *data = (ThreadData*)arg;
    
    if (data->is_complex) {
        ComplexMatrix *a = (ComplexMatrix*)data->mat_a;
        ComplexMatrix *b = (ComplexMatrix*)data->mat_b;
        ComplexMatrix *c = (ComplexMatrix*)data->mat_c;
        
        if (!a->is_sparse && !b->is_sparse) {
            // Dense Kronecker product
            int a_rows_per_thread = (a->rows + data->num_threads - 1) / data->num_threads;
            int start_i = data->thread_id * a_rows_per_thread;
            int end_i = (start_i + a_rows_per_thread > a->rows) ? a->rows : start_i + a_rows_per_thread;
            
            for (int i = start_i; i < end_i; i++) {
                for (int j = 0; j < a->cols; j++) {
                    for (int p = 0; p < b->rows; p++) {
                        for (int q = 0; q < b->cols; q++) {
                            int row = i * b->rows + p;
                            int col = j * b->cols + q;
                            c->data[row][col] = a->data[i][j] * b->data[p][q];
                        }
                    }
                }
            }
        }
    } else {
        RealMatrix *a = (RealMatrix*)data->mat_a;
        RealMatrix *b = (RealMatrix*)data->mat_b;
        RealMatrix *c = (RealMatrix*)data->mat_c;
        
        if (!a->is_sparse && !b->is_sparse) {
            // Dense Kronecker product[3]
            int a_rows_per_thread = (a->rows + data->num_threads - 1) / data->num_threads;
            int start_i = data->thread_id * a_rows_per_thread;
            int end_i = (start_i + a_rows_per_thread > a->rows) ? a->rows : start_i + a_rows_per_thread;
            
            for (int i = start_i; i < end_i; i++) {
                for (int j = 0; j < a->cols; j++) {
                    for (int p = 0; p < b->rows; p++) {
                        for (int q = 0; q < b->cols; q++) {
                            int row = i * b->rows + p;
                            int col = j * b->cols + q;
                            c->data[row][col] = a->data[i][j] * b->data[p][q];
                        }
                    }
                }
            }
        }
    }
    
    pthread_exit(NULL);
}

// Test function
void run_tests() {
    printf("=== Multithreaded Matrix Operations Test ===\n\n");
    
    int sizes[] = {50, 100, 200};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 8};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int n = sizes[size_idx];
        printf("Testing with matrix size: %dx%d\n", n, n);
        
        for (int tc_idx = 0; tc_idx < num_thread_counts; tc_idx++) {
            int num_threads = thread_counts[tc_idx];
            if (num_threads > MAX_THREADS) continue;
            
            printf("\n--- Using %d threads ---\n", num_threads);
            
            // Test 1: Dense Real Matrix Multiplication
            RealMatrix *a_real = create_real_matrix(n, n, 0);
            RealMatrix *b_real = create_real_matrix(n, n, 0);
            RealMatrix *c_real = create_real_matrix(n, n, 0);
            
            fill_real_matrix_random(a_real, 0.0);
            fill_real_matrix_random(b_real, 0.0);
            
            double start_time = get_time();
            
            pthread_t threads[MAX_THREADS];
            ThreadData thread_data[MAX_THREADS];
            
            int rows_per_thread = n / num_threads;
            for (int t = 0; t < num_threads; t++) {
                thread_data[t].thread_id = t;
                thread_data[t].start_row = t * rows_per_thread;
                thread_data[t].end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
                thread_data[t].num_threads = num_threads;
                thread_data[t].mat_a = a_real;
                thread_data[t].mat_b = b_real;
                thread_data[t].mat_c = c_real;
                thread_data[t].operation = 0;
                thread_data[t].is_complex = 0;
                
                pthread_create(&threads[t], NULL, matrix_mult_thread, &thread_data[t]);
            }
            
            for (int t = 0; t < num_threads; t++) {
                pthread_join(threads[t], NULL);
            }
            
            double end_time = get_time();
            printf("Dense Real Matrix Multiplication: %.6f seconds\n", end_time - start_time);
            
            // Test 2: Dense Complex Matrix Multiplication
            ComplexMatrix *a_complex = create_complex_matrix(n, n, 0);
            ComplexMatrix *b_complex = create_complex_matrix(n, n, 0);
            ComplexMatrix *c_complex = create_complex_matrix(n, n, 0);
            
            fill_complex_matrix_random(a_complex, 0.0);
            fill_complex_matrix_random(b_complex, 0.0);
            
            start_time = get_time();
            
            for (int t = 0; t < num_threads; t++) {
                thread_data[t].mat_a = a_complex;
                thread_data[t].mat_b = b_complex;
                thread_data[t].mat_c = c_complex;
                thread_data[t].is_complex = 1;
                
                pthread_create(&threads[t], NULL, matrix_mult_thread, &thread_data[t]);
            }
            
            for (int t = 0; t < num_threads; t++) {
                pthread_join(threads[t], NULL);
            }
            
            end_time = get_time();
            printf("Dense Complex Matrix Multiplication: %.6f seconds\n", end_time - start_time);
            
            // Test 3: Dense Real Kronecker Product (smaller size due to memory)
            if (n <= 100) {
                RealMatrix *a_kron = create_real_matrix(n/4, n/4, 0);
                RealMatrix *b_kron = create_real_matrix(n/4, n/4, 0);
                RealMatrix *c_kron = create_real_matrix((n/4)*(n/4), (n/4)*(n/4), 0);
                
                fill_real_matrix_random(a_kron, 0.0);
                fill_real_matrix_random(b_kron, 0.0);
                
                start_time = get_time();
                
                for (int t = 0; t < num_threads; t++) {
                    thread_data[t].mat_a = a_kron;
                    thread_data[t].mat_b = b_kron;
                    thread_data[t].mat_c = c_kron;
                    thread_data[t].operation = 1;
                    thread_data[t].is_complex = 0;
                    
                    pthread_create(&threads[t], NULL, kronecker_thread, &thread_data[t]);
                }
                
                for (int t = 0; t < num_threads; t++) {
                    pthread_join(threads[t], NULL);
                }
                
                end_time = get_time();
                printf("Dense Real Kronecker Product: %.6f seconds\n", end_time - start_time);
                
                // Free Kronecker matrices
                free(a_kron->data);
                free(b_kron->data);
                free(c_kron->data);
                free(a_kron);
                free(b_kron);
                free(c_kron);
            }
            
            // Test 4: Sparse Real Matrix Multiplication
            RealMatrix *a_sparse = create_real_matrix(n, n, 1);
            RealMatrix *b_sparse_dense = create_real_matrix(n, n, 0);
            RealMatrix *c_sparse = create_real_matrix(n, n, 0);
            
            fill_real_matrix_random(a_sparse, 0.9); // 90% sparse
            fill_real_matrix_random(b_sparse_dense, 0.0);
            
            start_time = get_time();
            
            for (int t = 0; t < num_threads; t++) {
                thread_data[t].mat_a = a_sparse;
                thread_data[t].mat_b = b_sparse_dense;
                thread_data[t].mat_c = c_sparse;
                thread_data[t].operation = 0;
                thread_data[t].is_complex = 0;
                
                pthread_create(&threads[t], NULL, matrix_mult_thread, &thread_data[t]);
            }
            
            for (int t = 0; t < num_threads; t++) {
                pthread_join(threads[t], NULL);
            }
            
            end_time = get_time();
            printf("Sparse-Dense Real Matrix Multiplication: %.6f seconds\n", end_time - start_time);
            
            // Cleanup
            free(a_real->data);
            free(b_real->data);
            free(c_real->data);
            free(a_real);
            free(b_real);
            free(c_real);
            
            free(a_complex->data);
            free(b_complex->data);
            free(c_complex->data);
            free(a_complex);
            free(b_complex);
            free(c_complex);
            
            free(a_sparse->row_ptr);
            free(a_sparse->col_idx);
            free(a_sparse->values);
            free(a_sparse);
            free(b_sparse_dense->data);
            free(c_sparse->data);
            free(b_sparse_dense);
            free(c_sparse);
        }
        printf("\n");
    }
}

int main() {
    printf("Multithreaded Matrix Operations Performance Test\n");
    printf("================================================\n");
    printf("Testing matrix multiplication and Kronecker products\n");
    printf("with dense/sparse matrices and real/complex numbers\n\n");
    
    run_tests();
    
    pthread_mutex_destroy(&print_mutex);
    return 0;
}
