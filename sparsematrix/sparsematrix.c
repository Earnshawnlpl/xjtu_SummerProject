#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define A_ROWS 2048
#define A_COLS 4096
#define B_ROWS 4096
#define B_COLS 2048

// CSR格式矩阵定义
typedef struct {
    int *row_ptr;
    int *col_ind;
    double *values;
    int nnz; // 非零元素数量
    int rows;
    int cols;
} CSRMatrix;

// 分配CSR格式矩阵内存
CSRMatrix allocate_csr_matrix(int rows, int cols, int nnz) {
    CSRMatrix matrix;
    matrix.row_ptr = (int *) malloc((rows + 1) * sizeof(int));
    matrix.col_ind = (int *) malloc(nnz * sizeof(int));
    matrix.values = (double *) malloc(nnz * sizeof(double));
    matrix.nnz = nnz;
    matrix.rows = rows;
    matrix.cols = cols;
    return matrix;
}

// 释放CSR格式矩阵内存
void free_csr_matrix(CSRMatrix matrix) {
    free(matrix.row_ptr);
    free(matrix.col_ind);
    free(matrix.values);
}

// 串行CSR格式矩阵乘法 (C = A * B)
void spmm_csr_serial(CSRMatrix A, CSRMatrix B, CSRMatrix *C) {
    int *row_nnz = (int *) calloc(A.rows, sizeof(int));

    // 计算每行非零元素的数量
    for (int i = 0; i < A.rows; i++) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            int a_col = A.col_ind[j];
            for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; k++) {
                int b_col = B.col_ind[k];
                row_nnz[i]++;
            }
        }
    }

    // 为结果矩阵分配内存
    int nnz = 0;
    for (int i = 0; i < A.rows; i++) {
        nnz += row_nnz[i];
    }
    *C = allocate_csr_matrix(A.rows, B.cols, nnz);

    // 初始化C->row_ptr
    C->row_ptr[0] = 0;
    for (int i = 0; i < A.rows; i++) {
        C->row_ptr[i + 1] = C->row_ptr[i] + row_nnz[i];
    }

    // 计算结果矩阵的值
    for (int i = 0; i < A.rows; i++) {
        int index = C->row_ptr[i];
        double* temp_values = (double*)calloc(B.cols, sizeof(double));

        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            int a_col = A.col_ind[j];   
            for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; k++) {
                int b_col = B.col_ind[k];
                temp_values[b_col] += A.values[j] * B.values[k];
            }
        }

        for (int j = 0; j < B.cols; j++) {
            if (temp_values[j] != 0) {
                C->col_ind[index] = j;
                C->values[index] = temp_values[j];
                index++;
            }
        }

        free(temp_values);
    }

    free(row_nnz);
}

// 并行CSR格式矩阵乘法 (C = A * B)
void spmm_csr_parallel(CSRMatrix A, CSRMatrix B, CSRMatrix *C) {
    int *row_nnz = (int *) calloc(A.rows, sizeof(int));
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    // 计算每行非零元素的数量
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = tid * (A.rows / num_threads);
        int end = (tid + 1) * (A.rows / num_threads);
        if (tid == num_threads - 1) {
            end = A.rows;
        }
        for (int i = start; i < end; i++) {
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
                int a_col = A.col_ind[j];
                for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; k++) {
                    int b_col = B.col_ind[k];
                    row_nnz[i]++;
                }
            }
        }
    }

    // 为结果矩阵分配内存
    int nnz = 0;
    for (int i = 0; i < A.rows; i++) {
        nnz += row_nnz[i];
    }
    *C = allocate_csr_matrix(A.rows, B.cols, nnz);

    // 初始化C->row_ptr
    C->row_ptr[0] = 0;
    for (int i = 0; i < A.rows; i++) {
        C->row_ptr[i + 1] = C->row_ptr[i] + row_nnz[i];
    }

    // 计算结果矩阵的值
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = tid * (A.rows / num_threads);
        int end = (tid + 1) * (A.rows / num_threads);
        if (tid == num_threads - 1) {
            end = A.rows;
        }
        for (int i = start; i < end; i++) {
            int index = C->row_ptr[i];
            double* temp_values = (double*)calloc(B.cols, sizeof(double));

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
                int a_col = A.col_ind[j];
                for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; k++) {
                    int b_col = B.col_ind[k];
                    temp_values[b_col] += A.values[j] * B.values[k];
                }
            }

            for (int j = 0; j < B.cols; j++) {
                if (temp_values[j] != 0) {
                    C->col_ind[index] = j;
                    C->values[index] = temp_values[j];
                    index++;
                }
            }

            free(temp_values);
        }
    }

    free(row_nnz);
}

// 比较两个CSR矩阵是否相同
int compare_csr_matrices(CSRMatrix A, CSRMatrix B) {
    if (A.rows != B.rows || A.cols != B.cols || A.nnz != B.nnz) return 0;
    for (int i = 0; i <= A.rows; i++) {
        if (A.row_ptr[i] != B.row_ptr[i]) return 0;
    }
    for (int i = 0; i < A.nnz; i++) {
        if (A.col_ind[i] != B.col_ind[i] || A.values[i] != B.values[i]) return 0;
    }
    return 1;
}

int main() {
    int nnz_A = 100000; // 假设A非零元素数量
    int nnz_B = 100000; // 假设B非零元素数量

    CSRMatrix A = allocate_csr_matrix(A_ROWS, A_COLS, nnz_A);
    CSRMatrix B = allocate_csr_matrix(B_ROWS, B_COLS, nnz_B);
    CSRMatrix C_serial = {0};
    CSRMatrix C_parallel = {0};

    // 为C_serial和C_parallel设置行数和列数
    C_serial.rows = A_ROWS;
    C_serial.cols = B_COLS;
    C_parallel.rows = A_ROWS;
    C_parallel.cols = B_COLS;

    // 初始化稀疏矩阵A和B
    srand(time(NULL));
    for (int i = 0; i <= A_ROWS; i++) {
        A.row_ptr[i] = (i * nnz_A) / A_ROWS;
    }
    for (int i = 0; i < nnz_A; i++) {
        A.col_ind[i] = rand() % A_COLS;
        A.values[i] = (double)(rand() % 100);
    }
    for (int i = 0; i <= B_ROWS; i++) {
        B.row_ptr[i] = (i * nnz_B) / B_ROWS;
    }
    for (int i = 0; i < nnz_B; i++) {
        B.col_ind[i] = rand() % B_COLS;
        B.values[i] = (double)(rand() % 100);
    }

    // 计算串行稀疏矩阵乘法时间
    double start_time = omp_get_wtime();
    spmm_csr_serial(A, B, &C_serial);
    double end_time = omp_get_wtime();
    printf("串行稀疏矩阵乘法时间: %f seconds\n", end_time - start_time);

    // 计算并行稀疏矩阵乘法时间
    start_time = omp_get_wtime();
    spmm_csr_parallel(A, B, &C_parallel);
    end_time = omp_get_wtime();
    printf("并行稀疏矩阵乘法时间: %f seconds\n", end_time - start_time);

    // 比较结果矩阵
    if (compare_csr_matrices(C_serial, C_parallel)) {
        printf("串行计算和并行计算的结果相同。\n");
    } else {
        printf("串行计算和并行计算的结果不同。\n");
    }

    // 释放内存
    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C_serial);
    free_csr_matrix(C_parallel);

    return 0;
}