#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define A_ROWS 2048
#define A_COLS 4096
#define B_ROWS 4096
#define B_COLS 2048
#define BLOCK_SIZE 32  // 调整块大小以适应缓存

// 串行化矩阵相乘
void matrix_multiply_serial(double **A, double **B, double **C) {
    for (int i = 0; i < A_ROWS; ++i) {
        for (int j = 0; j < B_COLS; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < A_COLS; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 并行化矩阵相乘（分块优化）
void matrix_multiply_parallel(double **A, double **B, double **C) {
    int i, j, k, ii, jj, kk;
    int num_blocks_row = A_ROWS / BLOCK_SIZE;
    int num_blocks_col = B_COLS / BLOCK_SIZE;
    
    omp_set_num_threads(4);
    #pragma omp parallel for private(i, j, k, ii, jj, kk)
    for (int block_i = 0; block_i < num_blocks_row; ++block_i) {
        for (int block_j = 0; block_j < num_blocks_col; ++block_j) {
            for (int block_k = 0; block_k < (A_COLS / BLOCK_SIZE); ++block_k) {
                // 计算每个块的起始位置
                int row_start = block_i * BLOCK_SIZE;
                int col_start = block_j * BLOCK_SIZE;
                int k_start = block_k * BLOCK_SIZE;

                // 计算局部块乘法
                for (i = row_start; i < row_start + BLOCK_SIZE; ++i) {
                    for (j = col_start; j < col_start + BLOCK_SIZE; ++j) {
                        double sum = 0;
                        for (k = k_start; k < k_start + BLOCK_SIZE; ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

// 为矩阵分配内存
double** allocate_matrix(int rows, int cols) {
    double **matrix = (double **) malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; ++i) {
        matrix[i] = (double *) malloc(cols * sizeof(double));
    }
    return matrix;
}

// 释放矩阵内存
void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

// 比较串行化计算的矩阵和并行化计算的矩阵是否相同
int compare_matrices(double **C1, double **C2, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (C1[i][j] != C2[i][j]) {
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    double **A = allocate_matrix(A_ROWS, A_COLS);
    double **B = allocate_matrix(B_ROWS, B_COLS);
    double **C_serial = allocate_matrix(A_ROWS, B_COLS);
    double **C_parallel = allocate_matrix(A_ROWS, B_COLS);

    // 用随机数初始化A，B矩阵
    srand(time(NULL));
    for (int i = 0; i < A_ROWS; ++i) {
        for (int j = 0; j < A_COLS; ++j) {
            A[i][j] = rand() % 100;
        }
    }
    for (int i = 0; i < B_ROWS; ++i) {
        for (int j = 0; j < B_COLS; ++j) {
            B[i][j] = rand() % 100;
        }
    }

    // 计算串行化时间
    double start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C_serial);
    double end_time = omp_get_wtime();
    printf("串行化时间: %f seconds\n", end_time - start_time);

    // 计算并行化时间
    start_time = omp_get_wtime();
    matrix_multiply_parallel(A, B, C_parallel);
    end_time = omp_get_wtime();
    printf("并行化时间: %f seconds\n", end_time - start_time);

    // 比较矩阵
    if (compare_matrices(C_serial, C_parallel, A_ROWS, B_COLS)) {
        printf("串行矩阵和并行矩阵相同.\n");
    } else {
        printf("串行矩阵和并行矩阵不相同.\n");
    }

    // 释放分配的内存
    free_matrix(A, A_ROWS);
    free_matrix(B, B_ROWS);
    free_matrix(C_serial, A_ROWS);
    free_matrix(C_parallel, A_ROWS);

    return 0;
}
