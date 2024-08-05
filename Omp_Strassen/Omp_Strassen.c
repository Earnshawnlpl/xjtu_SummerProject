#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define A_ROWS 2048
#define A_COLS 4096
#define B_ROWS 4096
#define B_COLS 2048

// 函数原型声明
double** allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void strassen(double **A, double **B, double **C, int size);

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

// 分配矩阵内存并进行错误检查
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "Error allocating memory for matrix rows\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; ++i) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Error allocating memory for matrix columns\n");
            exit(EXIT_FAILURE);
        }
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

// 矩阵加法函数
void add_matrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// 矩阵减法函数
void sub_matrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Strassen矩阵相乘
// Strassen 矩阵乘法函数
void strassen(double **A, double **B, double **C, int n) {
    if (n <= 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int m = n / 2;

    double** A11 = allocate_matrix(m, m);
    double** A12 = allocate_matrix(m, m);
    double** A21 = allocate_matrix(m, m);
    double** A22 = allocate_matrix(m, m);
    double** B11 = allocate_matrix(m, m);
    double** B12 = allocate_matrix(m, m);
    double** B21 = allocate_matrix(m, m);
    double** B22 = allocate_matrix(m, m);
    double** M1 = allocate_matrix(m, m);
    double** M2 = allocate_matrix(m, m);
    double** M3 = allocate_matrix(m, m);
    double** M4 = allocate_matrix(m, m);
    double** M5 = allocate_matrix(m, m);
    double** M6 = allocate_matrix(m, m);
    double** M7 = allocate_matrix(m, m);
    double** C11 = allocate_matrix(m, m);
    double** C12 = allocate_matrix(m, m);
    double** C21 = allocate_matrix(m, m);
    double** C22 = allocate_matrix(m, m);

    // 分割矩阵
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + m];
            A21[i][j] = A[i + m][j];
            A22[i][j] = A[i + m][j + m];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + m];
            B21[i][j] = B[i + m][j];
            B22[i][j] = B[i + m][j + m];
        }
    }

    double** temp1 = allocate_matrix(m, m);
    double** temp2 = allocate_matrix(m, m);

    // 计算 M1 到 M7
    add_matrix(A11, A22, temp1, m);
    add_matrix(B11, B22, temp2, m);
    strassen(temp1, temp2, M1, m);

    add_matrix(A21, A22, temp1, m);
    strassen(temp1, B11, M2, m);

    sub_matrix(B12, B22, temp2, m);
    strassen(A11, temp2, M3, m);

    sub_matrix(B21, B11, temp2, m);
    strassen(A22, temp2, M4, m);

    add_matrix(A11, A12, temp1, m);
    strassen(temp1, B22, M5, m);

    sub_matrix(A21, A11, temp1, m);
    add_matrix(B11, B12, temp2, m);
    strassen(temp1, temp2, M6, m);

    sub_matrix(A12, A22, temp1, m);
    add_matrix(B21, B22, temp2, m);
    strassen(temp1, temp2, M7, m);

    // 组合结果
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C12[i][j] = M3[i][j] + M5[i][j];
            C21[i][j] = M2[i][j] + M4[i][j];
            C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    // 将 C11, C12, C21, C22 合并到 C 中
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + m] = C12[i][j];
            C[i + m][j] = C21[i][j];
            C[i + m][j + m] = C22[i][j];
        }
    }

    // 释放内存
    free_matrix(A11, m);
    free_matrix(A12, m);
    free_matrix(A21, m);
    free_matrix(A22, m);
    free_matrix(B11, m);
    free_matrix(B12, m);
    free_matrix(B21, m);
    free_matrix(B22, m);
    free_matrix(M1, m);
    free_matrix(M2, m);
    free_matrix(M3, m);
    free_matrix(M4, m);
    free_matrix(M5, m);
    free_matrix(M6, m);
    free_matrix(M7, m);
    free_matrix(C11, m);
    free_matrix(C12, m);
    free_matrix(C21, m);
    free_matrix(C22, m);
    free_matrix(temp1, m);
    free_matrix(temp2, m);
}


// 并行化矩阵相乘（分成2个矩阵并采用Strassen算法）
void matrix_multiply_parallel(double **A, double **B, double **C) {
    int i, j;
    // 分块大小
    int block_rows = A_ROWS;
    int block_cols = B_COLS;
    
    omp_set_num_threads(2);
    #pragma omp parallel for shared(A, B, C) private(i, j)
    for (int block = 0; block < 2; block++) {
        int row_start = 0;
        int col_start = (block % 2) * block_cols;

        double **subA = allocate_matrix(block_rows, block_cols);
        double **subB = allocate_matrix(block_rows, block_cols);
        double **subC = allocate_matrix(block_rows, block_cols);
        
        for (i = 0; i < block_rows; ++i) {
            for (j = 0; j < block_cols; ++j) {
                subA[i][j] = A[i][j + col_start];
                subB[i][j] = B[i + col_start][j];
            }
        }

        // 使用 Strassen 算法进行乘法
        strassen(subA, subB, subC, block_rows);
        
        // 使用 reduction 子句进行累加
        #pragma omp critical
        {
            for (i = 0; i < block_rows; ++i) {
                for (j = 0; j < block_cols; ++j) {
                    C[i][j] += subC[i][j];
                }
            }
        }
        
        free_matrix(subA, block_rows);
        free_matrix(subB, block_rows);
        free_matrix(subC, block_rows);
    }
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
