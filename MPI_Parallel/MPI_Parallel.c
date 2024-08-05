#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define A_ROWS 16
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 16

// 串行矩阵相乘
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

// 分配矩阵内存
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

// 比较两个矩阵是否相等
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

void print_matrix(double **matrix, int rows, int cols, const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.0f  ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 主函数
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double **A, **B, **C_serial, **C_parallel;
    double *linear_A, *local_A, *local_C, *B_linear, *C_parallel_linear;
    int rows_per_proc = A_ROWS / size;

    // 进程0负责初始化和分发数据
    if (rank == 0) {
        A = allocate_matrix(A_ROWS, A_COLS);
        B = allocate_matrix(B_ROWS, B_COLS);
        C_serial = allocate_matrix(A_ROWS, B_COLS);
        C_parallel = allocate_matrix(A_ROWS, B_COLS);
    
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
        double start_time = MPI_Wtime();
        matrix_multiply_serial(A, B, C_serial);
        double end_time = MPI_Wtime();
        printf("串行化时间: %f seconds\n", end_time - start_time);

        // 线性化A矩阵
        linear_A = (double *)malloc(A_ROWS * A_COLS * sizeof(double));
        for (int i = 0; i < A_ROWS; ++i) {
            for (int j = 0; j < A_COLS; ++j) {
                linear_A[i * A_COLS + j] = A[i][j];
            }
        }
    } else {
        B = allocate_matrix(B_ROWS, B_COLS);
    }

    // 为线性化的B矩阵分配内存
    B_linear = (double *)malloc(B_ROWS * B_COLS * sizeof(double));
    if (rank == 0) {
        for (int i = 0; i < B_ROWS; ++i) {
            for (int j = 0; j < B_COLS; ++j) {
                B_linear[i * B_COLS + j] = B[i][j];
            }
        }
    }

    // 广播B矩阵和线性化的A矩阵
    MPI_Bcast(B_linear, B_ROWS * B_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Bcast(linear_A, A_ROWS * A_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // 分配local_A和local_C内存
    local_A = (double *) malloc(rows_per_proc * A_COLS * sizeof(double));
    local_C = (double *) malloc(rows_per_proc * B_COLS * sizeof(double));

    // 每个进程分配部分A矩阵到local_A
    MPI_Scatter(linear_A, rows_per_proc * A_COLS, MPI_DOUBLE, local_A, rows_per_proc * A_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 记录并行计算开始时间
    double parallel_start_time = MPI_Wtime();

    // 并行计算局部C矩阵
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < B_COLS; ++j) {
            local_C[i * B_COLS + j] = 0;
            for (int k = 0; k < A_COLS; ++k) {
                local_C[i * B_COLS + j] += local_A[i * A_COLS + k] * B_linear[k * B_COLS + j];
            }
        }
    }

    // 记录并行计算结束时间
    double parallel_end_time = MPI_Wtime();
    if (rank == 0) {
        printf("并行计算时间: %f seconds\n", parallel_end_time - parallel_start_time);
    }

    // 分配用于收集的内存
    if (rank == 0) {
        C_parallel_linear = (double *)malloc(A_ROWS * B_COLS * sizeof(double));
    }

    // 收集局部C矩阵到进程0
    MPI_Gather(local_C, rows_per_proc * B_COLS, MPI_DOUBLE, C_parallel_linear, rows_per_proc * B_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 进程0将线性化的C_parallel转换为二维数组
    if (rank == 0) {
        for (int i = 0; i < A_ROWS; ++i) {
            for (int j = 0; j < B_COLS; ++j) {
                C_parallel[i][j] = C_parallel_linear[i * B_COLS + j];
            }
        }

        // 打印矩阵以检查
        //print_matrix(C_serial, A_ROWS, B_COLS, "C_serial");
        //print_matrix(C_parallel, A_ROWS, B_COLS, "C_parallel");

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
        free(C_parallel_linear);
        free(linear_A);
    } else {
        free_matrix(B, B_ROWS);
    }

    free(local_A);
    free(local_C);
    free(B_linear);

    MPI_Finalize();
    return 0;
}
