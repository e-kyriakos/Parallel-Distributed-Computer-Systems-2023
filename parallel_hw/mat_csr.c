#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include "mmio.h"

// COO sparse matrix structure
typedef struct {
    int rows;
    int cols;
    int nnz;  // Number of non-zero elements
    double *values;  // Array of non-zero values
    int *row_indices;  // Array of row indices
    int *col_indices;  // Array of column indices
} COOMatrix;

// Sparse Matrix in CSR format
typedef struct {
    int rows;
    int cols;
    int nnz;  // Number of non-zero elements
    double *values;  // Array of non-zero values
    int *col_indices;  // Array of column indices
    int *row_ptr;  // Row pointers
} SparseMatrixCSR;

// Thread data structure for pthreads
typedef struct {
    int start;
    int end;
    double *R_values;
    int *R_col_indices;
    int *R_row_ptr;
} ThreadData;

// Function to convert COO to CSR
SparseMatrixCSR cooToCSR(const COOMatrix *coo) {
    SparseMatrixCSR csr;
    csr.rows = coo->rows;
    csr.cols = coo->cols;
    csr.nnz = coo->nnz;

    // Allocate memory for CSR arrays
    csr.values = (double *)malloc(sizeof(double) * csr.nnz);
    csr.col_indices = (int *)malloc(sizeof(int) * csr.nnz);
    csr.row_ptr = (int *)malloc(sizeof(int) * (csr.rows + 1));

    // Initialize row_ptr to zero
    for (int i = 0; i <= csr.rows; i++) {
        csr.row_ptr[i] = 0;
    }

    // Count non-zero elements in each row
    for (int i = 0; i < csr.nnz; i++) {
        csr.row_ptr[coo->row_indices[i]]++;
    }

    // Cumulative sum to get the starting index of each row
    int cumsum = 0;
    for (int i = 0; i < csr.rows; i++) {
        int temp = csr.row_ptr[i];
        csr.row_ptr[i] = cumsum;
        cumsum += temp;
    }
    csr.row_ptr[csr.rows] = csr.nnz;

    // Fill values and col_indices in CSR format
    for (int i = 0; i < csr.nnz; i++) {
        int row = coo->row_indices[i];
        int dest = csr.row_ptr[row];

        csr.values[dest] = coo->values[i];
        csr.col_indices[dest] = coo->col_indices[i];

        csr.row_ptr[row]++;
    }

    // Shift row_ptr back
    for (int i = csr.rows; i > 0; i--) {
        csr.row_ptr[i] = csr.row_ptr[i - 1];
    }
    csr.row_ptr[0] = 0;

    return csr;
}

// Function to perform matrix calculation R^T * A * R using pthreads
void *matrixCalculationPthreads(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < data->end; j++) {
            double temp = 0.0;
            for (int k = data->R_row_ptr[i]; k < data->R_row_ptr[i + 1]; k++) {
                temp += data->R_values[k] * data->R_values[data->R_col_indices[k]] * data->R_values[data->R_col_indices[k]];
            }
            data->R_values[i] = temp;
        }
    }

    pthread_exit(NULL);
}

// Function to perform matrix calculation R^T * A * R using OpenMP
void matrixCalculationOpenMP(int rows, int cols, double *R_values, int *R_col_indices, int *R_row_ptr) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            double temp = 0.0;
            for (int k = R_row_ptr[i]; k < R_row_ptr[i + 1]; k++) {
                temp += R_values[k] * R_values[R_col_indices[k]] * R_values[R_col_indices[k]];
            }
            R_values[i] = temp;
        }
    }
}

int main(int argc, char** argv) {
    int ret_code;
    MM_typecode matcode;
    FILE* f;
    int n_rows, n_cols, n_nz;
    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }else {
        if ((f = fopen(argv[1], "r")) == NULL)
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &n_rows, &n_cols, &n_nz)) != 0) exit(1);

    I = (int*)malloc(n_nz * sizeof(int));
    J = (int*)malloc(n_nz * sizeof(int));
    val = (double*)malloc(n_nz * sizeof(double));

    for (int i = 0; i < n_nz; i++) {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  // Adjust
        J[i]--;
    }

    if(f != stdin) fclose(f);

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, n_rows, n_cols, n_nz);
    printf("COO matrix\n");
    for (int i = 0; i < n_nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);


    COOMatrix COO;

    COO.rows = n_rows;
    COO.cols = n_cols;
    COO.nnz = n_nz;

    // Example matrices R in CSR format
    SparseMatrixCSR R;
    R.rows = 3;
    R.cols = 3;
    R.nnz = 5;
    R.values = (double[]){1.0, 1.0, 1.0, 1.0, 1.0};
    R.col_indices = (int[]){0, 1, 2, 0, 2};
    R.row_ptr = (int[]){0, 3, 5};

    // Perform matrix calculation using pthreads
    int num_threads = 16;  // Adjust as needed
    pthread_t pthreads_tid[num_threads];
    ThreadData thread_data[num_threads];

    int chunk_size = R.rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? R.rows : (i + 1) * chunk_size;
        thread_data[i].R_values = R.values;
        thread_data[i].R_col_indices = R.col_indices;
        thread_data[i].R_row_ptr = R.row_ptr;

        pthread_create(&pthreads_tid[i], NULL, matrixCalculationPthreads, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(pthreads_tid[i], NULL);
    }
/*
    // Perform matrix calculation using OpenMP
    matrixCalculationOpenMP(R.rows, R.cols, R.values, R.col_indices, R.row_ptr);
*/
    // Print the resulting matrix
    printf("Resulting Matrix:\n");
    for (int i = 0; i < R.rows; i++) {
        printf("%f\t", R.values[i]);
    }
    printf("\n");

    return 0;
}

