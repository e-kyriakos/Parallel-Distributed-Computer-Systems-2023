#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"
#include "mmio.c"


typedef struct Sparse_CSR {
    int n_rows;
    int n_cols;
    int n_nz;
    int* row_ptrs;
    int* col_indices;
    double* values;
} Sparse_CSR;


Sparse_CSR read_sparse_matrix(int argc, char *argv[]){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    const double *val;
    int n_rows, n_cols, n_nz;
    Sparse_CSR *A_csr;
        printf("SEG FAULT");

    if (argc < 2){
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else if ((f = fopen(argv[1], "r")) == NULL) exit(1);
    
    if (mm_read_banner(f, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_matrix(matcode) && !mm_is_sparse(matcode) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */ //SEG FAULT
    if ((ret_code = mm_read_mtx_crd_size(f, &n_rows,&n_cols,&n_nz)) !=0) exit(1);

    A_csr->n_rows = n_rows;
    A_csr->n_cols = n_cols;
    A_csr->n_nz = n_nz;

    /* reseve memory for matrices */
    A_csr->row_ptrs = calloc(n_rows+1, sizeof(int));
    A_csr->col_indices = calloc(n_nz, sizeof(int));
    A_csr->values = calloc(n_nz, sizeof(double));

    int nz_id = 0;

     for (int i=0; i < n_rows; ++i) {
        A_csr->row_ptrs[i] = nz_id;
        for (int j=0; j< n_cols; ++j) {
            if (val[i* n_cols + j] != 0.0) {
                A_csr->col_indices[nz_id] = j;
                A_csr->values[nz_id] = val[i*n_cols + j];
                nz_id++;
            }
        }
    }

    A_csr->row_ptrs[n_rows] = nz_id;

    return *A_csr;
}

int print_sparse_csr(Sparse_CSR *A_csr) {
    printf("row\tcol\tval\n");
    printf("----\n");
    for (int i=0; i < A_csr->n_rows; ++i) {
        int nz_start = A_csr->row_ptrs[i];
        int nz_end = A_csr->row_ptrs[i+1];
        for (int nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            int j = A_csr->col_indices[nz_id];
            double val = A_csr->values[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
    return EXIT_SUCCESS;
}

int matrix_vector_sparse_csr(
    const Sparse_CSR* A_csr,
    const double* vec,
    double* res
) {
    for (int i=0; i<A_csr->n_rows; ++i) {
        res[i] = 0.0;
        int nz_start = A_csr->row_ptrs[i];
        int nz_end = A_csr->row_ptrs[i+1];
        for (int nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            int j = A_csr->col_indices[nz_id];
            double val = A_csr->values[nz_id];
            res[i] = res[i] + val * vec[j];
        }
    }
    return EXIT_SUCCESS;
}

int free_sparse_csr(Sparse_CSR* A_csr) {
    free(A_csr->row_ptrs);
    free(A_csr->col_indices);
    free(A_csr->values);

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]){

    Sparse_CSR A;
    A = read_sparse_matrix(argc, argv);
    //print_sparse_csr(&A);
    free_sparse_csr(&A);
}

