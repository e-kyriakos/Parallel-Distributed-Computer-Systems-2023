#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "mmio.h"

#define NUM_THREADS 4

typedef struct {
  int id;
} param;

typedef struct {
    int n_rows;
    int n_cols;
    int n_nz;
    int* row_ptrs;
    int* col_indices;
    int* values;
} Sparse_CSR;

struct ThreadArgs {
    int* arr;
    size_t size;
};

Sparse_CSR coo_to_csr(int n_rows, int n_cols, int n_nz, int* I, int* J, int* val) {
    Sparse_CSR A_csr;
    A_csr.n_rows = n_rows;
    A_csr.n_cols = n_cols;
    A_csr.n_nz = n_nz;

    A_csr.row_ptrs = (int*)calloc(n_rows + 1, sizeof(int));
    A_csr.col_indices = (int*)malloc(n_nz * sizeof(int));
    A_csr.values = (int*)malloc(n_nz * sizeof(int));

    for (int i = 0; i < n_nz; i++) {
        A_csr.row_ptrs[I[i]]++;
    }

    size_t sum = 0;
    for (int i = 0; i <= n_rows; i++) {
        int temp = A_csr.row_ptrs[i];
        A_csr.row_ptrs[i] = sum;
        sum += temp;
    }

    for (int i = 0; i < n_nz; i++) {
        int row = I[i];
        int index = A_csr.row_ptrs[row];
        A_csr.col_indices[index] = J[i];
        A_csr.values[index] = val[i];
        A_csr.row_ptrs[row]++;
    }

    for (int i = n_rows; i > 0; i--) {
        A_csr.row_ptrs[i] = A_csr.row_ptrs[i - 1];
    }
    A_csr.row_ptrs[0] = 0;

    return A_csr;
}

void* parallelMergeSort(void* arg){
	struct ThreadArgs* targs = (struct ThreadArgs*) arg;
	mergeSort(targs->arr, 0, targs->size - 1);
	return NULL;
}

int compare(const void *a, const void* b){
	return (*(int*)a - *(int*)b);
}

// Merge two sorted subarrays
void merge(int arr[], int l, int m, int r) {
     int n1 = m - l + 1;
     int n2 = r - m;

     int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r){
	if(l < r){
		int m = l + (r - l) / 2;
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}

int print_sparse_csr(const Sparse_CSR* A_csr) {
    printf("row\tcol\tval\n");
    printf("----\n");
    for (size_t i=0; i<A_csr->n_rows; ++i) {
        size_t nz_start = A_csr->row_ptrs[i];
        size_t nz_end = A_csr->row_ptrs[i+1];
        for (size_t nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            size_t j = A_csr->col_indices[nz_id];
            int val = A_csr->values[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
    return EXIT_SUCCESS;
}


void free_csr(Sparse_CSR* A_csr) {
    free(A_csr->row_ptrs);
    free(A_csr->col_indices);
    free(A_csr->values);
}

int main(int argc, char *argv[]) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    int *val;

    //pthread_t *threads;
    //pthread_attr_t pthread_custom_attr;
    //param *p;

    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    /* reseve memory for matrices */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (int *) malloc(nz * sizeof(int));

    for (i=0; i<nz; i++)
    {
        scanf(f, "%d %d %d\n", &I[i], &J[i], (int)val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);

    Sparse_CSR A_csr = coo_to_csr(M, N, nz, I, J, val);

    // Now you can work with the CSR matrix
    print_sparse_csr(&A_csr);

    //threads = (pthread_t *)malloc(nz*sizeof(pthread_t));
    //pthread_attr_init(&pthread_custom_attr);
    //p = (param *)malloc(sizeof(param)*nz);

    /*
    struct ThreadArgs targs[NUM_THREADS];
    pthread_t threads[NUM_THREADS];

    //Device array into chunks for each available thread
    size_t chunk_size = nz / NUM_THREADS;
    for(size_t i = 0; i < NUM_THREADS; i++){
		targs[i].arr = &A_csr.values[i] + i *chunk_size;
		targs[i].size = (i == NUM_THREADS - 1) ? (nz - i *chunk_size) : chunk_size;
		pthread_create(&threads[i], NULL, parallelMergeSort, &targs[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Merge sorted subarrays
    for (int i = 1; i < NUM_THREADS; i++) {
        merge(A_csr.values, 0, (i - 1) * chunk_size - 1, i * chunk_size - 1);
    }

    for (int i = 0; i < nz; i++) {
	    //printf("%d\n", A_csr.values);
    }
*/
    // Free the allocated memory when done
    free_csr(&A_csr);

    return 0;
}

