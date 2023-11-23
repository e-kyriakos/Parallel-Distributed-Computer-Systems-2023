#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include <cilk/cilk.h>
#include "mmio.h"
#include <sys/time.h>

#define ALLOC_CHUNK 10

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int rows;
    int cols;
    int nnz;       // Number of non-zero elements
    int *values;   // Array of non-zero values
    int *col_indices; // Array of column indices
    int *row_ptr;  // Array of row pointers
} Sparse_CSR;

// Thread data structure for pthreads
typedef struct {
    int start;
    int end;
    const Sparse_CSR *A;
    const Sparse_CSR *B;
    Sparse_CSR *C;
} ThreadData;

void freeCSR(Sparse_CSR *A){
    free(A->values);
    free(A->col_indices);
    free(A->row_ptr);
}

void displayDenseMatrix(const Sparse_CSR *csr) {
    // Allocate memory for dense matrix
    double **denseMatrix = (double **)malloc(csr->rows * sizeof(double *));
    for (int i = 0; i < csr->rows; i++) {
        denseMatrix[i] = (double *)malloc(csr->cols * sizeof(double));
    }

    // Initialize dense matrix with zeros
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            denseMatrix[i][j] = 0.0;
        }
    }

    // Fill in values from CSR matrix
    for (int i = 0; i < csr->rows; i++) {
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
            int col = csr->col_indices[j];
            denseMatrix[i][col] = csr->values[j];
        }
    }

    // Display the dense matrix
    printf("Dense Matrix:\n");
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            printf("%.2f\t", denseMatrix[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < csr->rows; i++) {
        free(denseMatrix[i]);
    }

    free(denseMatrix);
}

int print_sparse_csr(Sparse_CSR* A_csr){
	printf("row\tcol\tval\n");
    	printf("----\n");
    	for (size_t i=0; i<A_csr->rows; ++i) {
        	size_t nz_start = A_csr->row_ptr[i];
        	size_t nz_end = A_csr->row_ptr[i+1];
        for (size_t nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            size_t j = A_csr->col_indices[nz_id];
            double val = A_csr->values[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
    return EXIT_SUCCESS;
}

void display_csr(Sparse_CSR A, int num_nodes, char* name){

    // Display the generated matrices in CSR format
    printf(" %s CSR Matrix (Values): ", name);
    for (int i = 0; i < A.nnz; i++) {
        printf("%d ", A.values[i]);
    }
    printf("\n");

    printf("%s CSR Matrix (Column Indices): ", name);
    for (int i = 0; i < A.nnz; i++) {
        printf("%d ", A.col_indices[i]);
    }
    printf("\n");

    printf("%s CSR Matrix (Row Pointer): ", name);
    for (int i = 0; i <= num_nodes; i++) {
        printf("%d ", A.row_ptr[i]);
    }
    printf("\n");

    print_sparse_csr(&A);
    displayDenseMatrix(&A);
}

Sparse_CSR sparse_R(int *C, int num_nodes, int num_clusters){
	//Count the number of non-zero elements  in CSR matrix R

	//Allocate memory for CSR matrix R
	Sparse_CSR R;
	R.rows = num_nodes;
	R.cols = num_clusters;
	R.nnz = num_nodes; // The total number of aces has to be the aces mentioned in each row. Colum size / num nodes size.
	R.values = (int *)malloc(num_nodes * sizeof(int));
	R.col_indices = (int *)malloc(num_clusters * sizeof(int));
	R.row_ptr = (int *)malloc((num_nodes + 1)* sizeof(int));

	//Populate CSR matrix R
	int index = 0;
	R.row_ptr[0] = 0;
	for(int i =  0; i < num_nodes; i++){
		R.values[index] = 1;
		R.col_indices[index] = C[i] - 1;
		R.row_ptr[i] = index;
		index++;
	}
	R.row_ptr[R.rows] = index;

	return R;
}

void transposeCSRMatrix(Sparse_CSR *input, Sparse_CSR *output) {
    // Set the dimensions
    output->rows = input->cols;
    output->cols = input->rows;
    output->nnz = input->nnz;

    // Allocate memory
    output->values = (int *)malloc(output->nnz * sizeof(int));
    output->col_indices = (int *)malloc(output->nnz * sizeof(int));
    output->row_ptr = (int *)malloc((output->rows + 1) * sizeof(int));

    // Initialize row pointers to zero
    for (int i = 0; i <= output->rows; i++) {
        output->row_ptr[i] = 0;
    }
    // Count the number of elements in each column
    for (int i = 0; i < input->nnz; i++) {
        output->row_ptr[input->col_indices[i] + 1]++;
    }

    // Cumulative sum to get the starting index of each column
    for (int i = 1; i <= output->rows; i++) {
        output->row_ptr[i] += output->row_ptr[i - 1];
    }

    // Copy values and column indices to the transpose matrix
    for (int i = 0; i < input->rows; i++) {
        for (int j = input->row_ptr[i]; j < input->row_ptr[i + 1]; j++) {
            int col = input->col_indices[j];
            int index = output->row_ptr[col];
            output->values[index] = input->values[j];
            output->col_indices[index] = i;
            output->row_ptr[col]++;
        }
    }

    for(int i = output->rows; i > 0; i--){
	    output->row_ptr[i] = output->row_ptr[i - 1];
    }
    output->row_ptr[0] = 0;
}

void* multiplyThread(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    data->C->values = (int *)malloc(sizeof(int));
    data->C->col_indices = (int *)malloc(sizeof(int));
    data->C->nnz = 0;

    for (int i = data->start; i < data->end; ++i) {
        int temp_nn = 0;
	int *temp_values = (int*)malloc(data->B->cols * sizeof(int));

        for (int j = 0; j < data->B->cols; ++j) {
            int dot_product = 0;

            for (int k = data->A->row_ptr[i]; k < data->A->row_ptr[i + 1]; ++k) {
                int colA = data->A->col_indices[k];

                for (int l = data->B->row_ptr[colA]; l < data->B->row_ptr[colA + 1]; ++l) {
                    if (data->B->col_indices[l] == j) {
                        dot_product += data->A->values[k] * data->B->values[l];
                        break;
                    }
                }
            }
            if (dot_product != 0) {

		pthread_mutex_lock(&mutex);

		data->C->nnz++;
		data->C->values = (int *)realloc(data->C->values, data->C->nnz * sizeof(int));
		data->C->col_indices = (int *)realloc(data->C->col_indices, data->C->nnz * sizeof(int));

		data->C->values[data->C->nnz - 1] = dot_product;
		data->C->col_indices[data->C->nnz - 1] = j;
                // Store the result in the temporary array
                temp_values[temp_nn] = dot_product;
                temp_nn++;

		pthread_mutex_unlock(&mutex);
            }
        }
	data->C->row_ptr[i + 1] = data->C->nnz;
    }
    pthread_exit(NULL);
}

void multiplyCSR_POSIX(Sparse_CSR *A, Sparse_CSR *B, Sparse_CSR *C, int num_threads){
	pthread_t threads[num_threads];
	ThreadData thread_data[num_threads];

	C->rows = A->rows;
    	C->cols = B->cols;
    	C->nnz = 0;

    // Initialize row pointers
    C->row_ptr = (int*)malloc((C->rows + 1) * sizeof(int));
    C->row_ptr[0] = 0;

    // Create threads and distribute the work
    int rows_per_thread = C->rows / num_threads;
    int remaining_rows = C->rows % num_threads;
    int start_row = 0;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].start = start_row;
        thread_data[i].end = start_row + rows_per_thread + (i < remaining_rows ? 1 : 0);
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;

	printf("Create thread %d\n", i);
        pthread_create(&threads[i], NULL, multiplyThread, (void*)&thread_data[i]);

        start_row = thread_data[i].end;
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
}

void multiplyCSR_OPENMP(Sparse_CSR *A, Sparse_CSR *B, Sparse_CSR *C) {

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = 0;

    // Initialize row pointers
    C->row_ptr = (int *)malloc((C->rows + 1) * sizeof(int));
    C->row_ptr[0] = 0;

    // Initialize values and col_indices arrays (allocate initial space)
    C->values = (int *)malloc(A->rows * B->cols * sizeof(int));
    C->col_indices = (int *)malloc(A->rows * B->cols * sizeof(int));

    #pragma omp parallel for collapse(2) reduction(+:C->nnz)
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            int dot_product = 0;

            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
                int colA = A->col_indices[k];

                for (int l = B->row_ptr[colA]; l < B->row_ptr[colA + 1]; ++l) {
                    if (B->col_indices[l] == j) {
                        dot_product += A->values[k] * B->values[l];
                        break;
                    }
                }
            }

            // If dot product is non-zero, add it to the result matrix
            if (dot_product != 0) {
                int index = 0;

                #pragma omp critical
                {
                    index = C->nnz;
                    C->nnz++;
                }

                C->values[index] = dot_product;
                C->col_indices[index] = j;
            }
        }

        // Set row pointer
        C->row_ptr[i + 1] = C->nnz;
    }

    // Resize arrays to fit the actual number of non-zero elements
    C->values = (int *)realloc(C->values, C->nnz * sizeof(int));
    C->col_indices = (int *)realloc(C->col_indices, C->nnz * sizeof(int));
}

void multiplyCSR_OPENCILK(Sparse_CSR *A, Sparse_CSR *B, Sparse_CSR *C){
    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = 0;

    // Initialize row pointers
    C->row_ptr = (int *)malloc((C->rows + 1) * sizeof(int));
    C->row_ptr[0] = 0;

    // Parallelize the loop using cilk_for
    cilk_for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            int dot_product = 0;
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                int colA = A->col_indices[k];
                for (int l = B->row_ptr[colA]; l < B->row_ptr[colA + 1]; l++) {
                    if (B->col_indices[l] == j) {
                        dot_product += A->values[k] * B->values[l];
                        break;
                    }
                }
            }
            // If dot product is non-zero, add it to the result matrix
            if (dot_product != 0) {
                C->nnz++;
                C->values = (int *)realloc(C->values, C->nnz * sizeof(int));
                C->col_indices = (int *)realloc(C->col_indices, C->nnz * sizeof(int));
                C->values[C->nnz - 1] = dot_product;
                C->col_indices[C->nnz - 1] = j;
            }
        }
        C->row_ptr[i + 1] = C->nnz;
    }
}

void multiplySerialCSR(Sparse_CSR *A, Sparse_CSR *B, Sparse_CSR *C ) {

	C->rows = A->rows;
	C->cols = B->cols;
	C->nnz = 0;

	//C->values = (int *)malloc(C->rows * C->cols * sizeof(int)); // Less non-zero elements than rows x cols
	C->values = NULL;
	//C->col_indices = (int *)malloc( C->rows * C->cols * sizeof(int));
	C->col_indices = NULL;
	C->row_ptr = (int *)malloc((C->rows + 1) * sizeof(int));

        // Initialize row pointers
        C->row_ptr[0] = 0;
	for(int i = 0; i < A->rows; ++i){
		for(int j = 0; j < B->cols; ++j){
			int dot_product = 0;
			for(int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++){
				int colA = A->col_indices[k];
				for(int l = B->row_ptr[colA]; l < B->row_ptr[colA + 1]; l++){
					if(B->col_indices[l] == j){
						dot_product += A->values[k] * B->values[l];
						break;
					}
				}
			}
			// If dot product us non zero, add it to the result matrix
			if(dot_product != 0){
				C->nnz++;
				C->values = (int *)realloc(C->values, C->nnz * sizeof(int));
				C->col_indices = (int *)realloc(C->col_indices, C->nnz * sizeof(int));
				C->values[C->nnz - 1] = dot_product;
				C->col_indices[C->nnz - 1] = j;
			}
		}
		C->row_ptr[i + 1] = C->nnz;
	}
}

Sparse_CSR graph_minor(Sparse_CSR A, int *C, int num_nodes, int num_clusters){

	Sparse_CSR R, Rt, Temp, M;
	R = sparse_R(C, num_nodes, num_clusters);
	//display_csr(R, num_clusters, "R");

 	if (pthread_mutex_init(&mutex, NULL) != 0) {
        	fprintf(stderr, "Error: Mutex initialization failed\n");
        	exit(EXIT_FAILURE);
    	}

	int num_threads = 4;

	printf("Number of POSIX threads in use: %d\n", num_threads);
	transposeCSRMatrix(&R, &Rt);
	display_csr(Rt, num_clusters, "Rt");

	// Check if matrices can be multiplied
    	if ((Rt.cols != A.rows) || (A.cols != R.rows)) {
        	fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication: ");
		printf("%d * %d x %d * %d", Rt.cols, A.rows, A.cols, R.rows);
        	exit(EXIT_FAILURE);
    	}
	//Pick what parallisation to choose from
	int parall_case = 1;


	struct timeval start_time;
	gettimeofday(&start_time, NULL);

	switch (parall_case) {
		case 0:
			printf("Trying POSIX\n");
			multiplyCSR_POSIX(&Rt, &A, &Temp, num_threads);
			display_csr(Temp, Temp.rows, "Temp");
			multiplyCSR_POSIX(&Temp, &R, &M, num_threads);
			display_csr(M,M.rows, "M");
			break;
		case 1:
			printf("Trying OpenMP\n");
			multiplyCSR_OPENMP(&Rt, &A, &Temp);
			display_csr(Temp, Temp.rows, "Temp");
			multiplyCSR_OPENMP(&Temp, &R, &M);
			display_csr(M,M.rows, "M");
			break;
		case 2:
			printf("Trying OpenCILK\n");
			//multiplyCSR_OPENCILK(&Rt, &A, &Temp);
			display_csr(Temp, Temp.rows, "Temp");
			//multiplyCSR_OPENCILK(&Temp, &R, &M);
			display_csr(M,M.rows, "M");
			break;
		default:
			printf("Trying Serial\n");
			multiplySerialCSR(&Rt, &A, &Temp);
			display_csr(Temp, Temp.rows, "Temp");
			multiplySerialCSR(&Temp, &R, &M);
			display_csr(M,M.rows, "M");
	}

	struct timeval end_time;
	gettimeofday(&end_time, NULL);

	double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

    	printf("Elapsed time: %f seconds\n", elapsed_time);

	freeCSR(&Rt);
	freeCSR(&A);
	freeCSR(&R);

	return M;
}

void coo_to_csr(int coo_num_rows, int coo_num_cols, int coo_nnz, int *coo_val, int *coo_row_indices, int *coo_col_indices, Sparse_CSR *A){
	int num_rows = coo_num_rows;
	int num_cols = coo_num_cols;
	int nnz = coo_nnz;

	A->rows = num_rows;
	A->cols = num_cols;
	A->nnz = nnz;
	printf("rows %d cols %d nnz %d\n", A->rows, A->cols, A->nnz);

	//Allocate memory for CSR arrays
	A->values = (int *)malloc(nnz * sizeof(int));
	A->col_indices = (int *)malloc(nnz * sizeof(int));
	A->row_ptr = (int *)malloc((num_rows + 1) * sizeof(int));

  	// Initialize row pointers
    	for (int i = 0; i <= num_rows; ++i) A->row_ptr[i] = 0;

    	// Count the number of non-zeros in each row
    	for (int i = 0; i < nnz; ++i) A->row_ptr[coo_row_indices[i]]++;

    	// Cumulative sum to get row pointers
    	int sum = 0;
    	for (int i = 0; i < num_rows; ++i) {
        	int temp = A->row_ptr[i];
        	A->row_ptr[i] = sum;
        	sum += temp;
    	}
    	A->row_ptr[num_rows] = nnz;

    	// Fill CSR arrays
    	for (int i = 0; i < nnz; ++i) {
        	int row = coo_row_indices[i];
        	int dest = A->row_ptr[row];

        	A->values[dest] = coo_val[i];
		printf("CSR Value: %d\n", A->values[dest]);
        	A->col_indices[dest] = coo_col_indices[i];
        	A->row_ptr[row]++;
    }

    // Reset row pointers

    sum = 0;
    for (int i = 0; i < num_rows; ++i) {
        int temp = A->row_ptr[i];
        A->row_ptr[i] = sum;
        sum += temp;
    }

    //display_csr(*A, A->rows ,"Adjacency");
}

Sparse_CSR generateSparseAdjacencyMatrix(int num_nodes, int num_clusters, double edge_probability, unsigned int rng_seed) {
	// Set RNG seed for reproducibility
    	srand(rng_seed);

	int nnz = 0;
    	// Generate random adjacency matrix with given edge probability
    	int **adjacencyMatrix = (int **)malloc(num_nodes * sizeof(int *));
    	for (int i = 0; i < num_nodes; i++) {
		adjacencyMatrix[i] = (int *)malloc(num_nodes * sizeof(int));
		for(int j = 0; j < num_nodes; j++){
            		adjacencyMatrix[i][j] = (rand() / (double)RAND_MAX) < edge_probability;
			if(adjacencyMatrix[i][j] != 0){
				nnz++;
			}
		}
        }


	// Convert the adjacency matrix to CSR format
    	Sparse_CSR A;
	A.nnz = nnz;
	A.rows = num_nodes;
	A.cols = num_nodes;

    	// Allocate memory for CSR matrix
    	A.values = (int *)malloc(A.nnz * sizeof(int));
    	A.col_indices = (int *)malloc(A.nnz * sizeof(int));
    	A.row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));

    	// Populate CSR matrix
    	int index = 0;
	A.row_ptr[0] = 0;
    	for (int i = 0; i < A.rows; i++) {
		A.row_ptr[i] = index;
		for (int j = 0; j < A.cols; j++) {
            		if (adjacencyMatrix[i][j] != 0) {
                		A.values[index] = adjacencyMatrix[i][j];
                		A.col_indices[index] = j;
                		index++;
			}
            	}
    	}
	A.row_ptr[A.rows] = index;

    	display_csr(A, num_nodes, "Adjacency");

	// Free memory for the adjacency matrix
    	for (int i = 0; i < num_nodes; i++) {
        	free(adjacencyMatrix[i]);
    	}

	free(adjacencyMatrix);
    	return A;
}

void mmio_read(int argc, char *argv[], Sparse_CSR *A){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if ((f = fopen(argv[1], "r")) == NULL)
    exit(1);

    if(mm_read_banner(f, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) exit(1);

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));

    for (i=0; i<nz; i++){
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);
    //Convert sparse matrix from a COO format to a CSR
    coo_to_csr(M, N, nz, val, I, J, A);
}


int main(int argc, char** argv) {

    int num_nodes, num_clusters;
    Sparse_CSR A;

    num_clusters = 4;
    printf("Number of clusters randomly generated per graph nodes: %d\n", num_clusters);

    if (argc < 2){
	printf("Given no matrix input an arbitary matrix will be generated\n");
	fprintf( "Proper Matrix Market Usage: %s [martix-market-filename]\n", argv[0]);

        num_nodes = 5;
    	printf("The number of clusters distributed along the nodes %d", num_clusters);

    	double edge_probability = 0.2;
    	unsigned int rng_seed = 42;
	printf("Edge Probability: %f\nRng Seed: %d\n", edge_probability, rng_seed);

    	A = generateSparseAdjacencyMatrix(num_nodes, num_clusters, edge_probability, rng_seed);
    	print_sparse_csr(&A);
    }
    else{
        mmio_read(argc, argv, &A);
	num_nodes = A.rows;
    }

    	// Generate random configuration by mapping each node to a cluster
    	int *C = (int *)malloc(num_nodes * sizeof(int));
    	for (int i = 0; i < num_nodes; i++) C[i] = rand() % num_clusters + 1;
    	printf("Node Cluster Configuration:\n");
    	for (int i = 0; i < num_nodes; i++) printf("%d ", C[i]);
    	printf("\n");
	Sparse_CSR M = graph_minor(A, C, num_nodes, num_clusters);

    // Free dynamically allocated memory
    	free(C);
    	freeCSR(&A);
    	freeCSR(&M);
}

