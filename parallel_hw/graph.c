#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

#define MAX_V 100 // Maximum number of vertices in the original graph
#define MAX_C 10  // Maximum number of clusters

typedef struct {
    int start;
    int end;
} Edge;

typedef struct {
    int cluster;
    int vertex;
} Vertex;

Edge edges[MAX_V];      // Edges in the original graph
Vertex vertices[MAX_V]; // Vertices with cluster information
int cluster_count = 0;

int M[MAX_C][MAX_C]; // Adjacency matrix for the graph minor G'
int Omega[MAX_V][MAX_C];

int**  generate_config_matrix(Vertex vertices){
	int** Omega;
	for(int i = 0; i < MAX_V; i++){
		//if(vertices[i].cluster
	}
}

int main() {
    int V, E; // Number of vertices and edges in the original graph G
    int i, j;

    // Initialize G' as an empty graph
    for (i = 1; i <= MAX_C; i++) {
        for (j = 1; j <= MAX_C; j++) {
            M[i][j] = 0;
        }
    }

    // Read input for the original graph G
    printf("Enter the number of vertices (V) in the original graph: ");
    scanf("%d", &V);
    printf("Enter the number of edges (E) in the original graph: ");
    scanf("%d", &E);

    printf("Enter the edges in the format 'start end' (1-based indexing):\n");
    for (i = 0; i < E; i++) {
        scanf("%d %d", &edges[i].start, &edges[i].end);
    }

    // Assign clusters to vertices based on the mapping V -> {0, 1, ..., c}
    printf("Enter the cluster for each vertex (1 to c):\n");
    for (i = 1; i <= V; i++) {
        scanf("%d", &vertices[i].cluster);
        if (vertices[i].cluster > cluster_count) {
            cluster_count = vertices[i].cluster;
        }
    }

    // Construct G' based on the edges in G
    for (i = 0; i < E; i++) {
        int cluster_start = vertices[edges[i].start].cluster;
        int cluster_end = vertices[edges[i].end].cluster;
        if (cluster_start != cluster_end) {
            M[cluster_start][cluster_end] = 1;
            M[cluster_end][cluster_start] = 1; // Graph is undirected
        }
    }

    // Print G'
    printf("Adjacency matrix for the graph minor G':\n");
    for (i = 1; i <= cluster_count; i++) {
        for (j = 1; j <= cluster_count; j++) {
            printf("%d ", M[i][j]);
        }
        printf("\n");
    }

    return 0;
}

