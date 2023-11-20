function [A, C] = generate_sparse_adjacency_matrix(num_nodes, num_clusters, edge_probability, rng_seed)
    % Set RNG seed for reproducibility
    rng(rng_seed);

   
    % Generate a random adjacency matrix with given edge probability
    adjacency_matrix = rand(num_nodes) < edge_probability

    % Generate random configuration by mapping each node to a cluster
    C = randi([1, num_clusters], [1, num_nodes]);

    % Convert to sparse matrix
    A = sparse(adjacency_matrix);

    % Display the sparse adjacency matrix and node configuration
    disp('Sparse Adjacency Matrix:');
    disp(full(A));
    disp(C);
end
