
function M = graphminor(A, C)
    n = length(C);
    L = max(C);
    R = sparse(1:n, C, 1, n, L);
    full(R)
    M = R' * A * R;
    full(M)
end

