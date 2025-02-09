@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    # Compute the row index i and column index j for this thread
    i, j = cuda.grid(2)
    
    # Check if (i, j) is within the bounds of the output matrix
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.  # Accumulator for the dot product
        # Iterate over the row of A and the column of B
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]  # Compute the dot product
        C[i, j] = tmp  # Write the result to the output matrix