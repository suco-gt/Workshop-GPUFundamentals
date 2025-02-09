from numba import cuda
import numpy as np

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    # Compute the row index i and column index j for this thread
    i, j = cuda.grid(2)

    # Ensure (i, j) is within the bounds of the output matrix
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.  # Accumulator for the dot product
        
        # TODO: Iterate over the row of A and the column of B
        # Compute the sum for the dot product and store the result in tmp
        # (Roughly 2 lines)

        ####### Student Code Here ############################

        #######################################################
        
        C[i, j] = tmp  # Write the result to the output matrix