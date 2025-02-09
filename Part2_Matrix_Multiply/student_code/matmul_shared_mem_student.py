from numba import cuda, float32

# Controls threads per block and shared memory usage
TPB = 16  # Threads Per Block (assumed to be 16x16 block size)

@cuda.jit
def fast_matmul(A, B, C):
    # Define shared memory arrays for A and B tiles
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # Compute the global thread coordinates
    x, y = cuda.grid(2)

    # Compute thread indices within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Check if (x, y) is within the bounds of the output matrix
    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0.  # Accumulator for the dot product

    # Loop over tiles of A and B
    for i in range(cuda.gridDim.x):
        # TODO: Load tiles from global memory into shared memory (2 lines)
        # Hint: sA[tx, ty] should load A[x, ty + i * TPB]
        # Hint: sB[tx, ty] should load B[tx + i * TPB, y]
        ########### STUDENT CODE HERE #########################

        #######################################################

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # TODO: Compute partial products on the shared memory tiles
        # Hint: Iterate over the range TPB and accumulate the product sA[tx, j] * sB[j, ty]
        # (similar to previous matmul)
        ########### STUDENT CODE HERE #########################

        #######################################################

        # Wait until all threads finish computing
        cuda.syncthreads()

    # Write the computed value to the output matrix
    C[x, y] = tmp