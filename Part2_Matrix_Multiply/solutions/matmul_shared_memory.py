from numba import cuda, float32

# Controls threads per block and shared memory usage.
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

    # Compute the number of blocks per grid (only for square matrices in this example)
    bpg = cuda.gridDim.x

    # Check if (x, y) is within the bounds of the output matrix
    if x >= C.shape[0] or y >= C.shape[1]:
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into partial products of TPB-long vectors.
    tmp = 0.

    for i in range(bpg):
        # Preload data into shared memory
        ### SOLUTION ######################
        sA[tx, ty] = A[x, ty + i * TPB]  
        sB[tx, ty] = B[tx + i * TPB, y] 
        ###################################

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Compute partial products on the shared memory
        ### SOLUTION ########################
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        #####################################

        # Wait until all threads finish computing
        cuda.syncthreads()

    # Write the computed value to the output matrix
    C[x, y] = tmp