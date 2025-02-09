import numpy as np
from numba import cuda
import time

# Import student solutions
from matmul_student import matmul
from matmul_shared_mem_student import fast_matmul

# Matrix size (assuming square matrices)
N = 512

# Initialize input matrices
A = np.random.random((N, N)).astype(np.float32)
B = np.random.random((N, N)).astype(np.float32)
C_basic = np.zeros((N, N), dtype=np.float32)
C_optimized = np.zeros((N, N), dtype=np.float32)

# Define threads per block and block grid size
TPB = 16
threads_per_block = (TPB, TPB)
blocks_per_grid = (N // TPB, N // TPB)

# =====================
# Basic Matrix Multiply
# =====================
# Transfer arrays to GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C_basic = cuda.to_device(C_basic)

# Measure execution time
start_time = time.time()
matmul[blocks_per_grid, threads_per_block](d_A, d_B, d_C_basic)
cuda.synchronize()  # Ensure all threads complete before measuring time
basic_time = time.time() - start_time

# Copy result back to host
C_basic = d_C_basic.copy_to_host()

# ==============================
# Optimized Matrix Multiply
# ==============================
# Transfer arrays to GPU
d_C_optimized = cuda.to_device(C_optimized)

# Measure execution time
start_time = time.time()
fast_matmul[blocks_per_grid, threads_per_block](d_A, d_B, d_C_optimized)
cuda.synchronize()
optimized_time = time.time() - start_time

# Copy result back to host
C_optimized = d_C_optimized.copy_to_host()

# =====================
# Compare Results
# =====================
if np.allclose(C_basic, C_optimized, atol=1e-5):
    print("Both implementations produced correct results.")
else:
    print("Results differ between basic and optimized implementations.")

print(f"Basic matrix multiplication time: {basic_time:.6f} seconds")
print(f"Optimized matrix multiplication time: {optimized_time:.6f} seconds")

speedup = basic_time / optimized_time if optimized_time > 0 else float('inf')
print(f"Speedup: {speedup:.2f}x")