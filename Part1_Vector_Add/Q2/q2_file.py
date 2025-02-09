import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

# Suppress Numba performance warnings to keep the output clean for students
warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

# Specification function for the expected operation: element-wise addition
def zip_spec(a, b):
    return a + b

# Define the CUDA kernel setup
# The `cuda` module provides access to GPU-related functionality
def zip_test(cuda):
    def call(out, a, b) -> None:
        # Get the current thread's index within the block
        local_i = cuda.threadIdx.x

        # Write code here to compute element-wise addition
        # Hint: Use local_i to read from `a` and `b` and store the result in `out`.
        ####### STUDENT CODE HERE (roughly one line) #########

        ######################################################

    return call


# Define the problem size (number of elements in each array)
SIZE = 4
# Initialize empty output array 
out = np.zeros((SIZE,))
# Initialize input arrays 
a = np.arange(SIZE)
b = np.arange(SIZE)
# Define the CUDA problem configuration
problem = CudaProblem(
    "Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)

problem.check()