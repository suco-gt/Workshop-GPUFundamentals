import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord


# Suppress performance warnings from Numba to avoid clutter during student execution
warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

# Define a simple mapping function to be applied to an array
# This function adds 10 to each element of the input array
def map_spec(a):
    return a + 10

# Define the GPU kernel structure for the map operation
# `cuda` is the Numba CUDA module passed to this function
def map_test(cuda):
    def call(out, a) -> None:
        # Get the current thread's index within the block
        local_i = cuda.threadIdx.x

        # Write code here to map the input value to the output
        # Hint: Use local_i to access the correct index in the input and output arrays.
        ####### STUDENT CODE HERE (roughly one line) #########

        ######################################################

    return call


# Define the problem size (number of elements in the input array)
SIZE = 4
out = np.zeros((SIZE,)) # Initialize output array
a = np.arange(SIZE) # Initialize input array

# Define the CUDA problem configuration using a custom CudaProblem class
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)

problem.check()