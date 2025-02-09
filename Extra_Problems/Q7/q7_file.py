import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

# PROBLEM: Implement the same kernel (as the previous question), but in 2d.  
# You have fewer threads per block than the size of ``a`` in both directions.

def map_spec(a):
    return a + 10

def map_block2D_test(cuda):
    def call(out, a, size):
        # FILL ME IN (roughly 4 lines)

    return call

SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))

problem = CudaProblem(
    "Blocks 2D",
    map_block2D_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    blockspergrid=Coord(2, 2),
    spec=map_spec,
)

problem.check()