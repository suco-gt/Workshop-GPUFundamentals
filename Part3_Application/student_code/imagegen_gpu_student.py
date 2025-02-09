import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import time
from save_image import save_image

# Define the CUDA kernel
@cuda.jit
def generate_one_tile(output, color_map):
    """
    CUDA kernel to generate image tiles representing a complex function.

    Parameters:
    - output: A 3D device array to store the computed pixel values (RGBA format).
    - color_map: A device array containing RGBA color mappings.

    Each thread processes a tile of 16x16 pixels.
    """
    # Calculate the base coordinates for the tile using grid dimensions
    tile_x, tile_y = cuda.grid(2)

    # Loop over each pixel in the 16x16 tile assigned to this thread
    for xi in range(16):
        for yi in range(16):
            # Compute the pixel coordinates in the full image
            # TODO: Calculate the x and y pixel indices. Replace None with your expressions
            # (hint: you need to use tile_x, tile_y, xi, yi, and 16)
            ####### STUDENT CODE HERE ###############################
            x = None
            y = None
            #########################################################

            # Ensure the pixel coordinates are within image bounds
            if x >= output.shape[0] or y >= output.shape[1]:
                continue

            # Map the pixel coordinates to the complex plane
            z = complex(float(x) / output.shape[0] * 2.0 - 1.0, float(y) / output.shape[1] * 2.0 - 1.0)

            # Compute the complex function value at the pixel coordinate
            f_z = z ** 1.5 ** (complex(-1, -1)) if z != 0 else 0

            # Compute the magnitude and map it to a color index
            magnitude = abs(f_z)
            color_index = int(magnitude * 10 % len(color_map))

            # Store the corresponding RGBA color in the output array
            output[x, y, 0] = color_map[color_index, 0]  # Red channel
            output[x, y, 1] = color_map[color_index, 1]  # Green channel
            output[x, y, 2] = color_map[color_index, 2]  # Blue channel
            output[x, y, 3] = color_map[color_index, 3]  # Alpha channel

# Start measuring execution time
start_time = time.time()

# Define color map for visualization (RGBA format)
colors = np.array([
   [236.0/255, 244.0/255, 214.0/255, 1.0],  # Soft Green
   [154.0/255, 208.0/255, 194.0/255, 1.0],  # Aquamarine
   [45.0/255, 149.0/255, 150.0/255, 1.0],   # Teal
   [38.0/255, 80.0/255, 115.0/255, 1.0],    # Deep Sky Blue
   [34.0/255, 9.0/255, 44.0/255, 1.0],      # Dark Purple
   [135.0/255, 35.0/255, 65.0/255, 1.0],    # Crimson
   [190.0/255, 49.0/255, 68.0/255, 1.0],    # Raspberry
   [240.0/255, 89.0/255, 65.0/255, 1.0],    # Coral
   [7.0/255, 102.0/255, 173.0/255, 1.0],    # Cobalt Blue
   [41.0/255, 173.0/255, 178.0/255, 1.0]    # Turquoise
], dtype=np.float32)

# TODO: Transfer the color map to the GPU memory (one line). Replace None with your expression.
# (Hint: use cuda.to_device() method)
####### STUDENT CODE HERE #######################
colors_gpu = None
#################################################

# Allocate GPU memory for the output image
shape = (4096, 4096, 4)  # 4096x4096 image with RGBA channels
image_gpu = cuda.device_array(shape, dtype=np.float32)

# Define grid and block dimensions
blocks_per_grid = (32, 32)  # 32x32 tiles
threads_per_block = (8, 8)  # Each block contains 8x8 threads

# Launch the CUDA kernel
generate_one_tile[blocks_per_grid, threads_per_block](image_gpu, colors_gpu)

# Copy the result of our computations back to the CPU
cpu_array = image_gpu.copy_to_host()

# Measure and display execution time
end_time = time.time() - start_time
print(f"Execution time: {end_time} seconds")

# Save the generated image as a PNG file
save_image(cpu_array, "output.png")
print("Image saved")