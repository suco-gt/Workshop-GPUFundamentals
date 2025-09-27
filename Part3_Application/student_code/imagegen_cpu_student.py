import numpy as np
from numba import jit
import time
from save_image import save_image

def generate_image_cpu(output, color_map):
    """
    Populates the output array with an image representing the graph of a complex function.
    
    Parameters:
    - output: A NumPy array that will store the RGBA pixel values for the image.
    - color_map: A color mapping array to represent different magnitudes of the complex function.
    
    This method iterates over every pixel and assigns a color based on the computed complex function value.
    """
    # Iterate over each pixel position on the 4096x4096 grid
    for x in range(4096):
        for y in range(4096):
            # Map pixel coordinates (x, y) to a point in the complex plane
            z = complex(float(x) / output.shape[0] * 2.0 - 1.0, float(y) / output.shape[1] * 2.0 - 1.0)
            
            # Compute the value of the complex function at the given coordinate
            # Use a mathematical operation on the complex number
            f_z = z ** 1.5 ** (complex(-1, -1)) if z != 0 else 0
            
            # Calculate the magnitude of the complex function value
            magnitude = abs(f_z)
            
            # Determine the color index based on the magnitude, wrapping around the color map
            color_index = int(magnitude * 10 % len(color_map))
            
            # Assign the corresponding color to the output pixel at (x, y)
            output[x, y] = color_map[color_index]

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

# Initialize the output image array with zeros (RGBA format)
shape = (4096, 4096, 4)
image = np.zeros(shape, dtype=np.float32)

# Generate the image by computing colors for each pixel based on the complex function
generate_image_cpu(image, colors)

# Measure and display execution time
end_time = time.time() - start_time
print(f"Execution time: {end_time} seconds")

# Save the generated image as a PNG file
save_image(image, "output.png")
print(f"Image Saved")
