import numpy as np

# read the text file
with open('Output.txt') as f:
    lines = f.readlines()

# create a 2D NumPy array to store the RGB values
rgb_values = np.zeros((1000000, 3), dtype=float)

# fill the array with the RGB values from the text file
for i, line in enumerate(lines):
    rgb_values[i] = [float(x) for x in line.split()]

# reshape the NumPy array to a 3D array
rgb_values = rgb_values * 255
rgb_values_3d = rgb_values.reshape((1000, 1000, 3))

# create a new Pillow image from the RGB values
from PIL import Image
img = Image.fromarray(rgb_values_3d.astype('uint8'), 'RGB')

# save the image as a PNG file
img.save('Paper.png')