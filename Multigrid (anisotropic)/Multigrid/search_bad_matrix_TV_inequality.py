import numpy as np
import itertools
import TV
from building_data import halve
import time

# Define your list of values
v = [0, 0.25, 0.5, 0.75, 1]

# Define your mask
mask = np.array([[False, False, False, False, False, False],
                 [False, False,  True,  True, False, False],
                 [False,  True,  True,  True,  True, False],
                 [False,  True,  True,  True,  True, False],
                 [False, False,  True,  True, False, False],
                 [False, False, False, False, False, False]
                 ])

# Find the positions where the mask is True
positions = np.argwhere(mask)
num_positions = positions.shape[0]

# Generate all possible combinations of values for the True positions in the mask
value_combinations = itertools.product(v, repeat=num_positions)

def halve_once(im):
    

def halve_once(im):
    m = im.shape[0]
    m = int(m/2)
    im_halved = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            im_halved[i,j] = np.mean([im[2*i,2*j], im[2*i+1,2*j], im[2*i,2*j+1], im[2*i+1,2*j+1]])
    return im_halved

def average(im): # averages a matrix
    x = np.zeros(np.shape(im))
    h = halve(im,times)
    for i in range(2**times):
        for j in range(2**times):
            x[i::2**times,j::2**times] = h
    return x

# Loop through each combination and create the corresponding matrix
t_old = time.time()
for values in value_combinations:
    t = time.time()
    if t > (t_old + 30):
        print(values)
        t_old = t
    # Start with a zero matrix
    matrix = np.zeros_like(mask, dtype=float)
    # Assign the values to the positions where the mask is True
    for pos, val in zip(positions, values):
        matrix[tuple(pos)] = val
    if TV.tv_2d_iso(matrix) < 2 * TV.tv_2d_iso(halve(matrix)):
        break
print(matrix)
print(f"TV(matrix) = {TV.tv_2d_iso(matrix)}, 2*TV(halve(matrix)) = {2 * TV.tv_2d_iso(halve(matrix))}")
    



    
