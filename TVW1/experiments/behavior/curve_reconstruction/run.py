'''Here I create a curve "b" in the spaces of N x N images. It has the shape of a (T,N,N) array. T stands for time.'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T, N, N = 20, 40, 40    # T times, NxN images
D = 4       # coordinate of the point

# I define a function from {1...T} to images NxN: it's constantly p0 and then jumps to p1
b_clean = np.zeros((T, N, N))   # initialize the data
b = b_clean.copy()
p0, p1 = np.array([D, D]), np.array([N-D, N-D])

def curve_function(t):    
    if t <= T/2: return p0
    else: return p1
    
def curve_diagonal(t):
    return [D+t, D+t]
    
def other_curve_function(t): # implement here other curves
    return None

# I define two vectors with shape (T,2) that serve as noise on the position given by curve_function. The first models a gaussian noise in the position, the second models an oscillation
noise_trajectory_gaussian = np.array( np.random.normal(loc=0.0, scale=20 , size=(T, 2)), dtype=int).clip(min=-(D-1), max=D-1)
noise_trajectory_oscillating = 3 * np.array(int(T/2)*[[1, -1], [-1, 1]])
noise_trajectory_zero = np.zeros((T,2), dtype=int)

# I choose one of the noises and the curve function
noise_trajectory = noise_trajectory_oscillating
curve = curve_diagonal
noise = 0.2 * np.random.rand(T,N,N)

# This creates a 2-D matrix with a ball of ones, centered at "center"
def create_ball_matrix(shape, center, radius, output=int):
    rows, cols = shape
    i, j = center
    y, x = np.ogrid[:rows, :cols] # Generate row and column indices
    distance_sq = (x - j)**2 + (y - i)**2 # Compute the squared distance from the center
    mask = distance_sq < radius**2 # Create the mask: True where distance <= radius^2
    if output in {bool}: return mask
    matrix = np.zeros(shape, dtype=int) # Initialize the matrix with zeros
    matrix[mask] = 1 # Set elements within the radius to 1
    return matrix

# this defines a curve of balls. It overwrites the previous b.
for t in range(T):
    center_clean_t = curve(t)
    center_noisy_t = center_clean_t + noise_trajectory[t]
    radius = 3.2
    b_clean[t] = create_ball_matrix((N,N), center_clean_t, radius)
    b[t] = create_ball_matrix((N,N), center_noisy_t, radius)
    b[t] = b[t] + 0 * noise[t]


print('Sum of elements in b: ', np.sum(b))