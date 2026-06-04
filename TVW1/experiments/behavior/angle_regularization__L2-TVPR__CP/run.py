import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from gaussians import gaussian_function_as_matrix, rotate_2d_array
from plottings import plot_u
from l2_tvpr import L2_TVPR_Dirichlet
np.random.seed(0)

''' I build and plot gaussians at 4 different angles'''
''' Then I build an array thetas of random angles to tweak the gaussians. I tweak the gaussians.
I print both the angles and the tweaked gaussians'''

M, N = 10, 30
shape_x, shape_y = (M,M), (N,N)
shape_xy = shape_x + shape_y
ndim_x, ndim_y = len(shape_x), len(shape_y)
gaussians = np.zeros(shape_xy)
gaussians_noisy = gaussians.copy()
thetas = np.zeros((M,M))
thetas[:5,:5], thetas[:5,5:], thetas[5:,5:] = np.pi/4, -np.pi/4, np.pi/2
thetas_noisy = np.zeros((M,M))
cov = np.array([[8,0],[0,1]])
coord = np.linspace(-5,5,N)

plt.imshow(thetas, cmap='gray')
plt.show()

for i in range(M):
    for j in range(M):
        cov_temp = rotate_2d_array(cov, thetas[i,j])
        gaussians[i,j] = gaussian_function_as_matrix(coord,coord, cov=cov_temp)
        thetas_noisy[i,j] = thetas[i,j] + (np.random.rand() - 0.5) * np.pi/4
        cov_temp_noisy = rotate_2d_array(cov, thetas_noisy[i,j])
        gaussians_noisy[i,j] = gaussian_function_as_matrix(coord,coord, cov=cov_temp_noisy)

plot_u(gaussians, ndim_y)
plot_u(gaussians_noisy, ndim_y)
plt.imshow(thetas_noisy, cmap='gray')
plt.show()

'''I reconstruct and plot the gaussians using L2_TVPR'''

gaussians_star, _ = L2_TVPR_Dirichlet(gaussians_noisy, 0.0015, 0.06, forward=False, maxit=1000) # a suitable value for eps is 0.01
plot_u(gaussians_star, ndim_y)

def compute_gaussian_covariance(grid, Z): # given a meshgrid of shape (M,M,2) (resulting from my_meshgrid), it returns the covariance matrix of a 2-d distribution Z
    """
    Given a 2D Gaussian function sampled on a grid, compute its covariance matrix.
    
    Parameters:
    - X, Y: 2D arrays representing the coordinates of the Gaussian grid.
    - Z: 2D array representing the Gaussian function values.

    Returns:
    - Sigma: 2x2 covariance matrix.
    """
    
    X, Y = grid[:,:,0], grid[:,:,1]
    
    # Normalize Z to ensure it sums to 1 (acts as a probability density function)
    Z_sum = np.sum(Z)
    if Z_sum == 0:
        raise ValueError("Z sum is zero, cannot normalize.")

    Z_norm = Z / Z_sum

    # Compute the mean (mu_x, mu_y)
    mu_x = np.sum(X * Z_norm)
    mu_y = np.sum(Y * Z_norm)

    # Compute variances and covariance
    sigma_xx = np.sum(Z_norm * (X - mu_x) ** 2)
    sigma_yy = np.sum(Z_norm * (Y - mu_y) ** 2)
    sigma_xy = np.sum(Z_norm * (X - mu_x) * (Y - mu_y))

    # Construct covariance matrix
    Sigma = np.array([[sigma_xx, sigma_xy],
                      [sigma_xy, sigma_yy]])

    return Sigma

covariances_retrieved = np.zeros((M,M,2,2))
from gaussians import my_meshgrid
grid = my_meshgrid(coord, coord)
for i in range(M):
    for j in range(M):
        covariances_retrieved[i,j] = compute_gaussian_covariance(grid, gaussians_star[i,j])

# Example usage:
# Assuming X, Y, and Z are predefined NumPy arrays representing a sampled 2D Gaussian.
# Sigma = compute_gaussian_covariance(X, Y, Z)
# print(Sigma)  # This should return the estimated covariance matrix.

