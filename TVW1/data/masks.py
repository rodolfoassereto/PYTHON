"""Undersampling masks for the (k,q) forward operator.
  undersampling_mask_xy / _y : the canonical q-space variable-density mask used by the diffusion experiments (broadcast across all spatial voxels).
"""
import numpy as np
from skimage.draw import line
from gaussians import gaussian_density_factory, function_as_array


# =====================================================================
# Canonical q-space variable-density mask (diffusion experiments)
# =====================================================================
def undersampling_mask_y(shape_y, retained_ratio, concentration_coeff): # needs to be ifftshifted before being applied
    """Boolean (shape_y) mask: retained_ratio of points, Gaussian-density sampled."""
    assert 0 < concentration_coeff < 1
    ndim_y = len(shape_y)
    size_y = np.prod(shape_y)
    N_ones_y = int(retained_ratio * size_y)
    f = gaussian_density_factory(ndim_y * [0], np.eye(ndim_y))
    domain = 4 * concentration_coeff * np.pi / 2
    coordinates = [np.linspace(-domain, domain, n) for n in shape_y]
    probabilities_flat = function_as_array(coordinates, f).flatten()
    probabilities_flat = probabilities_flat / np.sum(probabilities_flat)
    indices_flat = np.random.choice(size_y, N_ones_y, replace=False, p=probabilities_flat)
    indices = np.unravel_index(indices_flat, shape_y)
    mask_y = np.zeros(shape_y, dtype=bool)
    mask_y[indices] = True
    return mask_y


def undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff):
    """Boolean (shape_x+shape_y) mask, the q-space pattern shared across all voxels."""
    mask_xy = np.zeros(shape_x + shape_y, dtype=bool)
    mask_y = undersampling_mask_y(shape_y, retained_ratio, concentration_coeff)
    mask_xy[:] = mask_y  # NumPy broadcasting
    return mask_xy
