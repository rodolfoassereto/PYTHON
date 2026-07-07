"""Undersampling masks for the (k,q) forward operator.
  undersampling_mask_xy / _y : the canonical q-space variable-density mask used by the diffusion experiments (broadcast across all spatial voxels).
"""
import numpy as np
from skimage.draw import line
from .gaussians import gaussian_density_factory, function_as_array


# =====================================================================
# Canonical q-space variable-density mask (diffusion experiments)
# =====================================================================
def undersampling_mask_y(shape_y, retained_ratio, concentration_coeff):
    """Boolean (shape_y) mask in centered (fftshift) layout: ~retained_ratio of
    points, Gaussian-density sampled, Hermitian-symmetric (mask(k) = mask(-k)
    after ifftshift) and always containing the DC bin.

    Symmetry matters for the full-fft forward model: K*K is then a 0/1 diagonal
    in Fourier, so the fidelity resolvent is an exact pointwise division.
    Sampling is by {k, -k} orbits, so the retained count stays within 1 of
    int(retained_ratio * size_y).
    """
    assert 0 < concentration_coeff < 1
    ndim_y = len(shape_y)
    size_y = np.prod(shape_y)
    N_ones_y = int(retained_ratio * size_y)
    f = gaussian_density_factory(ndim_y * [0], np.eye(ndim_y))
    domain = 4 * concentration_coeff * np.pi / 2
    coordinates = [np.linspace(-domain, domain, n) for n in shape_y]
    probabilities_flat = function_as_array(coordinates, f).flatten()

    # Hermitian mirror in centered layout: per axis, i -> (2*(n//2) - i) % n
    grids = np.indices(shape_y).reshape(ndim_y, -1)
    mirror = np.ravel_multi_index(
        [(2 * (n // 2) - g) % n for n, g in zip(shape_y, grids)], shape_y)

    # one representative per two-element orbit {k, -k}; self-paired bins
    # (DC, Nyquist combinations) are excluded from the draw
    all_flat = np.arange(size_y)
    representatives = all_flat[all_flat < mirror]
    orbit_probabilities = probabilities_flat[representatives] + probabilities_flat[mirror[representatives]]
    orbit_probabilities = orbit_probabilities / np.sum(orbit_probabilities)

    n_pairs = int(round((N_ones_y - 1) / 2))
    chosen = np.random.choice(representatives, n_pairs, replace=False, p=orbit_probabilities)

    mask_flat = np.zeros(size_y, dtype=bool)
    mask_flat[np.ravel_multi_index(tuple(n // 2 for n in shape_y), shape_y)] = True  # DC always sampled
    mask_flat[chosen] = True
    mask_flat[mirror[chosen]] = True
    return mask_flat.reshape(shape_y)


def undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff): # needs to be ifftshifted before being applied
    """Boolean (shape_x+shape_y) mask, the q-space pattern shared across all voxels."""
    mask_xy = np.zeros(shape_x + shape_y, dtype=bool)
    mask_y = undersampling_mask_y(shape_y, retained_ratio, concentration_coeff)
    mask_xy[:] = mask_y  # NumPy broadcasting
    return mask_xy

'''
To obtain the frequency-layout mask used by the full-fft forward operator:

mask_fft = np.fft.ifftshift(undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff))
'''