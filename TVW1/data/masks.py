"""Undersampling masks for the (k,q) forward operator.

  undersampling_mask_xy / _y : the canonical q-space variable-density mask used
      by the diffusion experiments (broadcast across all spatial voxels).
  undersampling_mask(_small), radial_mask, ... : the older mask family.

Side-effect-free: the stray plt.imshow/plt.show/print calls that used to sit
inside undersampling_mask have been removed.
"""
import numpy as np
from skimage.draw import line

from gaussians import gaussian_function_as_matrix, gaussian_density_factory, function_as_array


# =====================================================================
# Canonical q-space variable-density mask (diffusion experiments)
# =====================================================================
def undersampling_mask_y(shape_y, retained_ratio, concentration_coeff):
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


# =====================================================================
# Older mask family
# =====================================================================
def radial_mask(shape, scheme="nomask", **kwargs):  # only works for 2-D shapes
    """k-space mask: 'nomask', 'radial' (num_radials, full_radius) or 'cartesian' (gap, orientation)."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    if scheme == "nomask":
        return mask

    elif scheme in {"radial"}:
        height, width = height + 2, width + 2  # radii didn't reach the border; truncate later
        mask = np.zeros((height, width), dtype=bool)
        num_radials = kwargs.get('num_radials', 30)
        full_radius = kwargs.get('full_radius', True)
        center_y, center_x = height // 2, width // 2
        radius = min(center_y, center_x)
        if not full_radius:
            radius = radius // 2
        for i in range(num_radials):
            angle = (2 * np.pi * i) / num_radials
            x_end = center_x + int(radius * np.cos(angle))
            y_end = center_y + int(radius * np.sin(angle))
            rr, cc = line(center_y, center_x, y_end, x_end)
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)
            mask[rr, cc] = True
        return mask[1:-1, 1:-1]

    elif scheme in {"cartesian"}:
        gap = kwargs.get('gap', 2)
        orientation = kwargs.get('orientation', 'horizontal').lower()
        if orientation not in ['horizontal', 'vertical']:
            raise ValueError("orientation must be 'horizontal' or 'vertical'.")
        if orientation == 'horizontal':
            for y in range(0, height, gap):
                mask[y, :] = True
        else:
            for x in range(0, width, gap):
                mask[:, x] = True
        return mask

    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Supported: 'nomask', 'radial', 'cartesian'.")


def symmetrize_2d_mask(msk):  # !! ONLY WORKS FOR 2-D
    """Reflect the lower half of the matrix onto the upper half (central row keeps its right part)."""
    index_bool = msk.copy()
    v = np.zeros(np.size(index_bool), dtype=bool)
    v[0:int(len(v) / 2)] = True
    mskmsk = v.reshape(index_bool.shape)
    index_bool[mskmsk] = np.flip(index_bool)[mskmsk]
    return index_bool


def avoid_angles_mask(shape):
    """Disc mask: True inside the inscribed circle, dropping the far corners."""
    s = np.array(shape)
    center = (s - 1) / 2.0
    edge_dists = np.minimum(center, (s - 1) - center)
    min_edge_dist = edge_dists.min()
    radius = min_edge_dist + 0.5
    coords = np.ogrid[[slice(0, x) for x in s]]
    dist_sq = sum((g - c) ** 2 for g, c in zip(coords, center))
    return dist_sq <= radius ** 2


def undersampling_mask_small(shape, ratio=0.5, scheme='gaussian', cov=None, sym=False, noangles=True, **kwargs):
    """A `shape` mask with ~ratio True entries, sampled uniformly / gaussian / radial / cartesian."""
    dim = len(shape)
    index_bool = np.zeros(shape, dtype=bool)
    N = int(np.prod(shape) * ratio)
    if scheme == 'nomask':
        index_bool = np.ones(shape, dtype=bool)
    if scheme == 'gaussian':
        if cov is None:
            cov = np.eye(dim)
        coordinates = [np.linspace(-3, 3, n) for n in shape]
        P = gaussian_function_as_matrix(*coordinates, cov=cov)
        if noangles:
            P[np.invert(avoid_angles_mask(shape))] = 0
        P = np.ravel(P) / np.sum(P)
        index_temp = np.unravel_index(np.random.choice(np.prod(shape), N, replace=False, p=P), np.shape(index_bool))
        index_bool[index_temp] = True
    if scheme == 'random':
        index_temp = np.unravel_index(np.random.choice(np.prod(shape), N, replace=False), np.shape(index_bool))
        index_bool[index_temp] = True
    if scheme in {'radial', 'cartesian'}:
        return radial_mask(shape, scheme=scheme)
    if sym:  # !! only works for 2d
        index_bool = symmetrize_2d_mask(index_bool)
    return index_bool


def undersampling_mask(shape, scheme_k='cartesian', scheme_q='gaussian', ratio_q=0.75, cov=None, **kwargs):
    """Composite (k,q) mask. Returns the *inverted* boolean mask (True = masked out)."""
    shape_k, shape_q = shape[:2], shape[2:]
    mask_k = undersampling_mask_small(shape_k, scheme=scheme_k, **kwargs)
    mask_q = np.fft.fftshift(undersampling_mask_small(shape_q, scheme=scheme_q, ratio=ratio_q, cov=cov))
    if len(shape_q) == 2:
        mask_q = mask_q.reshape(shape_q + (1,))
    shape_q = mask_q.shape
    n0, n1, n2 = shape_q
    index_bool = np.zeros(shape_k + shape_q, dtype=bool)
    for j0 in range(n0):
        for j1 in range(n1):
            for j2 in range(n2):
                if mask_q[j0, j1, j2]:
                    index_bool[:, :, j0, j1, j2] = mask_k
    return ~index_bool
