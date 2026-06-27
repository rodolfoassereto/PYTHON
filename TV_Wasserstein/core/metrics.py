"""Ground-truth distances for evaluating reconstructions (true OT, via POT).

These are *evaluation* metrics computed with python-optimal-transport (`ot`),
not the proxable surrogates used inside the solvers.

  voxelwise_wasserstein : sum of per-voxel W1/W2 between two EAP fields
                          (was objective_functions.py)
  build_cost_matrix     : pairwise ground cost on a displacement grid
  W1                     : a single balanced W1 via ot.emd2
  W1_naif(_components)   : the naif unbalanced distance  alpha1*W1 + alpha2*|mass gap|
                          (was 'W1 naif.py')
"""
import numpy as np
import ot


# ----------------------------------------------------------------------
# Voxelwise Wasserstein between two EAP fields  (P1, P2 of shape shape_x+shape_y)
# ----------------------------------------------------------------------
def normalize_4d(U1, U2):
    """Normalize both arrays so that at each voxel they have equal total mass (the average)."""
    sums_1 = np.sum(U1, axis=(-2, -1), keepdims=True)
    sums_2 = np.sum(U2, axis=(-2, -1), keepdims=True)
    sums_avg = 0.5 * (sums_1 + sums_2)
    mask_1 = (sums_1 == 0)
    mask_2 = (sums_2 == 0)
    sums_1_safe = np.where(mask_1, 1, sums_1)
    sums_2_safe = np.where(mask_2, 1, sums_2)
    U1_normalized = U1 * sums_avg / sums_1_safe
    U2_normalized = U2 * sums_avg / sums_2_safe
    U1_normalized[mask_1[:, :, 0, 0]] = sums_avg[mask_1[:, :, 0, 0]] / np.prod(U1.shape[-2:])
    U2_normalized[mask_2[:, :, 0, 0]] = sums_avg[mask_2[:, :, 0, 0]] / np.prod(U2.shape[-2:])
    return U1_normalized, U2_normalized, sums_avg[:, :, 0, 0]


def voxelwise_wasserstein(P1, P2, shape_x, shape_y, order=1):
    """
    Sum of voxelwise Wasserstein distances between two EAP fields.

    Parameters
    ----------
    P1, P2 : arrays of shape shape_x + shape_y
    order : 1 for W1, 2 for W2

    Returns
    -------
    total : scalar, sum of W_p distances across all voxels
    per_voxel : array of shape shape_x, individual distances
    sums_avg : array of shape shape_x containing the resulting masses
    """
    ndim_y = len(shape_y)
    coordinates = [np.arange(n) for n in shape_y]
    mesh = np.array(np.meshgrid(*coordinates, indexing='ij'))
    points = mesh.reshape(ndim_y, -1).T

    if order == 1:
        M = ot.dist(points, metric='euclidean')
    elif order == 2:
        M = ot.dist(points, metric='sqeuclidean')
    else:
        raise ValueError("order must be 1 or 2")

    P1n, P2n, sums_avg = normalize_4d(P1.clip(min=0), P2.clip(min=0))

    per_voxel = np.zeros(shape_x)
    for idx in np.ndindex(shape_x):
        a = P1n[idx].ravel().astype(np.float64)
        b = P2n[idx].ravel().astype(np.float64)
        b *= a.sum() / b.sum()
        cost = ot.emd2(a, b, M)
        if order == 2:
            cost = np.sqrt(cost)
        per_voxel[idx] = cost

    return np.sum(per_voxel), per_voxel, sums_avg


# ----------------------------------------------------------------------
# Single-distribution distances (W1 and the naif unbalanced distance)
# ----------------------------------------------------------------------
def build_cost_matrix(metric, *coordinates_or_shape):
    """
    metric: 'euclidean' -> L2 distance (for W1), 'sqeuclidean' -> L2^2 (for W2).
    coordinates_or_shape: either 1-D coordinate arrays, or a sequence of ints (a shape),
        in which case each axis is taken as np.arange(n)/(n-1).
    Returns M with shape (size, size), the pairwise ground cost.
    """
    if all(isinstance(coord, int) for coord in coordinates_or_shape):
        coordinates = [np.arange(n) / (n - 1) for n in coordinates_or_shape]
    else:
        coordinates = coordinates_or_shape

    meshgrid_array = np.array(np.meshgrid(*coordinates, indexing='ij'))
    domain_size = np.prod([len(coord) for coord in coordinates])
    meshgrid_flat = np.moveaxis(meshgrid_array.reshape(-1, domain_size), 0, -1)
    return ot.dist(meshgrid_flat, metric=metric)


def W1(f, g, M=None):  # if M is None, defaults to the L2 ground cost (hence W1)
    assert f.shape == g.shape
    if M is None:
        M = build_cost_matrix('euclidean', *f.shape)
    return ot.emd2(f.flatten(), g.flatten(), M)


def W1_naif_components(f, g, alpha1, alpha2, M=None):
    """Naif unbalanced distance, split into (total, ot_part, mass_part)."""
    assert np.all(f >= 0) and np.all(g >= 0)
    f_mass, g_mass = np.sum(f), np.sum(g)
    f0 = f - f_mass / f.size
    g0 = g - g_mass / g.size
    f0_pos, f0_neg = np.maximum(f0, 0), -np.minimum(f0, 0)
    g0_pos, g0_neg = np.maximum(g0, 0), -np.minimum(g0, 0)

    ot_part = alpha1 * W1(f0_pos + g0_neg, f0_neg + g0_pos, M=M)
    mass_part = alpha2 * np.abs(g_mass - f_mass)
    return ot_part + mass_part, ot_part, mass_part


def W1_naif(f, g, alpha1, alpha2, M=None):
    return W1_naif_components(f, g, alpha1, alpha2, M)[0]
