import numpy as np
import ot

def normalize_4d(U1, U2):
    """Normalizes both arrays so that at each voxel they have equal total mass (the average)."""
    sums_1 = np.sum(U1, axis=(-2, -1), keepdims=True)
    sums_2 = np.sum(U2, axis=(-2, -1), keepdims=True)
    sums_avg = 0.5 * (sums_1 + sums_2)
    mask_1 = (sums_1 == 0)
    mask_2 = (sums_2 == 0)
    sums_1_safe = np.where(mask_1, 1, sums_1)
    sums_2_safe = np.where(mask_2, 1, sums_2)
    U1_normalized = U1 * sums_avg / sums_1_safe
    U2_normalized = U2 * sums_avg / sums_2_safe
    U1_normalized[mask_1[:,:,0,0]] = sums_avg[mask_1[:,:,0,0]] / np.prod(U1.shape[-2:])
    U2_normalized[mask_2[:,:,0,0]] = sums_avg[mask_2[:,:,0,0]] / np.prod(U2.shape[-2:])
    return U1_normalized, U2_normalized, sums_avg[:,:,0,0]

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

    P1n, P2n, sums_avg = normalize_4d(P1.clip(min=0), P2.clip(min=0)) # sums_avg[i,j] contains the mass of P1n[i,j] and P2n[i,j]

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