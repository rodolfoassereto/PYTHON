import numpy as np
import ot


def TVW1(u, shape_x, shape_y, ground_metric=None, reg=None, iso=False):
    """
    Total Variation Wasserstein-1 seminorm.

    Sums the W1 distances between every pair of spatially adjacent slices of u
    along each dimension of shape_x.

    Parameters
    ----------
    u : array-like, shape shape_x + shape_y
        Each "pixel" u[x] is interpreted as a (non-negative) measure over the
        shape_y grid.
    shape_x : tuple of int
        Spatial (grid) dimensions.
    shape_y : tuple of int
        Support (distribution) dimensions.
    ground_metric : (prod(shape_y), prod(shape_y)) array, optional
        Ground cost matrix. Defaults to pairwise Euclidean distances on the
        shape_y integer grid.
    reg : float, optional
        Entropy regularization. None → exact OT (ot.emd2); float → Sinkhorn
        (ot.sinkhorn2), which is differentiable.
    iso : bool
        Isotropic variant (not yet implemented).

    Returns
    -------
    float
    """
    if iso:
        raise NotImplementedError("Isotropic TVW1 is not implemented yet.")

    u = np.asarray(u, dtype=float)
    assert u.shape == tuple(shape_x) + tuple(shape_y), (
        f"u.shape {u.shape} != shape_x + shape_y {tuple(shape_x) + tuple(shape_y)}"
    )

    n_x = len(shape_x)
    ny = int(np.prod(shape_y))

    if ground_metric is None:
        # Integer grid coordinates for shape_y, then pairwise Euclidean cost
        coords = np.array(list(np.ndindex(*shape_y)), dtype=float)
        ground_metric = ot.dist(coords, coords, metric='euclidean')

    M = np.ascontiguousarray(ground_metric, dtype=np.float64)
    total = 0.0
    sl_base = [slice(None)] * u.ndim

    for d in range(n_x):
        sl_a = sl_base.copy()
        sl_b = sl_base.copy()
        sl_a[d] = slice(None, -1)
        sl_b[d] = slice(1, None)

        a_batch = u[tuple(sl_a)]  # shape: (modified shape_x) + shape_y
        b_batch = u[tuple(sl_b)]

        n_pairs = int(np.prod(a_batch.shape[:n_x]))
        a_flat = a_batch.reshape(n_pairs, ny)  # (n_pairs, ny)
        b_flat = b_batch.reshape(n_pairs, ny)

        for i in range(n_pairs):
            ai = np.ascontiguousarray(a_flat[i], dtype=np.float64)
            bi = np.ascontiguousarray(b_flat[i], dtype=np.float64)
            # Kantorovich-Rubinstein: W1(a,b) = emd2((a-b)+, (a-b)-, M)
            # Works for probability measures AND zero-sum signed measures.
            # Requires ai.sum() == bi.sum() (balanced); raises if not.
            diff = ai - bi
            pos = np.maximum(diff, 0.0)
            neg = np.maximum(-diff, 0.0)
            mass = pos.sum()
            if mass < 1e-15:
                continue  # ai ≈ bi — W1 = 0
            if reg is None:
                total += float(ot.emd2(pos, neg, M))
            else:
                total += float(ot.sinkhorn2(pos, neg, M, reg)[0])

    return total

if __name__ == 'main':

    sh_x, sh_y = (4,4), (9,9)
    tmp = np.zeros(sh_x + sh_y)
    tmp[:,:,1,1] = 1
    tmp[1,3,1,1] = 0
    tmp[1,3,1,2] = 1

    from plottings import plot_u
    plot_u(tmp, sh_y)

    dist = TVW1(tmp, sh_x, sh_y)

    print(dist)

if __name__ == 'main':

    sh_x, sh_y = (4,4), (9,9)
    tmp = np.zeros(sh_x + sh_y)
    tmp[:,:,1,1] = 1
    tmp[:,:,3,3] = -1
    tmp[1,3,1,1] = 0
    tmp[1,3,1,2] = 1

    from plottings import plot_u
    plot_u(tmp, sh_y)