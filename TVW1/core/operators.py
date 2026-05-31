"""Problem-specific linear operators for the TV-Wasserstein model.

I  : per-voxel total mass         (sum over the displacement axes)
J  : per-voxel zero-mean projection   J = id - (1/N) I*I
P  : per-voxel mean component          (so that  id = J + P )

These act on an EAP array  P : shape_x + shape_y  -> R, applying I/J entrywise
over the spatial grid shape_x.  (Lifted out of the old L2_TVW1_naif__model.py,
where they were defined inline.)
"""
import numpy as np


def II(P, shape_x, shape_y):
    assert P.shape == shape_x + shape_y
    axes_to_sum = tuple(-(i + 1) for i in range(len(shape_y)))
    return np.sum(P, axis=axes_to_sum)
# note: the spectral norm of II is exactly size_y = np.prod(shape_y)


def IIstar(x, shape_y):
    ndim_y = len(shape_y)
    x1 = x.reshape(x.shape + ndim_y * (1,))
    return np.broadcast_to(x1, x.shape + shape_y)


def PP(P, shape_x, shape_y):
    size_y = np.prod(shape_y)
    return IIstar(II(P, shape_x, shape_y), shape_y) / size_y


def JJ(P, shape_x, shape_y):
    N = np.prod(shape_y)
    return P - IIstar(II(P, shape_x, shape_y) / N, shape_y)


JJstar = JJ  # JJ is self adjoint
