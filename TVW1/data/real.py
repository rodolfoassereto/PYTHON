"""Real diffusion-tensor data (Sebastian's slice) -> gaussian EAP distributions.

Requires `dt.mat` in data/assets/.  That file is NOT in the repository
(it is git-ignored and currently absent), so these loaders raise a clear
FileNotFoundError until it is provided.
"""
import numpy as np
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"
_DT_MAT = _ASSETS / "dt.mat"


def _load_dt():
    from scipy.io import loadmat
    if not _DT_MAT.is_file():
        raise FileNotFoundError(
            f"dt.mat not found at {_DT_MAT}. The Sebastian tensor data is not part of "
            "the repository; drop dt.mat into data/assets/ to use real.py."
        )
    return loadmat(str(_DT_MAT))


def import_sebastian(whichslice=3, normalize_entries=True):
    data = _load_dt()
    dt_slice = data['dt'][:, :, whichslice]
    mask = data['mask'][:, :, whichslice]
    m, n = mask.shape
    dt_slice_matrices = np.zeros((m, n, 3, 3))
    for i in range(m):
        for j in range(n):
            Dxx, Dyy, Dzz, Dyz, Dxz, Dxy = dt_slice[i, j, :]
            dt_slice_matrices[i, j][np.triu_indices(3)] = [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
            dt_slice_matrices[i, j] += np.triu(dt_slice_matrices[i, j], k=1).T
            if ~np.all(np.linalg.eigvals(dt_slice_matrices[i, j]) >= 0):
                mask[i, j] = False
                dt_slice_matrices[i, j] = 0
    if normalize_entries:
        dt_slice_matrices = dt_slice_matrices / np.max(dt_slice_matrices)
    return dt_slice_matrices, np.array(mask, dtype=bool)


def build_sebastian_distributions(N=21, normalize_tensors=False):
    from gaussians import gaussian_function_as_matrix
    dt, mask = import_sebastian()
    if normalize_tensors:
        dt = dt / np.max(dt)
    M = dt.shape[0]
    coord = np.linspace(-1.5, 1.5, N)
    distributions = np.zeros((M, M, N, N, N))
    for i in range(M):
        for j in range(M):
            if mask[i, j]:
                distributions[i, j] = gaussian_function_as_matrix(coord, coord, coord, cov=dt[i, j])
    return distributions, mask
