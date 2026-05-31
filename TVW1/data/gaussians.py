"""Gaussian-distribution test data — the single source of truth.

Two families live here:

  (1) Canonical crossing-fibre mixture used by the diffusion experiments,
      parametrised by eigenvalues + a spatial angle field via Givens rotations.
      DETERMINISTIC (no RNG).  Entry point: build_gaussian_mixture(shape_x, shape_y).
      (This generator used to be inlined — three near-identical copies — inside
      test_models_with_gaussian_mixture.py, experiment_1 and experiment_2a.)

  (2) Older covariance-based helpers (my_meshgrid, gaussian_function_as_matrix,
      rotate_2d_array, build_u_gaussians ...) kept for the behaviour demos.
      Those that build whole 4-D fields use np.random — SEED IN THE CALLER.

Nothing is built at import time.
"""
import numpy as np
from scipy.ndimage import shift

N_gaussians = 30
M_gaussians = 5


# =====================================================================
# (1) Canonical crossing-fibre gaussian mixture (deterministic)
# =====================================================================
def function_as_array(coordinates, f):
    mesh = np.meshgrid(*coordinates, indexing='ij')
    return np.vectorize(f)(*mesh)


def gaussian_density(mu, sig_inv, *x):
    mu, x = np.asarray(mu), np.asarray(x)
    d = len(mu)
    exponent = -0.5 * (x - mu).T @ sig_inv @ (x - mu)
    const = (2 * np.pi) ** (d / 2)
    return np.exp(exponent) * np.sqrt(np.abs(np.linalg.det(sig_inv))) / const


def gaussian_density_factory(mu, sig_inv):
    mu = np.asarray(mu)
    def f(*x):
        return gaussian_density(mu, sig_inv, *x)
    return f


def rotation_matrix_2d(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi),  np.cos(phi)]])


def Givens_rotation(n, i, j, phi):
    R = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    R[[i, i, j, j], [i, j, i, j]] = c, -s, s, c
    return R


def orthonormal_basis_from_angles(n, phis):
    assert len(phis) == n * (n - 1) // 2
    U = np.eye(n)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            U = Givens_rotation(n, i, j, phis[idx]) @ U
            idx += 1
    return U.T


def generate_field_2d(shape_x, a, b, c=0):
    """(M,N,1) array of angles, varying linearly along the diagonal."""
    M, N = shape_x
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, N), indexing='ij')
    A = np.pi / 4 * (a * X + b * Y + c)
    return A.reshape(A.shape + (1,))


def generate_gaussians_from_eigval_and_field(eigenvalues, field, shape_y):
    ndim_y = len(shape_y)
    number_of_angles = field.shape[-1]
    assert number_of_angles == ndim_y * (ndim_y - 1) // 2
    shape_x = field.shape[:-1]
    P = np.zeros(shape_x + shape_y)
    coordinates = [np.linspace(-8, 8, n) for n in shape_y]
    Delta = np.diag(eigenvalues)
    for idx in np.ndindex(shape_x):
        U = orthonormal_basis_from_angles(ndim_y, field[idx])
        sig = U.T @ Delta @ U
        f = gaussian_density_factory(ndim_y * [0], np.linalg.inv(sig))
        P[idx] = function_as_array(coordinates, f)
    return P


def build_gaussian_mixture(shape_x, shape_y):
    """Sum of two oriented-gaussian fields: the canonical crossing-fibre toy.

    Deterministic.  Callers typically scale by 100 and threshold tiny values:
        gt = 100 * build_gaussian_mixture(shape_x, shape_y)
        gt[gt < 1e-14] = 0
    """
    field_1 = generate_field_2d(shape_x, 0.8, 0.8, 2)
    field_2 = generate_field_2d(shape_x, 0, 0, -0.75)
    eigenvalues = (8, 0.6)
    g1 = generate_gaussians_from_eigval_and_field(eigenvalues, field_1, shape_y)
    g2 = generate_gaussians_from_eigval_and_field(eigenvalues, field_2, shape_y)
    return g1 + g2


# =====================================================================
# (2) Covariance-based helpers (older demos)
# =====================================================================
def my_meshgrid(*coordinates, normalize=True, indexing='xy'):
    """coord_0..coord_d (or a shape) -> (l1,...,ld,d) grid of coordinate tuples."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if all(isinstance(x, int) for x in coordinates):
        if normalize:
            coordinates = [np.linspace(0, 1, dim) for dim in coordinates]
        else:
            coordinates = [np.linspace(0, dim - 1, dim) for dim in coordinates]
    return np.array(np.meshgrid(*coordinates, indexing=indexing)).T


def quadratic_function(A, q):
    """einsum 'i1...ikd,dn,i1...ikn' for A of shape (N1,...,Nk,D) and q of shape (D,D)."""
    import string
    k = A.ndim - 1
    if k != len(q):
        raise ValueError('A is supposed to represent coordinates with same size as the side of q')
    indices = string.ascii_letters[:-2]
    if k > len(indices):
        raise ValueError(f"Too many dimensions in A (maximum supported is {len(indices)}).")
    free_indices = indices[:k]
    subscript_A1 = free_indices + 'Y'
    subscript_q = 'YZ'
    subscript_A2 = free_indices + 'Z'
    einsum_subscript = f'{subscript_A1},{subscript_q},{subscript_A2}->{free_indices}'
    return np.einsum(einsum_subscript, A, q, A)


def rotate_2d_array(A, theta):
    """Conjugate a 2x2 operator by R(theta) (useful for covariance matrices)."""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R @ A @ R.T


def gaussian_function_as_matrix(*coordinates, mean=None, cov=None, output_dxyz=False):
    """Matrix M with M[i,j] = normal_density(coord_0[i], coord_1[j]; mean, cov)."""
    d = len(coordinates)
    if mean is None:
        mean = np.zeros(d)
    if cov is None:
        cov = np.eye(d)
    if type(cov) in {float, int}:
        cov = cov * np.eye(d)
    coordinates_2 = [coord - mean[k] for k, coord in enumerate(coordinates)]
    grid = my_meshgrid(*coordinates_2)
    covinv = np.linalg.inv(cov)
    density = np.sqrt((2 * np.pi) ** -d * np.linalg.det(covinv)) * np.exp(-0.5 * quadratic_function(grid, covinv))
    if output_dxyz:
        dxyz = []
        for coord in coordinates:
            dxyz += [(coord[-1] - coord[0]) / (len(coord) - 1)]
        dxyz = np.prod(dxyz)
        return density, dxyz
    return density


def build_u_gaussians(M=M_gaussians, N=N_gaussians, cov=None):
    """4-D field (M,M,N,N) of randomly-rotated gaussians.  Seed in the caller."""
    if cov is None:
        cov = np.array([[1, 0], [0, 12]])
    gaussians = np.zeros((M, M, N, N))
    coord_0, coord_1 = 2 * [np.linspace(-8, 8, N)]
    for i in range(M):
        for j in range(M):
            mean = 0 * np.array([np.random.rand(), np.random.rand()])
            theta = np.random.rand() - 0.5
            cov_temp = rotate_2d_array(cov, theta)
            gaussians[i, j] = shift(gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp), shift=mean)
    return gaussians


def build_u_gaussian_mixture(M=M_gaussians, N=N_gaussians, cov=None):
    """build_u_gaussians plus a second, perpendicular gaussian in ~half the voxels.

    Random — seed in the caller for reproducibility.
    """
    if cov is None:
        cov = np.array([[1, 0], [0, 12]])
    gaussian_mixture = build_u_gaussians(M=M, N=N, cov=cov)
    cov_rot_90 = rotate_2d_array(cov, np.pi / 2)
    coord_0, coord_1 = 2 * [np.linspace(-8, 8, N)]
    for i in range(M):
        for j in range(M):
            if np.random.rand() < 0.5:
                theta = 0.8 * np.random.rand() - 0.4
                cov_temp = rotate_2d_array(cov_rot_90, theta)
                gaussian_mixture[i, j] = gaussian_mixture[i, j] + gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp)
    return gaussian_mixture.clip(min=0)


def generate_field(N, a, b, c=0):
    """(N,N) angle field symmetric along the lower-left/upper-right diagonal."""
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    return np.pi / 4 * (a * X + b * Y + c)


def generate_field_2(N, a, c=0):
    """(N,N) angle field for a simple curved flow."""
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    return np.pi / 4 * (a * np.exp(Y) + c)


def build_u_gaussians_fromfield(*fields, N=N_gaussians, cov=None):
    """Superpose gaussians whose orientation follows each given (M,M) angle field."""
    if cov is None:
        cov = np.array([[1, 0], [0, 20]])
    M = len(fields[0])
    gaussians = np.zeros((M, M, N, N))
    coord_0, coord_1 = 2 * [np.linspace(-8, 8, N)]
    for field in fields:
        for i in range(M):
            for j in range(M):
                cov_temp = rotate_2d_array(cov, field[i, j])
                gaussians[i, j] += gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp)
    return gaussians


def generate_simple_gaussian_crossing(M, N=N_gaussians):
    m = int(M / 3)
    coord_0, coord_1 = 2 * [np.linspace(-8, 8, N)]
    isotropic_diff = gaussian_function_as_matrix(coord_0, coord_1) * 0.01

    field_hor = np.ones((M, M)) * (-np.pi / 2)
    horizontal = build_u_gaussians_fromfield(field_hor, N=N)
    indices = np.concatenate((np.arange(m), np.arange(len(horizontal) - m, len(horizontal))))
    horizontal[indices] = isotropic_diff

    field_vert = np.zeros((M, M))
    vertical = build_u_gaussians_fromfield(field_vert)
    vertical[:, indices] = isotropic_diff

    return horizontal + vertical


def create_4d_gaussians(covariances, N=20):
    """(s0,s1,N,N) field of gaussian densities from a (s0,s1,2,2) covariance field."""
    coord_x = np.linspace(-3, 3, N)
    coord_y = np.linspace(-3, 3, N)
    s0, s1, _, _ = covariances.shape
    B = np.zeros((s0, s1, N, N))
    for i in range(s0):
        for j in range(s1):
            B[i, j] = gaussian_function_as_matrix(coord_x, coord_y, cov=covariances[i, j])
    return B


def build_flow_gaussians_crossing(M=11):
    """The crossing-flow gaussian field that older scripts imported as a module global."""
    field_1 = generate_field(M, 1, 1, 2)
    field_2 = generate_field(M, 0, 0, -0.75)
    return build_u_gaussians_fromfield(field_1) + build_u_gaussians_fromfield(field_2)
