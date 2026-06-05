"""Cartoon-shaped displacement test data: bars/crosses, arrows ("gammas"),
stars and spirals.  Used by the behaviour demos (inpainting / median / angle).

Nothing is built at import time; the old module-level arrays (u_gammas, u_stars,
the masked inpainting arrays ...) are now returned by build_* functions.  Seed
in the caller where np.random is involved.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.ndimage import map_coordinates, shift

from gaussians import gaussian_function_as_matrix

datashape = (6, 6, 15, 15)
R1, R2, Q1, Q2 = datashape
M_gammas = 5
N_gammas = 30


def rotate_image(x, theta):
    """Rotate a 2-D array by theta radians (bilinear, zero fill)."""
    H, W = x.shape
    cy, cx = H / 2.0, W / 2.0
    y_indices, x_indices = np.indices((H, W))
    x_indices = x_indices - cx
    y_indices = y_indices - cy
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_rot = cos_theta * x_indices + sin_theta * y_indices
    y_rot = -sin_theta * x_indices + cos_theta * y_indices
    x_rot += cx
    y_rot += cy
    return map_coordinates(x, [y_rot.ravel(), x_rot.ravel()], order=1, mode='constant', cval=0.0).reshape((H, W))


# ---- "-|o." displacements ----
def displacement_factory(displ_shape, shape=(Q1, Q2), fading=True, normalize=True):
    q1, q2 = shape
    out = np.full(shape, False)
    if displ_shape == '-':
        for i in range(q1):
            if i >= 2 * q1 / 5 and i + 1 <= 3 * q1 / 5:
                out[i, :] = True
    if displ_shape == '|':
        for j in range(q2):
            if j >= 2 * q2 / 5 and j + 1 <= 3 * q2 / 5:
                out[:, j] = True
    if displ_shape == '--':
        for i in range(q1):
            if i >= q1 / 3 and i + 1 <= 2 * q1 / 3:
                out[i, :] = True
    if displ_shape == '||':
        for j in range(q2):
            if j >= q2 / 3 and j + 1 <= 2 * q2 / 3:
                out[:, j] = True
    if displ_shape == '+':
        out = displacement_factory('|', shape, fading=False, normalize=False) + displacement_factory('-', shape, fading=False, normalize=False)
    if displ_shape == '++':
        out = displacement_factory('||', shape, fading=False, normalize=False) + displacement_factory('--', shape, fading=False, normalize=False)
    a, b = q1 / 2, q2 / 2
    if displ_shape == 'o':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i - a) / a, (j - b) / b]) <= 1 and np.linalg.norm([(i - a) / a, (j - b) / b]) >= 4 / 5:
                    out[i, j] = True
    if displ_shape == 'oo':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i - a) / a, (j - b) / b]) <= 1 and np.linalg.norm([(i - a) / a, (j - b) / b]) >= 3 / 5:
                    out[i, j] = True
    if displ_shape == '.':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i - a) / a, (j - b) / b]) <= 1 / 2:
                    out[i, j] = True
    if displ_shape == '..':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i - a) / a, (j - b) / b]) <= 3 / 4:
                    out[i, j] = True
    if fading:
        coord_0, coord_1 = np.linspace(-1, 1, q1), np.linspace(-1, 1, q2)
        P = gaussian_function_as_matrix(coord_0, coord_1, cov=2 * np.eye(2))
        P = P / np.max(P)
        out = out * P
    if normalize:
        return out / np.sum(out)
    return out


def toydata_displacements_fromfactory():
    displ_hor = displacement_factory('-', shape=(Q1, Q2))
    displ_plus = displacement_factory('+', shape=(Q1, Q2))
    displ_circ = displacement_factory('o', shape=(Q1, Q2))
    displ_dot = displacement_factory('.', shape=(Q1, Q2))
    displ_bigdot = displacement_factory('..', shape=(Q1, Q2))

    toydata = np.zeros(datashape)
    toydata[:int(R1 / 2), :int(R2 / 2)] = displ_hor
    toydata[int(R1 / 2):, :int(R2 / 2)] = displ_plus
    toydata[int(R1 / 2):, int(R2 / 2):] = displ_circ + displ_plus + displ_dot
    toydata[:int(R1 / 2), int(R2 / 2):] = displ_bigdot + displ_hor
    return toydata


# ---- "gamma-shaped" arrows ----
def build_gammashape(N=N_gammas):
    gammashape = np.zeros((N, N))
    gammashape[int(N / 10):int(7 * N / 10), int(4.5 * N / 10):int(5.5 * N / 10)] = 1
    gammashape[int(N / 10):int(2 * N / 10), int(4.5 * N / 10):int(7.5 * N / 10)] = 1
    gammashape[int(3 * N / 10):int(4 * N / 10), int(3.5 * N / 10):int(4.5 * N / 10)] = 1
    return gammashape


def build_u_gammas(M=M_gammas, N=N_gammas, angle=np.pi / 6):
    u_gammas = np.zeros((M, M, N, N))
    gammashape = build_gammashape(N)
    for i in range(M):
        for j in range(M):
            x = rotate_image(gammashape, i * angle)
            u_gammas[i, j] = shift(x, shift=(0, 7 * j - int(N / 2.5)))
    return u_gammas.clip(min=0)


def build_gammas_inpainting(M=M_gammas, N=N_gammas):
    """(u_gammas, masked_array) where voxel (1,1) is masked — the inpainting demo."""
    u_gammas = build_u_gammas(M=M, N=N)
    mask_inpainting = np.zeros(u_gammas.shape, dtype=bool)
    mask_inpainting[1, 1, :, :] = True
    return u_gammas, np.ma.masked_array(u_gammas, mask=mask_inpainting)


# ---- stars and spirals (median experiment, cf. Carlier & Chenchene) ----
def generate_star_array(N, n_points=5, outer_radius=None, inner_radius=None):
    grid = np.zeros((N, N), dtype=int)
    center = N / 2
    if outer_radius is None:
        outer_radius = N / 2 - 1
    if inner_radius is None:
        inner_radius = outer_radius / 2
    angles = np.linspace(0, 2 * np.pi, num=n_points * 2, endpoint=False) - np.pi / 2
    radii = np.empty(n_points * 2)
    radii[::2] = outer_radius
    radii[1::2] = inner_radius
    x = radii * np.cos(angles) + center
    y = radii * np.sin(angles) + center
    vertices = np.column_stack([x, y])
    vertices = np.vstack([vertices, vertices[0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 2) + [mpath.Path.CLOSEPOLY]
    path = mpath.Path(vertices, codes)
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    points = np.vstack((X.flatten(), Y.flatten())).T
    mask = path.contains_points(points).reshape((N, N))
    grid[mask] = 1
    return grid


def generate_spiral_array(N, thickness=2, num_turns=3, max_radius=None):
    grid = np.zeros((N, N), dtype=int)
    center = N / 2.0
    if max_radius is None:
        max_radius = (N / 2.0) - 1.0
    b = max_radius / (num_turns * 2 * np.pi)
    y_indices, x_indices = np.ogrid[0:N, 0:N]
    x = x_indices - center
    y = y_indices - center
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    theta_spiral = r / b
    k = np.round((theta_spiral - theta) / (2 * np.pi))
    theta_adjusted = theta + 2 * np.pi * k
    delta_theta = np.abs(theta_spiral - theta_adjusted)
    delta_theta_max = thickness / b
    mask = delta_theta < delta_theta_max
    mask &= r <= max_radius
    grid[mask] = 1
    return grid


def build_u_stars(M=3, N=51):
    """(u_stars, masked_array) with voxel (1,1) masked — the Wasserstein-median demo."""
    star_array = generate_star_array(N, outer_radius=15, inner_radius=6.5)
    spiral_array = generate_spiral_array(N, thickness=1.5, num_turns=2, max_radius=15)
    u_stars = np.zeros((M, M, N, N))
    u_stars[0, 1] = shift(star_array, shift=(0, 0))
    u_stars[1, 0] = shift(star_array, shift=(0, -10))
    u_stars[1, 2] = shift(star_array, shift=(0, 10))
    u_stars[2, 1] = shift(spiral_array, shift=(10, 10))
    mask = np.zeros((M, M, N, N), dtype=bool)
    mask[1, 1, :, :] = True
    return u_stars, np.ma.masked_array(u_stars, mask=mask)

