# %% ── 0. Reproducibility & imports
"""
Phase 1: Step-size sigma sensitivity for Graph-DR applied to TV-Wasserstein dMRI reconstruction.

Fixed: graph topology (complete state, chain base), alpha1, alpha2, data (gaussian_mixture).
Swept: sigma over a logarithmic grid.

Outputs:
  - Convergence curves (edge residual, P-variance, relative L2 error) vs iteration
  - Final relative L2 error vs sigma
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Reproducibility
np.random.seed(42)

# Add Libraries folder to sys.path
objective_functions_path = r"C:\Users\rodol\My Drive\PYHTON\TVW1"
if objective_functions_path not in sys.path:
    sys.path.insert(0, objective_functions_path)
from objective_functions import voxelwise_wasserstein

# %% ── 1. Data generation helpers (inlined to avoid import chain issues) ───────

def _function_as_array(coordinates, f):
    mesh = np.meshgrid(*coordinates, indexing='ij')
    return np.vectorize(f)(*mesh)

def _gaussian_density(mu, sig_inv, *x):
    mu, x = np.asarray(mu), np.asarray(x)
    d = len(mu)
    exponent = -0.5 * (x - mu).T @ sig_inv @ (x - mu)
    const = (2 * np.pi) ** (d / 2)
    return np.exp(exponent) * np.sqrt(np.abs(np.linalg.det(sig_inv))) / const

def _gaussian_density_factory(mu, sig_inv):
    mu = np.asarray(mu)
    def f(*x):
        return _gaussian_density(mu, sig_inv, *x)
    return f

def _Givens_rotation(n, i, j, phi):
    R = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    R[[i, i, j, j], [i, j, i, j]] = c, -s, s, c
    return R

def _orthonormal_basis_from_angles(n, phis):
    U = np.eye(n)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            U = _Givens_rotation(n, i, j, phis[idx]) @ U
            idx += 1
    return U.T

def _generate_field_2d(shape_x, a, b, c=0):
    M, N = shape_x
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, N), indexing='ij')
    A = np.pi / 4 * (a * X + b * Y + c)
    return A.reshape(A.shape + (1,))

def _generate_gaussians(eigenvalues, field, shape_y):
    ndim_y = len(shape_y)
    shape_x = field.shape[:-1]
    P = np.zeros(shape_x + shape_y)
    coordinates = [np.linspace(-8, 8, n) for n in shape_y]
    Delta = np.diag(eigenvalues)
    for idx in np.ndindex(shape_x):
        U = _orthonormal_basis_from_angles(ndim_y, field[idx])
        sig = U.T @ Delta @ U
        f = _gaussian_density_factory(ndim_y * [0], np.linalg.inv(sig))
        P[idx] = _function_as_array(coordinates, f)
    return P

def build_gaussian_mixture(shape_x, shape_y):
    field_1 = _generate_field_2d(shape_x, 0.8, 0.8, 2)
    field_2 = _generate_field_2d(shape_x, 0, 0, -0.75)
    eigenvalues = (8, 0.6)
    return _generate_gaussians(eigenvalues, field_1, shape_y) + \
           _generate_gaussians(eigenvalues, field_2, shape_y)

def undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff):
    ndim_y = len(shape_y)
    size_y = np.prod(shape_y)
    N_ones = int(retained_ratio * size_y)
    f = _gaussian_density_factory(ndim_y * [0], np.eye(ndim_y))
    domain = 4 * concentration_coeff * np.pi / 2
    coords = [np.linspace(-domain, domain, n) for n in shape_y]
    probs = _function_as_array(coords, f).flatten()
    probs /= probs.sum()
    idx = np.random.choice(size_y, N_ones, replace=False, p=probs)
    mask_y = np.zeros(shape_y, dtype=bool)
    mask_y[np.unravel_index(idx, shape_y)] = True
    mask_xy = np.zeros(shape_x + shape_y, dtype=bool)
    mask_xy[:] = mask_y
    return mask_xy

# %% ── 2. Build ground truth and measured data ─────────────────────────────────

shape_x = (7, 8)
shape_y = (19, 20)

ground_truth = 100 * build_gaussian_mixture(shape_x, shape_y)
ground_truth[ground_truth < 1e-14] = 0
ground_truth_norm = np.linalg.norm(ground_truth)

ndim_x, ndim_y = len(shape_x), len(shape_y)
shape_xy = shape_x + shape_y
ndim_xy = len(shape_xy)

retained_ratio = 0.15
mask_fft = undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7)
mask_fft = np.fft.ifftshift(mask_fft)
mask_rfft = mask_fft[
    tuple(
        slice(None) if i != ndim_xy - 1 else slice(0, shape_xy[-1] // 2 + 1)
        for i in range(ndim_xy)
    )
]

from L2_TVW1_naif__model import KK_factory, KKstar_factory

KK = KK_factory(mask_rfft)
KKstar = KKstar_factory(mask_rfft)

E_clean = KK(ground_truth)
SNR = 20
E_energy = np.sum(np.abs(E_clean) ** 2)
std = np.sqrt(E_energy / (E_clean.size * SNR))
noise = np.random.normal(size=E_clean.size, scale=std)
E = E_clean + noise

print(f"Data: shape_x={shape_x}, shape_y={shape_y}, retained={retained_ratio}, SNR={SNR}")
print(f"  ||P†|| = {ground_truth_norm:.4f},  E.size = {E.size}")

# %% ── 3. Graph parameters (complete state, chain base OR complete base) ────────────────────────
N = 4

E_state = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

parent_node = [[] for _ in range(N)]
for h, i in E_state:
    parent_node[i].append(h)

base_graph_type = 'complete' # Adjust this parameter

if base_graph_type == 'complete':
    E_base = E_state.copy()
    print('You have chosen the COMPLETE base graph')

if base_graph_type == 'chain':
    E_base = [(0, 1), (1, 2), (2, 3)]
    print('You have chosen the CHAIN base graph')

Z = np.zeros((N, N - 1))
for j, (u, v) in enumerate(E_base):
    Z[u, j] = -1
    Z[v, j] = +1

adj = [set() for _ in range(N)]
for h, i in E_state:
    adj[h].add(i)
    adj[i].add(h)
d = np.array([len(a) for a in adj], dtype=float)

graph_DR_parameters = (Z, parent_node, d)

lambda1_base = 2 - np.sqrt(2)
print(f"Graph: complete state (d={d.tolist()}), chain base (lambda_1={lambda1_base:.4f})")

# %% ── 4. Experiment parameters ────────────────────────────────────────────────
alpha1 = 0.45
alpha2 = 1.4

iterations = 100
record_every = 10

sigma_values = [0.02]

# %% ── 5. Extra metrics callback ───────────────────────────────────────────────
def make_l2_error_fn(P_true, P_true_norm, N_operators):
    """Returns x_list -> dict computing relative L2 error of the consensus P."""
    def fn(x_list):
        P_bar = sum(xi['P'] for xi in x_list) / N_operators
        err = np.linalg.norm(P_bar - P_true) / P_true_norm
        return {'relative_L2_error': err}
    return fn

extra_metrics_fn = make_l2_error_fn(ground_truth, ground_truth_norm, N)

# %% ── 6. Run experiments ──────────────────────────────────────────────────────
from L2_TVW1_naif__model import model_naif_graph

results = {}

for sigma in sigma_values:
    print(f"\n{'=' * 60}")
    print(f"  sigma = {sigma},  effective resolvent step = sigma/d = {sigma / d[0]:.4f}")
    print(f"{'=' * 60}")

    # Reset RNG so all runs have the same random w0 initialization
    np.random.seed(123)

    t0 = time.time()
    x, w, history = model_naif_graph(
        E, mask_rfft, shape_x, shape_y, alpha1, alpha2,
        graph_DR_parameters, sigma, iterations,
        printprogress=True,
        return_history=True,
        extra_metrics_fn=extra_metrics_fn,
        record_every=record_every,
    )
    elapsed = time.time() - t0

    P_bar = sum(xi['P'] for xi in x) / N
    final_l2 = np.linalg.norm(P_bar - ground_truth) / ground_truth_norm
    final_W1, _, sums_avg_W1 = voxelwise_wasserstein( P_bar, ground_truth, shape_x, shape_y, order=1 )
    final_W2, _, sums_avg_W2 = voxelwise_wasserstein( P_bar, ground_truth, shape_x, shape_y, order=2 )
    final_W1 = final_W1 / np.sum(sums_avg_W1)
    final_W2 = final_W2 / np.sum(sums_avg_W2)

    results[sigma] = {
        'history': history,
        'final_l2': final_l2,
        'final_W1': final_W1,
        'final_W2': final_W2,
        'elapsed': elapsed,
        'P_bar': P_bar.copy(),
    }

    print(f"  Done in {elapsed:.1f}s.  Final relative L2 error = {final_l2:.6f}")

# %% ── 7. Plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_values)))

# (a) Edge residual norm squared
ax = axes[0, 0]
for (sigma, res), c in zip(results.items(), colors):
    iters = res['history']['iterations_recorded']
    vals = res['history']['edge_residual_norm_sq']
    ax.loglog(iters, vals, label=f'sig={sigma}', color=c, linewidth=1.2)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\| \tilde{T} w^k - w^k \|^2$')
ax.set_title('Edge residual (fixed-point residual)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# (b) P-variance
ax = axes[0, 1]
for (sigma, res), c in zip(results.items(), colors):
    iters = res['history']['iterations_recorded']
    vals = res['history']['P_variance']
    ax.loglog(iters, vals, label=f'sig={sigma}', color=c, linewidth=1.2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Var(P)')
ax.set_title('P-component state variance')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# (c) Relative L2 error vs iteration
ax = axes[1, 0]
for (sigma, res), c in zip(results.items(), colors):
    iters = res['history']['iterations_recorded']
    vals = res['history']['relative_L2_error']
    ax.semilogy(iters, vals, label=f'sig={sigma}', color=c, linewidth=1.2)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\| \bar{P}^k - P^\dagger \| \, / \, \| P^\dagger \|$')
ax.set_title('Relative L2 error vs ground truth')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# (d) Final L2 error vs sigma
ax = axes[1, 1]
sigmas_sorted = sorted(results.keys())
final_errors = [results[s]['final_l2'] for s in sigmas_sorted]
ax.semilogx(sigmas_sorted, final_errors, 'ko-', markersize=6)
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel('Final relative L2 error')
ax.set_title(f'Final L2 error after {iterations} iterations')
ax.grid(True, alpha=0.3)

best_idx = np.argmin(final_errors)
best_sigma = sigmas_sorted[best_idx]
best_err = final_errors[best_idx]
ax.annotate(
    f'  sig*={best_sigma}\n  err={best_err:.4f}',
    xy=(best_sigma, best_err), fontsize=9,
    arrowprops=dict(arrowstyle='->', color='red'),
    xytext=(best_sigma * 3, best_err + 0.02), color='red',
)

plt.suptitle(
    f'Phase 1: sigma sensitivity  |  alpha1={alpha1}, alpha2={alpha2}, '
    f'retained={int(retained_ratio * 100)}%, SNR={SNR}, {iterations} iter\n'
    f'Graph: complete state, chain base (lambda_1={lambda1_base:.3f})',
    fontsize=13,
)
plt.tight_layout()
plt.savefig('sigma_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% ── 8. Summary table ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  SUMMARY: sigma sweep  (iterations={iterations})")
print(f"{'=' * 70}")
print(f"  {'sigma':>8s}  {'sig/d':>8s}  {'final L2 err':>14s}  {'W1 err':>8s}  {'W2 err':>8s}  {'time (s)':>10s}")
print(f"  {'-' * 8}  {'-' * 8}  {'-' * 14}  {'-' * 8}  {'-' * 8}  {'-' * 10}")
for sigma in sigmas_sorted:
    r = results[sigma]
    marker = ' <-- best (for L2)' if sigma == best_sigma else ''
    print(f"  {sigma:8.2f}  {sigma / d[0]:8.4f}  {r['final_l2']:14.6f}  {r['final_W1']:8.6f}  {r['final_W2']:8.6f}  {r['elapsed']:10.1f}{marker}")

# %% ── 9. Save reconstructions as images ───────────────────────────────────────
from plottings import plot_u

# Ground truth
plot_u(ground_truth, ndim_y, title="Ground truth", save=True, name="recon_ground_truth.png")

# Rough reconstruction (adjoint of measured data)
P_rough = KKstar(E).clip(min=0)
plot_u(P_rough, ndim_y, title=f"Rough recon (KK*E), retained={int(retained_ratio*100)}%, SNR={SNR}",
       save=True, name="recon_rough.png")

# One image per sigma
for sigma in sigmas_sorted:
    P_bar = results[sigma]['P_bar']
    err = results[sigma]['final_l2']
    plot_u(P_bar.clip(min=0), ndim_y,
           title=f"sigma={sigma}, rel L2 err={err:.4f}, {iterations} iter",
           save=True, name=f"recon_sigma_{sigma}.png")