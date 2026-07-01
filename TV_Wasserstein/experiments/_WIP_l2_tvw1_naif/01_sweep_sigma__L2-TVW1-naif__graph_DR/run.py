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

from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401  — wires Libraries + core + models + data

from metrics import voxelwise_wasserstein

# %% ── 1. Data generation (single source of truth in data/) ───────────────────
from gaussians import build_gaussian_mixture
from masks import undersampling_mask_xy

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

from forward import KK_factory, KKstar_factory

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
# graph_DR no longer computes diagnostics internally; the consensus variance
# (P_variance) and fixed-point residual (edge_residual_norm_sq) used in the plots
# below come from graph_DR_diagnostics, recorded via extra_metrics_fn.
from proximal_algorithms.graph_DR_auxiliary_functions import graph_DR_diagnostics

def make_metrics_fn(P_true, P_true_norm, N_operators, Z):
    """Returns x_list -> {relative_L2_error, P_variance, edge_residual_norm_sq}."""
    def fn(x_list):
        P_bar = sum(xi['P'] for xi in x_list) / N_operators
        err = np.linalg.norm(P_bar - P_true) / P_true_norm
        return {'relative_L2_error': err, **graph_DR_diagnostics(x_list, Z)}
    return fn

extra_metrics_fn = make_metrics_fn(ground_truth, ground_truth_norm, N, Z)

# %% ── 6. Run experiments ──────────────────────────────────────────────────────
from l2_tvw1_naif import compute_C_bounds, model_naif_graph

C_bounds = compute_C_bounds(shape_x, shape_y, alpha1, alpha2, E)

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
        C_bounds, graph_DR_parameters, sigma, iterations,
        printprogress=True,
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