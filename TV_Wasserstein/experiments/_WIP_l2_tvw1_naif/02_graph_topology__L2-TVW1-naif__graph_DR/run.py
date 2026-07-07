"""
Phase 2a: Graph topology comparison for Graph-DR on TV-Wasserstein dMRI reconstruction.

Setup: Fix complete state graph on N=4 nodes. Vary the base graph among all connected
subgraphs of K_4. Color convergence curves by algebraic connectivity λ₁.

This mirrors Figure 1(b) of Bredies, Chenchene & Naldi (2022).

Outputs:
  - Convergence curves colored by λ₁ (one representative per λ₁ group)
  - Optionally: all 38 graphs to show clustering within each λ₁ group
"""

# %% ── 0. Reproducibility & imports ──────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401  — wires Libraries + core + models + data

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations

np.random.seed(42)

# %% ── 1. Data generation (single source of truth in data/) ───────────────────
from gaussians import build_gaussian_mixture
from masks import undersampling_mask_xy

# %% ── 2. Build data ───────────────────────────────────────────────────────────

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

from forward import KK_factory, KKstar_factory
KK = KK_factory(mask_fft)
KKstar = KKstar_factory(mask_fft)

E_clean = KK(ground_truth)
SNR = 20
std = np.sqrt(np.sum(np.abs(E_clean)**2) / (E_clean.size * SNR))
E = E_clean + np.random.normal(size=E_clean.size, scale=std)

print(f"Data: shape_x={shape_x}, shape_y={shape_y}, retained={retained_ratio}, SNR={SNR}")

# %% ── 3. Graph enumeration ────────────────────────────────────────────────────

N = 4
all_edges_K4 = [(i,j) for i in range(N) for j in range(i+1, N)]

# Complete state graph (fixed for all runs)
E_state = list(all_edges_K4)
parent_node = [[] for _ in range(N)]
for h, i in E_state:
    parent_node[i].append(h)

adj_state = [set() for _ in range(N)]
for h, i in E_state:
    adj_state[h].add(i); adj_state[i].add(h)
d = np.array([len(a) for a in adj_state], dtype=float)

def is_connected(N, edges): # checks if a graph with N nodes is connected
    if not edges: return N <= 1
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    visited = set(); stack = [0]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node); stack.extend(adj[node] - visited)
    return len(visited) == N

def graph_laplacian(N, edges): # laplaciano del grafo non orientato: suppongo che un edge (u,v) compaia solo una volta (e NON anche (v,u))
    L = np.zeros((N, N))
    for u, v in edges:
        L[u,u] += 1; L[v,v] += 1; L[u,v] -= 1; L[v,u] -= 1
    return L

def onto_decomposition(L, N):
    """Z such that ZZ^T = L, shape N x (N-1)."""
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    Z = eigvecs[:, 1:] * np.sqrt(np.maximum(eigvals[1:], 0))
    return Z

# Enumerate all connected subgraphs of K_4
all_base_graphs = [] # list of dictionaries with all possible base graphs
for n_edges in range(N-1, len(all_edges_K4)+1):
    for edge_subset in combinations(all_edges_K4, n_edges):
        if is_connected(N, edge_subset):
            L = graph_laplacian(N, edge_subset)
            eigvals = np.sort(np.linalg.eigvalsh(L)) # eigenvals is best for symmetric matrices
            all_base_graphs.append({
                'edges': edge_subset,
                'n_edges': n_edges,
                'L': L,
                'lambda1': round(eigvals[1], 4),
                'Z': onto_decomposition(L, N),
            })

# Group by lambda1
from collections import defaultdict
lambda1_groups = defaultdict(list) # lambda1_groups[λ₁] = [ *base graphs with connectivity λ₁* ]
for bg in all_base_graphs:
    lambda1_groups[bg['lambda1']].append(bg)

lambda1_values = sorted(lambda1_groups.keys()) # all possible connectivities
print(f"\nTotal connected subgraphs of K_4: {len(all_base_graphs)}")
print(f"Distinct λ₁ values: {lambda1_values}")
for lam in lambda1_values:
    print(f"  λ₁={lam:.4f}: {len(lambda1_groups[lam])} graphs")

# Pick one representative per λ₁ group (there are 4 possible algebraic connectivities)
representatives = {}
# λ₁ ≈ 0.586: chain 0-1-2-3
representatives[lambda1_values[0]] = next(
    bg for bg in lambda1_groups[lambda1_values[0]]
    if bg['edges'] == ((0,1),(1,2),(2,3))
)
# λ₁ = 1.0: star at node 0
representatives[lambda1_values[1]] = next(
    bg for bg in lambda1_groups[lambda1_values[1]]
    if bg['edges'] == ((0,1),(0,2),(0,3))
)
# λ₁ = 2.0: 4-cycle  0-1, 1-2, 2-3, 0-3
representatives[lambda1_values[2]] = next(
    bg for bg in lambda1_groups[lambda1_values[2]]
    if bg['edges'] == ((0,1),(0,3),(1,2),(2,3))
)
# λ₁ = 4.0: complete graph
representatives[lambda1_values[3]] = all_base_graphs[-1]

print("\nRepresentatives:")
for lam, bg in representatives.items():
    print(f"  λ₁={lam:.4f}: edges={bg['edges']}")

# %% ── 4. Experiment parameters ────────────────────────────────────────────────

alpha1 = 0.45
alpha2 = 1.4
sigma = 0.3       # optimal from Phase 1
iterations = 30000
record_every = 100

# %% ── 5. L2 error callback ───────────────────────────────────────────────────

# graph_DR no longer computes diagnostics internally; the consensus variance
# (P_variance) and fixed-point residual (edge_residual_norm_sq) plotted below come
# from graph_DR_diagnostics. Z varies per base graph, so the callback is rebuilt
# inside each loop with the current Z.
from proximal_algorithms.graph_DR_auxiliary_functions import graph_DR_diagnostics

def make_metrics_fn(P_true, P_true_norm, N_op, Z):
    def fn(x_list):
        P_bar = sum(xi['P'] for xi in x_list) / N_op
        return {'relative_L2_error': np.linalg.norm(P_bar - P_true) / P_true_norm,
                **graph_DR_diagnostics(x_list, Z)}
    return fn

# %% ── 6. Run: one representative per λ₁ ──────────────────────────────────────

from l2_tvw1_naif import compute_C_bounds, model_naif_graph

C_bounds = compute_C_bounds(shape_x, shape_y, alpha1, alpha2, E)

results_repr = {}

for lam1, bg in representatives.items():
    label = f"λ₁={lam1:.3f}, edges={bg['edges']}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  sigma={sigma}, sigma/d={sigma/d[0]:.4f}")
    print(f"{'='*60}")

    Z = bg['Z']
    graph_DR_parameters = (Z, parent_node, d)
    extra_metrics_fn = make_metrics_fn(ground_truth, ground_truth_norm, N, Z)

    np.random.seed(123)
    t0 = time.time()
    x, w, history = model_naif_graph(
        E, mask_fft, shape_x, shape_y, alpha1, alpha2,
        C_bounds, graph_DR_parameters, sigma, iterations,
        printprogress=True,
        extra_metrics_fn=extra_metrics_fn, record_every=record_every,
    )
    elapsed = time.time() - t0

    P_bar = sum(xi['P'] for xi in x) / N
    final_l2 = np.linalg.norm(P_bar - ground_truth) / ground_truth_norm

    results_repr[lam1] = {
        'bg': bg,
        'history': history,
        'final_l2': final_l2,
        'elapsed': elapsed,
    }
    print(f"  Done in {elapsed:.1f}s.  Final rel L² = {final_l2:.6f}")

# %% ── 7. Run: ALL base graphs (to show clustering) ───────────────────────────

RUN_ALL = False  # set to False to skip the full enumeration

results_all = {}

if RUN_ALL:
    print(f"\n{'#'*60}")
    print(f"  Running ALL {len(all_base_graphs)} base graphs...")
    print(f"{'#'*60}")

    for idx, bg in enumerate(all_base_graphs):
        label = f"[{idx+1}/{len(all_base_graphs)}] λ₁={bg['lambda1']:.3f}, {bg['edges']}"
        print(f"\n  {label}")

        Z = bg['Z']
        graph_DR_parameters = (Z, parent_node, d)
        extra_metrics_fn = make_metrics_fn(ground_truth, ground_truth_norm, N, Z)

        np.random.seed(123)
        t0 = time.time()
        x, w, history = model_naif_graph(
            E, mask_fft, shape_x, shape_y, alpha1, alpha2,
            C_bounds, graph_DR_parameters, sigma, iterations,
            printprogress=False,
            extra_metrics_fn=extra_metrics_fn, record_every=record_every,
        )
        elapsed = time.time() - t0

        P_bar = sum(xi['P'] for xi in x) / N
        final_l2 = np.linalg.norm(P_bar - ground_truth) / ground_truth_norm

        results_all[idx] = {
            'bg': bg,
            'history': history,
            'final_l2': final_l2,
            'elapsed': elapsed,
        }
        print(f"    Done in {elapsed:.1f}s, L²={final_l2:.4f}")

# %% ── 8. Plots ────────────────────────────────────────────────────────────────

# Color map: one color per λ₁ value
lambda1_colors = {
    lambda1_values[0]: '#d62728',  # red    (λ₁ ≈ 0.586, paths)
    lambda1_values[1]: '#ff7f0e',  # orange (λ₁ = 1, stars / 4-edge)
    lambda1_values[2]: '#2ca02c',  # green  (λ₁ = 2, cycles / 5-edge)
    lambda1_values[3]: '#1f77b4',  # blue   (λ₁ = 4, complete)
}

lambda1_labels = {
    lambda1_values[0]: f'λ₁ = {lambda1_values[0]:.4f}',
    lambda1_values[1]: f'λ₁ = {lambda1_values[1]:.1f}',
    lambda1_values[2]: f'λ₁ = {lambda1_values[2]:.1f}',
    lambda1_values[3]: f'λ₁ = {lambda1_values[3]:.1f}',
}

# --- Figure 1: Representatives only (clean, like the paper) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ('P_variance', 'Var(P)', 'P-component state variance'),
    ('edge_residual_norm_sq', r'$\|\tilde{T}w^k - w^k\|^2$', 'Fixed-point residual'),
    ('relative_L2_error', r'Rel. $L^2$ error', 'L² error vs ground truth'),
]

for ax_idx, (key, ylabel, title) in enumerate(metrics):
    ax = axes[ax_idx]
    for lam1 in lambda1_values:
        res = results_repr[lam1]
        iters = res['history']['iterations_recorded']
        vals = res['history'][key]
        c = lambda1_colors[lam1]
        ax.loglog(iters, vals, label=lambda1_labels[lam1], color=c, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Phase 2a: Base graph comparison (representatives)  |  '
    f'σ={sigma}, α₁={alpha1}, α₂={alpha2}, {iterations} iter\n'
    f'State graph: complete K₄ (d=[3,3,3,3])',
    fontsize=13,
)
plt.tight_layout()
plt.savefig('graph_topology_representatives.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 2: All 38 graphs (clustering by λ₁) ---
if RUN_ALL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[ax_idx]
        # Thin lines for all graphs
        for idx_bg, res in results_all.items():
            lam1 = res['bg']['lambda1']
            iters = res['history']['iterations_recorded']
            vals = res['history'][key]
            c = lambda1_colors[lam1]
            ax.loglog(iters, vals, color=c, linewidth=0.5, alpha=0.5)

        # Thick lines for representatives (on top)
        for lam1 in lambda1_values:
            res = results_repr[lam1]
            iters = res['history']['iterations_recorded']
            vals = res['history'][key]
            c = lambda1_colors[lam1]
            ax.loglog(iters, vals, label=lambda1_labels[lam1],
                      color=c, linewidth=2.5, alpha=0.9)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Phase 2a: All {len(all_base_graphs)} base graphs (thin) + representatives (thick)\n'
        f'σ={sigma}, α₁={alpha1}, α₂={alpha2}, {iterations} iter  |  '
        f'State: complete K₄',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig('graph_topology_all38.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% ── 9. Summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  SUMMARY: Phase 2a (representatives)")
print(f"{'='*70}")
print(f"  {'λ₁':>8s}  {'edges':>35s}  {'final L²':>10s}  {'time':>8s}")
print(f"  {'-'*8}  {'-'*35}  {'-'*10}  {'-'*8}")
for lam1 in lambda1_values:
    r = results_repr[lam1]
    edges_str = str(r['bg']['edges'])
    print(f"  {lam1:8.4f}  {edges_str:>35s}  {r['final_l2']:10.6f}  {r['elapsed']:7.1f}s")

if RUN_ALL:
    print(f"\n  SUMMARY: All {len(all_base_graphs)} graphs")
    print(f"  {'λ₁':>8s}  {'count':>6s}  {'mean L²':>10s}  {'std L²':>10s}  {'min L²':>10s}  {'max L²':>10s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lam1 in lambda1_values:
        l2s = [r['final_l2'] for r in results_all.values() if r['bg']['lambda1'] == lam1]
        print(f"  {lam1:8.4f}  {len(l2s):6d}  {np.mean(l2s):10.6f}  {np.std(l2s):10.6f}  "
              f"{np.min(l2s):10.6f}  {np.max(l2s):10.6f}")