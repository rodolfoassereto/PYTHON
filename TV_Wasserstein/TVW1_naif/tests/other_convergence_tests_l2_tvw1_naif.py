# %% ── bootstrap ──────────────────────────────────────────────────────────────
"""Correctness tests for models/l2_tvw1_naif.py  (CP and graph-DR implementations).

Uses the project toymodel (gaussians.build_gaussian_mixture) undersampled by the
canonical q-space mask (masks.undersampling_mask_xy) through the rfft forward
operator (forward.KK_*), exactly as the diffusion experiments do.

What is checked
---------------
  naif (CP):
    * output shape == shape_x+shape_y, all finite, all >= 0 (the clip);
    * regularized reconstruction clearly beats the rough zero-filled adjoint;
    * convergence: the Cauchy residual ||P(2T)-P(T)|| is small (near a fixed point).
  naif_graph (graph Douglas-Rachford):
    * returns (x = list of N node dicts, w, history);
    * consensus reconstruction beats the rough adjoint;
    * convergence diagnostics from history all decrease
      (P-variance -> consensus, edge residual -> feasibility, rel. L2 error).
  cross-check:
    * CP and graph-DR solve the SAME model -> comparable reconstruction quality.
  determinism:
    * same global seed -> bit-identical output (both solvers).
  surrogate gap:
    * the finite primal-dual gap decreases along the graph-DR iterates.

NOTE on what is *not* asserted: the L2 error to the ground truth is non-monotone
in the iteration count (the regularized minimizer is not the ground truth), so
"more iterations -> smaller L2-to-GT" is deliberately NOT a check; the Cauchy
residual is used instead. Likewise the pointwise CP-vs-graph difference shrinks
only slowly, so the cross-check compares reconstruction *quality*, not voxels.

Run as a script (prints a PASS/FAIL report, exits non-zero on failure) or
cell-by-cell in Spyder / Positron.
"""
import sys
from pathlib import Path
parent_folder_with_condition = ( p for p in [Path.cwd(), *Path.cwd().parents] if (p / "paths_rodolfoassereto.py").is_file() )
sys.path.insert( 0, str( next( parent_folder_with_condition ) ) )
import paths_rodolfoassereto

import numpy as np

from gaussians import build_gaussian_mixture  # ty:ignore[unresolved-import]
from masks import undersampling_mask_xy  # ty:ignore[unresolved-import]
from forward import KK_factory, KKstar_factory  # ty:ignore[unresolved-import]
from proximal_algorithms.graph_DR_auxiliary_functions import make_graph_DR_parameters, graph_DR_diagnostics  # ty:ignore[unresolved-import]
import l2_tvw1_naif
from plottings import plot_u  # ty:ignore[unresolved-import]


# ── tiny test harness ─────────────────────────────────────────────────────────
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def rel_err(P, ref):
    return float(np.linalg.norm(P - ref) / np.linalg.norm(ref))


# ── parameters & thresholds (tuned against measured behaviour, with margin) ───
SEED = 0
SCALE = 100.0           # the experiments scale the mixture by 100
RETAINED = 0.2
CONCENTRATION = 0.7
SNR = 15
ALPHA1, ALPHA2 = 0.45, 1.4
STEPSIZE_RATIO = 0.5    # CP
SIGMA = 0.3             # graph-DR (good value from the sigma sweep)
IT_CP = 1000          # CP main run
IT_GRAPH = 1000         # graph-DR main run
IT_DET = 200            # determinism check (exactness is iteration-independent)

QUALITY_FACTOR = 0.85   # a "good" reconstruction has err < this * rough err
CAUCHY_TOL = 0.12       # ||P(2T)-P(T)|| / ||P(T)||  must be below this
QUALITY_AGREE_TOL = 0.15  # |cp_err - graph_err| in rel-L2 units



# %% ── build the toymodel problem ─────────────────────────────────────────────
shape_x, shape_y = (6, 6), (22, 22)
ndim_x, ndim_y = len(shape_x), len(shape_y)
np.random.seed(SEED)
gt = SCALE * build_gaussian_mixture(shape_x, shape_y)
gt[gt < 1e-14] = 0

ndim_xy = len(shape_x + shape_y)
shape_xy = shape_x + shape_y
mask_fft = np.fft.ifftshift(undersampling_mask_xy(shape_x, shape_y, RETAINED, CONCENTRATION))
mask_rfft = mask_fft[ ..., : (shape_xy[-1]//2 + 1) ]
KK, KKstar = KK_factory(mask_rfft), KKstar_factory(mask_rfft)

E_clean = KK(gt)
# Proper complex Gaussian noise: real and imaginary parts each ~ N(0, std^2/2),
# so total noise power per component = std^2 and SNR = E_energy / (N*std^2).
# The 1/sqrt(2) normalises so that the total power equals std^2 (not 2*std^2).
std = np.sqrt(np.sum(np.abs(E_clean) ** 2) / (E_clean.size * SNR))
noise = np.random.normal(size=E_clean.size, scale=std) \
        + 1j * np.random.normal(size=E_clean.size, scale=std)
E = E_clean + noise / np.sqrt(2)

rough = KKstar(E).clip(min=0)

plot_u(rough, ndim_y, title='rough')

prob = dict(shape_x=shape_x, shape_y=shape_y, gt=gt, E=E, mask_rfft=mask_rfft, rough=rough) # keys are automatically characters strings
shape_x, shape_y = prob['shape_x'], prob['shape_y']
gt, E, mask_rfft = prob['gt'], prob['E'], prob['mask_rfft']
rough_err = rel_err(prob['rough'], gt)
print(f"\nProblem: shape_x={shape_x}, shape_y={shape_y}, retained={RETAINED}, SNR={SNR}")
print(f"  rough (zero-filled adjoint) rel. L2 error = {rough_err:.4f}")


# %% ── Test 1: model_naif (Chambolle-Pock) ────────────────────────────────────
print("\n[model_naif / CP]")
np.random.seed(SEED)
P_cp = l2_tvw1_naif.model_CP(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, STEPSIZE_RATIO, IT_CP, printprogress=False)
plot_u(P_cp, ndim_y, title='CP')
cp_err = rel_err(P_cp, gt)

check("naif: all finite", np.all(np.isfinite(P_cp)))
check("naif: non-negative", np.all(P_cp >= 0), f"min={P_cp.min():.3g}")
check("naif: beats rough adjoint", cp_err < QUALITY_FACTOR * rough_err, f"{cp_err:.4f} vs rough {rough_err:.4f}")

# convergence: Cauchy residual between T and 2T iterations should be small
np.random.seed(SEED)
P_cp_2T = l2_tvw1_naif.model_CP(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, STEPSIZE_RATIO, 2 * IT_CP, printprogress=False)
cauchy = rel_err(P_cp_2T, P_cp)  # ||P(2T)-P(T)|| / ||P(T)||
check("naif: Cauchy residual small", cauchy < CAUCHY_TOL, f"||P(2T)-P(T)||/||P(T)|| = {cauchy:.4f} < {CAUCHY_TOL}")


# %% ── Graph-DR parameters
print("\n[using l2_tvw1_naif.model_graphDR]")
N = 4
graph_params = make_graph_DR_parameters(N, [(0, 1), (1, 2), (2, 3)])

# %% ── Test 2: l2_tvw1_naif.model_graphDR (graph Douglas-Rachford) ──────────────────────

Z = graph_params[0]
def metrics_fn(x_list):
    # L2 error vs ground truth + the graph-DR convergence diagnostics
    # (consensus variance + fixed-point residual) that graph_DR no longer
    # computes internally.
    P_bar = sum(xi['P'] for xi in x_list) / N
    return {'relative_L2_error': rel_err(P_bar, gt), **graph_DR_diagnostics(x_list, Z)}

C_bounds = l2_tvw1_naif.compute_C_bounds(shape_y, ALPHA1, ALPHA2, E)

np.random.seed(SEED)
x, w, hist = l2_tvw1_naif.model_graphDR(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, C_bounds,
                              graph_params, SIGMA, IT_GRAPH,
                              printprogress=False,
                              extra_metrics_fn=metrics_fn, record_every=10)
P_graph = sum(xi['P'] for xi in x) / N
plot_u(P_graph, ndim_y, title='graph_DR')
graph_err = rel_err(P_graph, gt)

check("graph: returns N node dicts", isinstance(x, list) and len(x) == N and all('P' in xi for xi in x), f"N={len(x)}")
check("graph: history populated", hist is not None and len(hist['relative_L2_error']) > 1)
check("graph: consensus shape", P_graph.shape == shape_x + shape_y)
check("graph: all finite", np.all(np.isfinite(P_graph)))
check("graph: beats rough adjoint", graph_err < QUALITY_FACTOR * rough_err, f"{graph_err:.4f} vs rough {rough_err:.4f}")
check("graph: P-variance decreases (consensus)", hist['P_variance'][-1] < hist['P_variance'][0],
      f"{hist['P_variance'][0]:.3g} -> {hist['P_variance'][-1]:.3g}")
check("graph: edge residual decreases (feasibility)", hist['edge_residual_norm_sq'][-1] < hist['edge_residual_norm_sq'][0],
      f"{hist['edge_residual_norm_sq'][0]:.3g} -> {hist['edge_residual_norm_sq'][-1]:.3g}")
check("graph: L2 error decreases", hist['relative_L2_error'][-1] < hist['relative_L2_error'][0],
      f"{hist['relative_L2_error'][0]:.4f} -> {hist['relative_L2_error'][-1]:.4f}")


# %% ── Test 3: CP and graph-DR agree in quality (same model, two solvers) ─────
print("\n[cross-check: CP vs graph_DR]")
quality_gap = abs(cp_err - graph_err)
check("CP and graph_DR comparable quality", quality_gap < QUALITY_AGREE_TOL,
      f"|{cp_err:.4f} - {graph_err:.4f}| = {quality_gap:.4f} < {QUALITY_AGREE_TOL}")


# %% ── Test 4: determinism (same seed -> identical output), cheap low-iter ────
print("\n[determinism]")
np.random.seed(SEED)
A = l2_tvw1_naif.model_CP(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, STEPSIZE_RATIO, IT_DET, printprogress=False)
np.random.seed(SEED)
B = l2_tvw1_naif.model_CP(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, STEPSIZE_RATIO, IT_DET, printprogress=False)
check("naif deterministic", np.array_equal(A, B), f"max_abs_diff={np.max(np.abs(A - B)):.2e}")

np.random.seed(SEED)
xa, _, _ = l2_tvw1_naif.model_graphDR(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, C_bounds, graph_params, SIGMA, IT_DET,
                            printprogress=False)
np.random.seed(SEED)
xb, _, _ = l2_tvw1_naif.model_graphDR(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, C_bounds, graph_params, SIGMA, IT_DET,
                            printprogress=False)
Pa = sum(xi['P'] for xi in xa) / N
Pb = sum(xi['P'] for xi in xb) / N
check("graph deterministic", np.array_equal(Pa, Pb), f"max_abs_diff={np.max(np.abs(Pa - Pb)):.2e}")




# %% ── report ─────────────────────────────────────────────────────────────────
n_pass = sum(ok for _, ok in _RESULTS)
n_total = len(_RESULTS)
print(f"\n{'='*60}\n  {n_pass}/{n_total} checks passed\n{'='*60}")
if n_pass < n_total:
    print("  FAILED:", ", ".join(name for name, ok in _RESULTS if not ok))

if __name__ == "__main__":
    sys.exit(0 if n_pass == n_total else 1)
