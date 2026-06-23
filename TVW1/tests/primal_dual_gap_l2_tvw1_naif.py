# %% ── bootstrap ──────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from gaussians import build_gaussian_mixture
from masks import undersampling_mask_xy
from forward import KK_factory, KKstar_factory
from graph_DR_auxiliary_functions import make_graph_DR_parameters
from l2_tvw1_naif import model_naif, model_naif_graph, primal_dual_gap
from plottings import plot_u


# ── parameters & thresholds (tuned against measured behaviour, with margin) ───
SEED = 0
SCALE = 10          # factor for the scaling of the gaussian mixture
RETAINED = 0.2
CONCENTRATION = 0.7
SNR = 15
ALPHA1, ALPHA2 = 0.01, 0.0001
STEPSIZE_RATIO = 0.5    # CP
SIGMA = 0.3             # (good value from the sigma sweep)
IT_GRAPH = 300000         # iterations

N = 4
graph_params = make_graph_DR_parameters(N, [(0, 1), (1, 2), (2, 3)])

# %% ── build the toymodel  ─────────────────────────────────────────────
shape_x, shape_y = (6, 6), (22, 22)
ndim_x, ndim_y = len(shape_x), len(shape_y)
np.random.seed(SEED)

# gt = SCALE * build_gaussian_mixture(shape_x, shape_y)
# gt[gt < 1e-14] = 0

from displacements import build_u_gammas
P_dagger = build_u_gammas(shape_x[0], shape_y[0], angle=0.2)
plot_u(P_dagger, 2)

ndim_xy = len(shape_x + shape_y)
shape_xy = shape_x + shape_y
mask_fft = np.fft.ifftshift(undersampling_mask_xy(shape_x, shape_y, RETAINED, CONCENTRATION))
mask_rfft = mask_fft[ ..., : (shape_xy[-1]//2 + 1) ]
KK, KKstar = KK_factory(mask_rfft), KKstar_factory(mask_rfft)

E_clean = KK(P_dagger)
# Proper complex Gaussian noise: real and imaginary parts each ~ N(0, std^2/2),
# so total noise power per component = std^2 and SNR = E_energy / (N*std^2).
# The 1/sqrt(2) normalises so that the total power equals std^2 (not 2*std^2).
std = np.sqrt(np.sum(np.abs(E_clean) ** 2) / (E_clean.size * SNR))
noise = np.random.normal(size=E_clean.size, scale=std) \
        + 1j * np.random.normal(size=E_clean.size, scale=std)
E = E_clean + noise / np.sqrt(2)

rough = KKstar(E).clip(min=0)

plot_u(rough, ndim_y, title='backprojection')

prob = dict(shape_x=shape_x, shape_y=shape_y, P_dagger=P_dagger, E=E, mask_rfft=mask_rfft, rough=rough) # keys are automatically characters strings
shape_x, shape_y = prob['shape_x'], prob['shape_y']
P_dagger, E, mask_rfft = prob['P_dagger'], prob['E'], prob['mask_rfft']
print(f"\nProblem: shape_x={shape_x}, shape_y={shape_y}, retained={RETAINED}, SNR={SNR}")

# %% ── Test: surrogate primal-dual gap throughout graph-DR

print("\n[surrogate primal-dual gap]")
E_norm = np.linalg.norm(E)
C_Q = E_norm ** 2 / (2 * ALPHA1)           # bound on ||Q*||_{2,1}
C_P = 2 * E_norm         # bound on ||P*||_inf
C_f = ALPHA1 * np.floor( max(shape_y)+1 )/2 * (1 + np.sqrt(2) / 2)   # bound on ||f*||_inf
RECORD = 50

def gap_fn(x_list):
    n = len(x_list)
    P = sum(xi['P'] for xi in x_list) / n
    Q = sum(xi['Q'] for xi in x_list) / n
    f = sum(xi['f'] for xi in x_list) / n
    g = sum(xi['g'] for xi in x_list) / n
    h = sum(xi['h'] for xi in x_list) / n
    return {'gap': primal_dual_gap(P, Q, f, g, h, E, KK, KKstar, C_P, C_Q, C_f, ALPHA1, ALPHA2, shape_x, shape_y)}

np.random.seed(SEED)
x, _, hist_gap = model_naif_graph(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, C_P, C_Q, C_f,
                                  graph_params, SIGMA, IT_GRAPH,
                                  printprogress=True, return_history=True,
                                  extra_metrics_fn=gap_fn, record_every=RECORD)

P_graph = sum(xi['P'] for xi in x) / N
plot_u(P_graph, ndim_y, title='graph_DR')

gaps = np.array(hist_gap['gap'])
iters = np.arange(len(gaps)) * RECORD

plt.figure()
plt.semilogy(iters, gaps)
plt.xlabel('iteration')
plt.ylabel('primal-dual gap')
plt.title('surrogate primal-dual gap (graph-DR)')
plt.show()

print("gaps all finite:", np.all(np.isfinite(gaps)))
print(f"Initial gap: {gaps[0]:.3g}")
print(f"Final gap: {gaps[-1]:.3g}")