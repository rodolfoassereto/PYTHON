# %% ── bootstrap ──────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from masks import undersampling_mask_xy
from forward import KK_factory, KKstar_factory
from graph_DR_auxiliary_functions import make_graph_DR_parameters
from l2_tvw1_naif import compute_C_bounds, model_naif, model_naif_graph, primal_dual_gap
from plottings import plot_u


# ── parameters & thresholds (tuned against measured behaviour, with margin) ───
SEED = 0
SCALE = 10          # factor for the scaling of the gaussian mixture
RETAINED = 0.2
SNR = 15
CONCENTRATION = 0.7
ALPHA1, ALPHA2 = 0.05, 0.005
STEPSIZE_RATIO = 0.5    # CP
SIGMA = 0.3             # (good value from the sigma sweep)
IT_GRAPH = 500         # iterations
RECORD = max( 5, int(IT_GRAPH / 400) )

N = 4
graph_params = make_graph_DR_parameters(N, [(0, 1), (1, 2), (2, 3)])

# %% ── build the toymodel  ─────────────────────────────────────────────

shape_x, shape_y = (6, 6), (22, 22)
ndim_x, ndim_y = len(shape_x), len(shape_y)
np.random.seed(SEED)

# from gaussians import build_gaussian_mixture
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

print(f"\nProblem: shape_x={shape_x}, shape_y={shape_y}, retained={RETAINED}, SNR={SNR}")

# %% ── Test: objective function throughout graph-DR

print("\n[objective function]")
C_bounds = compute_C_bounds(shape_y, ALPHA1, ALPHA2, E)

from l2_tvw1_naif import objective_l2_tvw1_naif

def objective_fn(x_list):
    n = len(x_list)
    P = sum(xi['P'] for xi in x_list) / n
    return {'objective': objective_l2_tvw1_naif(P, shape_x, shape_y, ALPHA1, ALPHA2, KK, E)}

np.random.seed(SEED)
x, _, hist_objective = model_naif_graph(E, mask_rfft, shape_x, shape_y, ALPHA1, ALPHA2, C_bounds,
                                  graph_params, SIGMA, IT_GRAPH,
                                  printprogress=True,
                                  extra_metrics_fn=objective_fn, record_every=RECORD)

P_graph = sum(xi['P'] for xi in x) / N
plot_u(P_graph, ndim_y, title='graph_DR')

objectives = np.array(hist_objective['objective'])
iters = np.arange(len(objectives)) * RECORD

plt.figure()
plt.semilogy(iters, objectives)
plt.xlabel('iteration')
plt.ylabel('objective')
plt.title('objective function (graph-DR)')
plt.show()

print(f"Initial objective: {objectives[0]:.3g}")
print(f"Final objective: {objectives[-1]:.3g}")

from my_optimal_transport import TVW1
from standard_TV_denoising import TVL1
from operators import II, JJ

obj_1_dagger = 0.5 * np.sum(np.abs( KK(P_dagger) - E )**2)
obj_2_dagger = TVW1(JJ(P_dagger, shape_x, shape_y), shape_x, shape_y)
obj_3_dagger = TVL1( II(P_dagger, shape_x, shape_y))

obj_1 = 0.5 * np.sum(np.abs( KK(P_graph) - E )**2)
obj_2 = TVW1(JJ(P_graph, shape_x, shape_y), shape_x, shape_y)
obj_3 = TVL1( II(P_graph, shape_x, shape_y))

print( f"P_dagger: fidelity: {obj_1_dagger:2g}, TVW1: {obj_2_dagger:2g}, mass_TV: {obj_3_dagger:2g} --- Objective (with alphas): {obj_1_dagger + ALPHA1*obj_2_dagger + ALPHA2*obj_3_dagger}" )
print( f"P_graph: fidelity: {obj_1:2g}, TVW1: {obj_2:2g}, mass_TV: {obj_3:2g} --- Objective (with alphas): {obj_1 + ALPHA1*obj_2 + ALPHA2*obj_3}" )
