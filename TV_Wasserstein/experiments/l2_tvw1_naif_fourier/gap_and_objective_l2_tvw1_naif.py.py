# %% ── bootstrap ──────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

from data.masks import undersampling_mask_xy
from core.forward import KK_factory, KKstar_factory
# Equivalent: from proximal_algorithms.graph_DR_auxiliary_functions import make_graph_DR_parameters 
import proximal_algorithms.graph_DR_auxiliary_functions

from plottings import plot_u


# ── parameters & thresholds (tuned against measured behaviour, with margin) ───
SEED = 0
SCALE = 10          # factor for the scaling of the gaussian mixture
RETAINED = 0.2
SNR = 15
CONCENTRATION = 0.7
ALPHA1, ALPHA2 = 0.02, 0.0002
STEPSIZE_RATIO = 0.5    # CP
SIGMA = 0.3             # (good value from the sigma sweep)
ITER = 1000000         # iterations
RECORD = max( 5, int(ITER / 20) )

N = 4
graph_params = proximal_algorithms.graph_DR_auxiliary_functions.make_graph_DR_parameters(N, [(0, 1), (1, 2), (2, 3)])

# %% ── build the toymodel  ─────────────────────────────────────────────

shape_x, shape_y = (6, 6), (22, 22)
ndim_x, ndim_y = len(shape_x), len(shape_y)
np.random.seed(SEED)

# from data.gaussians import build_gaussian_mixture
# gt = SCALE * build_gaussian_mixture(shape_x, shape_y)
# gt[gt < 1e-14] = 0

from data.displacements import build_u_gammas
P_dagger = build_u_gammas(shape_x[0], shape_y[0], angle=0.2)
plot_u(P_dagger, 2)

ndim_xy = len(shape_x + shape_y)
shape_xy = shape_x + shape_y
mask_fft = np.fft.ifftshift(undersampling_mask_xy(shape_x, shape_y, RETAINED, CONCENTRATION))
KK, KKstar = KK_factory(mask_fft), KKstar_factory(mask_fft)

E_clean = KK(P_dagger)
# Proper complex Gaussian noise: real and imaginary parts each ~ N(0, std^2/2),
# so total noise power per component = std^2 and SNR = E_energy / (N*std^2).
# The 1/sqrt(2) normalises so that the total power equals std^2 (not 2*std^2).
std = np.sqrt(np.sum(np.abs(E_clean) ** 2) / (E_clean.size * SNR))
noise = np.random.normal(size=E_clean.size, scale=std) \
        + 1j * np.random.normal(size=E_clean.size, scale=std)
E = E_clean + noise / np.sqrt(2)

P_backprojection = KKstar(E).clip(min=0)

plot_u(P_backprojection, ndim_y, title='backprojection')

print(f"\nProblem: shape_x={shape_x}, shape_y={shape_y}, retained={RETAINED}, SNR={SNR}")

print("--------------------------------------------------")
print(">>> Note: computing the objective significantly slows down this script. Turn it off if not needed <<<")
print("--------------------------------------------------")

# %% ── Test: objective function throughout graph-DR

from models.l2_tvw1_naif_fourier.primal_dual_gap import compute_C_bounds
from models.l2_tvw1_naif_fourier.model import model_graphDR, objective
from models.l2_tvw1_naif_fourier.primal_dual_gap import primal_dual_gap

print("\n[This script computes objective function and primal-dual gap]")
C_bounds = compute_C_bounds(shape_x, shape_y, ALPHA1, ALPHA2, E)

def objective_fn(x_list):
    n = len(x_list)
    P = sum(xi['P'] for xi in x_list) / n
    return objective(P, shape_x, shape_y, ALPHA1, ALPHA2, KK, E)

def gap_fn(x_list):
    n = len(x_list)
    P = sum(xi['P'] for xi in x_list) / n
    Q = sum(xi['Q'] for xi in x_list) / n
    f = sum(xi['f'] for xi in x_list) / n
    g = sum(xi['g'] for xi in x_list) / n
    return primal_dual_gap(P, Q, f, g, E, KK, KKstar, C_bounds, ALPHA1, ALPHA2, shape_x, shape_y)

def extra_metrics_gap(x_list):
    n = len(x_list)
    P = sum(xi['P'] for xi in x_list) / n
    Q = sum(xi['Q'] for xi in x_list) / n
    f = sum(xi['f'] for xi in x_list) / n
    g = sum(xi['g'] for xi in x_list) / n
    gap, terms = primal_dual_gap(P, Q, f, g, E, KK, KKstar, C_bounds, ALPHA1, ALPHA2, shape_x, shape_y,
                                 return_terms=True)
    # print("Current gap:", gap)
    return {'gap': gap, **terms}

def extra_metrics_fn(x_list):
    return {'objective': objective_fn(x_list), 'gap': gap_fn(x_list)}

np.random.seed(SEED)
x, _, history = model_graphDR(E, mask_fft, shape_x, shape_y, ALPHA1, ALPHA2, C_bounds,
                                            graph_params, SIGMA, ITER,
                                            printprogress=True,
                                            extra_metrics_fn=extra_metrics_gap, record_every=RECORD)

P_graph = sum(xi['P'] for xi in x) / N
plot_u(P_graph, ndim_y, title='graph_DR')

print("--------------------------------------------------")

if 'objective' in history:

    objectives = np.array(history['objective'])
    iters = np.arange(len(objectives)) * RECORD

    plt.figure()
    plt.semilogy(iters, objectives)
    plt.xlabel('iteration')
    plt.ylabel('objective')
    plt.title('objective function (graph-DR)')
    plt.show()

    print(f"Initial objective: {objectives[0]:.3g}")
    print(f"Final objective: {objectives[-1]:.3g}")

    print("--------------------------------------------------")

if 'gap' in history:

    gaps = np.array(history['gap'])
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

# %% ── gap term breakdown ──────────────────────────────────────────────────────
# The gap is the sum of 7 terms (see primal_dual_gap.primal_terms/dual_terms).
# All are >= 0 except dual_fidelity = 1/2||h||^2 + <E,h>, which can be negative:
# it is plotted as |.| with a dashed line where negative.
'''

'primal_fidelity':      # 1/2 ||KP-E||^2
'primal_flux':          # phi(alpha_1 ||Q||_{2,1})
'primal_coupling':      # C_f ||deltabar(grad_x JP + div_y Q)||_1
'primal_mass_TV':       # alpha_2 ||grad I P||_1

'dual_fidelity':        # 1/2 ||h||^2 + <E,h>
'dual_positivity':      # C_P ||(JJ div f + II* div g - K*h)_+||_1
'dual_transport':       # phi*( ||grad_y f||_{2,inf} / alpha_1 )
'''
TERM_KEYS = ['primal_fidelity', 'primal_flux', 'primal_coupling', 'primal_mass_TV',
             'dual_fidelity', 'dual_positivity', 'dual_transport']

if all(k in history for k in TERM_KEYS):
    iters = np.array(history['iterations_recorded'])
    gaps = np.array(history['gap'])

    plt.figure(figsize=(9, 5.5))
    for key in TERM_KEYS:
        vals = np.array(history[key], dtype=float)
        if np.any(vals < 0):
            plt.semilogy(iters, np.abs(vals), '--', label=f'|{key}|  (<0 where dashed)')
        else:
            plt.semilogy(iters, vals, label=key)
    plt.semilogy(iters, gaps, 'k', linewidth=2.5, label='gap (sum of all)')
    plt.xlabel('iteration')
    plt.ylabel('contribution')
    plt.title('primal-dual gap: term breakdown (graph-DR)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nFinal breakdown (gap = sum):")
    for key in TERM_KEYS:
        val = history[key][-1]
        print(f"  {key:>18}: {val:>12.4g}   ({100*val/gaps[-1]:+.1f}% of gap)")
    print(f"  {'gap':>18}: {gaps[-1]:>12.4g}")

print("--------------------------------------------------")

from my_optimal_transport import TVW1
from standard_TV_denoising import TVL1
from core.tvw1_naif.operators import II, JJ

obj_1_dagger = 0.5 * np.sum(np.abs( KK(P_dagger) - E )**2)
obj_2_dagger = TVW1(JJ(P_dagger, shape_x, shape_y), shape_x, shape_y)
obj_3_dagger = TVL1( II(P_dagger, shape_x, shape_y))

obj_1_backprojection = 0.5 * np.sum(np.abs( KK(P_backprojection) - E )**2)
obj_2_backprojection = TVW1(JJ(P_backprojection, shape_x, shape_y), shape_x, shape_y)
obj_3_backprojection = TVL1( II(P_backprojection, shape_x, shape_y))

obj_1 = 0.5 * np.sum(np.abs( KK(P_graph) - E )**2)
obj_2 = TVW1(JJ(P_graph, shape_x, shape_y), shape_x, shape_y)
obj_3 = TVL1( II(P_graph, shape_x, shape_y))

print( f"P_dagger: fidelity: {obj_1_dagger:.1f}, TVW1: {obj_2_dagger:.1f}, mass_TV: {obj_3_dagger:.1f} --- Objective (with alphas): {obj_1_dagger + ALPHA1*obj_2_dagger + ALPHA2*obj_3_dagger:.1f}" )
print( f"P_backprojection: fidelity: {obj_1_backprojection:.1f}, TVW1: {obj_2_backprojection:.1f}, mass_TV: {obj_3_backprojection:.1f} --- Objective (with alphas): {obj_1_backprojection + ALPHA1*obj_2_backprojection + ALPHA2*obj_3_backprojection:.1f}" )
print( f"P_graph: fidelity: {obj_1:.1f}, TVW1: {obj_2:.1f}, mass_TV: {obj_3:.1f} --- Objective (with alphas): {obj_1 + ALPHA1*obj_2 + ALPHA2*obj_3:.1f}" )

