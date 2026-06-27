# %% ── bootstrap ──────────────────────────────────────────────────────────────
"""Compare reconstructions of the gaussian-mixture toy by the L2-TVW1 naif model
(Chambolle-Pock vs graph Douglas-Rachford).

Was 'TVW1 for Diffusion/test_models_with_gaussian_mixture.py'.  Changes made
during the reorg:
  * inlined data builders removed -> imported from data/;
  * the missing seed for mask/noise/random init is now set (np.random.seed(0)),
    so this is reproducible;
  * the `from L2_TVW1_basic__model import model_basic` cell was dropped
    (model_basic was an obsolete variant kept only in a Cestino folder).
The reference figures are in ./results/ ; re-running writes fresh figures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401

import numpy as np
np.random.seed(0)  # reproducibility: mask, noise, and random model init

from gaussians import build_gaussian_mixture
from masks import undersampling_mask_xy
from forward import KK_factory, KKstar_factory
from plottings import plot_u

# %% ── build data ─────────────────────────────────────────────────────────────
shape_x = (7, 8)
shape_y = (19, 20)

gaussian_mixture = 100 * build_gaussian_mixture(shape_x, shape_y)
gaussian_mixture[gaussian_mixture < 1e-14] = 0

ndim_x, ndim_y = len(shape_x), len(shape_y)
shape_xy = shape_x + shape_y
ndim_xy = len(shape_xy)

retained_ratio = 0.15
mask_fft = np.fft.ifftshift(undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7))
mask_rfft = mask_fft[tuple(slice(None) if i != ndim_xy - 1 else slice(0, shape_xy[-1] // 2 + 1)
                          for i in range(ndim_xy))]

KK, KKstar = KK_factory(mask_rfft), KKstar_factory(mask_rfft)

E_clean = KK(gaussian_mixture)
SNR = 20
std = np.sqrt(np.sum(np.abs(E_clean) ** 2) / (E_clean.size * SNR))
E = E_clean + np.random.normal(size=E_clean.size, scale=std)

if __name__ == '__main__':
    plot_u(gaussian_mixture, ndim_y, title="gaussian_mixture")
    plot_u(KKstar(E).clip(min=0), ndim_y, f"rough reconstruction, SNR={SNR}, retained_ratio={retained_ratio}")

# %% ── model_naif_graph (graph Douglas-Rachford) ──────────────────────────────
from l2_tvw1_naif import model_naif_graph
from graph_DR_auxiliary_functions import make_graph_DR_parameters

if __name__ == '__main__':
    alpha1, alpha2 = 0.45, 1.4
    iterations = 400
    sigma = 1
    N = 4
    graph_DR_parameters = make_graph_DR_parameters(N, [(0, 1), (1, 2), (2, 3)])

    x, w, _ = model_naif_graph(E, mask_rfft, shape_x, shape_y, alpha1, alpha2,
                               graph_DR_parameters, sigma, iterations)
    Pstar = sum(xi['P'] for xi in x) / N
    plot_u(Pstar, ndim_y, title=f"naif_graph, retained={int(retained_ratio*100)}%, SNR={SNR}, "
                                f"alpha1={alpha1}, alpha2={alpha2}, it={iterations}")

# %% ── model_naif (Chambolle-Pock) ────────────────────────────────────────────
from l2_tvw1_naif import model_naif

if __name__ == '__main__':
    alpha1, alpha2 = 0.45, 1.4
    stepsize_ratio = 0.5
    iterations = 1000

    Pstar = model_naif(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations)
    plot_u(Pstar, ndim_y, title=f"naif (CP), retained={int(retained_ratio*100)}%, SNR={SNR}, "
                                f"alpha1={alpha1}, alpha2={alpha2}, it={iterations}")
