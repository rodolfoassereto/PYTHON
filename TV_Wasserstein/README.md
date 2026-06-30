# TV_Wasserstein — TV-Wasserstein regularization for diffusion MRI

Numerical experiments around an Optimal-Transport–derived Total-Variation
regularizer (an *anisotropic W1-TV*) for the diffusion-MRI / EAP reconstruction
problem. Each model is a variational problem of the form **fidelity + TV_W1**
(possibly plus a voxelwise smoothing term), recast as a saddle-point problem and
solved with **Chambolle–Pock (CP)** or the **graph Douglas–Rachford (graph-DR)**
method of Bredies–Chenchene–Naldi. The unbalanced W1 comes in a *naif* variant
(mass penalty + W1 of the normalized parts) and a *Piccoli–Rossi* (PR) variant
(Kantorovich–Rubinstein with an extra ∞-norm constraint).

## Layout

```
TV_Wasserstein/
├── paths_rodolfoassereto.py bootstrap: wires sys.path from the repo location
├── core/ project-wide shared building blocks
│ └── forward.py KK_factory (undersampled rFFT)
├── data/ test-data generators (side-effect-free)
│ ├── gaussians.py crossing-fibre mixtures (+ covariance helpers)
│ ├── displacements.py bars, gammas, stars, spirals
│ └── masks.py undersampling masks
├── TVW1_naif/ [main] L2 fidelity + naif unbalanced W1-TV model
│ ├── model_implementations/
│ │ ├── l2_tvw1_naif.py model_CP, model_naif_graph
│ │ ├── operators.py II, IIstar, JJ, JJstar (per-voxel mass / zero-mean)
│ │ └── solve_linear_systems.py linear system inverses for the graph-DR resolvents
│ ├── experiments/ 00 / 01 / 02 + _WIP_w1_naif_properties
│ └── tests/ convergence suite + gap/objective scripts
├── TVW1_balanced/ balanced W1 model (simplex-valued EAP)
│ ├── w1_tvw1.py
│ └── experiments/
├── TVPR/ Piccoli–Rossi TV-W1 variant
│ ├── l2_tvpr.py
│ └── experiments/
├── other/ standalone experiments (no TV-W1 solver)
│ └── Marions_odf_crossing_fibres/
└── markdowns/ notes and write-ups (see below)
```

`Libraries/` lives **outside** `TV_Wasserstein/` (a sibling under `PYTHON/`) and
is added to `sys.path` by the bootstrap. It holds the general, cross-project
toolbox: `differential_operators.py`, `prox_and_proj.py`, `plottings.py`, …, and
the `proximal_algorithms/` package:

```
proximal_algorithms/
├── init.py
├── CP.py — Chambolle-Pock
├── DR.py — Douglas-Rachford
├── pDR.py — preconditioned Douglas-Rachford (Richardson)
├── graph_DR.py — graph Douglas-Rachford (Bredies, Chenchene & Naldi 2022)
├── graph_DR_auxiliary_functions.py — topology helpers: (Z, parent_node, d), λ₁ enumeration, graph_DR_diagnostics
└── utils_for_dict.py — add_dicts, subtract_dicts, scalar_multiply_dict, inner_product_dicts
```

## Running things

No install needed. Every script begins with a 3-line bootstrap that finds the
repo from the current directory and wires `sys.path`:

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents]
if (p / "paths_rodolfoassereto.py").is_file())))
import paths_rodolfoassereto # noqa: F401 — wires Libraries + core + TVW1_naif subfolders


After that, flat imports resolve from anywhere:
`from gaussians import build_gaussian_mixture`,
`from l2_tvw1_naif import model_CP`,
`from forward import KK_factory`,
`from proximal_algorithms.CP import CP`.

Experiments are written as Spyder / VS Code `# %%` cell scripts; run them
cell-by-cell or as a whole file.

## Models

| Module | Functions | Fidelity | Regularizer | Solver |
| --- | --- | --- | --- | --- |
| `TVW1_naif/model_implementations/l2_tvw1_naif.py` | `model_CP`, `model_naif_graph` | ½‖K·−E‖² (undersampled Fourier) | naif unbalanced TV-W1 (`α₁ TV_W1(JP) + α₂ TV(IP)`) | CP / graph-DR |
| `TVPR/l2_tvpr.py` | `L2_TVPR`, `L2_TVPR_Dirichlet`, `L2`, `L2_2`, `TVPR` | ½‖K·−E‖² | Piccoli–Rossi TV-W1 (+ optional Dirichlet smoothing) | CP / pDR |
| `TVW1_balanced/w1_tvw1.py` | `model_W1_TVW1` | W1 (balanced, simplex) | TV-W1 | CP |

## Experiments

Folder-name convention: `<purpose>__<model>__<algorithm>`. The algorithm tag is
`CP` (Chambolle–Pock), `graph_DR` (graph Douglas–Rachford), or both — present only
when an iterative TV-W1 solver is used. Folders prefixed `_WIP_` are
work-in-progress and may not be fully reproducible.

| Experiment (folder) | Model | Algorithm | What it shows | Reproducible? |
| --- | --- | --- | --- | --- |
| `TVW1_naif/experiments/00_compare_models__L2-TVW1-naif__CP+graph_DR` | L2-TVW1 naif | CP + graph-DR | CP vs graph-DR on the gaussian-mixture toy | ✅ seeded |
| `TVW1_naif/experiments/01_sweep_sigma__L2-TVW1-naif__graph_DR` | L2-TVW1 naif | graph-DR | graph-DR step-size σ sensitivity | ✅ seeded — set `iterations=30000`, `sigma_values=[0.02,0.1,0.3,1]` to reproduce `results/` |
| `TVW1_naif/experiments/02_graph_topology__L2-TVW1-naif__graph_DR` | L2-TVW1 naif | graph-DR | base-graph algebraic-connectivity (λ₁) study | ✅ seeded — `RUN_ALL=True` for the all-38 figure |
| `TVW1_naif/experiments/_WIP_w1_naif_properties` | W1-naif (distance) | — (direct OT) | properties of the naif unbalanced W1 (Dirac / mass creation) | WIP |
| `TVW1_balanced/experiments/angle_regularization_w1_tvw1__CP` | W1-TVW1 | CP | W1-TV on noisy-angle gaussians, bars, images (incl. W1-TV ≈ standard TV) | ✅ seeded |
| `TVW1_balanced/experiments/_WIP_curve_reconstruction` | — (data only) | — | curve / video-inpainting data | WIP |
| `TVPR/experiments/angle_regularization__L2-TVPR__CP` | L2-TVPR (Dirichlet) | CP | W1-TV on noisy-angle gaussians ≈ TV on the angle | ✅ seeded |
| `other/Marions_odf_crossing_fibres` | — (zero-filled IFFT) | — | crossing-fibre ODF from undersampled q-space | ✅ deterministic (needs scikit-image) |

Each reproducible experiment keeps its reference figures in its own `results/`.
Re-running a script writes fresh figures next to it.

## Tests (`TVW1_naif/tests/`)

`other_convergence_tests_l2_tvw1_naif.py` — automated correctness suite for
`l2_tvw1_naif.py`, seeded gaussian-mixture toy (`shape_x=(6,6)`, `shape_y=(22,22)`,
RETAINED=0.2, SNR=15). Four check groups:

- **CP (`model_CP`)** output is finite and non-negative; rel-L2 error beats the
  zero-filled adjoint by factor 0.85; Cauchy residual `‖P(2T)−P(T)‖/‖P(T)‖ < 0.12`.
- **graph-DR (`model_naif_graph`)** return type, shape, finiteness; beats rough
  adjoint; P-variance, edge-residual norm², and rel-L2 error all decrease.
- **Cross-solver agreement** — `|err_CP − err_graph| < 0.15` in relative-L2 units.
- **Determinism** — same global seed → bit-identical output for both solvers.

`gap_and_objective_l2_tvw1_naif.py.py` — exploratory scripts (not part of the
automated suite) that plot the surrogate primal–dual gap and the objective along
the graph-DR iterates, using the `compute_C_bounds` / `C_bounds` API.

The surrogate bounds are built once via `compute_C_bounds(shape_y, alpha1, alpha2, E)`
(returns the tuple `C_bounds = (C_P, C_Q, C_f)`) and passed to `model_naif_graph`;
graph-DR diagnostics (`P_variance`, `edge_residual_norm_sq`) are recorded via
`extra_metrics_fn=graph_DR_diagnostics`.

## Markdowns (`markdowns/`)

- `1a - TVW1 naif for diffusion.md` — project write-up: motivation, the
  anisotropic-W1-TV model for EAP reconstruction, and results/figures.
- `1b - Primal-dual gap computation.md` — exact primal / dual / saddle-point
  formulation of the L2-TVW1-naif model; why the raw gap is infinite at the
  iterates (three indicator blow-ups); and the *surrogate* construction
  (reverse-Huber `φ_M`) that replaces them.
- `1c - Surrogate problem.md` — full derivation of the surrogate primal and dual,
  bounds on `C_P`, `C_Q`, `C_f`, and the conjugate of `φ_M`.
- `2a - Graph_DR_short.md` — concise summary of the graph Douglas–Rachford method
  and the λ₁ topology study.
- `2b - graphDR for TVW1 naif.md` — optimality system, four-block splitting, and
  resolvent derivations for each block; linear-system inversions for the
  `𝕀`/`𝕁` resolvents and the fidelity prox.

## Known issues (`TVW1_naif/model_implementations/l2_tvw1_naif.py`)

- **CP `g`-update** (`model_CP`, line 62): `proj_L_infty_ball(u['g'] / alpha2)`
  clips `g` to the *unit* ball instead of the α₂-ball, and divides by zero when
  `alpha2=0`.  Fix: `proj_L_infty_ball(u['g'], alpha2)`.
- **`dual()` divides by `alpha_1`** (line 211): `s = … / alpha_1` is `NaN` when
  `alpha1=0`; the whole `phi_surrogate_conjugate` term should be skipped in that case.
- **`J_0` no-op clip** (line 125): `f.clip(max=C_f)` discards its return value, so
  the upper bound on `f` is never enforced by the solver (only by the gap's own
  re-clipping).  Fix: `f = f.clip(-C_f, C_f)`.