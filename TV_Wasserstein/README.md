# TVW1 — TV-Wasserstein regularization for diffusion MRI

Numerical experiments around an Optimal-Transport–derived Total-Variation
regularizer (an *anisotropic W1-TV*) for the diffusion-MRI / EAP reconstruction
problem. Each model is a variational problem of the form **fidelity + TV_W1**
(possibly plus a voxelwise smoothing term), recast as a saddle-point problem and
solved with **Chambolle–Pock (CP)** or the **graph Douglas–Rachford (graph-DR)**
method of Bredies–Chenchene–Naldi. The unbalanced W1 comes in a *naif* variant
(mass penalty + W1 of the normalized parts) and a *Piccoli–Rossi / Hanin* (PR)
variant (Kantorovich–Rubinstein with an extra ∞-norm constraint).

## Layout

```
TVW1/
├── paths.py            bootstrap: wires sys.path from the repo location
├── core/               TVW1-specific shared building blocks
│   ├── operators.py        II, JJ, PP  (per-voxel mass / zero-mean projections)
│   ├── forward.py          KK_factory (undersampled rFFT) + KK (masked full FFT, legacy)
│   └── metrics.py          voxelwise W1/W2 (POT), W1_naif, build_cost_matrix
├── models/             the variational models (see table below)
├── data/               test-data generators (side-effect-free)
│   ├── gaussians.py        crossing-fibre mixtures (+ covariance helpers)
│   ├── displacements.py    bars, gammas, stars, spirals
│   └── masks.py            undersampling masks
├── experiments/
│   ├── diffusion/          the active line (00/01/02, each with results/)
│   ├── behavior/           "what does W1-TV do" demos (each with results/)
│   └── archive/            non-reproducible material (see below)
├── tests/                  correctness suite + exploratory gap script (see below)
└── markdowns/              notes and write-ups in Markdown (see below)
```

Top-level `Libraries/` (outside TVW1) stays the general, cross-project toolbox:
the discrete operators (`differential_operators.py`), `prox_and_proj.py`,
`solve_linear_systems.py`, `plottings.py`, …, and the `proximal_algorithms/`
package containing the solvers:

```
proximal_algorithms/
├── __init__.py
├── CP.py                          — Chambolle-Pock
├── DR.py                          — Douglas-Rachford
├── pDR.py                         — preconditioned Douglas-Rachford (Richardson)
├── graph_DR.py                    — graph Douglas-Rachford (Bredies, Chenchene & Naldi 2022)
├── graph_DR_auxiliary_functions.py — topology helpers: (Z, parent_node, d), λ₁ enumeration, graph_DR_diagnostics
└── utils_for_dict.py              — add_dicts, subtract_dicts, scalar_multiply_dict, inner_product_dicts
```

## Running things

No install needed. Every script begins with a 3-line bootstrap that finds the
repo from the current directory and wires `sys.path` (this replaces the old
hardcoded `C:\Users\...\My Drive\PYHTON\...` lines):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "TVW1").is_dir()) / "TVW1"))
import paths  # noqa: F401  — wires Libraries + core + models + data
```

After that, flat imports resolve from anywhere:
`from gaussians import build_gaussian_mixture`,
`from l2_tvw1_naif import model_naif`,
`from proximal_algorithms.CP import CP`.
Experiments are written as Spyder / VS Code `# %%` cell scripts; run them
cell-by-cell, or as a whole file.

## Models (`models/`)

| Module            | Functions                        | Fidelity                        | Regularizer                                          | Solver        |
| ----------------- | -------------------------------- | ------------------------------- | ---------------------------------------------------- | ------------- |
| `l2_tvw1_naif.py` | `model_naif`, `model_naif_graph` | ½‖K·−E‖² (undersampled Fourier) | naif unbalanced TV-W1 (`α₁ TV_W1(JP)+α₂ TV(IP)`)     | CP / graph-DR |
| `l2_tvpr.py`      | `L2_TVPR`, `L2_TVPR_Dirichlet`   | ½‖K·−E‖²                        | Piccoli–Rossi TV-W1 (+ optional Dirichlet smoothing) | CP / pDR      |
| `l2_tvpr.py`      | `L2`, `L2_2`, `TVPR`             | —                               | — / TV-PR only                                       | CP            |
| `w1_tvw1.py`      | `model_W1_TVW1`                  | W1 (balanced, simplex)          | TV-W1                                                | CP            |

## Experiments

Folder-name convention: `<purpose>__<model>__<algorithm>`. The algorithm tag is
`CP` (Chambolle–Pock), `graph_DR` (graph Douglas–Rachford), or both — present only
when an iterative TV-W1 solver is used. The three folders with no model/algorithm
tag (`w1_naif_properties`, `odf_crossing_fibres`, `curve_reconstruction`) use no
such solver.

| Experiment (folder)                                      | Model                | Algorithm     | What it shows                                                   | Reproducible?                                                                              |
| -------------------------------------------------------- | -------------------- | ------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `diffusion/00_compare_models__L2-TVW1-naif__CP+graph_DR` | L2-TVW1 naif         | CP + graph_DR | CP vs graph-DR on the gaussian-mixture toy                      | ✅ seeded                                                                                   |
| `diffusion/01_sweep_sigma__L2-TVW1-naif__graph_DR`       | L2-TVW1 naif         | graph_DR      | graph-DR step-size σ sensitivity                                | ✅ seeded — set `iterations=30000`, `sigma_values=[0.02,0.1,0.3,1]` to reproduce `results/` |
| `diffusion/02_graph_topology__L2-TVW1-naif__graph_DR`    | L2-TVW1 naif         | graph_DR      | base-graph algebraic-connectivity (λ₁) study                    | ✅ seeded — `RUN_ALL=True` for the all-38 figure                                            |
| `behavior/angle_regularization__L2-TVPR__CP`             | L2-TVPR (Dirichlet)  | CP            | W1-TV on noisy-angle gaussians ≈ TV on the angle                | ✅ seeded                                                                                   |
| `behavior/reconstruct_4d__W1-TVW1__CP`                   | W1-TVW1              | CP            | W1-TV on bars/rings/mixtures/images (incl. W1-TV ≈ standard TV) | ✅ seed added (was unseeded)                                                                |
| `behavior/w1_naif_properties`                            | W1-naif (distance)   | — (direct OT) | properties of the naif unbalanced W1 (Dirac / mass creation)    | ✅ deterministic (needs POT)                                                                |
| `behavior/odf_crossing_fibres`                           | — (zero-filled IFFT) | —             | crossing-fibre ODF from undersampled q-space                    | ✅ self-contained, seeded (needs scikit-image)                                              |
| `behavior/curve_reconstruction`                          | — (data only)        | —             | curve / video-inpainting **data**                               | ⚠️ data only — recon/gif pipeline in `archive/traffic_video/legacy_experiments.py`         |

Each reproducible experiment keeps its reference figures in its own `results/`.
Re-running a script writes fresh figures next to it.

## Reproducibility & missing data

`experiments/archive/` holds material that **cannot** be regenerated as-is:

- `traffic_video/` — needs `traffic_highway.mp4`, which is **not in the repo**;
  the script (`legacy_experiments.py`) also predates the current module layout.
- `alpha_search/`, `grid_search/` — depend on removed helper names
  (`wass1_4d`, `build_cost_matrix` in the old `objective_functions`, a `model`
  alias) and were unseeded. Results kept for reference only.
- `misc/` — obsolete models (`L2_TVW1_basic__*`), the old `_CESTINO` images,
  hand-saved `plot-NNN.png` figures, inpainting demos.

There is currently no real-data loader in the tree: `data/assets/` is empty and
the former `data/real.py` (Sebastian tensors from `dt.mat`) placeholder has been
removed. Add it back here if/when real diffusion data is wired in.

## Tests

`tests/test_l2_tvw1_naif.py` — automated correctness suite for `models/l2_tvw1_naif.py`,
seeded gaussian-mixture toy (`shape_x=(6,6)`, `shape_y=(22,22)`, RETAINED=0.2, SNR=15).
Four check groups:

- **CP (`model_naif`)** output is finite and non-negative; rel-L2 error beats the
  zero-filled adjoint by factor 0.85; Cauchy residual `‖P(2T)−P(T)‖/‖P(T)‖ < 0.12`.
- **graph-DR (`model_naif_graph`)** return type, shape, finiteness; beats rough
  adjoint; P-variance, edge-residual norm², and rel-L2 error all decrease.
- **Cross-solver agreement** — `|err_CP − err_graph| < 0.15` in relative-L2 units.
- **Determinism** — same global seed → bit-identical output for both solvers.

Run as a script (exits non-zero on failure) or cell-by-cell. The surrogate bounds
are built once via `compute_C_bounds(shape_y, alpha1, alpha2, E)` (returns the tuple
`C_bounds = (C_P, C_Q, C_f)`) and passed to `model_naif_graph`; the graph-DR
diagnostics (`P_variance`, `edge_residual_norm_sq`) are recorded by passing
`graph_DR_diagnostics` through `extra_metrics_fn`.

`tests/primal_dual_gap_l2_tvw1_naif.py` and `tests/objective_function_l2_tvw1_naif.py`
— separate exploratory scripts (not part of the automated suite) that plot the
surrogate primal–dual gap / the objective along the graph-DR iterates, using the
same `compute_C_bounds` / `C_bounds` API.

## Markdowns (`markdowns/`)

- `1a - TVW1 naif for diffusion.md` — the project write-up: motivation, the
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

## Known issues (`models/l2_tvw1_naif.py`)

**Fixed:**
- `primal_dual_gap`: P was projected with `proj_simplex(P, lam=C_P)`, forcing
  `sum(P) = C_P` (~60) while the true mass is ~1750.  This made the gap plateau at
  ~10⁵ instead of converging.  Fix: `P = P.clip(0, C_P)` (the box that matches the
  dual term `C_P ‖ξ₊‖₁` and the `J_3` clip in the solver).

**Open:**
- **CP `g`-update** (`model_naif`, line 62): `proj_L_infty_ball(u['g'] / alpha2)`
  clips `g` to the *unit* ball instead of the α₂-ball, and divides by zero when
  `alpha2=0`.  Fix: `proj_L_infty_ball(u['g'], alpha2)`.
- **`dual()` divides by `alpha_1`** (line 211): `s = … / alpha_1` is `NaN` when
  `alpha1=0`; the whole `phi_surrogate_conjugate` term should be skipped in that case.
- **`J_0` no-op clip** (line 125): `f.clip(max=C_f)` discards its return value, so
  the upper bound on `f` is never enforced by the solver (only by the gap's own
  re-clipping).  Fix: `f = f.clip(-C_f, C_f)`.
