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
│   ├── forward.py          KK (undersampled rFFT) and UF (masked FFT)
│   ├── metrics.py          voxelwise W1/W2 (POT), W1_naif, build_cost_matrix
│   └── graphs.py           graph-DR topology: (Z, parent_node, d), λ₁ enumeration
├── models/             the variational models (see table below)
├── data/               test-data generators (side-effect-free)
│   ├── gaussians.py        crossing-fibre mixtures (+ covariance helpers)
│   ├── displacements.py    bars, gammas, stars, spirals
│   ├── masks.py            undersampling masks
│   └── real.py / assets/   Sebastian tensors (needs dt.mat — not in repo)
├── experiments/
│   ├── diffusion/          the active line (00/01/02, each with results/)
│   ├── behavior/           "what does W1-TV do" demos (each with results/)
│   └── archive/            non-reproducible material (see below)
└── papers/                 reference PDFs
```

Top-level `Libraries/` (outside TVW1) stays the general, cross-project toolbox:
the solvers `CP`, `graph_DR`, `DR`, `pDR_Richardson` (`algorithms_general.py`),
the discrete operators (`differential_operators.py`), `prox_and_proj.py`,
`solve_linear_systems.py`, `plottings.py`, …

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
`from algorithms_general import CP`.
Experiments are written as Spyder / VS Code `# %%` cell scripts; run them
cell-by-cell, or as a whole file.

## Models (`models/`)

| Module | Functions | Fidelity | Regularizer | Solver |
|---|---|---|---|---|
| `l2_tvw1_naif.py` | `model_naif`, `model_naif_graph` | ½‖K·−E‖² (undersampled Fourier) | naif unbalanced TV-W1 (`α₁ TV_W1(JP)+α₂ TV(IP)`) | CP / graph-DR |
| `l2_tvpr.py` | `L2_TVPR`, `L2_TVPR_Dirichlet` | ½‖K·−E‖² | Piccoli–Rossi TV-W1 (+ optional Dirichlet smoothing) | CP / pDR |
| `l2_tvpr.py` | `L2`, `L2_2`, `TVPR` | — | — / TV-PR only | CP |
| `w1_tvw1.py` | `model_W1_TVW1` | W1 (balanced, simplex) | TV-W1 | CP |

## Experiments

| Experiment | What it shows | Reproducible? |
|---|---|---|
| `diffusion/00_compare_models` | CP vs graph-DR on the gaussian-mixture toy | ✅ seeded |
| `diffusion/01_sweep_sigma` | graph-DR step-size σ sensitivity | ✅ seeded — set `iterations=30000`, `sigma_values=[0.02,0.1,0.3,1]` to reproduce `results/` |
| `diffusion/02_graph_topology` | base-graph algebraic-connectivity (λ₁) study | ✅ seeded — `RUN_ALL=True` for the all-38 figure |
| `behavior/angle_regularization` | W1-TV on noisy-angle gaussians ≈ TV on the angle | ✅ seeded |
| `behavior/w1_naif_properties` | properties of the naif unbalanced W1 (Dirac / mass creation) | ✅ deterministic (needs POT) |
| `behavior/odf_crossing_fibres` | crossing-fibre ODF from undersampled q-space | ✅ self-contained, seeded (needs scikit-image) |
| `behavior/reconstruct_4d` | W1-TV on bars/rings/mixtures/images (incl. W1-TV ≈ standard TV) | ✅ seed added (was unseeded) |
| `behavior/curve_reconstruction` | curve / video-inpainting **data** | ⚠️ data only — original recon/gif pipeline is in `archive/traffic_video/legacy_experiments.py` |

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

Also note: `data/real.py` (Sebastian tensors) needs `dt.mat` in `data/assets/`,
which is **not in the repo** and is git-ignored.
