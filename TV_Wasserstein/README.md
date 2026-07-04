# TV_Wasserstein

Optimal-Transport Total-Variation regularization (**anisotropic W1-TV**) for diffusion-MRI / EAP reconstruction. Each model is a variational problem **fidelity + TV_W1**, recast as a saddle point and solved with **Chambolle–Pock (CP)** or **graph Douglas–Rachford (graph-DR)**. The unbalanced W1 comes in a *naif* and a *Piccoli–Rossi (PR)* variant.

## Layout

```
core/         shared blocks: forward.py (K), tvw1_naif/operators.py, auxiliary_functions.py
data/         test-data generators: gaussians, displacements, masks
models/       the variational models (see table)
experiments/  # %% cell scripts; reference figures live in results/
others/       standalone experiments (Marion's ODF crossing fibres)
markdowns/    notes and derivations
```

`Libraries/` (a sibling of the repo) holds the cross-project toolbox: `differential_operators`,`prox_and_proj`, `plottings`, `my_optimal_transport`, and the `proximal_algorithms/` package (CP, DR, graph_DR, graph_DR_auxiliary_functions, …).

## Models

| Module                                 | Functions                      | Regularizer           | Solver        |
| -------------------------------------- | ------------------------------ | --------------------- | ------------- |
| `models/l2_tvw1_naif_fourier/model.py` | `model_CP`, `model_graphDR`    | naif unbalanced TV-W1 | CP / graph-DR |
| `models/l2_tvpr_fourier/model.py`      | `L2_TVPR`, `L2_TVPR_Dirichlet` | Piccoli–Rossi TV-W1   | CP / pDR      |
| `models/w1_tvw1_balanced/model.py`     | `model_CP`                     | TV-W1 (simplex)       | CP            |

Common fidelity: ½‖K·P − E‖² (undersampled Fourier). Gap/bounds diagnostics for the naif model live in `models/l2_tvw1_naif_fourier/primal_dual_gap.py` (`compute_C_bounds`, `primal_dual_gap`).

## Imports

Two roots must be on `sys.path`: the repo and its sibling `Libraries/`. Run
`_add_folders_to_path_machine_specific.py` once per kernel, then:

- **intra-project → absolute**: `from core.forward import KK_factory`,
  `from models.l2_tvw1_naif_fourier.model import model_CP`
- **from Libraries → flat**: `from differential_operators import nabla_x`,
  `from proximal_algorithms.CP import CP`
