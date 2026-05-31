# Archive — not reproducible as-is

Material kept for reference but which cannot be regenerated without work.

- **alpha_search/** — α-grid search. The scripts use helper names that were
  removed during the reorg (`objective_functions.wass1_4d` / `build_cost_matrix`,
  a `model` alias for the naif model) and were unseeded. Figures in
  `results/alpha_study`.
- **grid_search/** — (kept-ratio, noise, α) grid search with cached
  `output_parallel*.npy`. Same stale-import situation.
- **traffic_video/** — video-inpainting demo. Needs `traffic_highway.mp4`
  (absent from the repo); `legacy_experiments.py` also predates the current
  module layout (`from models import *`, old `build_toymodels` names).
- **misc/** — obsolete `L2_TVW1_basic__*` model/experiments, the old `_CESTINO`
  images, hand-saved `plot-NNN.png` figures, and the inpainting (= Wasserstein
  median) example outputs.

See `../../README.md` for the reproducibility table of the live experiments.
