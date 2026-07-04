Item 1 — bootstrap migration
Several experiment run.py files still carry the old header that searches for a folder called TVW1 and does import paths. That predates the rename to TV_Wasserstein / paths_rodolfoassereto.py, so those headers should be swapped for the new snippet (the one in "Running things"). Affected: the three scripts under TVW1_naif/experiments/, plus the TVPR/ and TVW1_balanced/ experiment scripts. The other/Marions_odf_crossing_fibres/run.py script already uses the new bootstrap.

Item 2 — known issues in TVW1_naif/model_implementations/l2_tvw1_naif.py
CP g-update (model_CP, line 62): proj_L_infty_ball(u['g'] / alpha2) clips g to the unit ball instead of the α₂-ball, and divides by zero when alpha2=0. Fix: proj_L_infty_ball(u['g'], alpha2).
dual() divides by alpha_1 (line 211): s = … / alpha_1 is NaN when alpha1=0; the whole phi_surrogate_conjugate term should be skipped in that case.
J_0 no-op clip (line 125): f.clip(max=C_f) discards its return value, so the upper bound on f is never enforced by the solver (only by the gap's own re-clipping). Fix: f = f.clip(-C_f, C_f).
One note worth flagging: in the source, the CP entry point is named model_CP (the docstring says it "should be renamed model_naif_CP" and notes it doesn't enforce P ≥ 0 the way model_naif_graph does). I documented it as model_CP to match the actual code — let me know if you'd rather the README use the intended model_naif_CP name instead.

TVPR functions seems wrong