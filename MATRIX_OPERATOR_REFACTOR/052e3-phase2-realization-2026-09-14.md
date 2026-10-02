# 052e.3 Phase 2 — harness-realization choices (RECORDED PRE-LAUNCH, 2026-09-14)

Phase 2 (S1/S2/S3 simulation lift parity at L2) is registered in the LOCKED
prereg `052e3-hybrid-fixture-preregistration-2026-09-12.md` (sims, level,
metrics, pairs, a priori estimates P3/P4, no-gates reporting). The prereg
leaves the numeric kinematics/dt/step-count and wake-realization parameters
open ("identical kinematics, time step, and step count"). This note records
the harness's realization of those open choices BEFORE the registered run.
No locked value is altered.

Harness: `scripts/tier1_052e3_hybrid_fixture_phase2_2026-09-14.jl`
(sha256 recorded in gates.txt at run time). Launcher:
`scripts/tier1_052e3_phase2_orc.slurm.sh` (smoke stage then registered stage,
one sbatch, Phase 1 pattern).

## R1 — simulation parameters (mirror 052e.2a addendum Phase B, its R2)

- Kinematics: static pitched wing, addendum fixture — capped wing, AOA=30°,
  |U∞|=1 along +x, pivot at quarter chord; frames built first
  (`pitching_wing_frame`), then `set_wake_Das!(body, VINF; magnitude=0.05c)`
  (production Das convention, addendum R11). Free-running `simulate!` with a
  no-op maneuver (no wake prescription, no fixed point).
- dt = 0.5·c/U (production `c_per_dt = 0.5`); NSAMP = 61 solve samples
  (t = 0 .. 30·c/U; 60 sheds ≈ 30c wake).
- S1/S2 wake (W1): `PanelWake(body; nwakerows=NSAMP+2=63,
  include_final_filament=false)` — retains all rows; `update_TE!` after
  construction (addendum BW1 pattern).
- S3 wake (W2): `PanelParticleWake(body; nwakerows=4, max_particles=200_000,
  method_trailing=method_unsteady=OverlapPPS(2.4, 2))` — shedding scheme is
  locked by the prereg (addendum v2 supersession value); the row/particle
  budgets are realization choices mirrored from the addendum.
- Solver: `Backslash` + `DirectBackend`, identical across S1/S2/S3 (prereg).
- Level: L2 = (n_airfoil, n_span, n_endcap) = (121, 10, 7), 3,816 panels
  (locked).

## R2 — settled window

"Last 50% of steps" realized as the last NWIN = NSAMP − ⌊NSAMP/2⌋ = 31 of the
61 solve samples (i_step 30..60 inclusive).

## R3 — C_L and metric realization

- C_L(t) per solve sample via the Kutta–Joukowski scalar summary
  CL = 2·Σ Γ_j dy_j /(U·b·c), Γ_j = TE doublet jump (addendum R3 machinery,
  identical across sims; `_get_wakestrength_mu` cross-check recorded).
- Registered metrics, computed exactly as prereg lines 96–97:
  Δ_mean = |mean(C_L^A) − mean(C_L^B)| / |mean(C_L^B)|,
  Δ_max = max_t |C_L^A(t) − C_L^B(t)| / |mean(C_L^B)|,
  over the settled window, pairs (A vs B): S2 vs S1, S3 vs S2, S3 vs S1.
  Unnormalized differences also printed (diagnostic only).
- Full C_L(t) histories saved: `cl_timeseries_L2.csv`, per-sim
  `steps_S*.csv` (i, t, CL, Γ_tot, max|strength|, np).

## R4 — recorded, not gated

- Steadiness: |Γ_tot drift| over the last 5 steps per sim (addendum
  convention).
- GR diagnostics at the final solve of S2 and S3 (residual, gauge defect, λ),
  plus per-step λ history (`grdiag_S*.csv`) — continues the watch on the
  λ-growth observation from Phase 1.
- Kutta cross-check max_j |Γ_j − (μ_i − μ_j)| at the final step.
- Hard failure (abort, not a gate) only on non-finite strengths.

## R5 — execution

- ORC silo `~/052e3_silo` (per the 2026-09-12 supersession HPC terms), Julia
  1.12.5, partition m9, 24 CPU / 64 GB, ONE sbatch: smoke stage (unregistered,
  tiny mesh (41,5,3), NSAMP=8, small particle budget, temp outputs) must
  finish before the registered stage runs. Raw reporting, no retuning.
- Outputs to `MATRIX_OPERATOR_REFACTOR/data/052e3-phase2/` (gates.txt + logs
  + CSVs), rsynced back to the repo after the run.
- Silo code pins: the silo remains at its 2026-09-12 provenance (FLOWPanel
  a9baea0a, FastMultipole ac7230a6, FLOWVPM f51f4ee9), the same code state as
  the Phase 1 registered run. The local checkouts have since gained one
  unrelated commit each (026 GPU-splitting work: FLOWPanel a804a954, FLOWVPM
  edc9d955); these are deliberately NOT pulled into the silo — Phase 2 runs
  on the Phase 1 pins for within-item consistency. Only the three new Phase 2
  files (harness, launcher, this note) are rsynced up. The cluster smoke
  stage validates the harness against the silo's code state.
