# 052b extended-revs 1r acceptance (25 revs) — registered run provenance (2026-09-15)

**Purpose:** the "more revs" acceptance run (Ryan's 2026-09-14 ruling;
specifics approved 2026-09-15: 25 revs, warmstart-extend later if not
settled). First 1r IGE acceptance on post-Kutta-fix code (`dd5573c`) — NOT
comparable to pre-fix trajectories (job 13568974).

## Pins (annotated tags, pushed to origin in all three repos)

| Repo | Tag | Commit |
|---|---|---|
| FLOWPanel.jl | `campaign/052b-1r-ext25-20260915` | `b5ffa08c60d870415f3d6519235b1f1075817d8c` |
| FastMultipole | `campaign/052b-1r-ext25-20260915` | `89ede6bec404f765005f9a85c513af65053d4aad` |
| FLOWVPM.jl | `campaign/052b-1r-ext25-20260915` | `6c8cda4c449cbc225246007a15ae760de8caefc9` |

Same pin as `campaign/058-gsdiag5-20260915` (one code wave, two registered
runs). See `058-gsdiag5-provenance-2026-09-15.md` for lineage and the shared
worktree/env layout (`~/campaigns/052b058-20260915/`).

## Run configuration

- Case `p022g_1r_ige` (1 rotor + ground disc, h/R=1.5),
  `P022G_MODE=accept_ext`: schedule spinup 1.5 + ramp 1.0 + hold 1.5 +
  withdraw 4.0 + settle 18.5 = 25 acceptance revs, 954 total steps at NT=36.
  Hover onset rev 6.5; convergence window = final `CONVERGENCE_REVS=10`
  complete revs (revs 16–25, entirely in hover; fixes the prior run's
  window-start artifact). Tolerances: mean 0.005, ptp 0.02. Per-rev CT CSV
  saved for 018-style matched-window drift analysis offline.
- Wall-clock gate `P022G_CASE_TIME_GATE_S=25200` (7 h; estimate ~5.4 h from
  job 13568974's step-cost ramp extrapolated to 954 steps),
  `P022G_CONFIRM_ACCEPTANCE=YES`, walltime 10:00:00.
- `P022_RUN_NAME=p022g_1r_ige_ext25_20260915` (prior `data/p022g_1r_ige`
  untouched). Job name `fp-022g-1r-ext25`.
- Partition: mgh (GH200), same as prior 1r accept 13568974.
- Extension path if not settled: driver warmstart (`RESTART_STEP`/
  `simulate_warmstart!`) from the saved state, per Ryan's 2026-09-15 ruling.

## Job

- Job ID: **13711488** (resubmission). First attempt 13705004 FAILED at
  ~30 s: missing untracked mesh asset in the worktree (same cause as
  gsdiag5 13705003 — see `058-gsdiag5-provenance-2026-09-15.md`; fixed by
  symlinking `examples/data` assets from the live checkout). Resubmitted
  2026-09-15 ~16:2x MDT with `P022G_EXISTING_RESULT=preserve`.
