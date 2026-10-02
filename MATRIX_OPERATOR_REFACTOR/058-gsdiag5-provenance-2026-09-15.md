# 058 discriminator 0 — gsdiag5 registered run provenance (2026-09-15)

**Purpose:** re-run the 2r block-GS residual diagnostic (gsdiag pattern) on
code including the FLOWPanel Kutta/phi_ext fix `dd5573c`, and compare the
normalized-block-residual plateau against the pre-fix registered baseline
(gsdiag2 job 13568975 / gsdiag4 job 13582074, plateau 5.8223981764176e-4,
iterations 2–120). Single-variable design: identical FastMultipole/FLOWVPM
pins and identical FMM/GS knobs to gsdiag2; the ONLY code change vs the
gsdiag2/4 baseline is the `dd5573c` cherry-pick (+ behavior-neutral carrier
additions: accept_ext mode, env-overridable time gate, P022G_REPO_OVERRIDE).

## Pins (annotated tags, pushed to origin in all three repos)

| Repo | Tag | Commit |
|---|---|---|
| FLOWPanel.jl | `campaign/058-gsdiag5-20260915` | `b5ffa08c60d870415f3d6519235b1f1075817d8c` |
| FastMultipole | `campaign/058-gsdiag5-20260915` | `89ede6bec404f765005f9a85c513af65053d4aad` |
| FLOWVPM.jl | `campaign/058-gsdiag5-20260915` | `6c8cda4c449cbc225246007a15ae760de8caefc9` |

FLOWPanel pin lineage: orc `unified-052` (`4e6b5b7`) + cherry-pick of
`dd5573c` (as `1593dcd`, `src/FLOWPanel_formulation.jl` +9 lines) + carrier
commit `b5ffa08` (`examples/run_rotor_multi_ground_effect_gpu.slurm.sh`
only). Branch `unified-052-camp20260915` on origin. FastMultipole/FLOWVPM
pins are exactly the gsdiag2/4-era orc `unified-052` states.

## Execution environment (orc)

- Campaign worktrees: `~/campaigns/052b058-20260915/{FLOWPanel.jl,FastMultipole,FLOWVPM.jl}`
  (FLOWPanel worktree branch `campaign-058-gsdiag5-20260915-wt` adds one
  commit `9fbf42f`: `data/` → symlink to `~/projects/FLOWPanel.jl/data`).
- Julia env: `~/campaigns/052b058-20260915/env-aarch64` — copy of
  `~/projects/envs/aarch64` with the three Manifest dev-paths repointed at
  the campaign worktrees. Julia 1.11.7 (ARM tarball), depot
  `~/fm052depot-gh200`.
- Partition: mgh (GH200, `--qos=gpu --gres=gpu:gh200:1 -C arm`), same as
  gsdiag2/4.

## Run configuration

- Case `p022g_2r_ige` (2 rotors + ground disc, h/R=1.5, spacing 2.7R),
  `P022G_MODE=smoke` (step-0 diagnostic; dies at
  `require_outer_convergence` — expected, the artifact is the per-iteration
  residual trajectory in the slurm log).
- Env: `GS_VERBOSE=true GS_MAX_OUTER=120` (gsdiag2 baseline), FMM knobs at
  carrier defaults (body 17/0.7/109, wake 16/0.6/38),
  `FLOWPANEL_GPU_INFLUENCE=cuda`.
- `P022_RUN_NAME=p022g_2r_ige_gsdiag5_20260915` (prior `data/p022g_2r_ige`
  untouched).
- Walltime 04:00:00, job name `fp-022g-2r-gsdiag5`.

## Job

- Job ID: **13711487** (resubmission). First attempt 13705003 FAILED at
  ~90 s: untracked mesh asset
  `examples/data/dji9443_20260725_45_185_capped_captess4.msh` absent from
  the tag-pinned worktree. Fix: symlinked the 27 files present in the live
  checkout's `examples/data/` but not in the worktree (read-only input
  assets; not part of the code pin). Resubmitted 2026-09-15 ~16:2x MDT with
  `P022G_EXISTING_RESULT=preserve` (failed attempt may have created the
  run dir).
- Readout: plateau of `normalized block residual` from the slurm log
  (`solve!` per-iteration print), compared against 5.8223981764176e-4.
  Decision rule: plateau unchanged → Kutta fix ruled out, proceed to
  discriminators 1–3; plateau moved/gone → 058 re-scoped.
