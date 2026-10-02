# Starting prompt for the next 052-lane session (written 2026-09-15b, post-Kutta closure; supersedes 052b-start-prompt-2026-09-14.md and the earlier 2026-09-15 draft of this file)

**FIRST MESSAGE TO RYAN (two standing items):**
1. **1r reminder (his standing instruction 2026-09-15):** the
   extended-revs 1r run is ON HOLD by his direction and he asked to be
   reminded at each context reset — ask whether to propose specifics
   (7200 s gate hardcoded at carrier line 203 + CONVERGENCE_REVS
   config update; ruling was "more revs" acceptance, not cycle-mean)
   and submit now.
2. **Confirm the mandate:** per the approved sequencing board, the
   next task is **item 058** (2r block-GS operator-mismatch
   discriminators). Ryan prepped this handoff for "the next task"
   right after seeing the Kutta isolation result, consistent with the
   board's "058 next" — confirm before submitting anything to HPC.

## Mandate: item 058 discriminators

Read `058-invest-2r-gs-operator-mismatch.md` in full (59 lines + the
2026-09-15 addendum) — it is self-contained: problem (normalized block
residual plateaus at 5.8224e-4 while strength delta hits machine eps),
evidence (gsdiag2/gsdiag4 bit-identical residuals under different FMM
knobs; both operators are the dense GPU route via
`_gpu_rect_influence!`/`_gpu_direct_batch!`), scope, and ordered
discriminators. **Run discriminator 0 first** (added 2026-09-15):
re-run the 2r residual diagnostic on code that includes FLOWPanel
`dd5573c` (the Kutta/phi_ext fix — see below) and see whether the
plateau moves, BEFORE funding discriminators 1-3. Registered-run
discipline applies to HPC evidence runs (worktree/silo + provenance,
campaign tags per house rules); the 052e3 silo is deleted — make a
fresh provenance'd worktree. Combine GPU jobs into one multi-stage
sbatch where sensible (H200 queue waits are long).

## What the 2026-09-15 session closed (Kutta item — DONE)

Full detail: 2026-09-15 decision-log entries in
`052b-handoff-prompt-2026-09-02.md` (§Decision log — append new
decisions THERE). Summary:

- **Root cause isolated**: 7fbd68a's tuple `solve!` redesign reads
  each body's ENTRY potential as an external incident potential
  (phi_ext, matching BackslashCoupled: saved, zeroed, added back into
  the Dirichlet RHS). The legacy VTS simulate!/kutta path enters with
  the previous step's STALE evaluated potential (pre-7fbd68a
  single-body Dirichlet `solve!` zeroed it as workspace) → RHS
  re-added the body's own converged potential → ~2x doublet/wake
  strengths on the legacy default path.
- **Scope = PRODUCTION-AFFECTING (inverts the 09-04 reading)**: the
  LEGACY default path itself changed at 7fbd68a; the :jump fallback
  had kept producing true-legacy values (its committed c=0 solve is
  insulated from the stale entry potential). Every post-7fbd68a VTS
  tuple solve (incl. 2r rotor+ground) ran contaminated.
- **Fix COMMITTED (Ryan-approved): FLOWPanel `dd5573c` on
  `fastmultipole`** — zero `body.potential` in
  `solve_formulation!(::VelocityThroughSources)`'s tuple branch before
  tuple `solve!` (`src/FLOWPanel_formulation.jl`, +9 lines). phi_ext
  semantics preserved for formulations/tests that deliberately supply
  an incident potential. Rejected alternative: zeroing at the solve!
  loop top (breaks the phi_ext design; coupled-oracle test stalls at
  its seeded 0.03).
- **Validated**: 7fbd68a+fix → legacy==fallback bitwise, == parent
  legacy to 1 ulp (cross-commit 1-ulp drift is unrelated reordering;
  within-run bitwise contract intact). HEAD+fix (`dd5573c` on
  `1b59af5`): kutta 658/658 (was 2 FAIL), solver 489/489 (incl.
  coupled oracle), formulation 956/956.
- **NOT yet propagated**: orc checkouts and the 052 silos do NOT have
  `dd5573c`. Propagate before any new 2r evidence run (mandatory for
  discriminator 0). Pre-/post-fix 2r trajectories are NOT comparable —
  never mix them in one comparison.
- Bisect worktrees removed; to reproduce: `git worktree add <path>
  7fbd68a^` (and/or `7fbd68a`) in FLOWPanel.jl, then Pkg.develop the
  local FastMultipole + FLOWVPM.jl paths (old Manifests pin registered
  FastMultipole v2.0.4, predating `numtype`).

## Sequencing board (approved 2026-09-15)

- 052b: Kutta item DONE (committed `dd5573c`). Extended-revs 1r run ON
  HOLD pending Ryan's go (see first-message item 1). Remaining 052b
  menu (mention only if Ryan asks what's outstanding): notebook draft
  `notebook-draft-2026-09-04.md` (insertion journals/20260901.md EOF;
  now also needs a Kutta-isolation section — ask Ryan re verbosity;
  approval BEFORE writing anything in the notebook); two unapproved
  053-audit defaults (FLOWPanel FMM_RADIUS_TOL inflation; FLOWVPM
  RadixFMM expansion_order=6 vs docstring 4).
- **Item 058 — THIS SESSION** (see mandate).
- 053 milestone review AFTER 052b.
- Item 059 last (`059-impl-gpu-fmm-cross-interactions.md`: FMM-on-GPU
  cross-interactions + first-ever FMM-vs-dense benchmark), informed by
  058.
- 052e: Ryan signaled interest in promoting the hybrid wake potential
  (GreenReconstruction) to PRODUCTION — path 052e.4 (temporal
  coherence) then 052e.5 (Stage B + promotion package); promotion
  caveat: gauge multiplier λ grows with refinement (2.9e-2 L1 → 5.5e-2
  L4, in-sim L2 ≈ 3.7e-2). Not scheduled; surface when 052b/053 clear.

## Authoritative context

- `058-invest-2r-gs-operator-mismatch.md` — the task (read fully).
- `052b-handoff-prompt-2026-09-02.md` — mandate + decision log
  (through 2026-09-15); `052b-handoff-prompt-2026-09-04.md` — gsdiag4
  readout, file:line anchors for the dense-route interception
  (FLOWPanel_fmm.jl:60-79, FLOWPanel_gpu_influence.jl:618-650), 022g
  driver/carrier anchors.
- 052c CLOSED (expint default, `052c-handoff-prompt-2026-09-05b.md`);
  052e.2b ADOPT (09-07); 052e.3 ACCEPTED (09-14,
  `052e-handoff-prompt-2026-09-14b.md`).
- Storage RESOLVED 2026-09-14 (/home 138.7G post-archiver). Cautions:
  scope archiver follow-ups by --root never --only alone; symlinked
  data/ dirs can double-count as distinct runs; hpc-storage doc's
  archive baseline is stale (actual 3.77T/93.3k files).

## Cautions (carry forward; verify in place)

- Do NOT whole-file-copy the sigma_guard ceil port into the silos
  (local FLOWVPM_timeintegration.jl has splitting_state divergence the
  silos lack) — minimal hunk only; silo backups at
  `~/FLOWVPM-052-{h200,gh200}/src/FLOWVPM_timeintegration.jl.bak-preceil`.
- Read `FLOWPanel.jl/agent_policies/HPC.md` and FLOWPanel's
  AGENTS.md/CLAUDE.md/WORKFLOW.md/TESTING.md before touching its code.
- ORC ssh quirks: memory file `orc-cluster-access` (bash -lc, banner
  ANSI noise, sacct one job at a time, keyboard-interactive expiry →
  Ryan runs `! ssh orc echo ok`, rsync --checksum). Probe partitions
  with the slurm-availability skill before submitting.
- Both repos dirty with unrelated in-flight work (052e files
  uncommitted in FastMultipole; 026 GPU-splitting in FLOWPanel/FLOWVPM
  — `dd5573c` deliberately excluded those files) — touch only
  052-lane files unless Ryan directs otherwise. ≤4 threads locally.
  Commits/pushes only on Ryan's ask. Notebook: approval before
  writing; Ryan ticks checkboxes.
- Do not disturb other lanes' jobs (fp-018gpu-*, fp-p026ph1-*); verify
  job states directly at turn start, never trust silence.
