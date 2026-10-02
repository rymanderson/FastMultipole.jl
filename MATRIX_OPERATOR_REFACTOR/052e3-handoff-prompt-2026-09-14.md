# 052e.3 handoff — 2026-09-14 (context reset; supersedes 052e-handoff-prompt-2026-09-12b.md)

## Where the item stands

**052e.3 Phase 1 is CLOSED — ACCEPTED (Ryan 2026-09-14).** Scope was
simplified by Ryan (2026-09-12) from the old Tier 1/1.5 plan to: single
wing, (Phase 1) solve parity `GreenReconstruction(:area_mean)` (GR) vs
`DirectWakePotential` (D) with a prescribed 20c flat panel wake, then
(Phase 2) simulation lift parity. Phase 1 registered run = ORC job
13663782 (m9-15-1, 24 CPU, 1h46m, Julia 1.12.5): aligned E_mu at L4
4.5e-3 (self-consistent SC) / 2.3e-3 (frozen-wake FZ), Gamma rms 6.6e-3
/ 2.1e-3, |dCL|/CL 0.65% / 0.20%, all monotone L1→L4, ~first order;
raw (unaligned) E_mu plateaus at ~0.12 = the predicted gauge constant
<q_f>_A (matches to 3 digits — D removes no mean from q_f, GR
area-means its trace). All within the registered a priori estimates.
Notebook entry appended under `# 20260914` in
`~/Dropbox/research/notebooks/journals/20260901.md` (checkbox NOT
ticked; Ryan ticks).

## Authoritative documents (all in MATRIX_OPERATOR_REFACTOR/)

- Prereg (LOCKED 2026-09-12): `052e3-hybrid-fixture-preregistration-2026-09-12.md`
  — Phase 2 design is ALREADY REGISTERED there (see below).
- Supersessions (pre-run, same file): `052e3-hybrid-fixture-supersession-2026-09-12.md`
  — M1a/M1b metric additions + HPC/silo execution terms.
- Phase 1 results (ruled ACCEPTED): `052e3-phase1-results-2026-09-12.md`.
- Data: `data/052e3-phase1/` (`gates.txt` = canonical numbers; a
  cancelled laptop run's log is preserved there, it produced no values).
- Harness (now sha-registered via gates.txt `f30c78d7b510` — do not
  edit; Phase 2 gets its own script):
  `scripts/tier1_052e3_hybrid_fixture_phase1_2026-09-12.jl`.
- Slurm launcher: `scripts/tier1_052e3_phase1_orc.slurm.sh` (adapt a
  copy for Phase 2).

## Next action: Phase 2 (registered in the LOCKED prereg — no new prereg needed)

Three time-stepping simulations, same wing, L2 (3,816 panels), identical
kinematics/dt/steps, `simulate!` free-running (not the fixed point):

- S1 doublet-panel wake + `DirectWakePotential` (reference)
- S2 doublet-panel wake + `GreenReconstruction(:area_mean)`
- S3 particle wake (production shedding, `OverlapPPS(2.4,2)`) + GR

Metrics over settled window = last 50% of steps, all three pairs:
$\Delta_{mean}$ and $\Delta_{max}$ of $C_L(t)$ (Kutta–Joukowski CL, as
Phase 1 / addendum R3). No pass/fail gates: report raw values next to
the registered a priori estimates (P3: S2-vs-S1 ≲1e-2; P4: S3 pairs
2e-2–6e-2), Ryan rules. Outputs under `data/052e3-phase2/` with
gates.txt + logs. Sim parameters not locked by the prereg (step count,
wake rows, max particles): mirror the 052e.2a addendum Phase B choices
(NSAMP=61, dt=0.5c/U, W1 rows=NSAMP+2 no final filament, W2 nwakerows=4,
max_particles=200_000) and RECORD them as harness-realization choices
pre-launch. Reuse Phase 1's telemetry/CL machinery; new script
`scripts/tier1_052e3_hybrid_fixture_phase2_2026-09-14.jl` (or dated as
written).

## HPC execution (Ryan-directed 2026-09-12; supersedes laptop discipline for this item)

- Silo lives at `orc:~/052e3_silo/` (sibling FLOWPanel.jl /
  FastMultipole / FLOWVPM.jl rsync copies, no .git; provenance in
  `silo_provenance.txt`, echoed into gates.txt by the harness's
  gitinfo fallback). Env: `--project=$HOME/052e3_silo/FLOWPanel.jl`,
  Julia 1.12.5 via `~/.juliaup/bin/julia +1.12.5`; already instantiated.
- rsync any changed/new files up with `--checksum` (stale-file quirk);
  after runs, rsync `data/052e3-phase2/` back into the repo.
- Partition m9 (plenty idle), 24 CPU / 64 GB / tight time limit worked
  for Phase 1; smoke stage then registered stage in ONE sbatch (see the
  Phase 1 launcher). ssh quirks (bash -lc, banner glue, sacct one job at
  a time, keyboard-interactive expiry → Ryan runs `! ssh orc echo ok`):
  memory file `orc-cluster-access`.
- **DELETE the silo when 052e.3 closes** (condition of Ryan's
  authorization): `ssh orc 'rm -rf ~/052e3_silo'` — after results are
  rsynced back and Ryan has ruled.
- If any Phase 2 sim setup differs from what the silo holds, re-rsync
  and update `silo_provenance.txt` (regenerate, don't hand-edit).

## Later queue (carried forward)

- Offer (do not write) owed notebook entries: Tier 0B, Tier 0B-R,
  production integration, addendum lock (approval + verbosity first).
- Offer to copy the 2026-09-10 diagnostic scripts out of the volatile
  /tmp scratchpad (see 052e-handoff-prompt-2026-09-12b.md item 3).
- Nothing is committed in either repo — when Ryan asks, stage/commit
  only 052e files.
- 052e.1 regression completion (blocked on 052b); 052e.6 gauge design
  study; external-review returns only if issues arise.

## Cautions (carried forward)

- LOCKED prereg append-only; deviations need a dated supersession note
  BEFORE the affected run. Registered runs: raw reporting, no retuning.
- Frozen sha-registered scripts (tier0b, tier0br, both addendum
  harnesses, and now the Phase 1 harness): do not edit.
- FastMultipole (flowpanel-20260817) heavily dirty with unrelated work —
  touch only 052e files. FLOWPanel dirty with unrelated BRAINSTORM
  files — leave alone; read its AGENTS.md / CLAUDE.md /
  agent_policies/WORKFLOW.md (+ TESTING.md) before touching its code.
- Never read `data/**` CSVs or binary data directly; summarize by
  script (gates.txt is fine to read). Delegate runs to
  `julia-test-runner`, doc questions to `refactor-docs-librarian`.
- ≤4 threads locally; the limit does not apply on HPC.
- Notebook: approval before writing anything; append-only; Ryan ticks
  checkboxes.
- Recorded-not-gated observation to keep an eye on: GR compatibility
  multiplier λ grows with refinement (2.9e-2 L1 → 5.5e-2 L4).
