# 052e handoff — 2026-09-14b (context reset; supersedes 052e3-handoff-prompt-2026-09-14.md)

## Where things stand

**052e.3 is CLOSED — ACCEPTED, both phases (Ryan 2026-09-14).**

- Phase 1 (solve parity GR vs D, prescribed 20c wake, L1–L4): ORC job
  13663782, accepted. Results: `052e3-phase1-results-2026-09-12.md`;
  data `data/052e3-phase1/`.
- Phase 2 (simulation lift parity S1/S2/S3 at L2, free-running): ORC
  job 13687690 (m9-16-2, 24 CPU, 7m46s, Julia 1.12.5), smoke+registered
  one sbatch. Registered Δ_mean/Δ_max (settled window = last 31/61,
  normalized by |mean CL_ref|): S2–S1 1.14e-2/1.14e-2 (P3 est ≲1e-2),
  S3–S2 1.25e-3/1.58e-3 and S3–S1 1.26e-2/1.29e-2 (P4 est 2e-2–6e-2).
  Key finding: panel-vs-particle wake representation effect (S3–S2) an
  order below estimate; GR reconstruction offset dominates. Verifier
  reproduced all metrics from `cl_timeseries_L2.csv` exactly. Ruling +
  provenance: `052e3-phase2-results-2026-09-14.md`; realization record
  `052e3-phase2-realization-2026-09-14.md`; data `data/052e3-phase2/`
  (gates.txt canonical). Harness now sha-registered (848f4b3f59c2 —
  do not edit): `scripts/tier1_052e3_hybrid_fixture_phase2_2026-09-14.jl`.
- The ORC silo `~/052e3_silo` was DELETED on closure (Ryan's
  authorization condition). Phase 2 ran on the silo's unchanged
  2026-09-12 pins; the unrelated local 026 GPU-splitting commits
  (FLOWPanel a804a954, FLOWVPM edc9d955) were deliberately not pulled in.
- Notebook: the `# 20260914` entry in
  `~/Dropbox/research/notebooks/journals/20260901.md` was unified
  in-place (Ryan-directed) to cover both phases; checkbox NOT ticked
  (Ryan ticks).

## What remains TO BE DONE across 052 subitems

Closed and off the table: 052 (parent), 052a, 052c, 052d, 052e.0,
052e.2a, 052e.2b, 052e.3.

Open work:

1. **052b — multi-rotor IGE GPU extension: the only open non-052e
   item.** Phase A.1 complete/verified 2026-08-26; Phase A.2+ was
   blocked on 052 acceptance, which 052c satisfied (closed 2026-09-05)
   — so 052b is actionable now. See its newest handoff.
2. **052e.1 — code-health prereqs / regression completion:** blocked on
   052b closing. Not started.
3. **052e.4 — temporal coherence (Tier 2, translating ring):**
   dependencies (.2a/.2b) both closed — actionable. Not started. See
   `052e-impl-hybrid-wake-potential-experimental.md:122-123`.
4. **052e.5 — Stage B + promotion package:** gated on .2b–.4 (now only
   .4 outstanding). Not started.
5. **052e.6 — global gauge recovery design study:** parallel track, no
   dependency block. Not started. Motivating observation carried
   forward: GR compatibility multiplier λ grows with refinement
   (2.9e-2 L1 → 5.5e-2 L4; in-sim L2 ≈ 3.7e-2 in Phase 2) — recorded,
   not gated.
6. **Housekeeping (small):**
   - Owed notebook entries — OFFER, don't write (approval + verbosity
     first): Tier 0B, Tier 0B-R, production integration, addendum lock.
   - Offer to copy the 2026-09-10 diagnostic scripts out of the
     volatile /tmp scratchpad (052e-handoff-prompt-2026-09-12b.md item 3).
   - Nothing committed in either repo; when Ryan asks, stage/commit
     only 052e files (now including the Phase 2 harness/launcher,
     realization + results docs, and data/052e3-phase2/).
   - External review returns only if issues arise.

## Cautions (carried forward)

- LOCKED preregs append-only; deviations need a dated supersession note
  BEFORE the affected run. Registered runs: raw reporting, no retuning.
- Frozen sha-registered scripts (tier0b, tier0br, both addendum
  harnesses, Phase 1 and Phase 2 052e.3 harnesses): do not edit.
- FastMultipole (flowpanel-20260817) heavily dirty with unrelated work —
  touch only 052e files. FLOWPanel dirty with unrelated work — leave
  alone; read its AGENTS.md / CLAUDE.md / agent_policies/WORKFLOW.md
  (+ TESTING.md) before touching its code.
- Never read `data/**` CSVs or binaries directly; summarize by script
  (gates.txt fine). Delegate runs to `julia-test-runner`, doc questions
  to `refactor-docs-librarian`.
- ≤4 threads locally; limit does not apply on HPC.
- Notebook: approval before writing; append-only across days; Ryan
  ticks checkboxes.
- ORC ssh quirks (bash -lc, banner ANSI noise pollutes command output —
  strip it, sacct one job at a time, keyboard-interactive expiry →
  Ryan runs `! ssh orc echo ok`, rsync --checksum): memory file
  `orc-cluster-access`. Campaign/registered HPC runs need a fresh silo
  or worktree with provenance — the 052e3 silo is gone.
