# Starting prompt for the next 052/058-lane session (written 2026-09-15b, post-submission of the gsdiag5 + 1r-ext25 wave; supersedes 052b-start-prompt-2026-09-15.md)

**FIRST ACTIONS (before talking to Ryan):** check both jobs directly
(`ssh orc 'bash -lc "squeue -u rander39 -o \"%.10i %.22j %.8T %.10M %R\""'`,
then `sacct -j <id>` one at a time if absent). Never trust silence.

- **13711487 `fp-022g-2r-gsdiag5`** (mgh/GH200, 4 h) — 058 discriminator 0.
- **13711488 `fp-022g-1r-ext25`** (mgh/GH200, 10 h) — 052b extended-revs
  1r acceptance, 25 revs.

Logs: `~/campaigns/052b058-20260915/FLOWPanel.jl/logs/slurm/`. Data:
`~/projects/FLOWPanel.jl/data/p022g_2r_ige_gsdiag5_20260915/` and
`.../p022g_1r_ige_ext25_20260915/` (worktree `data/` is a symlink there).
First attempts 13705003/4 failed on missing untracked `examples/data` mesh
assets (now symlinked into the worktree); if a resubmission failed again,
read the .err log before touching anything.

## Readouts

**gsdiag5 (058 discriminator 0):** grep the .out log for
`normalized block residual` — 120 outer iterations at step 0 print a
per-iteration trajectory, then the job FAILS at
`require_outer_convergence` BY DESIGN (the trajectory is the artifact; job
state FAILED ≠ run failed). Compare the plateau against the pre-fix
baseline **5.8223981764176e-4** (iters 2–120, gsdiag2 job 13568975 =
gsdiag4 job 13582074 bit-identical).
- Plateau unchanged → Kutta fix ruled out; proceed to 058 discriminators
  1–3 in order (see `058-invest-2r-gs-operator-mismatch.md`, read fully).
  Also consider first the cheap probe Ryan hasn't ruled on: dump the
  per-panel residual field and look at its spatial support (TE-adjacent
  rows → Kutta/wake row folding; uniform → self-potential ±μ/2 convention;
  see the 2026-09-15b decision-log entries in
  `052b-handoff-prompt-2026-09-02.md` for the residual definition and
  suspect ranking — the residual is a PHYSICAL max|φ| via influence!
  re-evaluation, a different operator than the factored G).
- Plateau moved/gone → report to Ryan; 058 re-scopes.

**1r-ext25 (052b):** carrier gates: elapsed ≤ 25200 s, device reserve
≥ 20%, metadata TOML checks (all_finite, gs converged, final normalized
residual ≤ 1e-8, no GPU fallback). Convergence verdict in the log tail +
`*_case_metadata.toml`; per-rev CT CSV supports 018-style matched-window
drift analysis offline. Window = final 10 revs (16–25, all hover). If NOT
settled: Ryan's ruling is warmstart extension (driver `RESTART_STEP` /
`simulate_warmstart!`, rotor_hover_ground_effect.jl:1856-1878) — propose
specifics, get his go. Post-fix trajectories are NOT comparable to
pre-fix runs (13568974 etc.) — never mix.

## Campaign layout (registered; do not edit while jobs queued/running)

- Worktrees `~/campaigns/052b058-20260915/{FLOWPanel.jl,FastMultipole,FLOWVPM.jl}`
  pinned by annotated tags `campaign/058-gsdiag5-20260915` +
  `campaign/052b-1r-ext25-20260915` (same commits; pushed to all three
  GitHub origins). FLOWPanel pin b5ffa08 = orc `unified-052` (4e6b5b7) +
  `dd5573c` cherry-pick + carrier commit (accept_ext mode, env gate
  `P022G_CASE_TIME_GATE_S`, `P022G_REPO_OVERRIDE`); worktree branch adds
  data-symlink commit 9fbf42f. FastMultipole 89ede6be, FLOWVPM 6c8cda4
  (exact gsdiag2/4-era pins — single-variable design).
- Envs `~/campaigns/052b058-20260915/env-{aarch64,x86_64}`: Manifest
  dev-paths point at the worktrees.
- Provenance: `058-gsdiag5-provenance-2026-09-15.md`,
  `052b-1r-ext25-provenance-2026-09-15.md` (this repo dir).

## IMPORTANT: local `fastmultipole` branch ≠ campaign lineage

Local FLOWPanel `fastmultipole` has diverged ~1500 lines from orc
`unified-052` (026 in-flight work in wake/warmstart/simulate/solver).
The campaign deliberately did NOT use it. The carrier edit
(accept_ext/gate/REPO_OVERRIDE) sits UNCOMMITTED in the local FLOWPanel
checkout on `fastmultipole` — Ryan hasn't ruled whether to commit it
there (committed copy lives on origin branch `unified-052-camp20260915`).
Ask before committing or discarding. Local FLOWPanel/FLOWVPM repos now
have an `orc` ssh remote (fetch-only use).

## Standing items for Ryan (mention when relevant)

- Notebook draft `notebook-draft-2026-09-04.md` still unwritten to the
  journal (insertion journals/20260901.md EOF); now needs Kutta-isolation
  AND 058/1r-ext25 sections — ask verbosity; approval BEFORE writing.
- Two unapproved 053-audit defaults (FLOWPanel FMM_RADIUS_TOL inflation;
  FLOWVPM RadixFMM expansion_order=6 vs docstring 4).
- Sequencing board (2026-09-15): 058 in flight → 053 milestone review
  after 052b → 059 last → 052e.4/.5 unscheduled.

## Cautions (carry forward)

- Read `FLOWPanel.jl/agent_policies/HPC.md` + AGENTS/WORKFLOW/TESTING
  before touching FLOWPanel code. ORC quirks: memory `orc-cluster-access`
  (bash -lc, banner noise, sacct one job at a time, kbd-interactive expiry
  → Ryan runs `! ssh orc echo ok`). Probe with slurm-availability before
  submitting.
- Both local repos dirty with unrelated in-flight work (052e in
  FastMultipole, 026 in FLOWPanel/FLOWVPM) — touch only 052/058-lane
  files. ≤4 threads locally. Commits/pushes only on Ryan's ask (campaign
  tag/branch pushes on 2026-09-15 were policy-required and are done).
  Notebook: approval before writing; Ryan ticks checkboxes.
- Do not disturb other lanes' jobs (fp-018gpu-*, p021-*, fp-p026ph1-*).
- Registered-run discipline for any NEW HPC evidence (worktree + tags +
  provenance BEFORE submitting). Old 052 silos deprecated — don't submit
  from them.

## Authoritative context

- `058-invest-2r-gs-operator-mismatch.md` — the 058 task (read fully).
- `052b-handoff-prompt-2026-09-02.md` §Decision log — through 2026-09-15b
  (residual definition, suspect ranking, campaign wave entries). Append
  new decisions THERE.
- `052b-handoff-prompt-2026-09-04.md` — gsdiag4 readout, dense-route
  interception anchors (FLOWPanel_fmm.jl:60-79,
  FLOWPanel_gpu_influence.jl:618-650).
- Brainstorm 018 (`FLOWPanel.jl/BRAINSTORM/018_dji9443_hover_convergence_campaign.md`)
  — settledness standard, matched windows, quiet-limit observable (used to
  design the 25-rev run; its 45–60-rev extension rule is the warmstart
  path).
