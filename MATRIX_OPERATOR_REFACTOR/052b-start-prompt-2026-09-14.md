# Starting prompt for 052b (written 2026-09-14, post-052e.3 closure; sequencing updated 2026-09-15)

**SUPERSEDED 2026-09-15 by `052b-start-prompt-2026-09-15.md` (Kutta
item done — isolated, production-affecting scope, legacy restored,
fix uncommitted). Use that file.**

**FIRST MESSAGE TO RYAN (his standing instruction 2026-09-15): remind him
that the extended-revs 1r run is ON HOLD by his direction and he asked to
be reminded of it at this context reset — ask whether to propose specifics
and submit now.**

Board APPROVED by Ryan 2026-09-15. Your mandate, in order:

1. **Do the Kutta item and ONLY the Kutta item this session** (Ryan
   2026-09-15): isolate the regression inside FLOWPanel commit 7fbd68a
   per the "Kutta canonicality" ruling below — find the offending hunk,
   determine SCOPE (does the LEGACY path itself produce different
   strengths post-7fbd68a — production-affecting — or does only the
   :jump-fallback path diverge from legacy — contained?), and restore
   legacy behavior (or, if the fix is invasive, report the isolated
   cause + scope + proposed fix to Ryan before changing anything).
   Local work, ≤4 threads; the bisect worktree recipe is below.
2. **Then PREP FOR A CONTEXT RESET before starting the next task**
   (Ryan 2026-09-15): write a fresh dated handoff/start prompt
   superseding this one, carrying the results of (1), the sequencing
   board below, all cautions, and the 1r reminder.
3. Do NOT start the next board item in this session.

Sequencing board (approved 2026-09-15):
- 052b first: Kutta isolation now (this session); the extended-revs 1r
  run is ON HOLD pending Ryan's go (remind him — see the first-message
  instruction above). Remaining 052b menu: notebook draft
  `notebook-draft-2026-09-04.md`, two unapproved 053-audit defaults.
- Item 058 discriminators next — possibly right after the Kutta issue,
  depending on Ryan's ruling once he sees the isolation result.
- 053 milestone review AFTER 052b.
- Item 059 last, informed by 058.
- 052e: Ryan signaled interest in making the hybrid wake potential
  (GreenReconstruction of the wake's effective potential) PRODUCTION —
  the path is 052e.4 (temporal coherence) then 052e.5 (Stage B +
  promotion package); recorded caveat to fold into promotion evidence:
  gauge multiplier λ grows with refinement (2.9e-2 L1 → 5.5e-2 L4,
  in-sim L2 ≈ 3.7e-2). Not scheduled yet; surface when 052b/053 clear.

Paste the following to a fresh agent:

---

Resume item 052b (multi-rotor IGE GPU extension). Authoritative context:
`MATRIX_OPERATOR_REFACTOR/052b-handoff-prompt-2026-09-04.md` (newest
052b handoff; read it fully) and
`052b-handoff-prompt-2026-09-02.md` (mandate + decision log — append
decisions there). Ignore START_HERE.md's stale 052b row (2026-08-26).

What changed since that handoff was written:
- 052c CLOSED 2026-09-05: trial-2 (expint) PASSED the 1,080-step
  acceptance (job 13592503), all locked gates PASS, expint made the
  default — see `052c-handoff-prompt-2026-09-05b.md`. This resolves the
  "052c commit plan" ruling gate; check which other menu items it
  moots.
- 052e.2b closed ADOPT (2026-09-07) and 052e.3 closed ACCEPTED
  (2026-09-14) — relevant to the "Kutta canonicality" and 052e
  tolerance-ratification flags; see
  `052e-handoff-prompt-2026-09-14b.md` for the 052e state.

Storage: RESOLVED 2026-09-14 — an hpc-storage archiver cycle ran
(Ryan-authorized): /home 197.2G → 138.7G (freed 58.5G; the 09-04 ~618G
figure was already stale pre-cycle), 5 finished p018/p3 runs archived
and verified, no sweep needed, ledger line delivered to Ryan. The
p026_restart_gpu40_s950 anomalies were RESOLVED same day (Ryan-ruled):
the projects_FLOWPanel.jl ARCHIVED-STALE residue was --resume-delete'd
(re-verified byte-for-byte, 27MB freed), and the "second"
wt026gpu_FLOWPanel.jl copy turned out to be a SYMLINK to the
main-checkout run (its VERIFY-FAIL was a race between the two parallel
apply workers re-archiving the same underlying directory) — no second
run exists and no retry is needed. Caution stands: scope archiver
follow-ups by --root, never --only alone, and beware symlinked data/
dirs between checkouts double-counting as distinct runs. Also note
the hpc-storage doc's archive baseline (61.64G/92,998 files) is stale:
actual 3.77T/93.3k files.

Rulings received 2026-09-14:
- 1r gate/config policy: RULED — "we need more revs" (more-revs
  acceptance, not cycle-mean). Plan the extended-revs 1r run; the
  7200 s gate hardcoded at carrier line 203 and CONVERGENCE_REVS
  config will need updating to match — propose specifics to Ryan
  before submitting.
- Kutta canonicality: RULED — "we'll use the legacy default kutta
  condition" (RigidTransitionAttachment + JumpKutta, γ = μ_up − μ_lo,
  c ≡ 0; FLOWPanel_kutta.jl:434-438 `_is_legacy_kutta`). Consequence:
  the :jump-fallback bitwise contract STANDS; the ~half-legacy
  fallback strengths appearing at FLOWPanel commit 7fbd68a are a
  REGRESSION to fix, not a convention to adopt. Note: 7fbd68a's only
  FLOWPanel_kutta.jl hunk is an unrelated particle-body-overlap hook —
  the halving almost certainly enters via the commit's FMM/influence-
  side changes (it's a large cross-pass commit), so the job is to
  isolate the offending hunk within 7fbd68a and restore legacy
  behavior, NOT to update `_kutta_trial!` or relax the contract.
  (Bisect worktree recipe: `git worktree add <path> 7fbd68a^` +
  Pkg.develop-pathed FastMultipole/FLOWVPM, per the 09-04 handoff.)

- 2r solver policy: RESOLVED by staging (Ryan 2026-09-14) — the
  operator-mismatch investigation is now item 058
  (`058-invest-2r-gs-operator-mismatch.md`), and the FMM-on-GPU
  cross-interaction implementation + first-ever FMM-vs-dense benchmark
  is item 059 (`059-impl-gpu-fmm-cross-interactions.md`); both have
  STAGED rows in START_HERE.md. Background: no FMM-vs-dense GPU
  benchmark ever existed because no GPU-FMM cross route exists —
  gsdiag2/gsdiag4 proved the 5.8224e-4 residual plateau is
  dense-operator mismatch, not FMM accuracy.

This session's work is the Kutta isolation ONLY (see the mandate at the
top of this file). The remaining small ruling items (notebook draft
`notebook-draft-2026-09-04.md`; two unapproved 053-audit defaults) are
NOT this session's work — carry them into the next handoff. Mention
them to Ryan only if he asks what's outstanding.

Cautions (from the 09-04 handoff; verify in place):
- Do NOT whole-file-copy the sigma_guard ceil port into the silos (the
  local FLOWVPM_timeintegration.jl has splitting_state divergence the
  silos lack) — minimal hunk only; silo backups are at
  `~/FLOWVPM-052-{h200,gh200}/src/FLOWVPM_timeintegration.jl.bak-preceil`.
- The Kutta :jump regression bisect worktree needs Pkg.develop-pathed
  FastMultipole/FLOWVPM.jl (old Manifests predate `numtype`); the
  regression is pinned to commit 7fbd68a.
- Read `FLOWPanel.jl/agent_policies/HPC.md` (storage archiver pass is
  owed) and FLOWPanel's AGENTS.md/CLAUDE.md/WORKFLOW.md/TESTING.md
  before touching its code. Registered/campaign runs need worktrees or
  a fresh provenance'd silo — the 052e3 silo is deleted.
- ORC ssh quirks: memory file `orc-cluster-access` (bash -lc, banner
  ANSI noise in output, sacct one job at a time, keyboard-interactive
  expiry → Ryan runs `! ssh orc echo ok`, rsync --checksum).
- Both repos are dirty with unrelated work (052e files uncommitted;
  026 GPU-splitting in flight in FLOWPanel/FLOWVPM) — touch only 052b
  files unless Ryan directs otherwise. ≤4 threads locally. Notebook:
  approval before writing; Ryan ticks checkboxes.
