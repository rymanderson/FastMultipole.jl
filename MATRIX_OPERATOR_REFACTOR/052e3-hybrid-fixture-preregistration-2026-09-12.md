# 052e.3 — Hybrid solver fixture: preregistration (2026-09-12)

**Status: LOCKED (Ryan, 2026-09-12). Append-only; any change requires a
new dated supersession note.**

Scope set by Ryan 2026-09-12, superseding the Tier 1/1.5 spec in
`052e-accuracy-plan-v2-draft-2026-09-05.md` for this item: single wing
only; no two-body, no gauge-shift fixture, no unpaired-edge negative
control, no helical Tier 1.5 sweep. Those may return in later items.

## What is being tested

The hybrid solver: the wake-induced potential trace on the body is added
to the RHS of the Dirichlet doublet-strength solve (source strengths
prescribed from freestream + kinematic velocity). Two routes for the
wake trace:

- **Route D (direct):** wake is doublet panels; induced potential on the
  body evaluated directly (panel influence sums).
- **Route GR (Green reconstruction):** sample wake-induced velocity at
  body control points, set $\sigma = -u_w \cdot n$, solve
  $(I-B)q = S\sigma$ in the `:area_mean` gauge via the production
  Householder-reduced path (052e.2b ADOPT); feed the reconstructed trace
  $q$ to the RHS.

Route GR is the only route available for particle wakes; Route D is the
oracle where the wake is panels.

## Fixture (reused from 052e.2a addendum)

Capped NACA0012 rectangular wing, $b = 2.7$ m, $c = 0.76$ m, AOA 30°,
$|U_\infty| = 1$ along $+\hat{x}$; FLOWPanel Dirichlet capped
formulation, `Backslash` dense backend, `build_pitching_wing_body`
constructor. Resolutions L1–L4 = 1,744 / 3,816 / 8,960 / 19,384 panels.
Prescribed flat doublet-panel wake W1, length $20c$, rigid-wake/Kutta
machinery, as in `scripts/addendum_052e2a_realsim_2026-09-11.jl`
(frozen; a new harness script will be written — no edits to frozen
sha-registered scripts).

## Phase 1 — Prescribed panel wake: solve parity (L1–L4)

At each level L1–L4: build wing + W1, solve doublet strengths twice
(Route D and Route GR), identical everything except the wake-trace
route.

**Metrics** (definitions match the addendum's locked stage-1 machinery):

- **M1 — doublet strengths.** Area-weighted relative RMS difference
$$E_\mu = \frac{\mathrm{rmsA}(\mu_{GR} - \mu_D,\, a)}{\mathrm{rmsA}(\mu_D,\, a)},$$
  with $\mathrm{rmsA}(q,a) = \sqrt{\sum a\, q^2 / \sum a}$ and $a$ =
  panel areas. Also report $E_\infty$ (max-norm), all panels and
  TE-excluded. No gauge alignment is applied to $\mu$: the solved
  strengths must agree as-is.
- **M2 — trailing-edge jump circulation.** $\Gamma(y)$ from the TE
  doublet-strength jump; report dy-weighted relative RMS and max of
  $\Gamma_{GR} - \Gamma_D$, normalized by $\mathrm{rms}(\Gamma_D)$.

**Reporting rule (per Ryan 2026-09-12): no pass/fail gates.** Raw values
of M1 and M2 at all four levels are reported unmodified alongside the
registered a priori estimates below; Ryan rules on acceptability.

**Registered a priori estimates (Phase 1).** Tier 0B trace parity
(direct vs reconstructed *trace*, same wake class) was
$E_q = 0.17\%$–$0.57\%$ at L4 with monotone convergence
(`052e2a-tier0b-results-2026-09-07.md`). The solve is a bounded linear
map of the RHS, so:

- **P1:** $E_\mu$ at L4 of order $10^{-3}$–$10^{-2}$ (i.e. comparable to
  or modestly amplified from the trace error), decreasing monotonically
  L1→L4.
- **P2:** $\Gamma(y)$ relative RMS difference at L4 of the same order as
  $E_\mu$ (TE jumps difference two nearby strengths, so some
  amplification above $E_\mu$ is plausible; order $10^{-2}$ at L4 would
  not be surprising).

## Phase 2 — Simulation: lift parity (single resolution, L2)

Three time-stepping simulations of the same wing, identical kinematics,
time step, and step count, differing only in wake representation and
wake-trace route:

- **S1:** doublet-panel wake, Route D (reference).
- **S2:** doublet-panel wake, Route GR.
- **S3:** particle wake (production shedding, `OverlapPPS(2.4, 2)` as in
  the addendum v2 supersession), Route GR.

Resolution: **L2** (3,816 panels). Lift: $C_L(t)$ via the same
Kutta–Joukowski summary used in the addendum harness.

**Comparisons** (all pairwise): S2 vs S1 (reconstruction error, wake
representation held fixed), S3 vs S2 (wake-representation difference,
reconstruction held fixed), S3 vs S1 (total).

**Metrics per pair**, over the settled window = **last 50% of steps**:

$$\Delta_{\mathrm{mean}} = \frac{|\overline{C_L^A} - \overline{C_L^B}|}{|\overline{C_L^B}|}, \qquad
\Delta_{\max} = \frac{\max_t |C_L^A(t) - C_L^B(t)|}{|\overline{C_L^B}|}.$$

Full $C_L(t)$ histories are saved for both diagnostics and figures.

**Reporting rule: no pass/fail gates.** Raw $\Delta_{\mathrm{mean}}$,
$\Delta_{\max}$ for the three pairs reported alongside the estimates
below; Ryan rules.

**Registered a priori estimates (Phase 2).**

- **P3 (S2 vs S1):** same physics, reconstruction error only —
  $\Delta_{\mathrm{mean}}$ of order the L2 Phase-1 $E_\mu$
  (estimate $\lesssim 10^{-2}$; L2 is coarser than L4 so above the
  L4 figure).
- **P4 (S3 vs S2, S3 vs S1):** dominated by the panel-vs-particle wake
  representation difference, not by reconstruction. From the addendum:
  Phase B $E_q$ at L2-adjacent levels was several $\times 10^{-2}$
  (L1→L4: $2.1\times10^{-1}$ → $5.9\times10^{-2}$), and the P1 $\Gamma$
  gap at mid levels was order $10^{-2}$. Estimate lift differences of
  **a few percent (order $2\times10^{-2}$–$6\times10^{-2}$)** at L2.
  This pair measures wake-model fidelity, not hybrid-solver correctness;
  S2 vs S1 is the correctness comparison.

## Run discipline

- Registered runs nohup-detached; logs and canonical numbers
  (`gates.txt` naming convention retained, containing the reported
  metric values) under `data/052e3-phase1/` and `data/052e3-phase2/`.
- Laptop, ≤ 4 threads; formulation-proof tier (no worktree/campaign tag
  required at this tier; any promotion to an official campaign follows
  global worktree + annotated-tag policy).
- New harness script(s) under `scripts/` named
  `tier1_052e3_hybrid_fixture_*.jl`, sha-recorded in the results doc;
  frozen prior scripts untouched.
- FastMultipole branch `flowpanel-20260817`: touch only 052e files.
  FLOWPanel: read AGENTS.md / CLAUDE.md / agent_policies/WORKFLOW.md
  (+ TESTING.md) before touching code or testing.
- Results reported raw, no retuning. Any deviation from this prereg
  requires a dated supersession note before the affected run.

## Ruling

**LOCKED by Ryan, 2026-09-12** ("lock it"). Phases, fixtures, metrics,
a priori estimates (P1–P4), and run discipline registered as above.
No pass/fail gates: raw values reported, Ryan rules on results.
