# 052e.3 supersession note — 2026-09-12 (pre-run, Phase 1 metric addition)

Applies to: `052e3-hybrid-fixture-preregistration-2026-09-12.md` (LOCKED
2026-09-12). This note ADDS measurements; nothing registered is removed
or altered. Recorded before any Phase 1 run (smoke or registered).

## Reason

Route D (`DirectWakePotential`) feeds the wake potential $q_f$ to the
RHS with **no mean removed** (FLOWPanel_formulation.jl:267–303), while
Route GR (`GreenReconstruction(gauge=:area_mean)`) feeds an
**area-mean-gauged** trace. For a closed body, a constant shift $c$ in
the RHS potential maps (up to discretization error) to a per-body
constant shift in the solved doublet strengths, with no effect on the
exterior field or on the TE jump $\Gamma(y)$. The raw (unaligned) M1
metric $E_\mu$ therefore contains a predictable gauge constant
$\approx \langle q_f \rangle_A$ (the area-weighted mean of the direct
wake potential), which is not a solver error.

## Additions (Phase 1 metrics)

- **M1a (added):** constant-aligned strength error — both $\mu$ fields
  area-mean-aligned (per body) before differencing:
$$E_\mu^{\mathrm{al}} = \frac{\mathrm{rmsA}\!\big(\mathrm{align}(\mu_{GR},a) - \mathrm{align}(\mu_D,a)\big)}{\mathrm{rmsA}\!\big(\mathrm{align}(\mu_D,a)\big)}$$
  reported at all levels alongside the raw M1 (which is retained
  unchanged and still reported).
- **M1b (added, diagnostic):** the measured constant offset
  $\langle \mu_D - \mu_{GR} \rangle_A$ vs the predicted gauge offset
  $\pm\langle q_f \rangle_A$ (sign recorded as measured), per level.

## Registered expectation for the addition

- The raw $E_\mu$ is expected to be dominated by the gauge constant and
  should track $|\langle q_f \rangle_A| / \mathrm{rmsA}(\mu_D)$.
- $E_\mu^{\mathrm{al}}$ is the quantity expected to behave per the
  locked P1 estimate (order $10^{-3}$–$10^{-2}$ at L4, monotone).
- M2 ($\Gamma$ metrics) are gauge-invariant and unaffected.

## Harness-realization choices recorded pre-launch (no locked value altered)

- **H1.** Phase 1 runs two comparisons per level: (i) **self-consistent**
  — each route runs the Phase-A-style fixed point (flat prescribed W1,
  wake strengths re-prescribed each step from the previous solve's TE
  jump, as addendum R1) to `TOL_A=1e-8` within `NITER_A=80`, converged
  states compared; (ii) **frozen-wake** — Route GR solved once on Route
  D's converged frozen wake (identical wake input), compared against
  Route D's converged strengths. (ii) realizes the prereg's "identical
  everything except the wake-trace route" literally; (i) compares the
  solvers as they would actually run.
- **H2.** Fixture constants, metrics machinery, Kutta cross-check, and
  CSV/gates.txt conventions reused from the frozen addendum harness
  (read, not edited); new script
  `scripts/tier1_052e3_hybrid_fixture_phase1_2026-09-12.jl`.

---

# Supersession addendum — 2026-09-12 (later same day): HPC execution

Directed by Ryan 2026-09-12 ("cancel any local runs. Let's run these on
HPC"), superseding the prereg's "Laptop, ≤ 4 threads" run-discipline
line for this item:

- The registered laptop Phase 1 run (launched after a passing smoke) was
  **cancelled mid-L1** before producing any gates.txt; its partial log is
  preserved as `data/052e3-phase1/run_2026-09-12_cancelled-local.log`
  and partial CSVs were removed. No registered values were produced or
  seen before cancellation.
- Registered runs execute on the BYU ORC cluster (x86 CPU partition,
  resources recorded in the job script and gates.txt `threads=` line).
  The ≤4-thread limit is local-only policy and does not apply on HPC.
- Code is staged via an rsync **silo** (`~/052e3_silo/` on ORC,
  sibling-layout FLOWPanel.jl / FastMultipole / FLOWVPM.jl copies of the
  live checkouts, rsync `--checksum`), authorized by Ryan for this item
  with deletion required at item close. Worktree+tag campaign treatment
  is waived by that authorization; provenance is instead recorded at
  rsync time (git SHAs + sha256 of tracked diffs per repo) in the silo's
  `silo_provenance.txt`, echoed into gates.txt.
- Harness change before any registered result: `gitinfo` made
  fault-tolerant for silo (no-.git) execution, and gates.txt now appends
  the silo provenance block. No metric, fixture, or route logic touched.
