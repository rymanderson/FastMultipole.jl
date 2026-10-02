# 058 — Investigate the 2r block-Gauss-Seidel operator mismatch

**STAGED 2026-09-14 (Ryan-directed):** promoted from an unfunded menu item
inside the 052b handoffs to its own item. No prior item covered it.

## Problem

The two-rotor (2r) block Gauss-Seidel solve on the GPU dense route shows a
normalized block residual that plateaus at **5.8224e-4** from iteration 2
through 120, while the strength delta contracts to machine epsilon (~2e-15 by
iteration 6). The residual-assembly operator and the block-solve operator
disagree at the ~5.8e-4 level; root cause unidentified.

## Evidence (registered, 052b lane)

- gsdiag2 (job 13568975; CPU-FMM knobs P=17, MAC=0.7) and gsdiag4 (job
  13582074; P=20, MAC=0.5) produce **bit-identical (16-digit) residuals**
  despite different FMM knobs — the plateau has nothing to do with FMM
  accuracy (052b-handoff-prompt-2026-09-04.md:22-35;
  052b-handoff-prompt-2026-09-02.md:127-141, incl. the 2026-09-04 decision-log
  entries).
- Mechanism: under `FLOWPANEL_GPU_INFLUENCE=cuda`, GS cross-influence AND
  residual assembly are intercepted by `_gpu_rect_influence!` /
  `_gpu_direct_batch!` (FLOWPanel_fmm.jl:60-79,
  FLOWPanel_gpu_influence.jl:618-650) and never touch `fmm!` — so both sides
  of the mismatch are the dense GPU operators. "Pay for FMM accuracy" is
  STRUCK as a lever for this route (Ryan, 052b-handoff-prompt-2026-09-04.md:85-86).
- There is NO FMM-on-GPU route to compare against; that is item 059's scope.

## Scope

Root-cause the ~5.8e-4 operator mismatch: identify which term differs between
residual assembly and the block-solve operator (wake contribution, Kutta/wake
rows, self vs cross blocks, precision/accumulation order, panel self-solve S
matrix vs assembled residual, etc.), and either fix it or characterize it as
an expected discretization/formulation gap with a written argument.

First discriminators (cheap, from the 09-04 handoff):
1. Vary WAKE fmm knobs (wake knobs were identical 16/0.6 across
   gsdiag2/gsdiag4) — if the residual moves, the wake term is implicated.
2. A step-0 no-wake 2r case — rules the wake in/out entirely.
3. If wake is ruled out: assemble both operators' matvecs on a tiny 2r case
   on CPU (DirectBackend) and diff term-by-term.

**Addendum 2026-09-15 (Kutta-isolation session finding — new discriminator
0, cheapest, run FIRST):** all registered gsdiag evidence above was gathered
on post-7fbd68a code whose VTS tuple solve carried a stale-entry-potential
(phi_ext) contamination in every Dirichlet RHS — fixed in FLOWPanel
`dd5573c` (see the 2026-09-15 decision-log entries in
`052b-handoff-prompt-2026-09-02.md`). Within one tuple solve! the solve and
residual pass shared the same contaminated phi_ext, so the plateau is not
automatically explained — but the converged solution (and possibly the
plateau value) changes post-fix. Discriminator 0: re-run the 2r residual
diagnostic (gsdiag pattern) on code including `dd5573c` and compare the
plateau (5.8224e-4, iterations 2-120) before investing in 1-3. Note the
orc checkouts/silos did NOT have the fix as of 2026-09-15 — propagate
first, and treat pre-/post-fix 2r trajectories as non-comparable. Structural
suspect to keep in view for (3): in the block solve the self block is the
assembled/factored G (wake rows folded per formulation), while residual
assembly reconstructs the self contribution through influence!/
`_gpu_direct_batch!` — any G-vs-influence! self-block convention gap
(self-potential, Kutta/wake rows, core size) shows up as exactly this kind
of solution-independent residual floor.

## Non-goals

- Implementing FMM on GPU for cross-interactions (item 059).
- Changing solver defaults or tolerances before the root cause is known.

## Gates

None locked; investigation item. Findings + proposed fix (if any) go to Ryan
for ruling. Registered-run discipline applies to any HPC evidence runs
(worktree/silo + provenance).

## Blocking / context

Uses the 052b evidence trail and the 022g 2r driver; no hard dependency.
Serves 052b Phase B+ (multi-body block-GS solve correctness).
