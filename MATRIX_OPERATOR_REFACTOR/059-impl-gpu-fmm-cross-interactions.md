# 059 — GPU FMM for cross-interactions (heterogeneous multi-system fmm!)

**STAGED 2026-09-14 (Ryan-directed):** promoted from the "eventual goal"
mentions (050 "option A" / placeholder "phase Q") to its own item. No prior
item defined this scope.

## Problem / motivation

All GPU cross-interactions (rotor↔rotor cross-influence, wake↔panels,
GS residual assembly) currently run through dense rectangular kernels
(`_gpu_rect_influence!` / `_gpu_direct_batch!`), which scale as
O(N_targets × N_sources). The 050 scoping adopted this (option B') as the
least-invasive step, with the multi-system radix generalization — a unified
`fmm!` with heterogeneous sources/targets on GPU ("option A") — recorded as
an eventual goal (050-theory-panel-multisystem-scoping.md:140-145, user
direction 2026-08-22). 052b's budget question re-raises it: at 4r IGE the
per-pair rectangular passes multiply (~5 passes each way) and the dense
wake→bodies kernel alone can exceed the 20 s/step budget
(052b-impl-multirotor-ige-gpu.md:120-124).

## Scope

1. **Feasibility/design:** lift the radix v1 `targets===sources` restriction
   (translate_batched_resident.jl:1735-1743; `target_bodies` aliased at
   translate_batched_cuda.jl:5447) to support distinct target sets — the
   option-A shape from 050. Reuse the 052c-tuned plateau triple where
   applicable; one shared source tree per step (052b's "tree reuse" lever).
2. **Implement** rectangular (source-system → target-system) FMM evaluation
   on GPU for the cross passes that are currently dense.
3. **Benchmark** — the comparison that has never existed: FMM-on-GPU vs the
   dense GPU route for cross-influence and GS residual assembly, at 052b
   operating points (1r/2r/4r, OGE/IGE): wall time per pass, accuracy vs a
   CPU DirectBackend reference, and effect on the block-GS residual floor
   (ties into 058 — if the mismatch is dense-operator-specific, an FMM
   cross-operator changes it).

## Non-goals

- Root-causing the 2r residual plateau (item 058; runs first or in parallel —
  a known dense-operator defect would change this item's baseline).
- Multi-GPU scaling (later phase).

## Gates

None locked at staging. Expected shape: pass-by-pass parity vs CPU reference
at a pinned operating point, then same-job A/B timing vs dense with a
promotion threshold — specifics to Ryan before any registered run.

## Blocking

`050` (approved; option-A design constraints recorded there), `052c`
(closed; supplies the tuned FMM triple + device-resident radix route),
058 findings (soft — see Non-goals). Serves `052b` Phase B budget.
