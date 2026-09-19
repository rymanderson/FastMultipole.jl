#!/usr/bin/env bash
#SBATCH --job-name=p021-fgs-replay-gate2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --qos=test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# BRAINSTORM 021 gate-2c: full-F32 source-major arms (Ryan's question: does
# dropping the convert-on-load cost — F32 storage AND state — beat convert?
# Speed only; full-F32 numerics must separately pass the independent
# evaluator). Reruns the best convert arms in-job for a same-node reference.
# SUBMISSION IS RYAN-GATED. Requires CENSUS/EDGES env overrides on orc
# (the driver default path does not exist there — rev-b lesson).

set -euo pipefail

EVID_DEFAULT="$HOME/projects/FLOWPanel.jl/BRAINSTORM/021_rotor_hover_solver_benchmarks/fgs_r4_followup_evidence_20260914/diag-v15-13694724/j64-b1/results"
CENSUS="${CENSUS:-$EVID_DEFAULT/gemv_census.csv}"
EDGES="${EDGES:-$EVID_DEFAULT/dependency_edges.csv}"
[ -f "$CENSUS" ] && [ -f "$EDGES" ] || { echo "missing census/edges CSVs: $CENSUS / $EDGES"; exit 2; }

module load julia

SWEEPS="${SWEEPS:-12}"
BENCH=benchmark/fgs_sequence_replay.jl
OUT="benchmark/replay_gate2c_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "== topology =="
lscpu | egrep 'Model name|Socket|NUMA|Thread|Core' | tee "$OUT/topology.txt"
numactl --hardware | tee -a "$OUT/topology.txt" || true

# NPS4 EPYC 7763: socket 0 = NUMA nodes 0-3 (rev-b lesson: node 0 != socket 0)
SOCK="--cpunodebind=0-3 --membind=0-3"
ILV="--interleave=0-3 --cpunodebind=0-3"

run() { # run <logname> <numactl-args...> -- <julia args...>
    local log="$OUT/$1"; local snap="$OUT/numastat_${1%.log}.txt"; shift
    local pin=()
    while [ "$1" != "--" ]; do pin+=("$1"); shift; done; shift
    echo "--- $log: numactl ${pin[*]:-none} julia $* ---"
    OPENBLAS_NUM_THREADS=1 numactl "${pin[@]}" julia --startup-file=no "$@" \
        --census "$CENSUS" --edges "$EDGES" > "$log" 2>&1 &
    local pid=$!
    # rev-b's single 60 s snapshot raced process exit (all files empty);
    # poll and keep the last live snapshot instead
    ( while kill -0 $pid 2>/dev/null; do
          if numastat -p $pid > "$snap.tmp" 2>/dev/null && grep -q Total "$snap.tmp"; then
              mv "$snap.tmp" "$snap"
          fi
          sleep 15
      done; rm -f "$snap.tmp" ) &
    local snapper=$!
    wait $pid
    wait $snapper 2>/dev/null || true
    cat "$log"
}

# 1. serial 1-core baselines, all three precisions (node-0 binding correct at t1)
run serial_f64.log     --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS
run serial_f32conv.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS --f32
run serial_f32full.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS --f32-full

# 2. handoff sanity at the champion team size
run handoff_t16.log $SOCK -- -t16 $BENCH --mode handoff --sweeps 40

# 3. full-F32 rowpar ladder, owner + interleave placements
for T in 4 8 16 32; do
    run rowpar_f32full_owner_t$T.log $SOCK -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --f32-full --first-touch owner
done
for T in 16 32; do
    run rowpar_f32full_interleave_t$T.log $ILV -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --f32-full
done

# 4. convert-mode references in-job (rev-b champions, same node/session)
run rowpar_f32conv_owner_t16.log      $SOCK -- -t16 $BENCH --mode rowpar --sweeps $SWEEPS --f32 --first-touch owner
run rowpar_f32conv_interleave_t16.log $ILV  -- -t16 $BENCH --mode rowpar --sweeps $SWEEPS --f32
run rowpar_f32conv_interleave_t32.log $ILV  -- -t32 $BENCH --mode rowpar --sweeps $SWEEPS --f32

echo "== done: results in $OUT =="
