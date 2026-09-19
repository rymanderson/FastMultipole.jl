#!/usr/bin/env bash
#SBATCH --job-name=p021-fgs-replay-gate2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --qos=test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# BRAINSTORM 021 gate-2d: REAL split dual-layout executor (dagteam mode) vs
# rowpar at equal precision and placement — the inclusive schedule comparison
# the spec requires before promoting the split design. Runs under interleave
# binding (task ownership is dynamic, owner first-touch undefined for a
# work-queue schedule). SUBMISSION IS RYAN-GATED. Requires CENSUS/EDGES env
# overrides on orc (default path does not exist there — rev-b lesson).

set -euo pipefail

EVID_DEFAULT="$HOME/projects/FLOWPanel.jl/BRAINSTORM/021_rotor_hover_solver_benchmarks/fgs_r4_followup_evidence_20260914/diag-v15-13694724/j64-b1/results"
CENSUS="${CENSUS:-$EVID_DEFAULT/gemv_census.csv}"
EDGES="${EDGES:-$EVID_DEFAULT/dependency_edges.csv}"
[ -f "$CENSUS" ] && [ -f "$EDGES" ] || { echo "missing census/edges CSVs: $CENSUS / $EDGES"; exit 2; }

module load julia

SWEEPS="${SWEEPS:-12}"
BENCH=benchmark/fgs_sequence_replay.jl
OUT="benchmark/replay_gate2d_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "== topology =="
lscpu | egrep 'Model name|Socket|NUMA|Thread|Core' | tee "$OUT/topology.txt"
numactl --hardware | tee -a "$OUT/topology.txt" || true

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

# 1. serial reference (pricing/consistency with gate-2b)
run serial_f64.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS

# 2. dagteam ladder, F64, interleave placement
for T in 4 8 16 32; do
    run dagteam_f64_interleave_t$T.log $ILV -- -t$T $BENCH --mode dagteam --sweeps $SWEEPS
done
# socket-bound serial-fill control at t16 (placement sensitivity of the schedule)
run dagteam_f64_sock_t16.log $SOCK -- -t16 $BENCH --mode dagteam --sweeps $SWEEPS

# 3. dagteam at reduced precision (equal-precision comparison vs rowpar)
for T in 16 32; do
    run dagteam_f32conv_interleave_t$T.log $ILV -- -t$T $BENCH --mode dagteam --sweeps $SWEEPS --f32
    run dagteam_f32full_interleave_t$T.log $ILV -- -t$T $BENCH --mode dagteam --sweeps $SWEEPS --f32-full
done

# 4. rowpar references in-job at matching precision/placement
run rowpar_f64_interleave_t16.log     $ILV -- -t16 $BENCH --mode rowpar --sweeps $SWEEPS
run rowpar_f32conv_interleave_t16.log $ILV -- -t16 $BENCH --mode rowpar --sweeps $SWEEPS --f32
run rowpar_f32full_interleave_t16.log $ILV -- -t16 $BENCH --mode rowpar --sweeps $SWEEPS --f32-full

echo "== done: results in $OUT =="
