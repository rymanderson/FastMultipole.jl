#!/usr/bin/env bash
#SBATCH --job-name=p021-fgs-replay-gate2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --qos=test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# BRAINSTORM 021 gate-2: R4 sequence replay on a zen3 node.
# Submit from the top level of the FastMultipole checkout (or campaign
# worktree). Partition/QOS per slurm-availability probe at submission time
# (p021 profile: zen3 512 GiB nodes; qos=test fits the <1 h walltime).
# SUBMISSION IS RYAN-GATED — this script prepares the run, it does not
# choose to run.

set -euo pipefail

# census/edge CSVs (saved R4 evidence in the FLOWPanel checkout)
EVID_DEFAULT="$HOME/projects/FLOWPanel.jl/BRAINSTORM/021_rotor_hover_solver_benchmarks/fgs_r4_followup_evidence_20260914/diag-v15-13694724/j64-b1/results"
CENSUS="${CENSUS:-$EVID_DEFAULT/gemv_census.csv}"
EDGES="${EDGES:-$EVID_DEFAULT/dependency_edges.csv}"
[ -f "$CENSUS" ] && [ -f "$EDGES" ] || { echo "missing census/edges CSVs: $CENSUS / $EDGES"; exit 2; }

module load julia

SWEEPS="${SWEEPS:-12}"
THREAD_LADDER="${THREAD_LADDER:-4 8 16 32}"
BENCH=benchmark/fgs_sequence_replay.jl
OUT="benchmark/replay_gate2_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "== topology =="
lscpu | egrep 'Model name|Socket|NUMA|Thread|Core' | tee "$OUT/topology.txt"
numactl --hardware | tee -a "$OUT/topology.txt" || true

# BINDING (gate-2 rev b): on NPS4 EPYC 7763 each socket is FOUR NUMA nodes
# of 16 CPUs / 2 memory channels. Job 13773451 used --cpunodebind=0
# --membind=0 ("node 0" != "socket 0"), confining every arm to 1/4 socket:
# t32 oversubscribed 33 threads on 16 cores, bandwidth capped at 2 channels,
# and membind=0 nullified the first-touch comparison. Socket 0 = nodes 0-3.
SOCK="--cpunodebind=0-3 --membind=0-3"

run() { # run <logname> <numactl-args...> -- <julia args...>
    local log="$OUT/$1"; local snap="$OUT/numastat_${1%.log}.txt"; shift
    local pin=()
    while [ "$1" != "--" ]; do pin+=("$1"); shift; done; shift
    echo "--- $log: numactl ${pin[*]:-none} julia $* ---"
    OPENBLAS_NUM_THREADS=1 numactl "${pin[@]}" julia --startup-file=no "$@" \
        --census "$CENSUS" --edges "$EDGES" > "$log" 2>&1 &
    local pid=$!
    # verify page placement mid-run rather than assuming it (spec trap)
    ( sleep 60; numastat -p $pid > "$snap" 2>/dev/null || true ) &
    local snapper=$!
    wait $pid
    wait $snapper 2>/dev/null || true
    cat "$log"
}

# 1. serial baselines, 1 core on node 0 (reference: 29.4 GB/s one core, F64;
#    single-node binding is correct for a 1-thread arm)
run serial_f64.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS
run serial_f32.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS --f32

# 2. handoff microbenchmark across the ladder (kill switch: < ~5.8 µs avg)
for T in $THREAD_LADDER; do
    run handoff_t$T.log $SOCK -- -t$T $BENCH --mode handoff --sweeps 40
done

# 3. rowpar: precision × first-touch × ladder, whole socket 0 (64 cores)
for T in $THREAD_LADDER; do
    for FT in serial owner; do
        run rowpar_f64_${FT}_t$T.log $SOCK -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --first-touch $FT
        run rowpar_f32_${FT}_t$T.log $SOCK -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --f32 --first-touch $FT
    done
done

# 4. interleave placement controls across socket-0 nodes, at t16 and t32
for T in 16 32; do
    run rowpar_f64_interleave_t$T.log --interleave=0-3 --cpunodebind=0-3 -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS
    run rowpar_f32_interleave_t$T.log --interleave=0-3 --cpunodebind=0-3 -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --f32
done

# 5. pull-DAG schedule analysis priced with this node's measured numbers:
#    fill in B (serial useful GB/s) and h (handoff µs) from the logs above.
#    NOTE when reading dag output: the list-schedule sim prices W workers at
#    B GB/s EACH with no shared-bandwidth cap; sanity-check its aggregate
#    (W×B) against the measured socket ceiling before believing makespans.
B=$(sed -n 's/.*useful BW (min): \([0-9.]*\) GB.*/\1/p' "$OUT/serial_f64.log" | head -1)
H=$(sed -n 's/.*avg handoff = \([0-9.]*\) .*/\1/p' "$OUT/handoff_t$(echo $THREAD_LADDER | awk '{print $1}').log" | head -1)
run dag.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode dag --dag-bandwidth "${B:-29.4}" --dag-handoff-us "${H:-3.0}"

echo "== done: results in $OUT =="
