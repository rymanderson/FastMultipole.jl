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

run() { # run <logname> <numactl-args...> -- <julia args...>
    local log="$OUT/$1"; shift
    local pin=()
    while [ "$1" != "--" ]; do pin+=("$1"); shift; done; shift
    echo "--- $log: numactl ${pin[*]:-none} julia $* ---"
    OPENBLAS_NUM_THREADS=1 numactl "${pin[@]}" julia --startup-file=no "$@" \
        --census "$CENSUS" --edges "$EDGES" 2>&1 | tee "$log"
}

# 1. serial baselines, socket-0 pinned (reference: 29.4 GB/s one core, F64)
run serial_f64.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS
run serial_f32.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode serial --sweeps $SWEEPS --f32

# 2. handoff microbenchmark across the ladder (kill switch: < ~5.8 µs avg)
for T in $THREAD_LADDER; do
    run handoff_t$T.log --cpunodebind=0 --membind=0 -- -t$T $BENCH --mode handoff --sweeps 40
done

# 3. rowpar: precision × first-touch × ladder, single socket
for T in $THREAD_LADDER; do
    for FT in serial owner; do
        run rowpar_f64_${FT}_t$T.log --cpunodebind=0 --membind=0 -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --first-touch $FT
        run rowpar_f32_${FT}_t$T.log --cpunodebind=0 --membind=0 -- -t$T $BENCH --mode rowpar --sweeps $SWEEPS --f32 --first-touch $FT
    done
done

# 4. interleave placement control (best ladder point re-run under interleave)
TBEST=$(echo $THREAD_LADDER | awk '{print $NF}')
run rowpar_f64_interleave_t$TBEST.log --interleave=all --cpunodebind=0 -- -t$TBEST $BENCH --mode rowpar --sweeps $SWEEPS
run rowpar_f32_interleave_t$TBEST.log --interleave=all --cpunodebind=0 -- -t$TBEST $BENCH --mode rowpar --sweeps $SWEEPS --f32

# 5. pull-DAG schedule analysis priced with this node's measured numbers:
#    fill in B (serial useful GB/s) and h (handoff µs) from the logs above
B=$(sed -n 's/.*useful BW (min): \([0-9.]*\) GB.*/\1/p' "$OUT/serial_f64.log" | head -1)
H=$(sed -n 's/.*avg handoff = \([0-9.]*\) .*/\1/p' "$OUT/handoff_t$(echo $THREAD_LADDER | awk '{print $1}').log" | head -1)
run dag.log --cpunodebind=0 --membind=0 -- -t1 $BENCH --mode dag --dag-bandwidth "${B:-29.4}" --dag-handoff-us "${H:-3.0}"

echo "== done: results in $OUT =="
