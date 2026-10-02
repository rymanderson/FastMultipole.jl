#!/bin/bash
# 052e.3 Phase 2 — ORC CPU job (smoke stage, then registered stage)
# Prereg: 052e3-hybrid-fixture-preregistration-2026-09-12.md (Phase 2 registered)
# Realization record (pre-launch): 052e3-phase2-realization-2026-09-14.md
# Submit from silo: sbatch $HOME/052e3_silo/FastMultipole/MATRIX_OPERATOR_REFACTOR/scripts/tier1_052e3_phase2_orc.slurm.sh
#SBATCH --job-name=052e3p2
#SBATCH --partition=m9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
SILO=$HOME/052e3_silo
JULIA="$HOME/.juliaup/bin/julia +1.12.5"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SILO/FastMultipole
SCRIPT=MATRIX_OPERATOR_REFACTOR/scripts/tier1_052e3_hybrid_fixture_phase2_2026-09-14.jl
DATA=MATRIX_OPERATOR_REFACTOR/data/052e3-phase2
mkdir -p $DATA

echo "== node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  julia: $($JULIA --version) =="

echo "== stage 1: smoke (unregistered) =="
T1_SMOKE=1 T1_SMOKE_DIR=$SILO/smoke_out_p2 \
    $JULIA --project=$SILO/FLOWPanel.jl --threads=$SLURM_CPUS_PER_TASK \
    $SCRIPT > $DATA/smoke_orc.log 2>&1
echo "smoke OK"

echo "== stage 2: registered S1/S2/S3 at L2 =="
$JULIA --project=$SILO/FLOWPanel.jl --threads=$SLURM_CPUS_PER_TASK \
    $SCRIPT > $DATA/run_2026-09-14_orc.log 2>&1
echo "registered OK"
