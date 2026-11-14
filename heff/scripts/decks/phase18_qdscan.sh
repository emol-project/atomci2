#!/bin/bash
# Phase 18, 10 Nov A: exact sub-spaces of the Co3+ famdq H (co.bas, coqd6), as phase17q16 for Fe2+:
#   X-13s (without the most diffuse virtual s), X-11d,13s (without the diffuse d and s).
# Waits for the A/B queue (run_queue5.log) to finish, so no queue job runs beside it (~9.3 GB).
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; L=$O/phase18_qdscan.log
until grep -q "projected-finish warning" $O/run_queue5.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) begin coqd famdq scan" >> $L
env ION=coqd C_N=6 A_RUNFMT='v17_famdq_d6_{t}e_coqd6' A_ION=coqd SCAN_DUMP=$O/heff_dump/q18_coqd_famdq.npz \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter \
  python "$HEFF/scripts/decks/peakmem.py" "q18 coqd famdq" python "$HEFF/scripts/heff/phase17c_scan.py" eval famdq $O/q18_coqd_famdq.json \
  X-13s X-11d,13s >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc: coqd famdq scan" >> $L
exit $rc
