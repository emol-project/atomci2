#!/bin/bash
# Phase 18 item 2 (10 Nov): T2 decomposition of the d6 fam space by exact leave-one-out of each virtual orbital
# (and the f, g pairs), same sets for Co3+ (cond) and Fe2+ (fend), on the stored H (results/<run>.npz only).
# One ion at a time (~9.3 GB each). Blocks -> results/phase18/heff_dump/q18_<ion>_fam_loo.npz, sets -> q18_<ion>_fam_loo.json.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; L=$O/phase18_loo.log
SETS="X-16s X-15p X-11d X-12d X-13d X-14d X-11f X-12f X-11f,12f X-11g X-12g X-11g,12g X-11h"
rc_all=0
for ion in cond fend; do
  echo "$(date +%T) begin $ion" >> $L
  env C_N=6 A_RUNFMT="v15u_fam_d5f2g2h1_d6_{t}e_${ion}6" A_ION=$ion SCAN_DUMP=$O/heff_dump/q18_${ion}_fam_loo.npz \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter \
    python "$HEFF/scripts/decks/peakmem.py" "q18 $ion fam loo" python "$HEFF/scripts/heff/phase17c_scan.py" eval fam_d5f2g2h1 $O/q18_${ion}_fam_loo.json \
    $SETS >> $L 2>&1; rc=$?
  echo "$(date +%T) end $ion rc $rc" >> $L
  [ $rc -ne 0 ] && rc_all=1
done
exit $rc_all
