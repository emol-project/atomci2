#!/bin/bash
# Phase 18 item 4 (10 Nov): Q1_rule and Q1_rule + pf2 on the Step 8 space (besth_sall) for V2+, Cr3+, Mn4+ (as Fe5+,
# phase18_q1diag.sh). Waits for phase18_loo.sh to finish (memory), then one ion at a time (~4.7 GB).
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; L=$O/phase18_methods.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter
until grep -q "end fend rc" $O/phase18_loo.log 2>/dev/null; do sleep 60; done
rc_all=0
for ion in vnd crnd mnnd; do
  echo "$(date +%T) begin q1pf2 $ion" >> $L
  ION=$ion python "$HEFF/scripts/decks/peakmem.py" "q1pf2 $ion" python "$HEFF/scripts/heff/phase18_q1diag.py" eval $O/q1diag_${ion}_s8.json - pf2 >> $L 2>&1; rc=$?
  echo "$(date +%T) end rc $rc: q1pf2 $ion" >> $L
  [ $rc -ne 0 ] && rc_all=1
done
exit $rc_all
