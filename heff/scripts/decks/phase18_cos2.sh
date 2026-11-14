#!/bin/bash
# Phase 18 item 3 (10 Nov): Co3+ fam second-order variants (C2, S2-cav, S2-diag) as Fe2+ (phase17a_fe.py on the stored
# fam H) -> results/phase17/a_cond_fam_d5f2g2h1.json. Runs after phase18_q1pf2.sh (one large job at a time).
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; L=$O/phase18_methods.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
until grep -q "end rc .*: q1pf2 mnnd" $L 2>/dev/null; do sleep 60; done
echo "$(date +%T) begin cos2" >> $L
A_RUNFMT='v15u_fam_d5f2g2h1_d6_{t}e_cond6' A_ION=cond python "$HEFF/scripts/decks/peakmem.py" "cos2" python "$HEFF/scripts/heff/phase17a_fe.py" fam_d5f2g2h1 >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc: cos2" >> $L
exit $rc
