#!/bin/bash
# Phase 18 Step 2: Co3+ fam sets on the stored H (as phase17q13 for Fe2+), one phase17c_scan run under the queue
# memory reservation: X- (exact H_eff, check against the stored exact), X-11d,12d,13d (exact sub-space without the
# three virtual d, standard basis), R (Q1_rule balanced). Blocks -> results/phase18/heff_dump/q18_cond_fam.npz.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_cofam.log
NEED=10000
trap 'rm -f $R' EXIT
echo $NEED > $R
echo "$(date +%T) reserved $NEED MB, waiting" >> $L
while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - NEED)) ] && break; sleep 30; done
echo "$(date +%T) begin" >> $L
env C_N=6 A_RUNFMT='v15u_fam_d5f2g2h1_d6_{t}e_cond6' A_ION=cond SCAN_DUMP=$O/heff_dump/q18_cond_fam.npz \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter \
  python "$HEFF/scripts/decks/peakmem.py" "q18 cond fam" python "$HEFF/scripts/heff/phase17c_scan.py" eval fam_d5f2g2h1 $O/q18_cond_fam.json \
  X- X-11d,12d,13d R >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc" >> $L
exit $rc
