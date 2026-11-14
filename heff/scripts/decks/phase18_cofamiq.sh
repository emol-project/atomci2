#!/bin/bash
# Phase 18 Step 2: Co3+ fam - 12f + one i + the three pure n = 3 configurations (coti1 famiq, as phase17q18 for Fe2+) and
# its exact sub-spaces on the stored H: X- (full), X-11i (+ n3), X-h2 (+ i), X-11i,h2 (neither). Blocks ->
# results/phase18/heff_dump/q18_coti1_famiq.npz (the n = 3 and first-i items of the Co3+ chain, phase18_co.py chain).
# Waits until every famiq term has its npz, then runs under the queue memory reservation.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_cofamiq.log
NEED=9500
TERMS="1S 3P 1D 3D 5D 1F 3F 1G 3G 3H 1I"
alldone() { for t in $TERMS; do grep -q "end v17_famiq_d6_${t}e_coti16 rc 0" $O/phase18_runs.log || return 1; done; }
until alldone; do sleep 120; done
trap 'rm -f $R' EXIT
while [ -s $R ]; do sleep 60; done          # another reservation (phase18_codump.sh) still active
echo $NEED > $R
echo "$(date +%T) reserved $NEED MB, waiting" >> $L
while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - NEED)) ] && break; sleep 30; done
echo "$(date +%T) begin" >> $L
env ION=coti1 C_N=6 A_RUNFMT='v17_famiq_d6_{t}e_coti16' A_ION=coti1 SCAN_DUMP=$O/heff_dump/q18_coti1_famiq.npz \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter \
  python "$HEFF/scripts/decks/peakmem.py" "q18 coti1 famiq" python "$HEFF/scripts/heff/phase17c_scan.py" eval famiq $O/q18_coti1_famiq.json \
  X- X-11i X-h2 X-11i,h2 >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc" >> $L
exit $rc
