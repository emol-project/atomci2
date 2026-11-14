#!/bin/bash
# Phase 18 item 3 (8 Nov): Q1_rule + one Q2 class at a time, Fe5+ Step 8, under the queue memory reservation.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_methods.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter ION=fe5nd
NEED=4700
trap 'rm -f $R' EXIT
echo $NEED > $R
echo "$(date +%T) q1diag reserved $NEED MB, waiting" >> $L
while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - NEED)) ] && break; sleep 30; done
echo "$(date +%T) begin q1diag" >> $L
python "$HEFF/scripts/heff/phase18_q1diag.py" eval $O/q1diag_fe5nd_s8.json - mxdg mxsd mxsg mxdg+mxsd+mxsg+mxfh+mxpf+mxph 3s_v 3p2_v 3p_v pf2 inner >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc: q1diag" >> $L
exit $rc
