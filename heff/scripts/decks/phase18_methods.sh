#!/bin/bash
# Phase 18 Step 1.2: second-order variants, C-a, C-b, Q1_rule of Fe5+ in the Step 8 space, on the stored H, inside the
# memory windows of the running queue (reservation in results/phase18/queue_reserve_mb, as phase17q13).
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_methods.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter ION=fe5nd
NEED=4700
trap 'rm -f $R' EXIT
echo $NEED > $R
echo "$(date +%T) reserved $NEED MB, waiting" >> $L
while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - NEED)) ] && break; sleep 30; done
one() { echo "$(date +%T) begin $*" >> $L; "$@" >> $L 2>&1; rc=$?; echo "$(date +%T) end rc $rc: $*" >> $L; return $rc; }
one python "$HEFF/scripts/heff/phase17a.py" compute besth_sall 30 || exit 1
one python "$HEFF/scripts/heff/phase17a.py" compute besth_sall_s4 30 || exit 1
one python "$HEFF/scripts/heff/phase17c.py" besth_sall C-a C-b || exit 1
one python "$HEFF/scripts/heff/phase17c_scan.py" eval besth_sall $O/crule_fe5nd_s8.json R || exit 1
echo "$(date +%T) all done" >> $L
