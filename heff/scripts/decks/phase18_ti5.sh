#!/bin/bash
# Phase 18 Step 1: second-order per-orbital data of the Fe5+ i-function space (3d->i class), under the same reservation.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_methods.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter ION=fe5ti5
NEED=4700
trap 'rm -f $R' EXIT
echo $NEED > $R
while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - NEED)) ] && break; sleep 30; done
echo "$(date +%T) begin fe5ti5 compute k1fhi" >> $L
python "$HEFF/scripts/heff/phase17a.py" compute k1fhi 30 >> $L 2>&1; rc=$?
echo "$(date +%T) end rc $rc: fe5ti5 compute k1fhi" >> $L
exit $rc
