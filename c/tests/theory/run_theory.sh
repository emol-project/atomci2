#!/bin/bash
# run_theory.sh [NAME...] -- the checks of the formulas of the technical note,
# each formula against an independent reference, without the C code:
#   rme       special cases of Edmonds (7.1.5) against the 9j formula (sympy)
#   conv      Edmonds (7.1.6)-(7.1.8) phases, 9j as sums of 6j, normal-order
#             contraction, one-electron reduced elements (PythonRef's determinants)
#   cfp       the quasi-spin factor of LS CFPs and XCFP's closed form
#   quasispin the quasi-spin generators and the phase of the B doublet (s, p, d)
#   factor    invariants and reduced elements, products of L_k, MOTECC (48)
#   lsj       Lande, a 6j closed form, the K = 0 reduction
#   coulomb   H^k and the METECC form, F closed forms, the Coulomb contraction terms
#   bp2e      SO2 (spin-same and spin-other-orbit), SS, OO, contact terms, mass
#             velocity: coordinate-space operators on a grid (theory_coord.py)
#   props     two-body recoil, GC1e, H6 rank 2, hfs T(1) and T(2), A_J and B_J,
#             E1 length and velocity, E2, one-electron M1 corrections, SMS
#   props2e   GC2e, OZ2e, the two-electron spin terms of M1 (K = 0, 2)
#   multipole tensor forms of E3, M2, M3
# Default: all. CONTROLS=1 also runs the sign-breaking controls of bp2e, props
# and props2e (one relative sign flipped per operator), which must FAIL.
# Times on 4 threads: bp2e about 7 min, conv and factor about 4 min each,
# props2e about 3 min, the rest under a minute (about 20 min in all); the controls
# add about 15 min.
#   PYTHON  a Python with numpy and scipy, and sympy for rme (default python3)
#   LOGS    directory for the logs (default: a temporary one, removed at the end)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../lib/need.sh"
need_py numpy scipy || exit 0
py=${PYTHON:-python3}
if [ -n "${LOGS:-}" ]; then L=$LOGS; mkdir -p "$L"; else L=$(mktemp -d); trap 'rm -rf "$L"' EXIT; fi

# verdict NAME LOG: 0 when the log reports success
verdict() {
    "$py" -I - "$1" "$2" <<'PY'
import re, sys
name, log = sys.argv[1], open(sys.argv[2]).read()
if name in ("bp2e", "props", "props2e", "multipole"):
    ok = re.search(r": all passed\s*$", log) is not None
elif name == "quasispin":
    nums = [float(x) for x in re.findall(r"(?:err|SL)\s+([0-9.eE+-]+)", log)]
    nums += [float(x) for l in re.findall(r"errs \[([^\]]*)\]", log) for x in l.split(",")]
    ok = bool(nums) and max(abs(x) for x in nums) < 1e-12
else:
    bad = [int(x) for x in re.findall(r"\bbad (\d+)", log)]
    ok = bool(bad) and max(bad) == 0
sys.exit(0 if ok else 1)
PY
}

status=0
run() {  # run NAME [ARGS] ; EXPECT=fail for a control
    local name=$1; shift
    local tag=$name${1:+_$(echo "$*" | tr ' ' _)} log t0=$(date +%s)
    log=$L/$tag.log
    if [ "$name" = rme ] && ! "$py" -c "import sympy" > /dev/null 2>&1; then
        echo "SKIPPED $tag: $py cannot import sympy (set PYTHON to a Python with sympy)"
        return
    fi
    "$py" -I "$here/theory_check_$name.py" "$@" > "$log" 2>&1
    local rc=$? res
    if [ $rc = 0 ] && verdict "$name" "$log"; then res=passed; else res=failed; fi
    if [ "${EXPECT:-pass}" = fail ]; then
        if [ "$res" = failed ] && [ $rc = 0 ]; then res="failed as it must (control)"; else res="CONTROL DID NOT FAIL"; status=1; fi
    elif [ "$res" = failed ]; then
        res=FAILED; status=1
    fi
    echo "$tag: $res ($(( $(date +%s) - t0 )) s)"
    [ "$res" = FAILED ] && tail -15 "$log" | sed 's/^/    /'
}

for n in ${@:-rme conv cfp quasispin factor lsj coulomb bp2e props props2e multipole}; do run "$n"; done
if [ "${CONTROLS:-0}" = 1 ]; then
    EXPECT=fail run bp2e so2 ss oo bp1 neg
    EXPECT=fail run props g hfs trans is neg
    EXPECT=fail run props2e gc2e oz2e m1s2 neg
    EXPECT=fail run props2e gc2e negt
    EXPECT=fail run props2e gc2e negc
fi
exit $status
