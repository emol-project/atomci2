#!/bin/bash
# run_m5.sh -- M5 / L5: the CI stage of the Fortran ATOMCI run on a tape
# written by the C engine (technical note, Appendix C).
#
# For each sample (boron: EXTAPE path; f: WFETAP path):
#   prep/     the Fortran up to the tape and the integral transformation
#             (CI and RS switched off in &CNTL NRNFG)
#   control/  copy of prep/, then CI + RS restarted on the Fortran tape
#   ctape/    copy of prep/ with fort.15 replaced by acgen's tape, then the
#             same CI + RS restart
# The control run must reproduce the golden energy (it validates the restart
# procedure); the C-tape run must agree with it to 1e-9.
#
#   AC_WORK   work root (default: $HOME/atomci_work)
# Needs the M0 build ($AC_WORK/m0/build: atomci.plain, atgen, CFPTBL); builds
# the two C tools it needs (c/build: acgen, nfte_diff).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
ftn=$work/m0/build
cbin=$repo/c/build
make -s -C "$repo/c" build/acgen build/nfte_diff
root=$work/m5
atomci=$ftn/atomci.plain

rm -rf "$root"
mkdir -p "$root"

# ---- inputs ---------------------------------------------------------------

# boron: header = the OPEN lines up to END; the run's &CNTL is lines with
# NRNFG ... TITLE; &CI to the end is the CI/RS part.
b=$repo/samples/native/boron.stdin
nend=$(grep -n '^END' "$b" | head -1 | cut -d: -f1)
nci=$(grep -n '&CI' "$b" | head -1 | cut -d: -f1)
sed -n "1,$((nci - 1))p" "$b" \
    | sed 's/NRNFG=    1, 0, 1, 1, 0, 1, 1, 1,/NRNFG=    1, 0, 1, 1, 0, 1, 0, 0,/' \
    > "$root/boron.prep.stdin"
grep -q 'NRNFG=    1, 0, 1, 1, 0, 1, 0, 0,' "$root/boron.prep.stdin"
{
    sed -n "1,${nend}p" "$b"
    echo " &CNTL NRNFG=0,0,0,0,0,0,1,1, TITLE=' B (DOUBLET P ODD) ', /"
    sed -n "${nci},\$p" "$b"
} > "$root/boron.restart.stdin"

# f: pass 1 only (up to the first &CI); its &CI and &RS for the restart
f=$repo/samples/native/f.stdin
nend=$(grep -n '^END' "$f" | head -1 | cut -d: -f1)
nci=$(grep -n '&CI' "$f" | head -1 | cut -d: -f1)
nrs=$(grep -n '&RS' "$f" | head -1 | cut -d: -f1)
sed -n "1,$((nci - 1))p" "$f" \
    | sed 's/&CNTL NRNFG=1,1,1,1,0,0,1,1,/\&CNTL NRNFG=1,1,1,1,0,0,0,0,/' \
    > "$root/f.prep.stdin"
grep -q 'NRNFG=1,1,1,1,0,0,0,0,' "$root/f.prep.stdin"
{
    sed -n "1,${nend}p" "$f"
    echo " &CNTL NRNFG=0,0,0,0,0,0,1,1, TITLE=' F DOUBLET P ', /"
    sed -n "${nci},${nrs}p" "$f"
} > "$root/f.restart.stdin"

# ---- runs -----------------------------------------------------------------

run() { # dir stdin output
    (cd "$1" && "$atomci" < "$2" > "$3" 2> "${3%.output}.err")
}

for s in boron f; do
    d=$root/$s
    mkdir -p "$d/prep"
    ln -sf "$ftn/CFPTBL" "$d/prep/CFPTBL"
    if [ "$s" = f ]; then
        cp "$repo/samples/native/f.bas" "$d/prep/"
        (cd "$d/prep" && "$ftn/atgen" < "$repo/samples/native/f.atg" > /dev/null)
    fi
    run "$d/prep" "$root/$s.prep.stdin" prep.output

    cp -a "$d/prep" "$d/control"
    cp -a "$d/prep" "$d/ctape"
    "$cbin/acgen" "$ftn/CFPTBL" "$d/prep/fort.15" "$d/ctape/fort.15"
    echo "== $s: Fortran tape vs C tape"
    "$cbin/nfte_diff" "$d/prep/fort.15" "$d/ctape/fort.15" | tail -2

    run "$d/control" "$root/$s.restart.stdin" restart.output
    run "$d/ctape" "$root/$s.restart.stdin" restart.output
done

# ---- energies -------------------------------------------------------------

echo
echo "== energies (golden: boron ET = -24.6461953967, f SD-CI = -99.641223)"
for s in boron f; do
    for k in control ctape; do
        o=$root/$s/$k/restart.output
        printf '%-6s %-8s ' "$s" "$k"
        if [ "$s" = boron ]; then
            { grep -a -h 'ET *=' "$o" || true; } | tail -1
        else
            { grep -a -h 'SD-CI ENERGY (CORRELATION)' "$o" || true; } | head -1
        fi
    done
done

echo
echo "== L5: ET in full precision (CI vector file, unit 37)"
status=0
for s in boron f; do
    ec=$(python3 "$here/et_from_civ.py" "$root/$s/control/fort.37")
    et=$(python3 "$here/et_from_civ.py" "$root/$s/ctape/fort.37")
    verdict=$(python3 -c "import sys; d=abs($ec-($et)); print(f'{d:.2e}', 'ok' if d <= 1e-9 else 'FAIL')")
    printf '%-6s control %s  C tape %s  |diff| %s\n' "$s" "$ec" "$et" "$verdict"
    case $verdict in *FAIL) status=1 ;; esac
done
echo "L5 $([ $status = 0 ] && echo passed || echo FAILED)"

echo
echo "== stderr (expected empty)"
for e in "$root"/*/*/*.err; do
    [ -s "$e" ] && { echo "non-empty: $e"; head -5 "$e"; }
done
exit $status
