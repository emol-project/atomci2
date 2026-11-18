#!/bin/bash
# terms.sh -- acfs: the terms of the matrices kept apart (dumpmat, proj, ls with the terms, nog),
# checked from the files alone, as a program outside would use them (terms_check.py; numpy).
#
# Systems: boron on the V2b orbitals ($AC_WORK/s2m2; run c/tests/s2m2/run_v2b.sh first):
#   small   the 5 configurations of V2b (26 CSFs), all levels
#   medium  the 128 configurations of odd parity of dump.sh (1358 CSFs), 3 levels per block
# and, when the Ti run of c/examples/tm_acfs is at hand (ACFS_TI_DIR, default
# $AC_WORK/ex_tm/ti, with fort.15, fort.16, fort.18, fort.21):
#   d3      Ti+ 3d3 (15*3; 8 CSFs)
#   d       Ti 3d2 4s2 + 3d3 4s + 3d4, all levels
# All with so2 ss oo mvd contact.
#   T1  From the matrices of dumpmat (h_C, h_OO, h_MVD, h_CONTACT, w_SO1, w_SO2, w_SS) and
#       csf.tsv, every J block is put together with the formula of meta.json (the 6j from
#       Racah's sum, in Python) and diagonalized: the levels of acfs to 1e-10 hartree, its
#       vectors up to sign (the span of each group of close levels).
#   T2  The files of proj equal c^T M c' with the matrices of dumpmat and the LS vectors
#       (1e-12 relative to the largest element). With all LS levels (small, d3, d): the J
#       blocks put together from the LS energies and the files of proj alone give the levels
#       of the J run (1e-10): this is how an effective Hamiltonian uses them.
#   T3  The matrix of each term from a run with that term alone is the one of the run with
#       all terms; the levels of the run with the terms kept apart are those of a run without
#       dumpmat (the terms added to the integrals), 1e-10.
#   also: ls gives the levels of h_C whatever terms are on; dense and sparse runs write the
#       same matrices; with AC_LSJ_DISK (w and w2 in files) the output is the same to the last
#       digit; nog prints the same levels; ls with a term but without proj or
#       dumpmat, proj without ls, and dumpmat without dump= are refused.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
cfp=$work/m0/build/CFPTBL
r=$work/s2m2
ti=${ACFS_TI_DIR:-$work/ex_tm/ti}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bo() { "$repo/c/build/acfs" "$cfp" "$r/ref/fort.15" "$r/ref/fort.16" "$r/ref/fort.18" "$r/ref/fort.21" "$@"; }
tm() { "$repo/c/build/acfs" "$cfp" "$ti/fort.15" "$ti/fort.16" "$ti/fort.18" "$ti/fort.21" "$@"; }
T="so2 ss oo mvd contact"

# one system: $1 the acfs function, $2 name, $3 configs, $4 levels per block
system() {
    local f=$1 n=$2 c=$3 nl=$4
    $f "$c" "$nl" $T                                    > "$tmp/$n.plain.out"
    $f "$c" "$nl" $T nog                                > "$tmp/$n.nog.out"
    $f "$c" "$nl" $T dump="$tmp/$n.j" dumpmat           > "$tmp/$n.j.out"
    $f "$c" "$nl" $T ls dump="$tmp/$n.ls" proj dumpmat  > "$tmp/$n.ls.out"
    $f "$c" "$nl" ls dump="$tmp/$n.ls0"                 > "$tmp/$n.ls0.out"
}
system bo small "$r/configs" 99
for t in noso:so2 noso:ss noso:oo noso:mvd noso:contact so1:; do
    opt=${t//:/ }; opt=${opt/so1/}
    bo "$r/configs" 99 $opt dump="$tmp/small.one.${t%%:*}${t#*:}" dumpmat > /dev/null
done

lof() {
    if [ "$1" -le 6 ]; then echo 0; elif [ "$1" -le 12 ]; then echo 1; elif [ "$1" -le 18 ]; then echo 2
    elif [ "$1" -le 23 ]; then echo 3; elif [ "$1" -le 27 ]; then echo 4; elif [ "$1" -le 30 ]; then echo 5
    else echo 6; fi
}
{
    echo "34*2 35*1"; echo "35*3"
    for a in $(seq 1 32); do
        echo "$a*2 35*1"
        if [ $(($(lof "$a") % 2)) = 1 ]; then echo "34*2 $a*1"; echo "35*2 $a*1"; fi
    done
    for a in $(seq 1 12); do for b in $(seq $((a + 1)) 12); do
        if [ $((($(lof "$a") + $(lof "$b")) % 2)) = 1 ]; then echo "34*1 $a*1 $b*1"; else echo "35*1 $a*1 $b*1"; fi
    done; done
} > "$tmp/med"
AC_LSJ_DENSE=1 system bo medium "$tmp/med" 3
AC_LSJ_SPARSE=1 bo "$tmp/med" 3 $T dav=100 dump="$tmp/medium.sp" dumpmat > "$tmp/medium.sp.out"
AC_LSJ_SPARSE=1 bo "$tmp/med" 3 $T dav=100 > "$tmp/medium.mem.out"
mkdir "$tmp/disk"
AC_LSJ_SPARSE=1 AC_LSJ_DISK="$tmp/disk" bo "$tmp/med" 3 $T dav=100 > "$tmp/medium.disk.out"
if AC_LSJ_SPARSE=1 AC_LSJ_DISK="$tmp/nodir" bo "$tmp/med" 3 $T > /dev/null 2>&1; then
    echo "terms.sh: AC_LSJ_DISK with a directory that is not there must fail"; exit 1
fi

# The two-electron parts of g_J (gc2e oz2e rec2): from the tables, in one pass for all the J
# blocks (AC_G2_PASS=1), and in one pass by itself when the tables do not fit AC_PROP_CACHE_MB:
# the same output to the last digit, dense and sparse. With AC_G2_PASS=0 and tables that do not
# fit (the small system): the same output and the warning on stderr.
G="gc2e oz2e rec2"
head -36 "$tmp/med" > "$tmp/g2cfg" # the first configurations of medium: the whole list takes minutes per run
for mode in DENSE SPARSE; do
    (
        export AC_LSJ_$mode=1
        AC_G2_PASS=0 bo "$tmp/g2cfg" 3 $G > "$tmp/g2.tab.out"
        AC_G2_PASS=1 bo "$tmp/g2cfg" 3 $G > "$tmp/g2.pass.out"
        AC_PROP_CACHE_MB=0.01 bo "$tmp/g2cfg" 3 $G > "$tmp/g2.small.out" 2> "$tmp/g2.small.err"
        grep -q dg_GC2e "$tmp/g2.tab.out" || { echo "terms.sh: no dg_GC2e column"; exit 1; }
        cmp "$tmp/g2.tab.out" "$tmp/g2.pass.out" || { echo "terms.sh: AC_G2_PASS=1 changes the output ($mode)"; exit 1; }
        cmp "$tmp/g2.tab.out" "$tmp/g2.small.out" || { echo "terms.sh: a small AC_PROP_CACHE_MB changes the output ($mode)"; exit 1; }
        if grep -q warning "$tmp/g2.small.err"; then echo "terms.sh: warning although g is made in one pass ($mode)"; exit 1; fi
    )
done
bo "$r/configs" 99 $G > "$tmp/g2.s.out"
AC_PROP_CACHE_MB=0.001 AC_G2_PASS=0 bo "$r/configs" 99 $G > "$tmp/g2.s0.out" 2> "$tmp/g2.s0.err"
cmp "$tmp/g2.s.out" "$tmp/g2.s0.out" || { echo "terms.sh: AC_G2_PASS=0 with a small cache changes the output"; exit 1; }
grep -q "warning: the table of a two-electron property" "$tmp/g2.s0.err" || { echo "terms.sh: no warning for a table that does not fit"; exit 1; }
echo "   g_J two-electron parts: tables, one pass and small cache agree (dense, sparse); warning ok"

names="small medium"
if [ -s "$ti/fort.21" ]; then
    echo "15*3" > "$tmp/d3.cfg"
    printf '14*2 15*2\n14*1 15*3\n15*4\n' > "$tmp/d.cfg"
    system tm d3 "$tmp/d3.cfg" 99
    system tm d "$tmp/d.cfg" 99
    names="$names d3 d"
else
    echo "   (no Ti run in $ti: the d-shell systems are left out)"
fi

if bo "$r/configs" 2 ls oo > /dev/null 2>&1; then echo "terms.sh: ls with oo alone must be refused"; exit 1; fi
if bo "$r/configs" 2 proj dump="$tmp/x" > /dev/null 2>&1; then echo "terms.sh: proj without ls must be refused"; exit 1; fi
if bo "$r/configs" 2 dumpmat > /dev/null 2>&1; then echo "terms.sh: dumpmat without dump= must be refused"; exit 1; fi

"$py" "$here/terms_check.py" "$tmp" $names
