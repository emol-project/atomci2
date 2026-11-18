#!/bin/bash
# dump.sh -- acfs for programs outside: info, dump=, nlevJ=, nlevSL=, tol=, dav=, ls
# (tools/acfs.c; ac_lsj.h), and a list of more than 512 configurations.
#
# On the boron V2b system ($AC_WORK/s2m2; run c/tests/s2m2/run_v2b.sh first; the tape has
# 35 orbitals, 33 = 1s (closed), 34 = 2s, 35 = 2p):
#   small  the 5 configurations of V2b (26 CSFs)
#   medium 128 configurations of odd parity: 2s2 2p, 2p3, a2 2p, 2s2 a and 2p2 a (a odd), and
#          2s a b or 2p a b (whichever is odd) over the s and p correlating orbitals
#   large  558 configurations: the same with a, b over all 32 correlating orbitals (acfs used
#          to stop reading at 512 without a word)
#   mixed  163 configurations of both parities: 2s2 2p, 2s2 a, a2 2p and 2p2 a over all a, and
#          2s a b over the s and p correlating orbitals
# The other lists have one parity: H does not couple the two parities, and Davidson, which
# starts from as many vectors as levels are wanted, missed levels on the mixed list when it
# was run on the whole block (level 3 of J = 7/2, 7e-3 hartree too high). The parts that H
# does not couple are now solved one by one (ac_ci.h, ac_ci_davidson_parts).
# dump_check.py (needs numpy; PYTHON) checks:
#   1. the options change nothing else: the output without the info lines is that of a run
#      without them; nlevJ= gives the numbers of levels asked for;
#   2. the files: every block of meta.json loads; idx holds the CSFs that couple to J, in
#      order; the vectors are orthonormal; e.npy is what is printed; the residuals are below
#      1e-10 (tol=1e-6: below 1e-6, and meta.json records it);
#   3. csf.tsv: one line per CSF, the configuration of each, and the SHA-256 of its keys is
#      that of meta.json and the same for every set of terms (noso, so2 ss oo, ls);
#   4. ls: with noso every J level is an LS level (1e-10 hartree);
#   5. the medium list: dense and sparse matrices, dsyev and Davidson give the same levels
#      (1e-10 hartree, vectors to 1e-8 up to the sign), with and without ls; tol=1e-6 stops
#      earlier; the mixed list and the medium list with noso (blocks of parts that H does
#      not couple) give the levels of dsyev, also with Davidson inside the parts
#      (AC_DAV_PART_DENSE=0);
#   6. the large list: all 558 configurations are read (11174 CSFs; its lowest LS level
#      is below that of the medium list, which it contains).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
cfp=$work/m0/build/CFPTBL
r=$work/s2m2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
acfs() { "$repo/c/build/acfs" "$cfp" "$r/ref/fort.15" "$r/ref/fort.16" "$r/ref/fort.18" "$r/ref/fort.21" "$@"; }

# l of the correlating orbitals 1..32: s 1-6, p 7-12, d 13-18, f 19-23, g 24-27, h 28-30, i 31-32
lof() {
    if [ "$1" -le 6 ]; then echo 0; elif [ "$1" -le 12 ]; then echo 1; elif [ "$1" -le 18 ]; then echo 2
    elif [ "$1" -le 23 ]; then echo 3; elif [ "$1" -le 27 ]; then echo 4; elif [ "$1" -le 30 ]; then echo 5
    else echo 6; fi
}
# the configurations of odd parity with a, b <= $1
list() {
    echo "34*2 35*1"; echo "35*3"
    for a in $(seq 1 32); do
        echo "$a*2 35*1"
        if [ $(($(lof "$a") % 2)) = 1 ]; then echo "34*2 $a*1"; echo "35*2 $a*1"; fi
    done
    for a in $(seq 1 "$1"); do for b in $(seq $((a + 1)) "$1"); do
        if [ $((($(lof "$a") + $(lof "$b")) % 2)) = 1 ]; then echo "34*1 $a*1 $b*1"; else echo "35*1 $a*1 $b*1"; fi
    done; done
}
list 12 > "$tmp/med"
list 32 > "$tmp/big"
[ "$(wc -l < "$tmp/med")" = 128 ] && [ "$(wc -l < "$tmp/big")" = 558 ] ||
    { echo "dump.sh: the lists have $(wc -l < "$tmp/med") and $(wc -l < "$tmp/big") lines"; exit 1; }
{
    echo "34*2 35*1"
    for a in $(seq 1 32); do echo "34*2 $a*1"; echo "$a*2 35*1"; echo "35*2 $a*1"; done
    for a in $(seq 1 12); do for b in $(seq $((a + 1)) 12); do echo "34*1 $a*1 $b*1"; done; done
} > "$tmp/mix"

s=$r/configs
acfs "$s" 2                                  > "$tmp/s0.out"
acfs "$s" 2 dump="$tmp/s1"                   > "$tmp/s1.out"
acfs "$s" 2 info                             > "$tmp/s1i.out"
acfs "$s" 2 dump="$tmp/s2" nlevJ=1:3,5:0     > "$tmp/s2.out"
acfs "$s" 2 so2 ss oo dump="$tmp/s3"         > "$tmp/s3.out"
acfs "$s" 99 noso dump="$tmp/s4"             > "$tmp/s4.out"
acfs "$s" 99 ls dump="$tmp/s5"               > "$tmp/s5.out"
acfs "$s" 2 ls nlevSL=2P:3,4S:0 dump="$tmp/s6" > "$tmp/s6.out"
if acfs "$s" 2 ls oo > /dev/null 2>&1; then echo "dump.sh: ls with oo must be refused"; exit 1; fi

b=$tmp/med
AC_LSJ_DENSE=1  acfs "$b" 3 dav=0   dump="$tmp/b1"            > "$tmp/b1.out"
AC_LSJ_DENSE=1  acfs "$b" 3 dav=100 dump="$tmp/b2"            > "$tmp/b2.out"
AC_LSJ_SPARSE=1 acfs "$b" 3 dav=100 dump="$tmp/b3"            > "$tmp/b3.out" 2> "$tmp/b3.err"
AC_LSJ_SPARSE=1 acfs "$b" 3 dav=100 tol=1e-6 dump="$tmp/b4"   > "$tmp/b4.out"
AC_LSJ_DENSE=1  acfs "$b" 3 ls dav=0   dump="$tmp/b5"         > "$tmp/b5.out"
AC_LSJ_SPARSE=1 acfs "$b" 3 ls dav=100 dump="$tmp/b6"         > "$tmp/b6.out"
AC_LSJ_SPARSE=1 acfs "$tmp/mix" 3 dav=100                     > "$tmp/b7.out" 2> "$tmp/b7.err"
AC_TIME=1 AC_DAV_PART_DENSE=0 AC_LSJ_SPARSE=1 acfs "$tmp/mix" 3 dav=100 > "$tmp/b8.out" 2> "$tmp/b8.err"
AC_LSJ_DENSE=1 acfs "$tmp/mix" 3 dav=0                        > "$tmp/b9.out" 2> "$tmp/b9.err"
AC_LSJ_DENSE=1 acfs "$b" 3 noso dav=0                         > "$tmp/b10.out"
AC_TIME=1 AC_DAV_PART_DENSE=0 AC_LSJ_SPARSE=1 acfs "$b" 3 noso dav=100 > "$tmp/b11.out" 2> "$tmp/b11.err"
AC_TIME=1 AC_DAV_PART_DENSE=0 AC_LSJ_DENSE=1  acfs "$b" 3 noso dav=100 > "$tmp/b12.out" 2> "$tmp/b12.err"
AC_LSJ_SPARSE=1 acfs "$tmp/big" 1 ls dump="$tmp/c1"           > "$tmp/c1.out"

"$py" "$here/dump_check.py" "$tmp"
