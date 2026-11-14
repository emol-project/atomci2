#!/bin/bash
# run_atsp_bp.sh -- S2-M5: the two-electron Breit-Pauli operators against
# ATSP in the same numerical orbitals and CSF lists.
#
# ATSP's bpci always adds its scalar relativistic shift in relativistic
# mode; acatsp reproduces it with "mvd rkrel" (one-electron mass velocity +
# Darwin, and (2k+1)(alpha^2/4) int R R R R r^2 dr on every R^k). Both use
# alpha = 1/137.036. Operators: bpci (spin-orbit, spin-other-orbit,
# spin-spin, orbit-orbit) = (1,1,0,0) or (1,1,1,0) against acatsp "so2"
# (H_SO1 + H_SO2) and "so2 ss".
#
#   F 2p5 2P (HF): splitting, (1,1,0,0)
#   B 2s2 2p + 2p3 2P (HF of 2s2 2p): all four levels, (1,1,0,0)
#   C 2p2 3P, 1D, 1S (HF): all five levels, (1,1,0,0) and (1,1,1,0), and
#     the spin-spin shifts alone
#   Ti 3d2 4s2 + 3d3 4s + 3d4, every term (37 CSFs, Ti HF): the lowest level
#     of every J, (1,1,1,0)
#   Si ATSP's example e1.bp-5.c restricted to s and p orbitals (215 CSFs, the
#     n = 5 MCHF orbitals of 3P): 3P_J, (1,1,1,0). bpci has no two-body
#     elements (SSO, SOO, SS) between CSFs that differ by ab -> cd over four
#     distinct orbitals, e.g. 3s2 3p2 and 3s 3p 4p 5s (its hzeta/hspin lists
#     hold none; acatsp's, checked against PySCF in tests/v5, are nonzero).
#     acatsp "atspcompat" leaves them out and must then agree; without it the
#     3P_J must differ (by ~1e-7). This list writes the closed core 1s2 2s2
#     2p6 into every CSF; on lists with the core in the header (O) or without
#     a core (Li) bpci keeps these elements, and its levels change with the
#     way the core is written (technical note, Sec. 7.6),
#     so atspcompat is a model of this list only.
#   Orbit-orbit (bpci (1,1,0,1) - (1,1,0,0) against acatsp "oo"), C's HF
#     orbitals: 2p2 3P, 1D, 1S without core agree (2e-9, bpci's print
#     precision); 1s 2p 3P must differ (s-p exchange: bpci is off by ~1%,
#     and by up to 13% for Gaussians with exact values, check-s2m6b).
# The energies of the two programs differ by a constant (core quadrature,
# ~1e-8 to 1e-7 hartree). Passes when, per system, the spread of C - ATSP
# over the levels is below 1e-8 hartree (2e-9 for the spin-spin shifts).
#
#   ATSP_BIN  ATSP binaries (default $HOME/atsp_work/atsp/bin)
#   ATSP_SRC  ATSP source tree with examples/ (skipped when unset)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_dir ATSP_SRC "the ATSP source tree, with examples/" || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf bpci || exit 0
src=$ATSP_SRC
A=$repo/c/build/acatsp
T=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

hf_run() { # name, term, Z, closed, open
    hf > $1.hf 2>&1 <<S
$1,$2,$3
$4
$5
all
y
y
n
n
S
    cp wfn.out $1.w
}
bp_run() { # name, 2Jmax, 2Jmin, eigenvalue line, ops
    local lines="" j
    for ((j = $2; j >= $3; j -= 2)); do lines+="$4"$'\n'; done
    rm -f *.lst $1.j
    bpci > $1.bp 2>&1 <<EOF
$1,y,n
n
n
$2,$3
${lines}2
n
$5
y
y
y
y
EOF
    awk '/2\*J =/ { j = $3 } /^ +[0-9]+ +-/ { print j, $2 }' $1.j
}
c_run() { # name, options...
    local n=$1
    shift
    "$A" "$T" $n.w $n.c 4 alpha=137.036 mvd rkrel "$@" |
        awk '/^ ?[0-9]+\/2 / { split($1, a, "/"); print a[1], $2, $3 }'
}

# F
hf_run F 2P 9. "  1s  2s" "2p(5)"
printf '%64s\n  1s  2s\n  2p( 5)\n 2P1\n*\n' "" > F.c
bp_run F 3 1 1 1,1,0,0 > F.atsp
c_run F so2 > F.c_out
# B
hf_run B 2P 5. "  1s" "2s(2)2p(1)"
printf '%64s\n  1s\n  2s( 2)  2p( 1)\n 1S0 2P1 2P\n  2p( 3)\n 2P1\n*\n' "" > B.c
bp_run B 3 1 1,2 1,1,0,0 > B.atsp
c_run B so2 > B.c_out
# C
hf_run Cat 3P 6. "  1s  2s" "2p(2)"
printf '%64s\n  1s  2s\n  2p( 2)\n 3P2\n  2p( 2)\n 1D2\n  2p( 2)\n 1S0\n*\n' "" > Cat.c
bp_run Cat 4 0 1,2 1,1,0,0 > Cat.atsp
bp_run Cat 4 0 1,2 1,1,1,0 > Cat_ss.atsp
c_run Cat so2 > Cat.c_out
c_run Cat so2 ss > Cat_ss.c_out
# orbit-orbit in C's HF orbitals, two electrons: 2p2 (no core) and 1s 2p 3P
cp Cat.w OOpp.w
cp Cat.w OOsp.w
printf '%64s\n\n  2p( 2)\n 3P2\n  2p( 2)\n 1D2\n  2p( 2)\n 1S0\n*\n' "" > OOpp.c
printf '%64s\n\n  1s( 1)  2p( 1)\n 2S1 2P1 3P\n*\n' "" > OOsp.c
bp_run OOpp 4 0 1,2 1,1,0,0 > OOpp.atsp
bp_run OOpp 4 0 1,2 1,1,0,1 > OOpp_oo.atsp
bp_run OOsp 4 4 1 1,1,0,0 > OOsp.atsp
bp_run OOsp 4 4 1 1,1,0,1 > OOsp_oo.atsp
c_run OOpp so2 > OOpp.c_out
c_run OOpp so2 oo > OOpp_oo.c_out
c_run OOsp so2 > OOsp.c_out
c_run OOsp so2 oo > OOsp_oo.c_out
# Ti
hf_run Ti 3F 22. "  1s  2s  2p  3s  3p" "3d(2)4s(2)"
core="1s(2) 2s(2) 2p(6) 3s(2) 3p(6)"
printf '%s 3d(2) 4s(2)\n%s 3d(3) 4s(1)\n%s 3d(4)\n' "$core" "$core" "$core" > ti.cfg
"$A" "$T" Ti.w ti.cfg 1 ls noso writec=Ti > /dev/null
{
    head -2 Ti_3F.c
    for f in Ti_*.c; do sed -n '3,$p' $f | grep -v '^\*'; done
    echo '*'
} > TiBP.c
cp Ti.w TiBP.w
bp_run TiBP 12 0 1 1,1,1,0 > TiBP.atsp
"$A" "$T" TiBP.w TiBP.c 1 alpha=137.036 mvd rkrel so2 ss |
    awk '/^ ?[0-9]+\/2 / { split($1, a, "/"); print a[1], $2, $3 }' > TiBP.c_out
# Si: ATSP's example e1.bp-5.c restricted to s and p orbitals (215 CSFs) in
# the n = 5 MCHF orbitals of 3P
python3 - "$src/examples/files_c/e1.bp-5.c" > SiBP.c <<'PY'
import sys
lines = open(sys.argv[1]).read().split("\n")
out, i = lines[:2], 2
while i < len(lines) and not lines[i].startswith("*"):
    labs = [lines[i][j:j + 4].strip() for j in range(0, len(lines[i]), 8)]
    if all(x[-1] in "sp" for x in labs if x):
        out += lines[i:i + 2]
    i += 2
print("\n".join(out + ["*"]))
PY
cp "$src/examples/e1/e1.14_5_3P.w" SiBP.w
bp_run SiBP 4 0 1 1,1,1,0 > SiBP.atsp
c_si() {
    "$A" "$T" SiBP.w SiBP.c 1 alpha=137.036 mvd rkrel so2 ss "$@" |
        awk '/^ ?[0-9]+\/2 / { split($1, a, "/"); print a[1], $2, $3 }'
}
c_si > SiBP.c_out
c_si atspcompat > SiBPx.c_out

python3 - <<'PY'
from collections import defaultdict
bad = 0
def levels_atsp(f):
    d = defaultdict(list)
    for line in open(f):
        j, e = line.split()
        d[int(j)].append(float(e))
    return {j: sorted(v) for j, v in d.items()}
def levels_c(f):
    d = defaultdict(list)
    for line in open(f):
        j, n, e = line.split()
        d[int(j)].append(float(e))
    return {j: sorted(v) for j, v in d.items()}
def compare(label, fa, fc, tol=1e-8):
    global bad
    a, c = levels_atsp(fa), levels_c(fc)
    off = [y - x for j in a for x, y in zip(a[j], c.get(j, []))]
    n = sum(len(v) for v in a.values())
    ok = len(off) == n and max(off) - min(off) < tol
    bad += not ok
    print(f"   {'ok  ' if ok else 'FAIL'} {label}: {len(off)} levels, C-ATSP {min(off):.2e} .. {max(off):.2e}"
          f" (spread {max(off) - min(off):.1e})")
    return a, c
compare("F 2p5 2P, SO+SOO", "F.atsp", "F.c_out")
compare("B 2s2 2p + 2p3 2P, SO+SOO", "B.atsp", "B.c_out")
a0, c0 = compare("C 2p2, SO+SOO", "Cat.atsp", "Cat.c_out")
a1, c1 = compare("C 2p2, SO+SOO+SS", "Cat_ss.atsp", "Cat_ss.c_out")
# spin-spin shifts alone
d = [(c1[j][i] - c0[j][i]) - (a1[j][i] - a0[j][i]) for j in a0 for i in range(len(a0[j]))]
big = max(abs(a1[j][i] - a0[j][i]) for j in a0 for i in range(len(a0[j])))
ok = max(map(abs, d)) < 2e-9
bad += not ok
print(f"   {'ok  ' if ok else 'FAIL'} C 2p2 spin-spin shifts: largest {big:.3e}, max |C - ATSP| {max(map(abs, d)):.1e}")
compare("Ti 3d2 4s2 + 3d3 4s + 3d4 (37 CSFs), SO+SOO+SS, lowest per J", "TiBP.atsp", "TiBP.c_out")
# orbit-orbit shifts (bpci prints 9 decimals)
def oo_shift(pre):
    a0, a1 = levels_atsp(f"{pre}.atsp"), levels_atsp(f"{pre}_oo.atsp")
    c0, c1 = levels_c(f"{pre}.c_out"), levels_c(f"{pre}_oo.c_out")
    return [(a1[j][i] - a0[j][i], c1[j][i] - c0[j][i]) for j in a0 for i in range(len(a0[j]))]
s = oo_shift("OOpp")
d = max(abs(c - a) for a, c in s)
ok = d < 2e-9
bad += not ok
print(f"   {'ok  ' if ok else 'FAIL'} orbit-orbit, 2p2 (5 levels): shifts up to {max(abs(a) for a, c in s):.3e},"
      f" max |C - ATSP| {d:.1e}")
s = oo_shift("OOsp")
a, c = s[0]
ok = abs(c - a) > 1e-7
bad += not ok
print(f"   {'ok  ' if ok else 'FAIL'} orbit-orbit, 1s 2p 3P: ATSP {a:.4e}, C {c:.4e}: bpci differs for s-p"
      " exchange (C agrees with exact values, check-s2m6b)")
compare("Si e1.bp-5.c s,p part (215 CSFs), 3P_J, SO+SOO+SS, atspcompat", "SiBP.atsp", "SiBPx.c_out")
# without atspcompat the 3P_J splittings differ: bpci has no two-body
# elements between CSFs differing by ab -> cd over four distinct orbitals
a, c = levels_atsp("SiBP.atsp"), levels_c("SiBP.c_out")
off = [c[j][0] - a[j][0] for j in a]
ok = max(off) - min(off) > 1e-8
bad += not ok
print(f"   {'ok  ' if ok else 'FAIL'} Si, all elements (no atspcompat): spread {max(off) - min(off):.1e}"
      " (the omitted elements matter)")
print("ATSP Breit-Pauli comparison", "passed" if not bad else "FAILED")
raise SystemExit(1 if bad else 0)
PY
