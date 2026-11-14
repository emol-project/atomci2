#!/bin/bash
# o3p_ic_atsp.sh -- S2-M7 (Z11), report only (ATSP needed): the intermediate-
# coupling part of the O 2p4 g_J against ATSP's bpci, which prints the Lande
# g_J of each Breit-Pauli eigenvector (brci_zeeman.f, g_s = 2.002319304386;
# no alpha^2 Zeeman terms). acatsp's g_J column is the same quantity, so both
# programs are run with the same orbitals, CSF list and Hamiltonian:
# bpci (1,1,1,0) = spin-orbit + spin-other-orbit + spin-spin with its scalar
# relativistic shift, acatsp "mvd rkrel so2 ss" (all two-body elements; see the
# note on atspcompat in the technical note, Sec. 7.6), alpha = 1/137.036.
#   HF:  2p4 3P, 1D, 1S on the HF orbitals of 3P
#   n3:  every LS term of the valence SD configurations to n = 3 (cfg_ic3 of
#        o3p_gj.sh) on its n = 3 MCHF orbitals
# Then (Z16) the Zeeman couplings between different levels and (S2-M8 P1) the
# hyperfine A and B factors against ATSP's hfs, see below.
# Needs the work directory of o3p_gj.sh (O_WORK, default $HOME/atomci_work/o3p).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" bpci hfs || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
src=${O_WORK:-${AC_WORK:-$HOME/atomci_work}/o3p}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

bp_run() { # name, 2Jmax, 2Jmin, eigenvalues per J
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
1,1,1,0
y
y
y
y
EOF
    # 2J, energy, g_J of each eigenvector
    awk '/2\*J =/ { j = $3 } /g_J=/ { g = $4 } /^ +[0-9]+ +-[0-9]/ { print j, $2, g }' $1.j
}
c_run() { # name
    "$A" "$T" $1.w $1.c 3 alpha=137.036 mvd rkrel so2 ss |
        awk '/^ ?[0-9]+\/2 / && NF >= 6 { split($1, a, "/"); print a[1], $3, $6 }'
}

# HF: 2p4, all three terms
cp "$src/O_hf.w" OHF.w
printf '%64s\n  1s  2s\n  2p( 4)\n 3P2\n  2p( 4)\n 1D2\n  2p( 4)\n 1S0\n*\n' "" > OHF.c
bp_run OHF 4 0 1,2 > OHF.atsp
c_run OHF > OHF.c_out

# n3: all LS terms of the valence SD configurations, the n = 3 orbitals
cp "$src/O_3.w" ON3.w
"$A" "$T" ON3.w "$src/cfg_ic3" 1 ls noso writec=ON3p > /dev/null
{
    head -2 ON3p_3P.c
    for f in ON3p_*.c; do sed -n '3,$p' $f | grep -v '^\*'; done
    echo '*'
} > ON3.c
echo "n3: $(grep -c '(' ON3.c) CSFs (all terms)"
bp_run ON3 4 2 1,2 > ON3.atsp
c_run ON3 > ON3.c_out

python3 - <<'PY'
from collections import defaultdict
def rd(f):
    d = defaultdict(list)
    for line in open(f):
        j, e, g = line.split()[:3]
        d[int(j)].append((float(e), float(g)))
    return {j: sorted(v) for j, v in d.items()}
for tag in ("OHF", "ON3"):
    a, c = rd(f"{tag}.atsp"), rd(f"{tag}.c_out")
    print(f"== {tag}")
    worst = 0.0
    for j in sorted(x for x in a if x > 0):
        for (ea, ga), (ec, gc) in zip(a[j], c[j]):
            worst = max(worst, abs(gc - ga))
            print(f"   2J={j}  E ATSP {ea:.9f}  C {ec:.9f}   g_J ATSP {ga:.10f}  C {gc:.13f}  diff {gc - ga:+.1e}")
    print(f"   max |g_J(C) - g_J(ATSP)| = {worst:.1e}")
PY

# Z16: the Zeeman couplings <A||L + g_s S||B> between different levels. ATSP
# prints no such element; it is rebuilt from bpci's eigenvectors (.j, 8
# decimals) with L and S diagonal in the CSFs, <(L S) J||S||(L S) J'> and
# <(L S) J||L||(L S) J'> by Edmonds (7.1.7)-(7.1.8) in ATSP's order |L S J>,
# and compared in absolute value (the eigenvector phases are each program's).
for t in OHF ON3; do
    "$A" "$T" $t.w $t.c 3 alpha=137.036 mvd rkrel so2 ss zoff > $t.full
    awk '/^Zeeman reduced/ { z = 1; next } /^levels relative/ { z = 0 }
         z && /^ ?[0-9]+\/2 / { print }' $t.full > $t.zoff
done
python3 - <<'PY'
import re
from fractions import Fraction as Fr
from math import factorial as fa, sqrt
gs = 2.002319304386
def f(x): return fa(int(x))
def tri(a, b, c):
    if c < abs(a - b) or c > a + b or (a + b + c) != int(a + b + c): return 0.0
    return f(a + b - c) * f(a - b + c) * f(-a + b + c) / f(a + b + c + 1)
def w6j(a, b, c, d, e, g):
    t = [tri(a, b, c), tri(a, e, g), tri(d, b, g), tri(d, e, c)]
    if 0 in t: return 0.0
    s = 0.0
    for k in range(0, 60):
        x = [k - a - b - c, k - a - e - g, k - d - b - g, k - d - e - c,
             a + b + d + e - k, a + c + d + g - k, b + c + e + g - k]
        if min(x) < 0: continue
        p = 1.0
        for y in x: p *= f(y)
        s += (-1) ** k * f(k + 1) / p
    return sqrt(t[0] * t[1] * t[2] * t[3]) * s
LL = "SPDFGHIKLMN"
def csf_terms(cfile):
    lines = [l.rstrip("\n") for l in open(cfile)][2:]
    out = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith("*"): break
        tok = lines[i + 1].split()[-1]
        m = re.match(r"(\d)([A-Z])", tok)
        out.append((Fr(int(m.group(1)) - 1, 2), LL.index(m.group(2))))
    return out
def atsp_levels(jfile, n):
    lev, lines, i = [], open(jfile).read().split("\n"), 0
    tj = None
    while i < len(lines):
        l = lines[i]
        m = re.search(r"2\*J =\s+(\d+)", l)
        if m: tj = int(m.group(1))
        if "Ssms=" in l:
            e = float(lines[i + 1].split()[1])
            vec, k = "", i + 2
            while len(vec) < 11 * n:
                vec += lines[k]; k += 1
            c = [float(vec[11 * q:11 * q + 11]) for q in range(n)]
            if e < 0: lev.append((tj, e, c))
            i = k; continue
        i += 1
    return lev
def red(S, L, tA, tB, op):
    JA, JB = Fr(tA, 2), Fr(tB, 2)
    ph = lambda x: -1.0 if int(x) % 2 else 1.0
    pre = sqrt((tA + 1) * (tB + 1))
    if op == "S":
        return ph(L + S + JA + 1) * pre * w6j(S, JA, L, JB, S, 1) * sqrt(S * (S + 1) * (2 * S + 1))
    return ph(L + S + JB + 1) * pre * w6j(L, JA, S, JB, L, 1) * sqrt(L * (L + 1) * (2 * L + 1))
for t in ("OHF", "ON3"):
    terms = csf_terms(f"{t}.c")
    al = atsp_levels(f"{t}.j", len(terms))
    # number the ATSP levels within each 2J by energy, as C does
    num = {}
    for tj in set(x[0] for x in al):
        for r, x in enumerate(sorted((x for x in al if x[0] == tj), key=lambda x: x[1])):
            num[(tj, r + 1)] = x
    print(f"== {t}: <A||L + g_s S||B>, C against ATSP's eigenvectors")
    worst = 0.0
    for l in open(f"{t}.zoff"):
        x = l.split()
        tA, nA, tB, nB = int(x[0].split("/")[0]), int(x[1]), int(x[2].split("/")[0]), int(x[3])
        cv = float(x[6])
        if (tA, nA) not in num or (tB, nB) not in num: continue
        a, b = num[(tA, nA)], num[(tB, nB)]
        v = sum(ca * cb * (red(S, L, tA, tB, "L") + gs * red(S, L, tA, tB, "S"))
                for (S, L), ca, cb in zip(terms, a[2], b[2]))
        d = abs(cv) - abs(v)
        worst = max(worst, abs(d))
        print(f"   {tA}/2 #{nA} - {tB}/2 #{nB}   C {cv:+.10f}   ATSP vectors {v:+.10f}   |C|-|ATSP| {d:+.1e}")
    print(f"   max ||C| - |ATSP|| = {worst:.1e}")
PY

# S2-M8 (P1): hyperfine A and B (ac_lsj_hfs_reduced, acatsp "hfs") against ATSP's
# hfs on the same bpci eigenvectors (CI, J dependent, the 3P levels of the
# dominant CSF of 3P2), with 2I = 2, mu_I = 1 n.m., Q = 1 b, so that ATSP's A is
# per mu_I/I and B per barn. ATSP's off-diagonal factors (hfs/lsjfact.f)
# are, in terms of the reduced elements:
#   A(J, J-1)  = A0 <J||T1||J-1> / sqrt(J (2J+1)(2J-1))
#   B(J, J-1)  = -B0 (1/2) sqrt(J(J-1)/((J+1)(2J+1)(2J-1))) <J||T2||J-1>
#   B(J, J-2)  = -B0 (1/4) sqrt(J(J-1)(2J-1)/((2J+1)(2J-3))) <J||T2||J-2>
# (compared in absolute value: the phases of the eigenvectors and of
# ATSP's L + S order differ). ATSP's A0 = 2 x 47.70534 MHz is older than
# CODATA 2018 (95.41053) by 1.6e-6; the comparison uses ATSP's.
for t in OHF ON3; do
    # the dominant CSF of the first (lowest 2J = 4, 3P2) eigenvector
    nr=$(awk '/Ssms=/ { getline; print $1; exit }' $t.j)
    hfs > $t.hfslog 2>&1 <<EOF
$t
N
1
C
Y
$nr
2,1.0,1.0
EOF
    "$A" "$T" $t.w $t.c 3 alpha=137.036 mvd rkrel so2 ss hfs |
        awk '/^Hyperfine reduced/ { z = 1; next } /^levels relative/ { z = 0 }
             z && /^ ?[0-9]+\/2 / { print }' > $t.hfsc
done
python3 - <<'PY'
import re
from math import sqrt
A0, B0 = 2 * 47.70534, 234.9647
def atsp(t):
    a, b, part = {}, {}, None
    for l in open(f"{t}.h"):
        if "A factors" in l: part = "A"
        elif "B factors" in l: part = "B"
        elif "Hyperfine parameters" in l: part = None
        x = l.split()
        if part and len(x) >= 3 and re.match(r"^\d", x[0]) and re.match(r"^\d", x[1]):
            j = lambda s: int(s.split("/")[0]) if "/" in s else 2 * int(s)
            key = (j(x[0]), j(x[1]))
            if part == "A": a[key] = [float(v) for v in x[2:6]]
            else: b[key] = float(x[2])
    return a, b
for t in ("OHF", "ON3"):
    a, b = atsp(t)
    # the J of bpci's run (n3 has no J = 0); the 3P levels are the lowest of
    # each J in both programs
    js = {int(m.group(1)) for m in re.finditer(r"2\*J =\s+(\d+)", open(f"{t}.j").read())}
    c = {}
    for l in open(f"{t}.hfsc"):
        x = l.split()
        tA, nA, tB, nB = int(x[0].split("/")[0]), int(x[1]), int(x[2].split("/")[0]), int(x[3])
        if nA == 1 and nB == 1 and tA in js and tB in js:
            c[(max(tA, tB), min(tA, tB))] = [float(v) for v in x[4:9]]
    print(f"== {t}: hyperfine, C against ATSP hfs (MHz per mu_I/I and per barn)")
    worst = 0.0
    for (tj, tjp), (orb, sd, con, t1, quad) in sorted(c.items()):
        J = tj / 2
        if tj == tjp:
            nrm = sqrt(J * (J + 1) * (2 * J + 1))
            ca = [A0 * v / nrm for v in (orb, sd, con, t1)]
            cb = -2 * B0 * quad * sqrt(J * (2 * J - 1) / ((J + 1) * (2 * J + 1) * (2 * J + 3)))
        elif tj - tjp == 2:
            nrm = sqrt(J * (2 * J + 1) * (2 * J - 1))
            ca = [A0 * v / nrm for v in (orb, sd, con, t1)]
            cb = -B0 * 0.5 * sqrt(J * (J - 1) / ((J + 1) * (2 * J + 1) * (2 * J - 1))) * quad
        else:
            ca = None
            cb = -B0 * 0.25 * sqrt(J * (J - 1) * (2 * J - 1) / ((2 * J + 1) * (2 * J - 3))) * quad
        lab = f"{tj}/2-{tjp}/2"
        if ca is not None and (tj, tjp) in a:
            for nm, x, y in zip(("orb", "sd", "con", "A"), ca, a[(tj, tjp)]):
                if abs(y) < 1e-7 and abs(x) < 1e-6: continue
                d = abs(x) - abs(y); worst = max(worst, abs(d) / max(abs(y), 1.0))
                print(f"   A {lab} {nm:4} C {x:+15.7f}  ATSP {y:+15.7f}  |C|-|ATSP| {d:+.1e}")
        if (tj, tjp) in b and (abs(cb) > 1e-6 or abs(b[(tj, tjp)]) > 1e-7):
            d = abs(cb) - abs(b[(tj, tjp)]); worst = max(worst, abs(d) / max(abs(b[(tj, tjp)]), 1.0))
            print(f"   B {lab}      C {cb:+15.7f}  ATSP {b[(tj, tjp)]:+15.7f}  |C|-|ATSP| {d:+.1e}")
    print(f"   max relative difference {worst:.1e}")
PY
