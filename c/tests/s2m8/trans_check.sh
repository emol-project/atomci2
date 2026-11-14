#!/bin/bash
# trans_check.sh -- S2-M8 (P2), report only (ATSP needed): the E1, M1, E2, M2,
# E3 and M3 lines of acatsp "trans" (ac_lsj_trans_reduced) against
#   (1) hydrogen 1s - 2p (exact orbitals from ATSP's hf): f = 24576/59049 in
#       sum, 1 : 2 over 2P1/2 and 2P3/2, length = velocity; the 2p fine-
#       structure M1, S = (g_s - 1)^2 4/3; 1s - 4f (E3): S summed over
#       4F5/2, 4F7/2 = 2 [2 8! (4/5)^9 / (768 sqrt35)]^2, 6 : 8; and M3
#       (= sqrt15 r^2 [C^2 x ((1/2) l + g_s s)]^(3)): 1s1/2 - 3d5/2 (spin
#       part only) S = (21/2) g_s^2 R^2, R = <1s|r^2|3d> = 5760 (3/4)^7 /
#       (81 sqrt30), and 3d3/2 - 3d5/2 S = (96/35) 126^2 (g_s - 3/2)^2;
#   (2) ATSP's trans (LSJ mode, length form) on bpci eigenvectors for Li
#       1s2 2s 2S - 1s2 2p 2P (complete active spaces to n = 3, l <= d) on one
#       orbital set (L_3.w of li2p_yan.sh, optimized for 2P; its 2s is a
#       correlation orbital, so 2S lies high: a test of the code, not of the
#       physics): E1 and M2 between 2S and 2P, M1 and E2 within 2P. ATSP's
#       magnetic lines use g_s = 2, so C is run with AC_TRANS_GS2=1 here.
#       M3 (and E2) on the same orbitals between that 2S and 2D = 1s2 3d
#       (one CSF, so both programs have the same vector) and within the 2D.
# Line strengths S are compared; gf and A also depend on the transition
# energy, which ATSP converts with the Rydberg constant of the finite mass
# and C with R_inf (8e-5 in Li). (3) checks C's conversion of S to A (E1, M1,
# E2, M2, E3, M3) against ATSP's formula (trp.f90) applied to C's own S and
# transition energies.
#   LI_WORK  li2p_yan.sh's work directory (default $HOME/atomci_work/li2p)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf bpci lsgen trans || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
src=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p}
[ -s "$src/L_3.w" ] || { echo "trans_check: $src/L_3.w missing (run li2p_yan.sh first)"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

# (1) hydrogen
hfrun() { # term, configuration
    hf > hf_$1.log 2>&1 <<S
H,$1,1.

$2
all
y
y
n
n
S
    cp wfn.out h_$1.w
}
hfrun 2S "1s(1)"
hfrun 2P "2p(1)"
hfrun 2F "4f(1)"
cat h_2S.w h_2P.w > h.w
printf '1s(1)\n2p(1)\n' > h.cfg
"$A" "$T" h.w h.cfg 3 alpha=137.035999084 trans > h.out
cat h_2S.w h_2F.w > hf.w
printf '1s(1)\n4f(1)\n' > hf.cfg
"$A" "$T" hf.w hf.cfg 3 alpha=137.035999084 trans > hf.out
hfrun 2D "3d(1)"
cat h_2S.w h_2D.w > hd.w
printf '1s(1)\n3d(1)\n' > hd.cfg
"$A" "$T" hd.w hd.cfg 3 alpha=137.035999084 trans > hd.out

# (2) Li: CSF lists, bpci (SO + SOO + SS), trans; C with the same Hamiltonian
gen() { # term, electrons in 2s, in 2p
    lsgen > lsgen_$1.log 2>&1 <<S



3
d


2
2
$2
$3
$1
3

S
    cp clist.out $1.c
    cp "$src/L_3.w" $1.w
}
gen 2S 1 0
gen 2P 0 1
gen 2D 1 0
{ head -2 2D.c; grep -A1 '^ *1s( 2) *3d( 1) *$' 2D.c; echo '*'; } > 2D1.c
mv 2D1.c 2D.c
bp() { # name, 2Jmax, 2Jmin, eigenvalue lines
    rm -f *.lst $1.j
    bpci > bp_$1.log 2>&1 <<S
$1,y,n
n
n
$2,$3
$4
2
n
1,1,1,0
y
y
y
y
S
}
bp 2S 1 1 "1"
bp 2P 3 1 "1
1"
bp 2D 5 3 "1
1"
tr() { # initial, final, type
    rm -f $1.$2.lsj
    trans > tr_$1_$2_$3.log 2>&1 <<S
$1
$2
n
n
y
y
$3
n
S
    awk -v t=$3 '/^ +[0-9]+ +-?[0-9]+\.[0-9]+ / { j[++n] = $1 } / S = / { print t, j[n - 1], j[n], $4 }' \
        $1.$2.lsj
}
cp 2P.c 2Q.c; cp 2P.w 2Q.w; cp 2P.j 2Q.j
{ tr 2S 2P E1; tr 2S 2P M2; tr 2P 2Q M1; tr 2P 2Q E2; } > li.atsp
cp 2D.c 2Q.c; cp 2D.w 2Q.w; cp 2D.j 2Q.j
{ tr 2S 2D M3; tr 2S 2D E2; tr 2D 2Q M3; tr 2D 2Q E2; } > lid.atsp
{ head -2 2S.c; for f in 2S.c 2P.c; do sed -n '3,$p' $f | grep -v '^\*'; done; echo '*'; } > SP.c
cp 2S.w SP.w
"$A" "$T" SP.w SP.c 2 alpha=137.036 mvd rkrel so2 ss trans > li.out
AC_TRANS_GS2=1 "$A" "$T" SP.w SP.c 2 alpha=137.036 mvd rkrel so2 ss trans > li_gs2.out
{ head -2 2S.c; for f in 2S.c 2D.c; do sed -n '3,$p' $f | grep -v '^\*'; done; echo '*'; } > SD.c
AC_TRANS_GS2=1 "$A" "$T" SP.w SD.c 2 alpha=137.036 mvd rkrel so2 ss trans > lid_gs2.out

python3 - <<'PY'
import re
def lines(f):
    out, on = [], False
    for l in open(f):
        if l.startswith("Transitions between"): on = True; continue
        if l.startswith("levels relative"): on = False
        x = l.split()
        if on and len(x) >= 7 and re.match(r"^\d+/2$", x[0]):
            out.append(x)
    return out
print("== (1) hydrogen 1s - 2p")
h = lines("h.out")
e1 = [x for x in h if x[4] == "E1"]
fl = sum(float(x[8]) for x in e1); fv = sum(float(x[9]) for x in e1)
print(f"   f(1s->2p) length {fl:.7f}  velocity {fv:.7f}  exact 24576/59049 = {24576 / 59049:.7f}  "
      f"ratio 2P3/2 : 2P1/2 = {float(e1[1][8]) / float(e1[0][8]):.6f} (2)")
m1 = [x for x in h if x[4] == "M1"][0]
gs = 2.00231930436256
print(f"   M1 2P1/2-2P3/2 S {float(m1[6]):.9f}  (g_s - 1)^2 4/3 = {(gs - 1) ** 2 * 4 / 3:.9f}")
from math import factorial, sqrt
e3 = {x[2]: float(x[6]) for x in lines("hf.out") if x[4] == "E3"}
ex = 2 * (2 * factorial(8) * 0.8 ** 9 / (768 * sqrt(35))) ** 2
print(f"   E3 1s - 4f: S(5/2) + S(7/2) {e3['5/2'] + e3['7/2']:.9f}  exact {ex:.9f}"
      f"  (rel {(e3['5/2'] + e3['7/2']) / ex - 1:+.1e})  ratio 7/2 : 5/2 = {e3['7/2'] / e3['5/2']:.7f} (4/3)")
R = 5760 * 0.75 ** 7 / (81 * sqrt(30))
m3 = {(x[0], x[2]): float(x[6]) for x in lines("hd.out") if x[4] == "M3"}
for key, ex, what in [(("1/2", "5/2"), 21 / 2 * gs ** 2 * R ** 2, "1s1/2 - 3d5/2, (21/2) g_s^2 R^2"),
                      (("3/2", "5/2"), 96 / 35 * 126 ** 2 * (gs - 1.5) ** 2, "3d3/2 - 3d5/2, (96/35) 126^2 (g_s - 3/2)^2")]:
    print(f"   M3 {what}: S {m3[key]:.9e}  exact {ex:.9e}  (rel {m3[key] / ex - 1:+.1e})")
print("== (2) Li 2S - 2P (E1, M2), within 2P (M1, E2), 2S - 2D and within 2D (M3, E2): line"
      " strengths, C (g_s = 2 in M1, M2, M3) against ATSP trans")
# C's levels (2J, number in that J): 2P1/2 (1, 1), 2P3/2 (3, 1), 2S1/2 (1, 2)
# (2S lies above 2P on these orbitals); ATSP's files hold just these three.
# With 2D: 2S1/2 (1, 1), 2D3/2 (3, 1), 2D5/2 (5, 1).
worst = 0.0
for cf, af, want in (("li_gs2.out", "li.atsp", {(1, 1), (3, 1), (1, 2)}),
                     ("lid_gs2.out", "lid.atsp", {(1, 1), (3, 1), (5, 1)})):
    c = {}
    for x in lines(cf):
        lo = (int(x[0].split("/")[0]), int(x[1]))
        up = (int(x[2].split("/")[0]), int(x[3]))
        if lo in want and up in want:
            c[(x[4], frozenset((lo[0], up[0])))] = float(x[6])
    for l in open(af):
        typ, ja, jb, s = l.split()
        s = float(s.replace("D", "e"))
        key = (typ, frozenset((int(ja), int(jb))))
        if key not in c: continue
        cs = c[key]
        d = cs / s - 1; worst = max(worst, abs(d))
        print(f"   {typ} 2J {ja} - {jb}   C {cs:.6e}   ATSP {s:.5e}   C/ATSP - 1 {d:+.1e}")
print(f"   max |C/ATSP - 1| = {worst:.1e} (ATSP prints 6 digits)")
print("== (3) A from C's S and dE through ATSP's conversion (biotr_tr/trp.f90, R_inf):"
      " E1, M1, E2, M2, E3, M3")
# TRP = (2l+1)(l+1)/l / ((2l+1)!!)^2 pi c / R^(2l) * sigma^(2l+1) times
# 4 alpha^(2l+1)/2^(2l) (E l) or alpha^(2l+3)/2^(2l) (M l); A = S TRP / g_up
from math import pi
def trp(typ, sig):
    l = int(typ[1]); dbl = [3.0, 15.0, 105.0][l - 1]
    f = (2 * l + 1) * (l + 1) / l / dbl ** 2 * pi * 2.99792458e10 / 1.0973731534e5 ** (2 * l)
    al = 7.29735308e-3
    f *= 4 / 2 ** (2 * l) * al ** (2 * l + 1) if typ[0] == "E" else al ** (2 * l + 3) / 2 ** (2 * l)
    return f * sig ** (2 * l + 1)
def energies(f):  # (2J, #) -> E (hartree), from the level table
    e = {}
    for l in open(f):
        if l.startswith(("Zeeman", "Hyperfine", "Transitions")): break
        x = l.split()
        if len(x) >= 3 and re.match(r"^\d+/2$", x[0]):
            e[(x[0], x[1])] = float(x[2])
    return e
worst = 0.0
for f in ("h.out", "hf.out", "hd.out", "li.out", "lid_gs2.out"):
  ev = energies(f)
  for x in lines(f):
    typ, s = x[4], float(x[6])
    sig = (ev[(x[2], x[3])] - ev[(x[0], x[1])]) * 219474.6313632
    a = float(x[10] if typ == "E1" else x[8] if typ == "M1r" else x[7])
    ref = s * trp(typ, sig) / (int(x[2].split("/")[0]) + 1)
    worst = max(worst, abs(a / ref - 1))
    print(f"   {f[0]} {typ} {x[0]} #{x[1]} - {x[2]} #{x[3]}   A C {a:.6e}   from trp {ref:.6e}"
          f"   ratio - 1 {a / ref - 1:+.1e}")
print(f"   max |ratio - 1| = {worst:.1e} (alpha and R_inf of 1993 in trp; C prints 7 digits)")
PY
