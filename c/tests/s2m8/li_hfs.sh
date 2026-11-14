#!/bin/bash
# li_hfs.sh -- S2-M8 (P5), report only (ATSP needed): the hyperfine constants
# of 7Li 1s2 2s 2S1/2 and 1s2 2p 2P1/2, 2P3/2 (acatsp "hfs") with growing
# correlation, against the nonrelativistic Hylleraas values (the reference;
# "nr") and experiment:
#   2S1/2  nr A = 401.738 MHz, from f_c = 2.905922 (Yan, McKenzie, Drake,
#          PRA 54, 1322 (1996)) with g_s = 2.00231930;
#          exp A = 401.7520433 MHz (half the ground-state splitting)
#   2P     nr A(1/2) = 45.96337, A(3/2) = -3.04214, B = -0.21259 MHz and the
#          operator terms of their Table II (Puchalski, Pachucki, PRA 79,
#          032510 (2009), exact g_s, Q = -4.00 fm2); in MHz for J = 3/2
#          a_c = -9.8916, a_o = 8.7184, a_d = -1.8635 (J = 1/2: -a_c, 2 a_o,
#          -10 a_d);
#          exp A(1/2) = 45.914(25), A(3/2) = -3.055(14), B = -0.221(29) MHz
#          (Orth, Ackermann, Otten, Z. Phys. A 273, 221 (1975); the J-diagonal
#          constants, without J mixing like the CI here)
# with mu_I = 3.2564268 mu_N, I = 3/2, Q = -0.0400 b. Experiment differs from
# nr by the relativistic, QED and nuclear-structure corrections (a few 1e-4);
# the point here is the convergence of each term -- the contact term (core
# polarization, 1s -> ns with the open shell), the orbital and spin-dipole
# terms and B -- in an MCHF expansion.
# Orbitals: ATSP MCHF, complete active spaces of three electrons from
# 1s2 2s or 1s2 2p, n = 3 .. NMAX (l <= d, f, g, g, ...), n >= 4 varying
# only the new layer (as li2p_yan.sh, whose L_n.w/L_n.c the 2P part reuses).
#   TERMS  (default "2S 2P"); NMAX (default 5)
#   LI_WORK  2P work directory (default $HOME/atomci_work/li2p)
#   LS_WORK  2S work directory (default $HOME/atomci_work/li2s_hfs)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen hfs || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
for term in ${TERMS:-2S 2P}; do
    if [ $term = 2S ]; then
        w=${LS_WORK:-${AC_WORK:-$HOME/atomci_work}/li2s_hfs}; el=2s; ref="1
0"
    else
        w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p}; el=2p; ref="0
1"
    fi
    mkdir -p "$w" && cd "$w"
    [ -s L_2.w ] || { hf > hf.log 2>&1 <<S
Z3,$term,3.
  1s
$el(1)
all
y
y
n
n
S
        cp wfn.out L_2.w; }
    printf '\n\n  1s( 2)  %s( 1)\n 1S0 2%s1 2%s \n*\n' $el ${term:1:1} ${term:1:1} > L_2.c
    for n in $(seq 2 ${NMAX:-5}); do
        if [ $n -ge 3 ] && ! { [ -s L_$n.w ] && [ -s L_$n.c ]; }; then
            case $n in 3) l=d;; 4) l=f;; *) l=g;; esac
            lsgen > lsgen_$n.log 2>&1 <<S



$n
$l


2
2
$ref
$term
3

S
            cp clist.out cfg.inp
            cp L_$((n - 1)).w wfn.inp
            if [ $n = 3 ]; then vary=all; elif [ $n = 4 ]; then vary=4s,4p,4d,4f
            else vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
            nonh > nonh_$n.log 2>&1
            mchf > mchf_$n.log 2>&1 <<S
Z3,3.
1
$vary
1s,$el
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
S
            cp wfn.out L_$n.w
            cp cfg.inp L_$n.c
        fi
        [ -s hfs_$n.out ] || "$A" "$T" L_$n.w L_$n.c 2 alpha=137.035999084 hfs > hfs_$n.out
        python3 - $term $n hfs_$n.out <<'PY'
import re, sys
term, n, path = sys.argv[1:4]
mu, I, Q = 3.2564268, 1.5, -0.0400
exp = {1: (401.7520433 if term == "2S" else 45.914), 3: -3.055}
nr = {1: (401.738 if term == "2S" else 45.96337), 3: -3.04214}
# Puchalski-Pachucki terms (orb, sd, con) for J = 1/2 and 3/2
nrt = {1: (17.4368, 18.6349, 9.8916), 3: (8.7184, -1.8635, -9.8916)}
e, sec, ncsf = {}, 0, None
out = []
for l in open(path):
    if l.startswith("Hyperfine"):
        sec = 1
        continue
    if l.startswith("levels relative"):
        break
    x = l.split()
    if not sec and len(x) >= 3 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
        e[int(x[0][0])] = float(x[2])
    if sec and len(x) == 11 and x[0] == x[2] and x[1] == "1" and x[3] == "1":
        tj = int(x[0][0])
        orb, sd, con, quad, aj, bj = map(float, (x[4], x[5], x[6], x[8], x[9], x[10]))
        a = aj * mu / I
        t = [a * v / (orb + sd + con) for v in (orb, sd, con)]
        s = (f"   {term} n={n} J={tj}/2: E {e[tj]:.8f}  A {a:10.4f} MHz  A/nr - 1 {a / nr[tj] - 1:+.2e}"
             f"  A/exp - 1 {a / exp[tj] - 1:+.2e}")
        if term == "2P":
            s += "  [" + ", ".join(f"{k} {v:+.3f} ({v / r - 1:+.3f})" for k, v, r in
                                   zip(("orb", "sd", "con"), t, nrt[tj]) if r) + "]"
        if tj == 3:
            s += f"  B {bj * Q:+.4f} MHz  B/nr - 1 {bj * Q / -0.21259 - 1:+.3f} (exp -0.221)"
        out.append(s)
print("\n".join(out))
PY
    done
done
