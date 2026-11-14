#!/bin/bash
# blike_grasp_corr.sh -- S2-M7 (Z5d), report only (GRASP and ATSP needed).
# B-like 1s2 2s2 2p 2P: does the residual C[1e + GC2e(SSO)] - GRASP (no-pair
# Dirac-Coulomb) stay the same when both sides are correlated in the same
# CSF space? 1s2 is inactive on both sides:
#   1cfg  2s2 2p
#   n2    2s2 2p + 2p3 (the n = 2 complex), all orbitals varied
#   n3    single and double excitations of 2s2 2p to n = 3 (l <= d), only
#         the n = 3 orbitals varied
# GRASP: MCDHF (EOL over the two J levels) + rci (no Breit) + rhfszeeman95.
# C: ATSP HF / MCHF (lsgen with at least two electrons in n = 1) + acatsp.
#   ZS        nuclear charges (default "10 14 18")
#   B_WORK    work directory (default $HOME/atomci_work/blike_gc)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL

# grasp_level tag "reference lines" active excitations "orbitals to vary"
grasp_level() {
    local tag=$1 refs=$2 act=$3 ex=$4 vary=$5
    rcsfgenerate > csf_$tag.out 2>&1 <<S
*
0
$refs

$act
1,3
$ex
n
S
    cp rcsf.out rcsf.inp
    rangular > ang_$tag.out 2>&1 <<'S'
y
S
    if [ -s rwfn.out ]; then
        cp rwfn.out prev.w
        rwfnestimate > est_$tag.out 2>&1 <<'S'
y
1
prev.w
*
3
*
S
    else
        rwfnestimate > est_$tag.out 2>&1 <<'S'
y
3
*
S
    fi
    rmcdhf > scf_$tag.out 2>&1 <<S
y
1
1
5
$vary
1s,2s,2p*
200
S
    grep -qE "Convergence not obtained|Failure" scf_$tag.out && echo "   GRASP $tag: MCDHF failed"
    rsave $tag > /dev/null 2>&1
    rci > rci_$tag.out 2>&1 <<S
y
$tag
n
n
n
n
n
n
1
1
S
    rhfszeeman95 > zee_$tag.out 2>&1 <<S
y
$tag
y
n
S
}

# atsp_level tag n l wfn.inp "orbitals to vary"
atsp_level() {
    local tag=$1 n=$2 l=$3 start=$4 vary=$5 lims
    if [ $n = 2 ]; then lims=$'2'; else lims=$'2\n3'; fi
    lsgen > lsgen_$tag.log 2>&1 <<S



$n
$l

y
$lims
2
2
2
1
2P
2

S
    cp clist.out cfg.inp
    cp $start wfn.inp
    nonh > nonh_$tag.log 2>&1
    mchf > mchf_$tag.log 2>&1 <<S
B,$Z.
1
$vary
1s,2s,2p
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
S
    cp wfn.out B_$tag.w
    cp cfg.inp B_$tag.c
    # C's CI: every configuration of the space with all its LS terms (as GRASP's
    # relativistic CSFs, e.g. 2p3 4S and 2D mix into the J = 3/2 level)
    python3 - $n > cfg_$tag <<'PY'
import itertools, sys
orbs = [("2s", 0), ("2p", 1)] + ([("3s", 0), ("3p", 1), ("3d", 2)] if sys.argv[1] == "3" else [])
ref = [2, 1] + [0] * (len(orbs) - 2)
for occ in itertools.product(*[range(0, min(4 * l + 2, 3) + 1) for _, l in orbs]):
    if sum(occ) != 3 or sum(o * l for o, (_, l) in zip(occ, orbs)) % 2 != 1:
        continue
    if sum(max(r - o, 0) for r, o in zip(ref, occ)) > 2:
        continue
    if sys.argv[1] == "2" and occ not in ((2, 1), (0, 3)):
        continue
    print("1s(2) " + " ".join(f"{n}({o})" for (n, _), o in zip(orbs, occ) if o))
PY
    "$A" "$T" B_$tag.w cfg_$tag 2 alpha=137.035999084 gc2e gc2epart=1 > s_$tag.out
}

for Z in ${ZS:-10 14 18}; do
    w=${B_WORK:-${AC_WORK:-$HOME/atomci_work}/blike_gc}/Z$Z
    rm -rf "$w" && mkdir -p "$w/grasp" && cd "$w/grasp"
    M=$((2 * Z + 1))
    rnucleus > nuc.out 2>&1 <<S
$Z
$M
n
$M
1
1
1
S
    grasp_level 1cfg "1s(2,i)2s(2,i)2p(1,i)" "2s,2p" 0 "*"
    grasp_level n2 "1s(2,i)2s(2,i)2p(1,i)
1s(2,i)2p(3,i)" "2s,2p" 0 "*"
    grasp_level n3 "1s(2,i)2s(2,*)2p(1,*)" "3s,3p,3d" 2 "3s,3p*,3d*"
    cd "$w"
    hf > hf.log 2>&1 <<S
B,2P,$Z.
  1s
2s(2)2p(1)
all
y
y
n
n
S
    cp wfn.out B_1cfg.w
    echo "1s(2) 2s(2) 2p(1)" > hf.cfg
    "$A" "$T" B_1cfg.w hf.cfg 2 alpha=137.035999084 gc2e gc2epart=1 > s_1cfg.out
    atsp_level n2 2 p B_1cfg.w all
    atsp_level n3 3 d B_n2.w 3s,3p,3d
    for t in n2 n3; do
        echo "   ATSP $t: MCHF on $(grep -c '(' B_$t.c) 2P CSFs, $(grep ETOTAL mchf_$t.log | tail -1); C's CI on $(wc -l < cfg_$t) configurations"
    done
    python3 - "$Z" <<'PY'
import math, re, sys
Z = int(sys.argv[1])
a = 1 / 137.035999084
def gD(kappa, n=2):
    k = abs(kappa); j = k - 0.5
    gam = math.sqrt(k * k - (Z * a) ** 2)
    E = 1 / math.sqrt(1 + (Z * a / (n - k + gam)) ** 2)
    return kappa * (2 * kappa * E - 1) / (2 * j * (j + 1))
lande = {1: 2 / 3, 3: 4 / 3}
kap = {1: 1, 3: -2}
def grasp(tag):
    g = {}
    for l in open(f"grasp/{tag}.ch"):
        m = re.match(r"\s+1\s+(\d)/2\s+[-+]\s+\S+\s+\S+\s+(\S+)", l)
        if m:  # the energy lines match too; g_J (A, B, g_J) comes last
            g[int(m.group(1))] = float(m.group(2).replace("D", "e"))
    return g
def c(tag):
    r = {}
    for l in open(f"s_{tag}.out"):
        x = l.split()
        if len(x) >= 10 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
            r[int(x[0][0])] = [float(v) for v in x[6:]]
    return r
for tj in (1, 3):
    d = gD(kap[tj])
    for t in ("1cfg", "n2", "n3"):
        gg = grasp(t)[tj] - d
        s = c(t)[tj]
        cs = lande[tj] + sum(s[:4]) - d
        print(f"Z={Z:2} 2p{tj}/2 {t:4}  GRASP {gg * 1e6:9.3f}e-6  C 1e+SSO {cs * 1e6:9.3f}e-6"
              f"  residual {(cs - gg) * 1e6:+8.3f}e-6  /SSO {(cs - gg) / s[3]:+.3f}")
PY
done
