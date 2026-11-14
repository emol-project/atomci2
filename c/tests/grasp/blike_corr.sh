#!/bin/bash
# blike_corr.sh -- S2-M7 (Z5c), report only (GRASP and ATSP needed). B-like
# 1s2 2s2 2p 2P: is the Z5 residual of C against GRASP the negative-energy
# contribution that no-pair Dirac-Coulomb leaves out?
# Delta g_int = g_J - g_D (hydrogenic Dirac), against Agababaev et al.,
# arXiv:1812.06429 (2p1/2) and arXiv:1812.06431 (2p3/2), Table 1 "Delta g_int"
# (Kohn-Sham column): first order rigorous QED including the negative-energy
# part, second order in the Breit approximation; the total does not depend on
# the screening potential, its split into [+] and [-] does.
#   GRASP: single-configuration DHF + rci (no Breit) + rhfszeeman95.
#   C: full BP (RMC + GC1e + ORMC + GC2e + OZ2e) and 1e + GC2e(SSO) on ATSP
#      orbitals: HF, then MCHF on the n = 2 complex (2s2 2p + 2p3) and on
#      single and double excitations of all five electrons to n = 3 (l <= d)
#      and n = 4 (l <= f), each n >= 3 varying only the new layer.
#   ZS        nuclear charges (default "10 14 18")
#   B_WORK    work directory (default $HOME/atomci_work/blike)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
NS=${NS:-2 3 4}
for Z in ${ZS:-10 14 18}; do
    w=${B_WORK:-${AC_WORK:-$HOME/atomci_work}/blike}/Z$Z
    rm -rf "$w/grasp" && mkdir -p "$w/grasp" && cd "$w/grasp"
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
    rcsfgenerate > csf.out 2>&1 <<S
*
0
1s(2,i)2s(2,i)2p(1,i)

2s,2p
1,3
0
n
S
    cp rcsf.out rcsf.inp
    rangular > ang.out 2>&1 <<'S'
y
S
    rwfnestimate > est.out 2>&1 <<'S'
y
3
*
S
    rmcdhf > scf.out 2>&1 <<'S'
y
1
1
5
*
*
200
S
    grep -qE "Convergence not obtained|Failure" scf.out && echo "Z=$Z: GRASP DHF failed"
    rsave X > /dev/null 2>&1
    rci > rci.out 2>&1 <<'S'
y
X
n
n
n
n
n
n
1
1
S
    rhfszeeman95 > zee.out 2>&1 <<'S'
y
X
y
n
S
    cd "$w"
    rm -f wfn.inp
    hf > hf.log 2>&1 <<S
B$Z,2P,$Z.
  1s
2s(2)2p(1)
all
y
y
n
n
S
    cp wfn.out B_hf.w
    echo "1s(2) 2s(2) 2p(1)" > hf.cfg
    "$A" "$T" B_hf.w hf.cfg 2 alpha=137.035999084 gc2e oz2e > g_hf.out
    "$A" "$T" B_hf.w hf.cfg 2 alpha=137.035999084 gc2e gc2epart=1 > s_hf.out
    outs="hf"
    for n in $NS; do
        case $n in 2) l=p; ex=2;; 3) l=d; ex=2;; *) l=f; ex=2;; esac
        lsgen > lsgen_$n.log 2>&1 <<S



$n
$l


2
2
2
1
2P
$ex

S
        cp clist.out cfg.inp
        rm -f wfn.inp
        if [ $n = 2 ]; then cp B_hf.w wfn.inp; vary=all
        elif [ $n = 3 ]; then cp B_2.w wfn.inp; vary=3s,3p,3d
        else cp B_$((n - 1)).w wfn.inp; vary=${n}s,${n}p,${n}d,${n}f; fi
        nonh > nonh_$n.log 2>&1
        mchf > mchf_$n.log 2>&1 <<S
B$Z,$Z.
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
        cp wfn.out B_$n.w
        cp cfg.inp B_$n.c
        echo "Z=$Z n=$n: $(grep -c '(' B_$n.c) CSFs, $(grep ETOTAL mchf_$n.log | tail -1), NaN lines $(grep -c NaN mchf_$n.log || true)"
        "$A" "$T" B_$n.w B_$n.c 2 alpha=137.035999084 gc2e oz2e > g_$n.out
        "$A" "$T" B_$n.w B_$n.c 2 alpha=137.035999084 gc2e gc2epart=1 > s_$n.out
        outs="$outs $n"
    done
    python3 - "$Z" $outs <<'PY'
import math, re, sys
Z = int(sys.argv[1]); tags = sys.argv[2:]
a = 1 / 137.035999084
# Agababaev et al., Table 1, Delta g_int (KS), x 1e-6
ref = {1: {10: 356.523, 12: 429.573, 14: 502.915, 16: 576.572, 18: 650.584, 20: 724.998},
       3: {10: 264.143, 12: 317.547, 14: 371.108, 16: 424.828, 18: 478.726, 20: 532.824}}
def gD(kappa, n=2):
    k = abs(kappa); j = k - 0.5
    gam = math.sqrt(k * k - (Z * a) ** 2)
    E = 1 / math.sqrt(1 + (Z * a / (n - k + gam)) ** 2)
    return kappa * (2 * kappa * E - 1) / (2 * j * (j + 1))
lande = {1: 2 / 3, 3: 4 / 3}
kap = {1: 1, 3: -2}
g = {}
for l in open("grasp/X.ch"):
    m = re.match(r"\s+1\s+(\d)/2\s+[-+]\s+\S+\s+\S+\s+(\S+)", l)
    if m:
        g[int(m.group(1))] = float(m.group(2).replace("D", "e"))
def rd(f):
    c = {}
    for l in open(f):
        x = l.split()
        if len(x) >= 10 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
            c[int(x[0][0])] = [float(v) for v in x[6:]]
    return c
for tj in (1, 3):
    d = gD(kap[tj])
    gg = g[tj] - d
    r = ref[tj].get(Z, float("nan")) * 1e-6
    print(f"Z={Z:2} 2p{tj}/2  Agababaev Delta g_int {r * 1e6:9.3f}e-6   GRASP(no-pair DC, 1 cfg) {gg * 1e6:9.3f}e-6")
    for t in tags:
        f, s = rd(f"g_{t}.out")[tj], rd(f"s_{t}.out")[tj]
        full = lande[tj] + sum(f[:5]) - d
        sso = lande[tj] + sum(s[:4]) - d
        print(f"   {t:3}  C full BP {full * 1e6:9.3f}e-6 ({(full - r) / r:+.2%})   C 1e+SSO {sso * 1e6:9.3f}e-6"
              f"   GC2e(SSO) {s[3] * 1e6:8.3f}e-6   (C 1e+SSO - GRASP)/SSO {(sso - gg) / s[3]:+.3f}"
              f"   SOO+OZ2e {(f[3] - s[3] + (f[4] if len(f) > 4 else 0)) * 1e6:8.3f}e-6")
PY
done
