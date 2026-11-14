#!/bin/bash
# f2p_gj.sh -- S2-M7 (Z8), report only (ATSP needed): atomic fluorine
# 1s2 2s2 2p5 2P g_J against experiment (Radford, Hughes, Beltran-Lopez,
# Phys. Rev. 123, 153 (1961)):
#   g_J(19F, 2P3/2) / g_J(1H, 2S1/2) = 2/3 - (497 +- 1) x 1e-6
# and the HF calculation of V. Beltran-Lopez et al., Phys. Rev. 172, 44 (1968),
# which this script reproduces on HF orbitals (2P3/2 ratio -497.5e-6, 2P1/2
# g_J - 2/3 = -1074.2e-6 with Eq. (3); quoted 2P1/2 value -1072(2)e-6).
# Theory: Lande value with the free-electron g_s and the recoil-corrected
# g_L = 1 - (m/M)(1 + X), X = <sum_{i!=j} (r_i x p_j)_z>/<L_z> of each wave
# function (acatsp rec2, ac_ci_rec2; on HF also the closed formula), plus C's
# alpha^2 terms with the g_s factors of Pachucki's operators (as in run_he.sh:
# GC1e and SSO x (g_s - 1), SOO x g_s/2, RMC x (1 + (g_s - 2)/6), ORMC and
# OZ2e x 1; the rank-2 part of -(g_s - 2)/4 (p.sigma)(p.B) is left out), over
# g_J(H 1s) = g_s (1 - alpha^2/3).
# Orbitals: HF, then MCHF on single and double excitations of 2s2 2p5 (1s2
# inactive) to n = 3 (l <= d) and n = 4 (l <= f), n = 4 varying the new layer.
#   F_WORK    work directory (default $HOME/atomci_work/f2p)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${F_WORK:-${AC_WORK:-$HOME/atomci_work}/f2p}
rm -rf "$w" && mkdir -p "$w" && cd "$w"
hf > hf.log 2>&1 <<'S'
F,2P,9.
  1s  2s
2p(5)
all
y
y
n
n
S
cp wfn.out F_hf.w
echo "1s(2) 2s(2) 2p(5)" > hf.cfg
"$A" "$T" F_hf.w hf.cfg 2 alpha=137.035999084 gc2e oz2e rec2 > g_hf.out
"$A" "$T" F_hf.w hf.cfg 2 alpha=137.035999084 gc2e gc2epart=1 > s_hf.out
# two-body recoil term on the HF orbitals (Beltran-Lopez et al. Eq. (3)):
#   X = -(2/3) sum_ns <ns|r|2p> int R_ns (R_2p' + 2 R_2p/r) r^2 dr
"$A" "$T" F_hf.w hf.cfg 2 q=r,1s,2p,1 q=r,2s,2p,1 q=D,1s,2p q=D,2s,2p | grep '^query' > recoil.out
for n in ${NS:-3 4}; do
    case $n in 3) l=d; lims=$'2\n5';; 4) l=f; lims=$'2\n5\n0';; 5) l=g; lims=$'2\n5\n0\n0';; esac
    # CORE=1: also single excitations from 1s (core-valence; at most two in all);
    # CORE=2: any double excitation, 1s2 included (core-core)
    # (minimum electrons with n = 1: 1 or 0; with n <= 2: 7, i.e. two holes at most)
    [ "${CORE:-0}" -ge 1 ] && lims="$((2 - CORE))${lims:1}" && lims=${lims/$'\n5'/$'\n7'}
    lsgen > lsgen_$n.log 2>&1 <<S



$n
$l

y
$lims
2
2
2
5
2P
2

S
    cp clist.out cfg.inp
    case $n in 3) cp F_hf.w wfn.inp; vary=all;; 4) cp F_3.w wfn.inp; vary=4s,4p,4d,4f;;
        5) cp F_4.w wfn.inp; vary=5s,5p,5d,5f,5g;; esac
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<S
F,9.
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
    cp wfn.out F_$n.w
    cp cfg.inp F_$n.c
    echo "n=$n: $(grep -c '(' F_$n.c) CSFs, $(grep ETOTAL mchf_$n.log | tail -1), NaN lines $(grep -c NaN mchf_$n.log || true)"
    "$A" "$T" F_$n.w F_$n.c 2 alpha=137.035999084 gc2e oz2e rec2 > g_$n.out
    "$A" "$T" F_$n.w F_$n.c 2 alpha=137.035999084 gc2e gc2epart=1 > s_$n.out
done
python3 - hf ${NS:-3 4} <<'PY'
import re, sys
a2 = (1 / 137.035999084) ** 2
gs = 2.00231930436256
me_M = 5.48579909e-4 / (18.99840316 - 9 * 5.48579909e-4)  # electron / 19F nucleus
# g_L = 1 - (m/M)(1 + X/<L_z>), X from the HF orbitals (recoil.out)
q = {l.split()[1]: float(l.split()[2]) for l in open("recoil.out")}
X = -2 / 3 * (q["r,1s,2p,1"] * q["D,1s,2p"] + q["r,2s,2p,1"] * q["D,2s,2p"])
print(f"recoil on HF by the closed formula (check of ac_ci_rec2): X = {X:+.5f}, g_L = 1 - (m/M) x {1 + X:.5f};"
      f" 2P3/2 isotopic correction {-me_M * (1 + X) * 2 / 3 * 1e6:+.2f}e-6 (paper Table II, Eq. (3): -6.9e-6)")
gL = 1 - me_M * (1 + X)
gH = gs * (1 - a2 / 3)
cL = {1: 4 / 3, 3: 2 / 3}   # 2P: g = g_L cL + g_s cS
cS = {1: -1 / 3, 3: 1 / 3}
# Radford, Hughes, Beltran-Lopez, Phys. Rev. 123, 153 (1961): -g_J(F)/g_p =
# 438.4839(3), -g_J(H)/g_p = 658.2167(6) (quoted errors: 4 x statistical)
rF, rH = 438.4839, 658.2167
exp = {3: rF / rH - 2 / 3}
dexp = ((3e-4 / rH) ** 2 + (6e-4 * rF / rH ** 2) ** 2) ** 0.5
calc68 = {1: -1072e-6}
def rd(f):
    c = {}
    for l in open(f):
        x = l.split()
        if len(x) >= 10 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
            c[int(x[0][0])] = [float(v) for v in x[6:]]
    return c
for tag in sys.argv[1:]:
    g, s = rd(f"g_{tag}.out"), rd(f"s_{tag}.out")
    for tj in (3, 1):
        rmc, gc1, ormc, gc2, oz, rec = g[tj][:6]
        Xw = rec / cL[tj]                 # the recoil term of this wave function (ac_ci_rec2)
        gLw = 1 - me_M * (1 + Xw)
        sso = s[tj][3]
        soo = gc2 - sso
        d2 = rmc + gc1 + ormc + gc2 + oz                       # g_s = 2 convention
        dr = rmc * (1 + (gs - 2) / 6) + gc1 * (gs - 1) + sso * (gs - 1) + soo * gs / 2 + ormc + oz
        lande = gLw * cL[tj] + gs * cS[tj]
        if tj == 3:   # measured as g_J(F) / g_J(H)
            r, what, ref = (lande + dr) / gH - 2 / 3, "g_J(F)/g_J(H) - 2/3", f"exp {exp[3] * 1e6:+.2f}({dexp * 1e8:.0f})"
        else:         # the 1968 calculation gives g_J itself
            r, what, ref = lande + dr - 2 / 3, "g_J - 2/3", f"1968 calc {calc68[1] * 1e6:+.1f}(2)"
        print(f"{tag:3} 2P{tj}/2: {what} = {r * 1e6:+8.2f}e-6   ({ref})   "
              f"alpha^2 terms {dr * 1e6:+7.2f}e-6 (g_s=2: {d2 * 1e6:+7.2f}; SOO+OZ2e {(soo + oz) * 1e6:+6.2f})  X {Xw:+.4f}")
PY
