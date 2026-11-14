#!/bin/bash
# o3p_gj.sh -- S2-M7 (Z9), report only (ATSP needed): atomic oxygen 2p4 3P1, 3P2
# g_J against H. E. Radford and V. W. Hughes, Phys. Rev. 114, 1274 (1959):
#   g_J(16O, 3P1) / g_J(1H, 2S1/2) = 0.7496369(7),  3P2: 0.7496045(7)
# (errors twice the statistical standard error).
# Theory as in f2p_gj.sh: Lande value (3P_J: g = (g_L + g_s)/2 for J = 1, 2) with
# the free-electron g_s and the recoil-corrected g_L = 1 - (m/M)(1 + X), X the
# two-body recoil term of each wave function (acatsp rec2); C's alpha^2 terms with Pachucki's
# g_s factors; over g_J(H 1s) = g_s (1 - alpha^2/3).
# Intermediate coupling: 3P2 mixes with 1D2 through H_SO1 + H_SO2 + H_SS (3P1
# has no partner in 2p4). The correction g(LSJ) - g(pure LS) is taken from an
# LSJ CI over all LS terms of the configurations (acatsp configuration mode),
# on HF orbitals and on the n = 3 MCHF orbitals (valence SD to n = 3), and
# added to the 3P-only rows. The 3P-only rows: HF and MCHF on valence SD to
# n = 3, 4, 5 (CORE=1: also 1s single excitations; CORE=2: all-electron SD).
#   NS        3P-only correlated layers (default "3 4")
#   IC4=1     also the IC CI on the n = 4 orbitals (about 1 hour)
#   O_WORK    work directory (default $HOME/atomci_work/o3p)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${O_WORK:-${AC_WORK:-$HOME/atomci_work}/o3p}
rm -rf "$w" && mkdir -p "$w" && cd "$w"
al=alpha=137.035999084
hf > hf.log 2>&1 <<'S'
O,3P,8.
  1s  2s
2p(4)
all
y
y
n
n
S
cp wfn.out O_hf.w
echo "1s(2) 2s(2) 2p(4)" > cfg_hf
"$A" "$T" O_hf.w cfg_hf 9 $al gc2e oz2e rec2 > g_hf.out
"$A" "$T" O_hf.w cfg_hf 9 $al gc2e gc2epart=1 > s_hf.out
"$A" "$T" O_hf.w cfg_hf 9 $al noso gc2e oz2e > ls_hf.out
"$A" "$T" O_hf.w cfg_hf 9 $al so2 ss gc2e oz2e > ic_hf.out
"$A" "$T" O_hf.w cfg_hf 2 q=r,1s,2p,1 q=r,2s,2p,1 q=D,1s,2p q=D,2s,2p | grep '^query' > recoil.out
for n in ${NS:-3 4}; do
    case $n in 3) l=d; lims=$'2\n6';; 4) l=f; lims=$'2\n6\n0';; 5) l=g; lims=$'2\n6\n0\n0';; esac
    [ "${CORE:-0}" = 1 ] && lims="1${lims:1}"
    [ "${CORE:-0}" = 2 ] && lims="0${lims:1}"   # CORE=2: 1s2 double excitations too
    lsgen > lsgen_$n.log 2>&1 <<S



$n
$l

y
$lims
2
2
2
4
3P
2

S
    cp clist.out cfg.inp
    case $n in 3) cp O_hf.w wfn.inp; vary=all;; 4) cp O_3.w wfn.inp; vary=4s,4p,4d,4f;;
        5) cp O_4.w wfn.inp; vary=5s,5p,5d,5f,5g;; esac
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<S
O,8.
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
    cp wfn.out O_$n.w
    cp cfg.inp O_$n.c
    echo "n=$n: $(grep -c '(' O_$n.c) 3P CSFs, $(grep ETOTAL mchf_$n.log | tail -1), NaN lines $(grep -c NaN mchf_$n.log || true)"
    "$A" "$T" O_$n.w O_$n.c 2 $al gc2e oz2e rec2 > g_$n.out
    "$A" "$T" O_$n.w O_$n.c 2 $al gc2e gc2epart=1 > s_$n.out
done
# intermediate coupling on the n = 3 orbitals: every configuration of valence
# SD to n = 3 with all its LS terms
python3 - > cfg_ic3 <<'PY'
import itertools
orbs = [("2s", 0), ("2p", 1), ("3s", 0), ("3p", 1), ("3d", 2)]
ref = [2, 4, 0, 0, 0]
for occ in itertools.product(*[range(0, min(4 * l + 2, 6) + 1) for _, l in orbs]):
    if sum(occ) != 6 or sum(o * l for o, (_, l) in zip(occ, orbs)) % 2:
        continue
    if sum(max(r - o, 0) for r, o in zip(ref, occ)) > 2:
        continue
    print("1s(2) " + " ".join(f"{n}({o})" for (n, _), o in zip(orbs, occ) if o))
PY
echo "IC CI: $(wc -l < cfg_ic3) configurations"
"$A" "$T" O_3.w cfg_ic3 2 $al noso gc2e oz2e > ls_ic3.out
"$A" "$T" O_3.w cfg_ic3 2 $al so2 ss gc2e oz2e > ic_ic3.out
# IC4=1: the same on the n = 4 orbitals, valence SD to n = 4 (56 configurations;
# about 1 hour and 2.3 GB for the two runs; needs 4 in NS)
if [ "${IC4:-0}" = 1 ]; then
    python3 - > cfg_ic4 <<'PY'
import itertools
orbs = [("2s", 0), ("2p", 1), ("3s", 0), ("3p", 1), ("3d", 2), ("4s", 0), ("4p", 1), ("4d", 2), ("4f", 3)]
ref = [2, 4] + [0] * 7
for occ in itertools.product(*[range(0, min(4 * l + 2, 6) + 1) for _, l in orbs]):
    if sum(occ) != 6 or sum(o * l for o, (_, l) in zip(occ, orbs)) % 2:
        continue
    if sum(max(r - o, 0) for r, o in zip(ref, occ)) > 2:
        continue
    print("1s(2) " + " ".join(f"{n}({o})" for (n, _), o in zip(orbs, occ) if o))
PY
    echo "IC CI n = 4: $(wc -l < cfg_ic4) configurations"
    "$A" "$T" O_4.w cfg_ic4 2 $al noso gc2e oz2e > ls_ic4.out
    "$A" "$T" O_4.w cfg_ic4 2 $al so2 ss gc2e oz2e > ic_ic4.out
fi
python3 - hf ${NS:-3 4} <<'PY'
import re, sys
a2 = (1 / 137.035999084) ** 2
gs = 2.00231930436256
me = 5.48579909e-4
me_M = me / (15.99491462 - 8 * me)  # electron / 16O nucleus
gH = gs * (1 - a2 / 3)
cm = 219474.6313705
q = {l.split()[1]: float(l.split()[2]) for l in open("recoil.out")}
X = -2 / 3 * (q["r,1s,2p,1"] * q["D,1s,2p"] + q["r,2s,2p,1"] * q["D,2s,2p"])
gL = 1 - me_M * (1 + X)
exp = {2: 0.7496369, 4: 0.7496045}  # 2J -> ratio g_J(O)/g_J(H)
def rows(f):
    """level -> (E, term, [g_J, RMC, GC1e, ORMC, GC2e, OZ2e]) for the lowest level of each 2J"""
    c = {}
    for l in open(f):
        x = l.split()
        if re.match(r"^\d+/2$", x[0] if x else "") and len(x) >= 5:
            tj, k = int(x[0].split("/")[0]), int(x[1])
            c[(tj, k)] = (float(x[2]), x[3], [float(v) for v in x[5:]])
    return c
def ic_report(tag, ls_f, ic_f):
    ls, ic = rows(ls_f), rows(ic_f)
    e = {key: v[0] for key, v in ic.items()}
    d1 = [v[0] for key, v in ic.items() if v[1] == "1D" and key[0] == 4]
    fs = f"3P2-3P1 {(e[(2, 1)] - e[(4, 1)]) * cm:7.2f} (exp 158.27)  3P1-3P0 {(e[(0, 1)] - e[(2, 1)]) * cm:6.2f} (exp 68.0)"
    if d1:
        fs += f"  1D2-3P2 {(min(d1) - e[(4, 1)]) * cm:8.1f} (exp 15867.9)"
    d = {tj: sum(ic[(tj, 1)][2][:6]) - sum(ls[(tj, 1)][2][:6]) for tj in (2, 4)}
    print(f"IC [{tag}]: {fs}\n   g(LSJ) - g(LS): 3P1 {d[2] * 1e6:+.3f}e-6  3P2 {d[4] * 1e6:+.3f}e-6")
    # the 1D2 admixture scales as (spin-orbit / (E(1D2) - E(3P2)))^2: rescale the
    # 3P2 correction with the observed 3P2-3P1 interval and 1D2 position
    if d1:
        f = ((158.27 / ((e[(2, 1)] - e[(4, 1)]) * cm)) * ((min(d1) - e[(4, 1)]) * cm / 15867.9)) ** 2
        ds = {2: d[2], 4: d[4] * f}
        print(f"   rescaled to the observed levels (x {f:.4f}): 3P2 {ds[4] * 1e6:+.3f}e-6")
        return d, ds
    return d, None
print(f"recoil on HF by the closed formula (check of ac_ci_rec2): X = {X:+.5f}")
d_hf, s_hf = ic_report("HF, 2p4", "ls_hf.out", "ic_hf.out")
d_n3, s_n3 = ic_report("n = 3 orbitals, valence SD CI", "ls_ic3.out", "ic_ic3.out")
ics = {"HF": d_hf, "n3": d_n3, "n3 rescaled": s_n3}
import os
if os.path.exists("ic_ic4.out"):
    d_n4, s_n4 = ic_report("n = 4 orbitals, valence SD CI", "ls_ic4.out", "ic_ic4.out")
    ics.update({"n4": d_n4, "n4 rescaled": s_n4})
res = {}
for tag in sys.argv[1:]:
    g, s = rows(f"g_{tag}.out"), rows(f"s_{tag}.out")
    for tj in (2, 4):
        rmc, gc1, ormc, gc2, oz, rec = g[(tj, 1)][2][1:7]
        Xw = rec / 0.5                    # c_L = 1/2 for 3P1, 3P2 (ac_ci_rec2)
        gLw = 1 - me_M * (1 + Xw)
        sso = s[(tj, 1)][2][4]
        soo = gc2 - sso
        dr = rmc * (1 + (gs - 2) / 6) + gc1 * (gs - 1) + sso * (gs - 1) + soo * gs / 2 + ormc + oz
        lande = (gLw + gs) / 2
        out = []
        for k, d in ics.items():
            r = (lande + dr + d[tj]) / gH
            out.append(f"IC {k}: {r:.7f} ({(r - exp[tj]) * 1e7:+5.1f}e-7)")
        print(f"{tag:3} 3P{tj // 2}: exp {exp[tj]:.7f}(7)   " + "   ".join(out)
              + f"   alpha^2 {dr * 1e6:+7.2f}e-6 (SOO+OZ2e {(soo + oz) * 1e6:+6.2f})  X {Xw:+.4f}")
        res.setdefault(tag, {})[tj] = (lande + dr + s_n3[tj]) / gH
    r = res[tag]
    print(f"{tag:3} 3P1 - 3P2 (IC n3 rescaled): {(r[2] - r[4]) * 1e7:.2f}e-7  exp {(exp[2] - exp[4]) * 1e7:.2f}(10)e-7")
PY
