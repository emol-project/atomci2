#!/bin/bash
# is_atsp.sh -- S2-M10, report only (ATSP's hf, lsgen, nonh, mchf needed):
# the isotope-shift parameters of acatsp "is" (ac_lsj_is_level)
#   K_NMS = <sum p_i^2/2>, K_SMS = <sum_{i<j} p_i.p_j>, rho(0) = <sum delta(r_i)>
# 1. one electron (H 1s, He+ 1s, Li2+ 2p): K_NMS = Z^2/(2 n^2), K_SMS = 0,
#    rho(0) = Z^3/(pi n^3) for s (0 for p);
# 2. Li 1s2 2s 2S and 1s2 2p 2P: HF and MCHF (complete active space of three
#    electrons, n = 3, l <= d), against mchf's kinetic energy and its
#    "Ssms" (the expectation value of -sum_{i<j} grad_i.grad_j, written to
#    the .l file) in the same orbitals and CSFs. On HF, K_SMS(2S) = 0 (no
#    s-p exchange) and K_SMS(2P) is the 1s-2p exchange term.
# The MCDHF values of Gaidamauskas et al. (J. Phys. B 44, 175003 (2011),
# arXiv:1106.1988, Tables 4-5, SDT n = 10, the non-relativistic operator
# K^1) are printed for scale: K_SMS 0.30241 (2S), 0.24751 (2P).
#   IS_WORK   work directory (default: a temporary one)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
if [ -n "${IS_WORK:-}" ]; then w=$IS_WORK; mkdir -p "$w"; else w=$(mktemp -d); trap 'rm -rf "$w"' EXIT; fi
cd "$w"
co="alpha=137.035999084 noso is"
# 1. one electron
for at in "H 1 1s 2S" "He 2 1s 2S" "Li 3 2p 2P"; do
    set -- $at
    hf > hf_$1.log 2>&1 <<S
$1,$4,$2.

$3(1)
all
y
y
n
n
S
    cp wfn.out $1.w
    printf '%s(1)\n' $3 > $1.cfg
    "$A" "$T" $1.w $1.cfg 1 $co > $1.out
done
# 2. Li 2S and 2P
for st in "2S 2s 0 1s,2s" "2P 2p 1 1s,2p"; do
    set -- $st
    mkdir -p $1 && cd $1
    hf > hf.log 2>&1 <<S
Li,$1,3.
  1s
$2(1)
all
y
y
n
n
S
    cp wfn.out hf.w
    printf '\n\n  1s( 2)  %s( 1)\n 1S0 2%s1 2%s \n*\n' $2 ${1:1:1} ${1:1:1} > hf.c
    "$A" "$T" hf.w hf.c 1 $co > hf.out
    lsgen > lsgen.log 2>&1 <<S



3
d


2
2
$((1 - $3))
$3
$1
3

S
    cp clist.out cfg.inp
    cp hf.w wfn.inp
    nonh > nonh.log 2>&1
    mchf > mchf.log 2>&1 <<S
Li,3.
1
all
$4
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
S
    cp wfn.out mc.w
    cp cfg.inp mc.c
    "$A" "$T" mc.w mc.c 1 $co > mc.out
    cat ./*.l > mc.l
    cd ..
done
python3 - <<'PY'
import math, re

def table(fn):
    """the isotope shift rows of acatsp: {2J: (K_NMS, K_SMS, rho0)} of level #1"""
    out, on = {}, False
    for l in open(fn):
        if l.startswith("Isotope shift"):
            on = True
        x = l.split()
        if on and len(x) == 6 and re.match(r"^\d+/2$", x[0]) and x[1] == "1":
            out[int(x[0][:-2])] = (float(x[2]), float(x[3]), float(x[5]))
    return out

worst = 0.0
for el, Z, n, l in (("H", 1, 1, 0), ("He", 2, 1, 0), ("Li", 3, 2, 1)):
    for tj, (nms, sms, rho) in table(f"{el}.out").items():
        want = (Z * Z / (2 * n * n), 0.0, Z ** 3 / (math.pi * n ** 3) if l == 0 else 0.0)
        d = max(abs(nms / want[0] - 1), abs(sms), abs(rho - want[2]) / max(want[2], 1.0))
        worst = max(worst, d)
        print(f"   {el} {n}{'sp'[l]} J={tj}/2: K_NMS {nms:.10f} ({want[0]:.10f})  K_SMS {sms:+.1e}"
              f"  rho(0) {rho:.10f} ({want[2]:.10f})")
print(f"   one electron: max deviation {worst:.1e}")
lit = {"2S": 0.30241, "2P": 0.24751}
for st in ("2S", "2P"):
    hf = table(f"{st}/hf.out")
    mc = table(f"{st}/mc.out")
    log = open(f"{st}/mchf.log").read()
    kin = float(re.findall(r"Kinetic\s+(\S+)", log)[-1])
    ssms = float(re.findall(r"Ssms\s*=\s*(\S+)", open(f"{st}/mc.l").read())[0])
    tj = min(mc)
    nms, sms, rho = mc[tj]
    print(f"   Li {st} HF:   K_NMS {hf[tj][0]:.9f}  K_SMS {hf[tj][1]:+.9f}  rho(0) {hf[tj][2]:.6f}")
    print(f"   Li {st} MCHF: K_NMS {nms:.9f} (mchf kinetic {kin:.9f}, {nms - kin:+.1e})"
          f"  K_SMS {sms:.9f} (Ssms {ssms:.9f}, {sms - ssms:+.1e}; MCDHF n=10 {lit[st]})"
          f"  rho(0) {rho:.6f}")
PY
