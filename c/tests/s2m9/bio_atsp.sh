#!/bin/bash
# bio_atsp.sh -- S2-M9 (B4), report only (ATSP needed): 7Li 1s2 2s 2S ->
# 1s2 2p 2P with SEPARATELY optimized orbitals, acatsp "bio" against
# ATSP's biotr, and the oscillator strength against the Hylleraas values of
# Yan, Tambasco and Drake, Phys. Rev. A 57, 1652 (1998) (f_l 0.7469569396,
# f_v 0.7469597, omega 0.0679037919), as tests/s2m8/li_e1.sh (P6) did on
# one orbital set for both terms.
#   Each term: MCHF of its own CAS list (lsgen, three electrons, n = 3 ..
#   NMAX; n = 2 is the single-CSF Hartree-Fock), n >= 4 varying only the
#   new layer, started from the term's own n-1 orbitals (n = 2 from P6's
#   li_2.w, the averaged HF). The CI vectors that biotr reads (.l) come from
#   bpci in nonrelativistic mode on the final orbitals (mchf's own .l is
#   from before its last orbital update). acatsp solves both CIs again on the
#   same orbitals.
#   Compared: the line strengths of 2S1/2 -> 2P1/2, 2P3/2 summed (the LS
#   multiplet; no spin-orbit), length S_l and velocity S_v / omega^2 (biotr's
#   velocity S is length-like); biotr prints 7 digits.
#   At n = 2 also the nonorthogonal-orbital (Loewdin cofactor) formula of
#   tests/s2m9/lowdin_n2.py, independent of the C code and of biotr.
#   NMAX (default 5; 7 takes long), LE_WORK (default $HOME/atomci_work/bio4)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen bpci biotr trans || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${LE_WORK:-${AC_WORK:-$HOME/atomci_work}/bio4}
p6=${P6_WORK:-${AC_WORK:-$HOME/atomci_work}/lie1}
mkdir -p "$w" && cd "$w"
gen() { # n, l, term
    if [ "$3" = 2S ]; then ref="1
0"; else ref="0
1"; fi
    lsgen > lsgen_$3_$1.log 2>&1 <<EOF



$1
$2


2
2
$ref
$3
3

EOF
    cp clist.out $3_$1.c
}
mchf_term() { # n, term
    local n=$1 t=$2 vary spec=1s,2s,2p
    cp ${t}_$n.c cfg.inp
    # n = 2 has only the orbitals of the term's own configuration
    [ $n = 2 ] && { [ $t = 2S ] && spec=1s,2s || spec=1s,2p; }
    if [ $n = 2 ]; then cp $p6/li_2.w wfn.inp; else cp ${t}_$((n - 1)).w wfn.inp; fi
    if [ $n -le 3 ]; then vary=all
    elif [ $n = 4 ]; then vary=4s,4p,4d,4f
    else vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
    nonh > nonh_${t}_$n.log 2>&1
    mchf > mchf_${t}_$n.log 2>&1 <<EOF
Li,3.
1
$vary
$spec
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
EOF
    cp wfn.out ${t}_$n.w
}
bpl() { # n, term: the CI vector on the final orbitals, name.l
    rm -f *.lst $2_$1.l
    bpci > bpl_$2_$1.log 2>&1 <<EOF
$2_$1,n,n
n
n
1
0
y
y
y
y
EOF
}
for n in $(seq 2 ${NMAX:-5}); do
    case $n in 3) l=d;; 4) l=f;; *) l=g;; esac
    if [ $n = 2 ]; then
        printf '\n\n  1s( 2)  2s( 1)\n 1S0 2S1 2S \n*\n' > 2S_2.c
        printf '\n\n  1s( 2)  2p( 1)\n 1S0 2P1 2P \n*\n' > 2P_2.c
    else
        [ -s 2S_$n.c ] || gen $n $l 2S
        [ -s 2P_$n.c ] || gen $n $l 2P
    fi
    for t in 2S 2P; do
        [ -s ${t}_$n.w ] || mchf_term $n $t
        [ -s ${t}_$n.l ] || bpl $n $t
    done
    [ -s biotr_$n.log ] || biotr > biotr_$n.log 2>&1 <<EOF || true
2S_$n
2P_$n
n
n
C
E1
EOF
    # from n = 5 lsgen's lists hold at most two electrons in an f or g
    # shell, so 4f2 5f -> 4f3 leaves them: not closed, and the part of the
    # counter-transformation outside the list is dropped (biotr drops it
    # too, without a check; Olsen et al. section V)
    [ -s ac_$n.out ] || "$A" "$T" 2S_$n.w 2S_$n.c 1 noso trans digits bio=2P_$n.w,2P_$n.c \
        biorthforce > ac_$n.out
    grep -E "not closed" ac_$n.out | sed 's/^ */   /' || true
    rm -f lowdin.out
    if [ $n = 2 ]; then
        ${PYTHON:-python3} $here/lowdin_n2.py \
            2S_2.w 2P_2.w > lowdin.out
    fi
    python3 - "$n" ac_$n.out biotr_$n.log <<'PY'
import re, sys
n, acp, btp = sys.argv[1:4]
fl_ref, fv_ref, w0 = 0.7469569396, 0.7469597, 0.0679037919
hartree_cm = 219474.6313632
sl = sv = 0.0; cm = None; on = False; ov = {}
for l in open(acp):
    if l.startswith("Between the levels"):
        on = True
    m = re.match(r"\s+l = (\d): .*<X_i\|Y_i>:(.*)", l)
    if m:
        x = m.group(2).split()
        ov.update({x[i]: float(x[i + 1]) for i in range(0, len(x), 2)})
    x = l.split()
    if on and len(x) >= 11 and x[4] == "E1" and (x[0], x[1]) == ("1/2", "1") \
            and x[3] == "1" and x[2] in ("1/2", "3/2"):
        sl += float(x[6]); sv += float(x[7]); cm = float(x[5])
w = cm / hartree_cm
txt = open(btp).read()
m = re.search(r"LS RESULTS \(length\s+formalism\):\s+S\s+=\s+(\S+).*?\(velocity formalism\):\s+S\s+=\s+(\S+)",
              txt, re.S)
bl, bv = (float(v.replace("D", "E")) for v in m.groups())
svl = sv / w ** 2
flx, fvx = w0 * sl / 3, sv / (3 * w0)
print(f"   n={n}: S_l {sl:.10f} biotr {bl:.6f} ({sl / bl - 1:+.1e})"
      f"  S_v/w^2 {svl:.10f} biotr {bv:.6f} ({svl / bv - 1:+.1e})"
      f"  <2s|2s'> {ov.get('2s', float('nan')):+.6f}")
import os
if os.path.exists("lowdin.out"):
    x = open("lowdin.out").read().split()
    ll, lv = float(x[x.index("S_l") + 1]), float(x[x.index("S_v") + 1])
    print(f"        Loewdin (tests/s2m9/lowdin_n2.py): S_l {ll:.10f} ({sl / ll - 1:+.1e})"
          f"  S_v {lv:.10e} ({sv / lv - 1:+.1e}; its P' is good to ~1e-6)")
print(f"        omega {w:.8f} ({w / w0 - 1:+.2e})  f_l {w * sl / 3:.7f} ({w * sl / 3 / fl_ref - 1:+.2e})"
      f"  f_v {sv / (3 * w):.7f} ({sv / (3 * w) / fv_ref - 1:+.2e})"
      f"  exact omega: f_l {flx:.7f} ({flx / fl_ref - 1:+.2e})  f_v {fvx:.7f} ({fvx / fv_ref - 1:+.2e})")
PY
done
