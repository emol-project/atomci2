#!/bin/bash
# li_e1.sh -- S2-M8 (P6), report only (ATSP needed): the E1 oscillator
# strength of 7Li 1s2 2s 2S -> 1s2 2p 2P (acatsp "trans", lines E1) with
# growing correlation, against the nonrelativistic Hylleraas values of
# Z.-C. Yan, M. Tambasco, G. W. F. Drake, Phys. Rev. A 57, 1652 (1998)
# (infinite nuclear mass):
#   f_length = 0.7469569396, f_velocity = 0.7469597,
#   E(2S) = -7.4780603237, E(2P) = -7.4101565318 (omega = 0.0679037919).
# Both terms are CI expansions on one ATSP orbital set, MCHF-optimized for
# the energy average of 2S and 2P (one block per term; the transition needs
# one orbital set): complete active spaces of three electrons from 1s2 2s
# and 1s2 2p, n = 3 .. NMAX (l <= d, f, g, g, ...), n >= 4 varying only the
# new layer; n = 2 is the averaged Hartree-Fock. CI without spin-orbit; the
# multiplet f is the sum over J_up of f(2S1/2 -> 2PJ). Printed: f with the
# CI energies (length, velocity), and f with the exact omega (which isolates
# the transition moment: f_l ~ omega S_l, f_v ~ S_v / omega).
#   NMAX  (default 7; n = 7 takes ~5 min); LE_WORK (default $HOME/atomci_work/lie1)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${LE_WORK:-${AC_WORK:-$HOME/atomci_work}/lie1}
mkdir -p "$w" && cd "$w"
gen() { # n, l, term (2S: reference 1s2 2s, 2P: 1s2 2p)
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
}
for n in $(seq 2 ${NMAX:-7}); do
    case $n in 2) l=p;; 3) l=d;; 4) l=f;; *) l=g;; esac
    if [ $n = 2 ]; then
        printf '\n\n  1s( 2)  2s( 1)\n 1S0 2S1 2S \n*\n' > S.c
        printf '\n\n  1s( 2)  2p( 1)\n 1S0 2P1 2P \n*\n' > P.c
    else
        gen $n $l 2S; cp clist.out S.c
        gen $n $l 2P; cp clist.out P.c
    fi
    if [ ! -s li_$n.w ]; then
        # orbitals: one block per term, the energy average of 2S and 2P
        { head -2 S.c; sed -n '3,$p' S.c | grep -v '^\*'; echo '*'
          head -2 S.c; sed -n '3,$p' P.c | grep -v '^\*'; echo '*'; } > cfg.inp
        rm -f wfn.inp
        [ $n -ge 3 ] && cp li_$((n - 1)).w wfn.inp
        if [ $n -le 3 ]; then vary=all
        elif [ $n = 4 ]; then vary=4s,4p,4d,4f
        else vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
        nonh > nonh_$n.log 2>&1
        mchf > mchf_$n.log 2>&1 <<EOF
Li,3.
1
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
EOF
        cp wfn.out li_$n.w
    fi
    # both terms on these orbitals, one CSF list
    { head -2 S.c; sed -n '3,$p' S.c | grep -v '^\*'; sed -n '3,$p' P.c | grep -v '^\*'
      echo '*'; } > both_$n.c
    [ -s e1_$n.out ] || "$A" "$T" li_$n.w both_$n.c 2 noso trans > e1_$n.out
    python3 - "$n" e1_$n.out <<'PY'
import re, sys
n, path = sys.argv[1:3]
fl_ref, fv_ref, es, ep = 0.7469569396, 0.7469597, -7.4780603237, -7.4101565318
w0 = ep - es
e, sl, sv, fl, fv, tr = {}, 0.0, 0.0, 0.0, 0.0, False
for l in open(path):
    x = l.split()
    if l.startswith("Transitions"):
        tr = True
    if not tr and len(x) >= 3 and re.match(r"^\d+/2$", x[0]) and re.match(r"^\d+$", x[1]) \
            and (x[0], x[1]) not in e:
        e[(x[0], x[1])] = float(x[2])
    if tr and len(x) >= 10 and x[4] == "E1" and (x[0], x[1]) == ("1/2", "1") \
            and (x[2], x[3]) in (("1/2", "2"), ("3/2", "1")):
        sl += float(x[6]); sv += float(x[7]); fl += float(x[8]); fv += float(x[9])
s, p = e[("1/2", "1")], e[("3/2", "1")]
w = p - s
# f = (2/3) w S / g_lo (length), (2/3) S_v / (w g_lo) (velocity); g_lo = 2
flx, fvx = w0 * sl / 3, sv / (3 * w0)
print(f"   n={n}: E(2S) {s:.8f}  E(2P) {p:.8f}  omega {w:.8f} ({w / w0 - 1:+.2e})"
      f"  f_l {fl:.7f} ({fl / fl_ref - 1:+.2e})  f_v {fv:.7f} ({fv / fv_ref - 1:+.2e})"
      f"  exact omega: f_l {flx:.7f} ({flx / fl_ref - 1:+.2e})  f_v {fvx:.7f} ({fvx / fv_ref - 1:+.2e})")
PY
done
