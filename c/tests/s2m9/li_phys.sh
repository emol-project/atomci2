#!/bin/bash
# li_phys.sh -- S2-M9 (B4, physics), report only (ATSP needed): the
# oscillator strength of 7Li 1s2 2s 2S -> 1s2 2p 2P against the Hylleraas
# values of Yan, Tambasco and Drake, Phys. Rev. A 57, 1652 (1998)
# (infinite nuclear mass): f_l 0.7469569396, f_v 0.7469597,
# E(2S) -7.4780603237, E(2P) -7.4101565318.
#   EXP=cas  complete active space of the three electrons, n <= NMAX
#   EXP=cv   core-valence: at least one 1s electron (no 1s^0, i.e. no
#            core-core pair), which aims the correlation orbitals at core
#            polarization (lsgen "Limitations on population": n=1 >= 1)
#   ORB=sep  each term its own orbitals (MCHF of its own list; the line by
#            acatsp "bio", biorthonormal)
#   ORB=avg  one orbital set, MCHF on the energy average of the two terms
#            (as tests/s2m8/li_e1.sh, P6); the line by acatsp "trans" on
#            both lists together
# Orbitals: n = 2 the Hartree-Fock (sep: each term's own; avg: P6's
# averaged li_2.w); from n = 3 the spectroscopic orbitals (sep: 1s, 2s of
# 2S and 1s, 2p of 2P; avg: 1s, 2s, 2p) stay at Hartree-Fock and only the
# correlation orbitals vary: at n = 3 every n = 3 orbital and the n = 2
# orbital a term lacks (2p of 2S, 2s of 2P), then only the new layer.
# l <= d (n = 3), f (4), g (n >= 5). Printed per n: the energies against
# Yan, f with the CI energies and with the exact omega (the latter isolates
# the transition moment: f_l ~ omega S_l, f_v ~ S_v / omega).
#   EXP (cas), ORB (sep), NMAX (7), PHYS_WORK (default
#   $HOME/atomci_work/liphys/$EXP_$ORB), P6_WORK ($HOME/atomci_work/lie1)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
exp=${EXP:-cas}
orb=${ORB:-sep}
p6=${P6_WORK:-${AC_WORK:-$HOME/atomci_work}/lie1}
w=${PHYS_WORK:-${AC_WORK:-$HOME/atomci_work}/liphys/${exp}_$orb}
mkdir -p "$w" && cd "$w"
gen() { # n, l, term
    local lim="" i
    if [ "$exp" = cv ]; then
        lim="y
1"
        for i in $(seq 2 $(($1 - 1))); do lim="$lim
1"; done
    fi
    if [ "$3" = 2S ]; then ref="1
0"; else ref="0
1"; fi
    lsgen > lsgen_$3_$1.log 2>&1 <<EOF



$1
$2

$lim
2
2
$ref
$3
3

EOF
    cp clist.out $3_$1.c
}
layer() { # n, lmax letter: the orbitals of shell n up to l
    local s="" x
    for x in s p d f g; do
        s="$s${s:+,}$1$x"
        [ $x = $2 ] && break
    done
    echo "$s"
}
mchf_run() { # name, blocks, vary, spectroscopic
    local eig=1
    [ $2 = 2 ] && eig="1
1"
    nonh > nonh_$1.log 2>&1
    mchf > mchf_$1.log 2>&1 <<EOF
Li,3.
$eig
$3
$4
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
EOF
}
for n in $(seq 2 ${NMAX:-7}); do
    case $n in 2) l=p;; 3) l=d;; 4) l=f;; *) l=g;; esac
    if [ $n = 2 ]; then
        printf '\n\n  1s( 2)  2s( 1)\n 1S0 2S1 2S \n*\n' > 2S_2.c
        printf '\n\n  1s( 2)  2p( 1)\n 1S0 2P1 2P \n*\n' > 2P_2.c
    else
        [ -s 2S_$n.c ] || gen $n $l 2S
        [ -s 2P_$n.c ] || gen $n $l 2P
    fi
    lay=$(layer $n $l)
    if [ $orb = sep ]; then
        for t in 2S 2P; do
            [ -s ${t}_$n.w ] && continue
            [ $t = 2S ] && { spec=1s,2s; miss=2p; } || { spec=1s,2p; miss=2s; }
            cp ${t}_$n.c cfg.inp
            if [ $n = 2 ]; then cp $p6/li_2.w wfn.inp; vary=all
            else
                cp ${t}_$((n - 1)).w wfn.inp
                [ $n = 3 ] && vary=$miss,$lay || vary=$lay
            fi
            mchf_run ${t}_$n 1 "$vary" "$spec"
            cp wfn.out ${t}_$n.w
        done
        [ -s ac_$n.out ] || { "$A" "$T" 2S_$n.w 2S_$n.c 1 noso trans digits \
            bio=2P_$n.w,2P_$n.c biorthforce > ac.tmp && mv ac.tmp ac_$n.out; }
    else
        if [ ! -s li_$n.w ]; then
            if [ $n = 2 ]; then cp $p6/li_2.w li_2.w
            else
                { head -2 2S_$n.c; sed -n '3,$p' 2S_$n.c | grep -v '^\*'; echo '*'
                  head -2 2S_$n.c; sed -n '3,$p' 2P_$n.c | grep -v '^\*'; echo '*'; } > cfg.inp
                cp li_$((n - 1)).w wfn.inp
                [ $n = 3 ] && vary=$lay || vary=$lay
                mchf_run avg_$n 2 "$vary" 1s,2s,2p
                cp wfn.out li_$n.w
            fi
        fi
        { head -2 2S_$n.c; sed -n '3,$p' 2S_$n.c | grep -v '^\*'; sed -n '3,$p' 2P_$n.c | grep -v '^\*'
          echo '*'; } > both_$n.c
        [ -s ac_$n.out ] || { "$A" "$T" li_$n.w both_$n.c 2 noso trans digits > ac.tmp &&
            mv ac.tmp ac_$n.out; }
    fi
    python3 - "$n" ac_$n.out "$orb" <<'PY'
import re, sys
n, path, orb = sys.argv[1:4]
fl_ref, fv_ref, es0, ep0 = 0.7469569396, 0.7469597, -7.4780603237, -7.4101565318
w0 = ep0 - es0
e, sl, sv, tr = {}, 0.0, 0.0, False
for l in open(path):
    x = l.split()
    # the E1 lines: between the states (sep), or the one list's table (avg)
    if l.startswith("Between the levels" if orb == "sep" else "Transitions"):
        tr = True
    # level lines: J, #, E, leading term; the lowest of each (term, J)
    if len(x) >= 4 and re.match(r"^\d+/2$", x[0]) and re.match(r"^\d+$", x[1]) \
            and re.match(r"^\d[A-Z]$", x[3]):
        e.setdefault((x[3], x[0]), float(x[2]))
    if tr and len(x) >= 11 and x[4] == "E1" and (x[0], x[1]) == ("1/2", "1"):
        if (orb == "sep" and x[3] == "1" and x[2] in ("1/2", "3/2")) or \
           (orb == "avg" and (x[2], x[3]) in (("1/2", "2"), ("3/2", "1"))):
            sl += float(x[6]); sv += float(x[7])
s, p = e[("2S", "1/2")], e[("2P", "3/2")]
w = p - s
print(f"   n={n}: E(2S) {s:.8f} ({s - es0:+.2e})  E(2P) {p:.8f} ({p - ep0:+.2e})"
      f"  omega {w:.8f} ({w / w0 - 1:+.2e})")
print(f"        f_l {w * sl / 3:.7f} ({w * sl / 3 / fl_ref - 1:+.2e})"
      f"  f_v {sv / (3 * w):.7f} ({sv / (3 * w) / fv_ref - 1:+.2e})"
      f"   exact omega: f_l {w0 * sl / 3:.7f} ({w0 * sl / 3 / fl_ref - 1:+.2e})"
      f"  f_v {sv / (3 * w0):.7f} ({sv / (3 * w0) / fv_ref - 1:+.2e})")
PY
    grep -h "not closed" ac_$n.out | sed 's/^ */        /' || true
done
