#!/bin/bash
# bi_ogjmf.sh -- S2-M9 (B4, physics), report only (ATSP needed): B I
# 1s2 2s2 2p 2P -> 1s2 2s 2p2 2D with each term's own MCHF orbitals, after
# Olsen, Godefroid, Joensson, Malmqvist and Froese Fischer, Phys. Rev. E 52,
# 4499 (1995), section VI, Tables I and II.
#   Expansions: single and double excitations from
#     2P: {1s2 2s2 2p, 1s2 2p3},  2D: {1s2 2s 2p2, 1s2 2s2 3d}
#   (1s open), every nl with l <= n-1 up to n = NMAX; lsgen per reference,
#   the union kept. The CSF counts are those of their Table I.
#   Orbitals: n = 1 is each term's Hartree-Fock (the reference CSF alone);
#   from n = 2 all orbitals vary, started from the term's own n-1 orbitals
#   (varying only the new layer misses their 2P n = 3 energy by 2.4e-4).
#   2D at n = 3 has two MCHF solutions. D3=ryd (default) starts the 3d from
#   the Hartree-Fock of 1s2 2s2 3d (diffuse, Rydberg-like), varies the rest
#   with it frozen, then all: their -24.379506. D3=free starts it from
#   nothing and lands 0.013 lower, the 3d a compact correlation orbital.
#   ORB=avg instead: one orbital set for both terms, the MCHF on the equally
#   weighted average of the two energies (the same lists, all orbitals
#   varied from the n-1 set; D3 does not apply).
#   The line: acatsp "bio" (2P list, bio = 2D list; for avg the same
#   orbitals on both sides) and ATSP biotr on bpci's nonrelativistic CI
#   vectors; printed against their Tables I and II.
#   NMAX (default 5), D3, ORB (sep), BI_WORK (default $HOME/atomci_work/bi_$D3,
#   or bi_avg)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen bpci biotr trans || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
d3=${D3:-ryd}
orb=${ORB:-sep}
[ $orb = avg ] && tag=avg || tag=$d3
w=${BI_WORK:-${AC_WORK:-$HOME/atomci_work}/bi_$tag}
mkdir -p "$w" && cd "$w"
LET=(s p d f g h i k)
sd() { # n, lmax letter, term, highest n of the reference, occupations...
    local n=$1 l=$2 t=$3 nr=$4
    shift 4
    printf '\n\n\n%s\n%s\n\n\n%s\n' $n $l $nr > lsgen.in
    printf '%s\n' "$@" >> lsgen.in
    printf '%s\n2\n\n' $t >> lsgen.in
    lsgen < lsgen.in > lsgen.log 2>&1
    cat clist.out
}
gen() { # n, term: the union of the SD lists of the references
    local n=$1 t=$2 l=${LET[$(($1 - 1))]}
    if [ $t = 2P ]; then
        sd $n $l 2P 2 2 2 1 > r1.c
        sd $n $l 2P 2 2 0 3 > r2.c
    else
        sd $n $l 2D 2 2 1 2 > r1.c
        if [ $n -ge 3 ]; then
            sd $n $l 2D 3 2 2 0 0 0 1 > r2.c
        else : > r2.c; fi
    fi
    python3 - r1.c r2.c > ${t}_$n.c <<'PY'
import sys
head, seen, out = None, set(), []
for p in sys.argv[1:]:
    x = open(p).read().split("\n")
    if len(x) < 3:
        continue
    if head is None:
        head = x[:2]
    i = 2
    while i + 1 < len(x) and not x[i].startswith("*"):
        k = (x[i].rstrip(), x[i + 1].rstrip())
        if k not in seen:
            seen.add(k); out.append(x[i] + "\n" + x[i + 1])
        i += 2
print("\n".join(head)); print("\n".join(out)); print("*")
PY
}
mchf_run() { # name, vary, spectroscopic
    nonh > nonh_$1.log 2>&1
    mchf > mchf_$1.log 2>&1 <<EOF
B,5.
${4:-1}
$2
$3
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
bpl() { # name: the CI vector on the final orbitals, name.l
    rm -f *.lst $1.l
    bpci > bpl_$1.log 2>&1 <<EOF
$1,n,n
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
pick() { # list, configuration: its first CSF alone
    awk -v p="$2" 'NR <= 2 {print; next} index($0, p) && !d {print; getline; print; d = 1}
        END {print "*"}' $1
}
orbital() { # .w file, label: that orbital's record
    python3 - "$@" <<'PY'
import struct, sys
d, i = open(sys.argv[1], "rb").read(), 0
while i < len(d):
    (m,) = struct.unpack_from("<i", d, i)
    if d[i + 16:i + 19].decode().strip() == sys.argv[2]:
        sys.stdout.buffer.write(d[i:i + 8 + m])
    i += 8 + m
PY
}
for n in $(seq 1 ${NMAX:-5}); do
    for t in 2P 2D; do
        if [ $n = 1 ]; then
            [ -s ${t}_2.c ] || gen 2 $t
            # the reference CSF alone: 1s2 2s2 2p / 1s2 2s 2p2 (1D)
            [ $t = 2P ] && pat='1s( 2)  2s( 2)  2p( 1)' || pat='1s( 2)  2s( 1)  2p( 2)'
            pick ${t}_2.c "$pat" > ${t}_1.c
        else
            [ -s ${t}_$n.c ] || gen $n $t
        fi
        if [ $orb = sep ] && [ ! -s ${t}_$n.w ]; then
            rm -f wfn.inp
            [ $n = 1 ] || cp ${t}_$((n - 1)).w wfn.inp
            if [ $t$n = 2D3 ] && [ $d3 = ryd ]; then
                # 3d from the Hartree-Fock of 1s2 2s2 3d, frozen, then all
                pick 2D_3.c '1s( 2)  2s( 2)  3d( 1)' > cfg.inp
                mchf_run ryd3d all 1s,2s,3d
                orbital wfn.out 3d >> wfn.inp
                cp ${t}_$n.c cfg.inp
                mchf_run ${t}_${n}_fix 1s,2s,2p,3s,3p 1s,2s,2p
                cp wfn.out wfn.inp
            fi
            cp ${t}_$n.c cfg.inp
            mchf_run ${t}_$n all 1s,2s,2p
            cp wfn.out ${t}_$n.w
        fi
    done
    if [ $orb = avg ]; then
        if [ ! -s avg_$n.w ]; then
            rm -f wfn.inp
            [ $n = 1 ] || cp avg_$((n - 1)).w wfn.inp
            { head -2 2P_$n.c; sed -n '3,$p' 2P_$n.c | grep -v '^\*'; echo '*'
              head -2 2P_$n.c; sed -n '3,$p' 2D_$n.c | grep -v '^\*'; echo '*'; } > cfg.inp
            mchf_run avg_$n all 1s,2s,2p "1
1"
            cp wfn.out avg_$n.w
        fi
        for t in 2P 2D; do
            [ -s ${t}a_$n.c ] || { cp ${t}_$n.c ${t}a_$n.c; cp avg_$n.w ${t}a_$n.w; }
        done
        p=2Pa_$n d=2Da_$n
    else
        p=2P_$n d=2D_$n
    fi
    for x in $p $d; do [ -s $x.l ] || bpl $x; done
    [ -s biotr_$n.log ] || biotr > biotr_$n.log 2>&1 <<EOF || true
$p
$d
n
n
C
E1
EOF
    # avg: "bio" with the same orbitals on both sides (S = 1 to ~1e-9)
    [ -s ac_$n.out ] || { "$A" "$T" $p.w $p.c 1 noso trans digits bio=$d.w,$d.c \
        > ac.tmp && mv ac.tmp ac_$n.out; }
    python3 - "$n" ac_$n.out biotr_$n.log $p.c $d.c <<'PY'
import re, sys
n, acp, btp, c1, c2 = sys.argv[1:6]
n = int(n)
hcm = 219474.6313632
E2P = [-24.529061, -24.560354, -24.621891, -24.638480, -24.645478, -24.648728, -24.650290]
E2D = [-24.311869, -24.311921, -24.379506, -24.416497, -24.424768, -24.429451, -24.431353]
NP = [1, 6, 129, 520, 1301, 2584, 4479]
ND = [1, 4, 157, 812, 2320, 5004, 9161]
T2 = {3: (0.6876, 0.8156, 2.5534, 53197), 4: (0.2456, 0.2696, 0.9959, 48720),
      5: (0.2625, 0.2695, 1.0705, 48440), 6: (0.2891, 0.2866, 1.1868, 48125),
      7: (0.2928, 0.2900, 1.2036, 48051)}
ncsf = lambda p: sum(1 for l in open(p) if "(" in l)
e, sl, sv, on = {}, 0.0, 0.0, False
for l in open(acp):
    x = l.split()
    if l.startswith("Between the levels"):
        on = True
    if len(x) >= 4 and re.match(r"^\d+/2$", x[0]) and re.match(r"^\d+$", x[1]) \
            and re.match(r"^\d[A-Z]$", x[3]):
        e.setdefault(x[3], float(x[2]))
    if on and len(x) >= 11 and x[4] == "E1" and x[1] == "1" and x[3] == "1":
        sl += float(x[6]); sv += float(x[7])
ep, ed = e["2P"], e["2D"]
w = ed - ep
txt = open(btp).read()
m = re.search(r"LS RESULTS \(length\s+formalism\):\s+S\s+=\s+(\S+).*?\(velocity formalism\):\s+S\s+=\s+(\S+)",
              txt, re.S)
bl, bv = (float(v.replace("D", "E")) for v in m.groups()) if m else (float("nan"),) * 2
svl = sv / w ** 2
i = n - 1
print(f" n={n}: CSFs {ncsf(c1)}/{ncsf(c2)} (OGJMF {NP[i]}/{ND[i]})"
      f"  E(2P) {ep:.6f} ({ep - E2P[i]:+.1e})  E(2D) {ed:.6f} ({ed - E2D[i]:+.1e})"
      f"  dE {w * hcm:.0f} cm-1")
print(f"       S_l {sl:.8f} biotr {bl:.6f} ({sl / bl - 1:+.1e})"
      f"  S_v/w^2 {svl:.8f} biotr {bv:.6f} ({svl / bv - 1:+.1e})")
gl, gv = 2 / 3 * w * sl, 2 / 3 * sv / w
s = f"       gf_l {gl:.4f}  gf_v {gv:.4f}  (2/3 w S)"
if n in T2:
    a = T2[n]
    s += f"   OGJMF: gf_l {a[0]:.4f} gf_v {a[1]:.4f} S_l {a[2]:.4f} dE {a[3]}"
print(s)
PY
    grep -h "not closed" ac_$n.out | sed 's/^ */       /' || true
done
