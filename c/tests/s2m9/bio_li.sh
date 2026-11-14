#!/bin/bash
# bio_li.sh -- S2-M9 (B3/B4), ATSP's lsgen and the S2-M8 P6 orbitals
# needed: acatsp "bio" (two states on biorthonormal orbitals) on
# 7Li 1s2 2s 2S -> 1s2 2p 2P, against the one-list run on a common orbital
# set (acatsp "trans" on both_n.c, as tests/s2m8/li_e1.sh):
#   same     both states on li_3.w: the biorthonormal machinery is the
#            identity and must reproduce the one-list line strengths
#   swap     state 2 as the bra
#   mode     the three splittings of the pivots (biomode=0, 1, 2)
#   rot      the orbitals of one state rotated within an l (rot1=, rot2=)
#            before its CI: the CAS energy is unchanged and the overlap
#            and the counter-transformation are not trivial (the test of
#            Olsen et al., Phys. Rev. E 52, 4499 (1995), section V)
#   rect     state 2 on li_4.w (one more layer, 4s..4f; the lower ones
#            are those of li_3.w, n >= 4 varies only the new layer): the
#            unequal orbital sets of their appendix B, against both lists
#            on li_4.w
#   m1       the relativistic M1 (M1r) of 2P1/2 - 2P3/2 with its two-electron
#            terms (m1r2e), both states rotated: |<||delta mu||>| within 1e-9
#            relative of the one-list run, both ways round
#   zoff 2e  the Zeeman elements between the 2P levels with gc2e, oz2e, rec2
#            (two-electron integrals with the bra orbitals of state 1 and the
#            ket orbitals of state 2), both states rotated: every column within
#            1e-6 of its largest magnitude (the printed digits)
# Both noso and with H_SO1 (J-mixed levels). All runs orthonormalize the
# orbitals on the grid first (acatsp "orth"): ATSP's orbitals, interpolated,
# are orthonormal only to ~1e-9, which the one-list reference ignores but
# the biorthonormal transformation absorbs (a 2e-10 difference without
# orth). Pass: every E1 line strength of 2S -> 2P_J within 1e-11 relative
# (printed with 17 digits, "digits"); for swap the velocity form within
# 1e-9 (its integral is taken the other way round, and the grid satisfies
# integration by parts only to ~1e-10).
#   LE_WORK (default $HOME/atomci_work/lie1, made by li_e1.sh with NMAX >= 4)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" lsgen hfs || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${LE_WORK:-${AC_WORK:-$HOME/atomci_work}/lie1}
cd "$w"
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
for n in 3 4; do
    case $n in 3) l=d;; 4) l=f;; esac
    [ -s 2S_$n.c ] || gen $n $l 2S
    [ -s 2P_$n.c ] || gen $n $l 2P
done
# both terms of 2S_3 (on li_4.w) and 2P_4: the common-orbital reference of rect
{ head -2 2S_3.c; sed -n '3,$p' 2S_3.c | grep -v '^\*'; sed -n '3,$p' 2P_4.c | grep -v '^\*'
  echo '*'; } > both_34.c
{ head -2 2S_3.c; sed -n '3,$p' 2S_3.c | grep -v '^\*'; sed -n '3,$p' 2P_3.c | grep -v '^\*'
  echo '*'; } > both_33.c
mkdir -p bio && cd bio
run() { # name, args...
    local name=$1; shift
    "$A" "$T" "$@" digits orth > $name.out 2> $name.err ||
        { echo "acatsp failed: $name"; cat $name.err; exit 1; }
}
for so in noso so; do
    opt=$([ $so = noso ] && echo noso || echo mvd)
    run ref_$so ../li_3.w ../both_33.c 2 $opt trans
    run same_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c
    run swap_$so ../li_3.w ../2P_3.c 2 $opt trans bio=../li_3.w,../2S_3.c
    run mode1_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c biomode=1
    run mode2_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c biomode=2
    run rot2_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c rot2=2p,3p,25
    run rot1_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c rot1=1s,2s,10
    run rot12_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_3.w,../2P_3.c rot1=2s,3s,-30 \
        rot2=1s,3s,15 rot2=2p,3p,-40
    run refr_$so ../li_4.w ../both_34.c 2 $opt trans
    run rect_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_4.w,../2P_4.c
    run rectrot_$so ../li_3.w ../2S_3.c 2 $opt trans bio=../li_4.w,../2P_4.c rot2=3d,4d,30 \
        rot1=2p,3p,20
    run hz_ref_$so ../li_3.w ../2P_3.c 2 $opt hfs zoff
    run hz_bio_$so ../li_3.w ../2P_3.c 2 $opt hfs zoff bio=../li_3.w,../2P_3.c rot2=2p,3p,25 \
        rot2=1s,2s,-15
    run gz_ref_$so ../li_3.w ../2P_3.c 2 $opt zoff gc2e oz2e rec2
    run gz_bio_$so ../li_3.w ../2P_3.c 2 $opt zoff gc2e oz2e rec2 bio=../li_3.w,../2P_3.c \
        rot2=2p,3p,25 rot2=1s,2s,-15 rot1=2p,3p,-10
done
# the relativistic M1 of 2P1/2 - 2P3/2 with its two-electron terms (m1r2e):
# one list vs two lists, the second rotated, and the same without m1r2e
for e in m1r2e none; do
    m=$([ $e = m1r2e ] && echo m1r2e || true)
    run m1_ref_$e ../li_3.w ../2P_3.c 2 mvd trans $m
    run m1_bio_$e ../li_3.w ../2P_3.c 2 mvd trans $m bio=../li_3.w,../2P_3.c rot2=2p,3p,25 \
        rot2=1s,2s,-15 rot1=2p,3p,-10
done
# a list that is not closed: 2P_3 without 1s2 2p (3p -> 2p leaves it)
{ head -2 ../2P_3.c; sed -n '5,$p' ../2P_3.c; } > open.c
if "$A" "$T" ../li_3.w ../2S_3.c 2 noso trans bio=../li_3.w,open.c > open.out 2>&1; then
    echo "  open list accepted: FAIL"; exit 1
elif grep -q "not closed" open.out; then
    echo "  open list refused: $(grep 'state 2 not closed' open.out | sed 's/^ *//')  ok"
else
    echo "  open list: unexpected failure"; cat open.out; exit 1
fi
python3 - <<'PY'
import sys
def lines(path, cross):
    """E1 lines 2S(1/2, #1) -> 2P_J: {J_up: (S_l, S_v)}; cross: the block
    after "Between the levels" (lower level in state 1 or 2)"""
    out, on, txt = {}, not cross, open(path).read().splitlines()
    energies = {}
    for l in txt:
        if l.startswith("Between the levels"):
            on = True
            continue
        x = l.split()
        if on and len(x) >= 11 and x[4] == "E1":
            out.setdefault("all", []).append((x[0], x[1], x[2], x[3], float(x[6]), float(x[7])))
    return out.get("all", [])
def pick(ls, ref):
    # the lines from the lowest J = 1/2 level to the lowest 2P_J
    if ref:   # one list: 2S is J=1/2 #1, 2P_1/2 J=1/2 #2, 2P_3/2 J=3/2 #1
        want = {("1/2", "1", "1/2", "2"): "P1/2", ("1/2", "1", "3/2", "1"): "P3/2"}
    else:     # two lists: each numbered within its own state
        want = {("1/2", "1", "1/2", "1"): "P1/2", ("1/2", "1", "3/2", "1"): "P3/2"}
    r = {}
    for j0, n0, j1, n1, sl, sv in ls:
        k = want.get((j0, n0, j1, n1))
        if k and k not in r:
            r[k] = (sl, sv)
    return r
bad = 0
for so in ("noso", "so"):
    ref = pick(lines(f"ref_{so}.out", False), True)
    refr = pick(lines(f"refr_{so}.out", False), True)
    for name, base in (("same", ref), ("swap", ref), ("mode1", ref), ("mode2", ref),
                       ("rot2", ref), ("rot1", ref), ("rot12", ref), ("rect", refr),
                       ("rectrot", refr)):
        got = pick(lines(f"{name}_{so}.out", True), False)
        if set(got) != set(base):
            dl = dv = float("inf")
        else:
            dl = max(abs(got[k][0] / base[k][0] - 1) for k in base)
            dv = max(abs(got[k][1] / base[k][1] - 1) for k in base)
        # swap: the velocity integral taken the other way round, which on
        # the grid satisfies integration by parts only to ~1e-10
        ok = dl < 1e-11 and dv < (1e-9 if name == "swap" else 1e-11)
        bad += not ok
        print(f"  {so:4s} {name:8s} S_l(P3/2) {got.get('P3/2', (0, 0))[0]:.12e}"
              f"  S_v {got.get('P3/2', (0, 0))[1]:.12e}  rel dev S_l {dl:.1e} S_v {dv:.1e}"
              f"  {'ok' if ok else 'FAIL'}")
def table(path, head, cross, ncol):
    """rows of the table whose header line starts with head: {(J_A, #A,
    J_B, #B): [values]}; cross: the one after "Between the levels" """
    out, on, inb = {}, False, not cross
    for l in open(path):
        if l.startswith("Between the levels"):
            inb = True
        if l.startswith(head):
            on = inb
            continue
        x = l.split()
        if on and len(x) >= 4 + ncol and x[0].endswith("/2") and x[2].endswith("/2"):
            out[tuple(x[:4])] = [float(v) for v in x[4:4 + ncol]]
        elif on and not (x and x[0] in ("J_A",)):
            on = False
    return out
# hyperfine and Zeeman between the 2P levels: one list vs two lists (the
# second rotated); printed with 10 digits
for so in ("noso", "so"):
    for head, ncol, what in (("Hyperfine", 5, "hfs"), ("Zeeman reduced", 3, "zoff")):
        a = table(f"hz_ref_{so}.out", head, False, ncol)
        b = table(f"hz_bio_{so}.out", head, True, ncol)
        common = [k for k in a if k in b]
        # up to the sign of each level's eigenvector (the second CI is
        # solved again on rotated orbitals): compare magnitudes
        dev = max((abs(abs(b[k][i]) - abs(a[k][i])) / max(abs(a[k][i]), 1e-6)
                   for k in common for i in range(ncol)), default=float("inf"))
        ok = len(common) >= 3 and dev < 2e-9
        bad += not ok
        print(f"  {so:4s} {what:8s} {len(common)} pairs, max rel dev {dev:.1e}"
              f"  {'ok' if ok else 'FAIL'}")
# the Zeeman elements with their two-electron terms (GC2e, OZ2e, rec2: integrals
# with the bra orbitals of state 1 and the ket orbitals of state 2), both states
# rotated; nine columns printed with 7 to 13 digits, so each column relative to
# its largest magnitude
names = ["<L>", "<S>", "<L+gS>", "RMC", "GC", "ORMC", "GC2e", "OZ2e", "rec2"]
for so in ("noso", "so"):
    a = table(f"gz_ref_{so}.out", "Zeeman reduced", False, 9)
    b = table(f"gz_bio_{so}.out", "Zeeman reduced", True, 9)
    common = [k for k in a if k in b]
    dev = {}
    for i, n in enumerate(names):
        scale = max((abs(a[k][i]) for k in common), default=0.0)
        dev[n] = max((abs(abs(b[k][i]) - abs(a[k][i])) for k in common),
                     default=float("inf")) / (scale if scale > 0 else 1.0)
    worst = max(dev.values(), default=float("inf"))
    ok = len(common) >= 3 and worst < 1e-6
    bad += not ok
    print(f"  {so:4s} zoff 2e  {len(common)} pairs, max rel dev GC2e {dev['GC2e']:.1e} OZ2e "
          f"{dev['OZ2e']:.1e} rec2 {dev['rec2']:.1e} (all columns {worst:.1e})"
          f"  {'ok' if ok else 'FAIL'}")
def m1r(path, cross):
    """<||delta mu||> of the M1r lines between J = 1/2 #1 and 3/2 #1 (either
    way round), as magnitudes (the signs of the eigenvectors are free)"""
    on, r = not cross, []
    for l in open(path):
        if l.startswith("Between the levels"):
            on = True
            continue
        x = l.split()
        if on and len(x) >= 8 and x[4] == "M1r" and {(x[0], x[1]), (x[2], x[3])} == \
                {("1/2", "1"), ("3/2", "1")}:
            r.append(abs(float(x[7])))
    return r
dm = {}
for e in ("m1r2e", "none"):
    a, b = m1r(f"m1_ref_{e}.out", False), m1r(f"m1_bio_{e}.out", True)
    dm[e] = a[0] if a else float("nan")
    dev = max((abs(v / a[0] - 1) for v in b), default=float("inf")) if a else float("inf")
    ok = len(b) == 2 and dev < 1e-9
    bad += not ok
    print(f"  M1r {e:6s} dmu {dm[e]:.10e}, bio both ways max rel dev {dev:.1e}"
          f"  {'ok' if ok else 'FAIL'}")
gap = abs(dm["m1r2e"] / dm["none"] - 1)
ok = gap > 1e-4
bad += not ok
print(f"  M1r two-electron terms change dmu by {gap:.1e} (relative)  {'ok' if ok else 'FAIL'}")
print("bio_li:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(bad != 0)
PY
