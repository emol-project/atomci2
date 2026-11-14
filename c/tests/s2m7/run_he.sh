#!/bin/bash
# run_he.sh -- S2-M7 (Z6), report only: He 2 3P and 3 3P against Puchalski and
# Jentschura, arXiv:1209.5142 (relativistic rows of Table III; Table I term by
# term for 3 3P, whose column is consistent with Table III; the 2 3P column of
# Table I is not, e.g. its <p^2> equals 2|E(3 1P)|).
# Orbitals: ATSP MCHF, complete active spaces n <= 7 (l <= g) from 1s np:
# he23p_mchf.sh (2 3P alone) and he33p_mchf.sh (2 3P + 3 3P state-averaged).
# C's Delta g of a level is split into the delta g_S, delta g_L, delta g_x of the
# paper (Eq. (29): delta g_J = A delta g_S - 2B delta g_L + C delta g_x / 3, with
# A = 1/2, -2B = 1/2, C = -1/10 (J = 2), 1/2 (J = 1)):
#   orbital (ORMC + OZ2e) = delta g_L / 2 at both J;
#   spin (RMC + GC1e + GC2e) = delta g_S / 2 + c_J delta g_x, c_2 = -1/30, c_1 = 1/6.
# C's operators use g_s = 2; the paper's Eq. (15) has g_s - 1 (GC1e, SSO), g_s/2
# (SOO) and -(g_s - 2)/4 (p.sigma)(p.B), whose scalar part is (g_s - 2)/6 x RMC.
# dgx is printed in Yan-Drake's sign (their Eq. (1): delta g_J = -g_x/6 at J = 1,
# +g_x/30 at J = 2), the opposite of c_J above.
# The "g_s" lines apply these factors (the rank-2 part of the last term, about
# 1e-9 in delta g_x, is left out).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23p}
[ -s "$w/he_7.w" ] || NS="3 4 5 6 7" bash "$here/he23p_mchf.sh"
[ -s "$w/he3_7.w" ] || NS="4 5 6 7" bash "$here/he33p_mchf.sh"
out=$(mktemp); trap 'rm -f "$out"' EXIT
{ NS="3 4 5 6 7" bash "$here/he23p_g.sh"; echo "== SA"; NS="5 6 7" bash "$here/he33p_g.sh"; } > "$out"
python3 - "$out" <<'PY'
import re, sys
a2 = (1 / 137.035999084) ** 2
gs = 2.00231930436256
blocks, sa, cur, part = {}, "", None, "all"
for line in open(sys.argv[1]):
    if line.startswith("== SA"):
        sa = "SA"
    elif line.startswith("== n="):
        cur, part = sa + line.split()[1], "all"
    elif line.startswith("-- SSO"):
        part = "sso"
    f = line.split()
    if len(f) >= 10 and re.match(r"[24]/2", f[0]):
        blocks[(cur, part, int(f[1]), int(f[0][0]) // 2)] = list(map(float, f[6:]))

def split(tag, root):
    g = {J: blocks[(tag, "all", root, J)] for J in (1, 2)}
    s = {J: blocks[(tag, "sso", root, J)][3] for J in (1, 2)}
    def sx(v1, v2):  # (rank-0 delta g_S contribution, rank-2 delta g_x contribution)
        return (v1 + 5 * v2) / 3, 5 * (v1 - v2)
    t = {}
    t["RMC"] = sx(g[1][0], g[2][0])
    t["GC1e"] = sx(g[1][1], g[2][1])
    t["SSO"] = sx(s[1], s[2])
    t["SOO"] = sx(g[1][3] - s[1], g[2][3] - s[2])
    t["ORMC"] = 2 * g[2][2]
    t["OZ2e"] = 2 * g[2][4]
    return t

# Table I (3 3P column) through Eq. (16), g_s = 2, in units of alpha^2:
#   RMC -d2/2, GC1e (Z/4)(2/3) d3 and -(3)(Z/4) t3, SSO -(1/4)(2/3) d5 and 3(1/4) t5,
#   SOO -(1/2)(2/3) d5 and 3(1/2) t5, ORMC -(-v2/2)/2, OZ2e (v61 - v62)/4
d2, d3, t3, d5, t5 = 4.1161621685, 2.1160455752, 0.0801115169, 0.11592898192, 0.075528036275
v2, v61, v62 = -0.25996906425, 0.0306847511, -0.001716909955
tab = {"RMC": (-d2 / 2, 0.0), "GC1e": (d3 / 3, -1.5 * t3), "SSO": (-d5 / 6, 0.75 * t5),
       "SOO": (-d5 / 3, 1.5 * t5), "ORMC": v2 / 4, "OZ2e": (v61 - v62) / 4}
lit = {"2 3P": (-80.4293, -6.91292, -5.385487), "3 3P": (-75.0839, -3.02957, -2.648665)}
yd = {"2 3P": (-80.436904, None, -5.391808), "3 3P": (-75.096557, None, -2.650193)}

def totals(t, real):
    f = {"RMC": 1 + (gs - 2) / 6 if real else 1, "GC1e": gs - 1 if real else 1,
         "SSO": gs - 1 if real else 1, "SOO": gs / 2 if real else 1}
    dS = sum(f[k] * t[k][0] for k in f)
    dx = sum((1 if k == "RMC" else f[k]) * t[k][1] for k in f)
    dL = t["ORMC"] + t["OZ2e"]
    return dS, dL, dx

for state, tag, root in [("2 3P", f"n={n}", 1) for n in (3, 4, 5, 6, 7)] + \
                        [("2 3P", f"SAn={n}", 1) for n in (5, 6, 7)] + \
                        [("3 3P", f"SAn={n}", 2) for n in (5, 6, 7)]:
    t = split(tag, root)
    print(f"{state} {tag:6}", end="")
    for real in (False, True):
        dS, dL, dx = totals(t, real)
        print(f"  [{'g_s' if real else 'g=2'}] dgS {dS * 1e6:+.4f} dgL {dL * 1e6:+.5f} dgx {-dx * 1e6:+.5f}", end="")
    print()
for state in lit:
    print(f"{state} Puchalski-Jentschura Table III (relativistic): dgS {lit[state][0]:+.4f} "
          f"dgL {lit[state][1]:+.5f} dgx {lit[state][2]:+.6f}  (x 1e-6)")
for state in yd:
    print(f"{state} Yan-Drake PRA 50, R1980: dgS {yd[state][0]:+.6f} dgx {yd[state][2]:+.6f}")
for n in (5, 6, 7):
    print(f"3 3P term by term (alpha^2 units, g_s = 2): C (SA n={n})  /  Table I via Eq. (16)")
    t = split(f"SAn={n}", 2)
    for k in ("RMC", "GC1e", "SSO", "SOO"):
        c = (t[k][0] / a2, t[k][1] / a2)
        print(f"   {k:5} dgS {c[0]:+.7f} / {tab[k][0]:+.7f}   dgx {c[1]:+.7f} / {tab[k][1]:+.7f}")
    for k in ("ORMC", "OZ2e"):
        print(f"   {k:5} dgL {t[k] / a2:+.7f} / {tab[k]:+.7f}")
PY
