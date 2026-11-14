"""Configuration lists for acatsp hfsrel (technical note, Sec. 5.2).

The space for the second order of the alpha^2 hfs correction: every configuration of NEL electrons
of the given parity with all electrons in the inner orbitals except at most MAXOUT in the near and
far orbitals together, and at most MAXFAR in the far ones. Closed under the single excitations
within each class, as hfsrel's resolution of the identity needs.

usage: hfsrel_cfg.py NEL PARITY INNER NEAR FAR [MAXOUT [MAXFAR]] > configs
  INNER, NEAR, FAR  comma-separated ATSP labels ("1s,2s,2p"; FAR may be "-")
  PARITY            0 even, 1 odd
"""
import sys
from itertools import combinations_with_replacement


def lval(lab):
    return "spdfghik".index(lab[-1])


def main():
    nel, par = int(sys.argv[1]), int(sys.argv[2])
    cls = [[x for x in s.split(",") if x and x != "-"] for s in sys.argv[3:6]]
    maxout = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    maxfar = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    labs = cls[0] + cls[1] + cls[2]
    kind = [0] * len(cls[0]) + [1] * len(cls[1]) + [2] * len(cls[2])
    n = 0
    for c in combinations_with_replacement(range(len(labs)), nel):
        occ = {}
        for i in c:
            occ[i] = occ.get(i, 0) + 1
        if any(v > 4 * lval(labs[i]) + 2 for i, v in occ.items()):
            continue
        if sum(lval(labs[i]) for i in c) % 2 != par:
            continue
        if sum(kind[i] > 0 for i in c) > maxout or sum(kind[i] == 2 for i in c) > maxfar:
            continue
        print(" ".join(f"{labs[i]}({occ[i]})" for i in sorted(occ)))
        n += 1
    print(f"hfsrel_cfg: {n} configurations", file=sys.stderr)


if __name__ == "__main__":
    main()
