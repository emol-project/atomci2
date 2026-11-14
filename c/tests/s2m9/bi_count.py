"""Count LS CSFs of B I expansions (OGJMF 1995, Table I) two ways:
SD from the multireference set, and RAS (1s RAS1 min 0, RAS2 2s2p[3d], RAS3 <= 2)."""
import itertools, sys
from functools import lru_cache
from collections import Counter

LET = "spdfghik"

@lru_cache(None)
def subshell_terms(l, q):
    """Counter {(2S, L): multiplicity} of the LS terms of l^q."""
    ms = [(m, s) for m in range(-l, l + 1) for s in (1, -1)]
    cnt = Counter()
    for c in itertools.combinations(ms, q):
        cnt[(sum(x[0] for x in c), sum(x[1] for x in c))] += 1
    terms = Counter()
    keys = sorted(cnt, key=lambda k: (-k[0], -k[1]))
    rem = dict(cnt)
    for L in range(l * q, -1, -1):
        for S2 in range(q, -1, -2):
            n = rem.get((L, S2), 0)
            if n > 0:
                terms[(S2, L)] += n
                for ML in range(-L, L + 1):
                    for MS in range(-S2, S2 + 1, 2):
                        rem[(ML, MS)] -= n
    return terms

def ncsf(cfg, S2t, Lt):
    """cfg: list of (n, l, q); number of CSFs with total (2S, L)."""
    cur = Counter({(0, 0): 1})
    for (_, l, q) in cfg:
        if q == 0:
            continue
        new = Counter()
        for (S1, L1), a in cur.items():
            for (S2, L2), b in subshell_terms(l, q).items():
                for S in range(abs(S1 - S2), S1 + S2 + 1, 2):
                    for L in range(abs(L1 - L2), L1 + L2 + 1):
                        new[(S, L)] += a * b
        cur = new
    return cur.get((S2t, Lt), 0)

def orbitals(nmax, lmax_of):
    return [(n, l) for n in range(1, nmax + 1) for l in range(min(n - 1, lmax_of(n)) + 1)]

def configs(orbs, nel, par):
    caps = [2 * (2 * l + 1) for (_, l) in orbs]
    def rec(i, left):
        if i == len(orbs):
            if left == 0:
                yield ()
            return
        for q in range(min(caps[i], left), -1, -1):
            for rest in rec(i + 1, left - q):
                yield (q,) + rest
    for occ in rec(0, nel):
        if sum(q * l for q, (_, l) in zip(occ, orbs)) % 2 == par:
            yield occ

def count(term, nmax, lmax_of, scheme):
    S2, L, par = {"2P": (1, 1, 1), "2D": (1, 2, 0)}[term]
    orbs = orbitals(nmax, lmax_of)
    idx = {o: i for i, o in enumerate(orbs)}
    if term == "2P":
        refs = [{(1, 0): 2, (2, 0): 2, (2, 1): 1}, {(1, 0): 2, (2, 1): 3}]
        ras2 = {(2, 0), (2, 1)}
    else:
        refs = [{(1, 0): 2, (2, 0): 1, (2, 1): 2}, {(1, 0): 2, (2, 0): 2, (3, 2): 1}]
        ras2 = {(2, 0), (2, 1)} | ({(3, 2)} if nmax >= 3 else set())
    tot = 0
    for occ in configs(orbs, 5, par):
        if scheme == "sd":
            ok = False
            for r in refs:
                if all(o in idx for o in r):
                    exc = sum(max(0, r.get(o, 0) - occ[idx[o]]) for o in orbs) + \
                        sum(v for o, v in r.items() if o not in idx)
                    if exc <= 2:
                        ok = True
        else:
            ras3 = sum(q for q, o in zip(occ, orbs) if o != (1, 0) and o not in ras2)
            ok = ras3 <= 2
        if ok:
            tot += ncsf([(n, l, q) for q, (n, l) in zip(occ, orbs)], S2, L)
    return tot

olsen = {"2P": [6, 129, 520, 1301, 2584, 4479], "2D": [4, 157, 812, 2320, 5004, 9161]}
for lcap in (sys.argv[1:] or ["3", "4"]):
    lc = int(lcap)
    print(f"l <= {LET[lc]}")
    for term in ("2P", "2D"):
        for scheme in ("sd", "ras"):
            row = [count(term, n, lambda n: lc, scheme) for n in range(2, 8)]
            print(f"  {term} {scheme:3s} {row}   Olsen {olsen[term]}")
