"""Independent LS-CSF counting from determinant M_L / M_S distributions.

For a spatial configuration (orbital -> occupation), the number of CSFs of
term (L, S) is
    N(M_L=L, M_S=S) - N(L+1, S) - N(L, S+1) + N(L+1, S+1)
where N counts the configuration's determinants with those projections.
"""
import collections
import itertools
from functools import lru_cache


@lru_cache(maxsize=None)
def shell_dist(l, q):
    """Counter over (M_L, 2M_S) of the determinants of q electrons in one orbital of angular momentum l."""
    so = [(m, s) for m in range(-l, l + 1) for s in (1, -1)]
    c = collections.Counter()
    for sub in itertools.combinations(so, q):
        c[(sum(m for m, _ in sub), sum(s for _, s in sub))] += 1
    return c


def config_dist(occ):
    """occ: iterable of (l, q). Counter over (M_L, 2M_S) of the configuration's determinants."""
    dist = collections.Counter({(0, 0): 1})
    for l, q in occ:
        new = collections.Counter()
        for (a, b), n in dist.items():
            for (c, d), m in shell_dist(l, q).items():
                new[(a + c, b + d)] += n * m
        dist = new
    return dist


def n_csf(occ, L, twoS):
    d = config_dist(occ)
    return d[(L, twoS)] - d[(L + 1, twoS)] - d[(L, twoS + 2)] + d[(L + 1, twoS + 2)]


def configurations(orbitals, nel):
    """All spatial occupations of nel electrons: orbitals = list of (name, l). Yields dict name -> q."""
    names = [o for o, _ in orbitals]
    caps = [2 * (2 * l + 1) for _, l in orbitals]

    def rec(i, left):
        if left == 0:
            yield {}
            return
        if i == len(names):
            return
        for q in range(min(caps[i], left), -1, -1):
            for rest in rec(i + 1, left - q):
                if q:
                    yield {names[i]: q, **rest}
                else:
                    yield rest
    yield from rec(0, nel)


def count_space(orbitals, nel, L, twoS, parity, keep=lambda occ: True):
    """Per-configuration CSF counts (dict frozenset(occ.items()) -> n) for the given term and parity."""
    lval = dict(orbitals)
    out = {}
    for occ in configurations(orbitals, nel):
        if sum(lval[o] * q for o, q in occ.items()) % 2 != parity or not keep(occ):
            continue
        n = n_csf([(lval[o], q) for o, q in occ.items()], L, twoS)
        if n:
            out[frozenset(occ.items())] = n
    return out
