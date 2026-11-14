"""Radial integrals over ATOMCI's contracted spherical Gaussians (one centre).

A basis function is N * r**l * sum_k c_k n_k exp(-a_k r**2) * Y_lm, where
n_k normalizes the primitive and N the contraction. That is the convention
ATOMCI applies to the input coefficients.
"""
from math import gamma

import numpy as np

LETTERS = 'SPDFGHIK'


def read_bas(path):
    """Parse an ATOMCI native GTO basis file -> list of (l, exponents, coefs)."""
    toks = []
    for line in open(path):
        if line[:1] in ('#', '"'):
            continue
        toks += line.split()
    fns, i = [], 0
    while not toks[i].startswith(')'):
        l = LETTERS.index(toks[i].upper())
        n = int(toks[i + 1])
        sc = float(toks[i + 2])
        e = np.array([float(x) for x in toks[i + 3:i + 3 + n]]) * sc * sc
        c = np.array([float(x) for x in toks[i + 3 + n:i + 3 + 2 * n]])
        fns.append((l, e, c))
        i += 3 + 2 * n
    return fns


def _rint(l, p, k):
    """int_0^inf r^(2l+2+k) exp(-p r^2) dr."""
    m = 2 * l + 2 + k
    return gamma((m + 1) / 2) / (2 * p ** ((m + 1) / 2))


def _prim(l, a, b, kind, Z):
    p = a + b
    if kind == 'S':
        return _rint(l, p, 0)
    if kind == 'r':
        return _rint(l, p, 1)
    if kind == 'V':
        return -Z * _rint(l, p, -1)
    if kind == 'T':
        return (2 * l + 3) * a * b / p * _rint(l, p, 0)
    raise ValueError(kind)


def matrices(fns, l, Z):
    """S, <r>, T, V (=-Z/r) matrices over the normalized contracted functions of symmetry l."""
    F = [(e, c) for (ll, e, c) in fns if ll == l]
    out = {}
    for kind in ('S', 'r', 'T', 'V'):
        M = np.zeros((len(F), len(F)))
        for i, (ea, ca) in enumerate(F):
            for j, (eb, cb) in enumerate(F):
                M[i, j] = sum(x * y * _prim(l, a, b, kind, Z) /
                              np.sqrt(_rint(l, 2 * a, 0) * _rint(l, 2 * b, 0))
                              for a, x in zip(ea, ca) for b, y in zip(eb, cb))
        out[kind] = M
    nrm = np.sqrt(np.diag(out['S']))
    return {k: v / np.outer(nrm, nrm) for k, v in out.items()}
