"""Fock-space tools for the d shell (10 spin-orbitals): term bases of d^N with seniority labels, n-body lift,
Judd-Leavitt (1986) operator tables and decomposition.

Conventions
- Determinants as in cfp_l (spin-orbital order (m, s) of cfp_l.spin_orbitals(2)); |I> = a+_{i1} ... a+_{in} |0>
  with i1 < ... < in.
- A pure n-body operator is given by its matrix M on the d^n determinants: V = sum_{I,J} M[I,J] a+_I a_J with
  a_J = (a+_J)^dagger. lift(M, n, N) is its matrix on d^N.
- Term states: for each (L, 2S) the stretched states of cfp_l.stretched (seniority ordered, lowest first); the full
  multiplet by L-, S- lowering, so a scalar operator is fixed by its blocks on the stretched states.
  The seniority states of repeated terms get signs SIGN[(N, L, 2S)] (fixed by comparison with JL Tables 6-8, see
  scripts/phase13b.py step a); the default sign is that of cfp_l (largest component positive, ties to the lowest index).
- Components: one per term (diagonal) and one per pair v < v' of a repeated term (off-diagonal <v|X|v'>);
  metric weight (2L + 1)(2S + 1), off-diagonal counted twice (JL Eq. (1)).
"""
import csv
import itertools
import os
from functools import lru_cache
from math import comb, sqrt

import numpy as np

from atomci_heff import cfp_l

LD = 2
NSO = 10
LCH = 'SPDFGHI'
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
# Tables 7-9 of Judd & Leavitt, J. Phys. B 19, 485 (1986), transcribed: published data, not part of ATOMCI and not
# covered by LEGENDS.txt (Section 5); see refdata/README.md.
JLDIR = os.path.join(_C.REFDATA, 'JL1986')

# relative signs of seniority states (v order) relative to cfp_l.stretched; key (N, L, 2S) -> tuple of +-1.
# Fixed by phase13b step a: with these signs the pure 2-body e1..e4 and pure 3-body t1..t4 (JL d^3) lift exactly to
# JL Tables 7 (d^4) and 8 (d^5); each pattern is the unique one that does. d^3: cfp_l signs already match JL Table 6.
# (4, 0, 0) and (5, 4, 1) changed from (1, -1) to (1, 1) with the deterministic tie-break of cfp_l.stretched
# (fix/stretched-tiebreak): their v = 0 / v = 3 states have tied largest components, which the old argmax resolved
# by rounding (BLAS thread count); the physical (JL) states are unchanged.
SIGN_JL = {(4, 0, 0): (1, 1), (4, 1, 2): (1, -1), (4, 2, 0): (1, 1), (4, 3, 2): (1, -1), (4, 4, 0): (1, 1),
           (5, 2, 1): (1, 1, 1), (5, 3, 1): (1, -1), (5, 4, 1): (1, 1)}
# (7, 2, 1): not a JL sign; it keeps the pre-tie-break relative sign of the d^7 2D seniority states (whose v = 1 state
# flipped with the cfp_l.stretched tie-break), so the 13b d^7 term maps stay valid.
SIGN = {**SIGN_JL, (7, 2, 1): (1, -1)}


@lru_cache(maxsize=None)
def dets(N):
    return cfp_l.dets(LD, N)


@lru_cache(maxsize=None)
def index(N):
    return {d: i for i, d in enumerate(dets(N))}


def _one_body(N, f):
    return cfp_l.one_body(LD, N, f) if N > 0 else np.zeros((1, 1))


@lru_cache(maxsize=None)
def ladders(N):
    """L+, S+ on d^N."""
    SO = cfp_l.spin_orbitals(LD)
    lp = np.zeros((NSO, NSO))
    sp = np.zeros((NSO, NSO))
    for p, (m, s) in enumerate(SO):
        for q, (m2, s2) in enumerate(SO):
            if s == s2 and m == m2 + 1:
                lp[p, q] = sqrt(LD * (LD + 1) - m2 * (m2 + 1))
            if m == m2 and s == 1 and s2 == -1:
                sp[p, q] = 1.0
    return _one_body(N, lp), _one_body(N, sp)


@lru_cache(maxsize=None)
def _stretched(N, L, twoS):
    """cfp_l.stretched, with N < 2 (no pairing operator) handled here: (states, seniorities)."""
    if N >= 2:
        St, q = cfp_l.stretched(LD, N, L, twoS)
        return St, [seniority(N, x) for x in q]
    St = np.ones((1, 1)) if N == 0 else np.eye(NSO)[:, [cfp_l.spin_orbitals(LD).index((LD, 1))]]
    return St, [N]


def seniority(N, q):
    """Seniority from the pairing eigenvalue q = (N - v)(12 - N - v)/20 (d shell, normalized 1S pair)."""
    for v in range(N % 2, min(N, 10 - N) + 1, 2):
        if abs((N - v) * (12 - N - v) / 20 - q) < 1e-8:
            return v
    raise ValueError((N, q))


@lru_cache(maxsize=None)
def terms(N):
    """[(L, 2S, [v...])] for d^N, L then S ascending; v list in stretched order."""
    out = []
    for (L, twoS) in ([(0, 0)] if N == 0 else [(2, 1)] if N == 1 else sorted(cfp_l.multiplets(LD, N))):
        out.append((L, twoS, _stretched(N, L, twoS)[1]))
    return out


def label(N, L, twoS, v):
    return f'{v}:{twoS + 1}{LCH[L]}'


def stretched(N, L, twoS):
    St, _ = _stretched(N, L, twoS)
    s = SIGN.get((N, L, twoS))
    return St * np.array(s) if s is not None else St.copy()


@lru_cache(maxsize=None)
def _lowered(N, L, twoS):
    """Unsigned full multiplet: array (k, ndet, (2L+1)(2S+1)) for the k seniority copies."""
    St, _ = _stretched(N, L, twoS)
    Lp, Sp = ladders(N)
    Lm, Sm = Lp.T, Sp.T
    S = twoS / 2
    out = []
    for c in St.T:
        cols = []
        x = c.copy()
        for a in range(2 * L + 1):
            y = x.copy()
            for b in range(twoS + 1):
                cols.append(y / np.linalg.norm(y))
                y = Sm @ y
            x = Lm @ x
        out.append(np.array(cols).T)
    return np.array(out)


def multiplet(N, L, twoS):
    B = _lowered(N, L, twoS)
    s = SIGN.get((N, L, twoS))
    return B * np.array(s)[:, None, None] if s is not None else B


def components(N):
    """[(key, weight)], key = (L, 2S, i, j) with i <= j indices into the seniority list."""
    out = []
    for L, twoS, vs in terms(N):
        g = (2 * L + 1) * (twoS + 1)
        for i in range(len(vs)):
            out.append(((L, twoS, i, i), g))
        for i, j in itertools.combinations(range(len(vs)), 2):
            out.append(((L, twoS, i, j), 2 * g))
    return out


def comp_label(N, key):
    L, twoS, i, j = key
    vs = dict(((a, b), c) for a, b, c in terms(N))[(L, twoS)]
    a = label(N, L, twoS, vs[i])
    return a if i == j else a + '|' + label(N, L, twoS, vs[j])


def to_vec(N, X):
    """Component vector of a scalar operator given as a d^N determinant matrix (Hermitian part)."""
    X = 0.5 * (X + X.T)
    v = []
    for (L, twoS, i, j), _ in components(N):
        St = stretched(N, L, twoS)
        v.append(St[:, i] @ X @ St[:, j])
    return np.array(v)


def from_vec(N, v):
    """Determinant matrix of the scalar operator with components v."""
    X = np.zeros((len(dets(N)),) * 2)
    for ((L, twoS, i, j), _), x in zip(components(N), v):
        if x == 0:
            continue
        B = multiplet(N, L, twoS)
        P = B[i] @ B[j].T
        X += x * (P if i == j else P + P.T)
    return X


def weights(N):
    return np.array([w for _, w in components(N)], float)


def ip(N, x, y):
    return float(np.sum(weights(N) * x * y))


@lru_cache(maxsize=None)
def _subsets(N, n):
    """For each d^N det D: list of (J index in d^n, sign, R) with a_J D = sign |R>."""
    Dn = index(n)
    out = []
    for D in dets(N):
        lst = []
        for J in itertools.combinations(D, n):
            r = cfp_l._apply([('a', p) for p in J], D)    # a_{j1} ... a_{jn}: removes jn first
            # a_J = a_{jn} ... a_{j1} (j1 removed first) = (-1)^{n(n-1)/2} a_{j1} ... a_{jn}
            lst.append((Dn[J], r[0] * (-1) ** (n * (n - 1) // 2), r[1]))
        out.append(lst)
    return out


def lift(M, n, N):
    """Matrix on d^N of the pure n-body operator whose d^n matrix is M."""
    if N < n:
        return np.zeros((len(dets(N)),) * 2)
    Dn, DN, ixN = dets(n), dets(N), index(N)
    X = np.zeros((len(DN), len(DN)))
    nz = [np.nonzero(np.abs(M[:, j]) > 1e-15)[0] for j in range(len(Dn))]
    sub = _subsets(N, n)
    for c, lst in enumerate(sub):
        for J, sg, R in lst:
            Rs = set(R)
            for I in nz[J]:
                It = Dn[I]
                if Rs.intersection(It):
                    continue
                r = cfp_l._apply([('c', p) for p in It], R)
                X[ixN[r[1]], c] += M[I, J] * sg * r[0]
    return X


def nbody_parts(mats):
    """Pure n-body parts V_n (as d^n matrices) of a Fock-space operator given by its matrices on d^0..d^K
    (mats[N]); V_n = mats[n] - sum_{k<n} lift(V_k, k, n)."""
    V = []
    for n, M in enumerate(mats):
        R = M.copy()
        for k, Vk in enumerate(V):
            R -= lift(Vk, k, n)
        V.append(R)
    return V


def apply_parts(V, N):
    return sum(lift(Vk, k, N) for k, Vk in enumerate(V))


# ---------------------------------------------------------------- JL tables

def _val(x):
    x = x.strip()
    if '*s' in x:
        c, r = x.split('*s')
        return int(c) * sqrt(int(r))
    return float(int(x))


def jl_table(N):
    """{op: component vector on d^N in our component order} from refs/JL1986 (N = 4, 5) or the Phase 12a d^3
    vectors (N = 3, which use the same 2D sign convention as cfp_l)."""
    if N == 3:
        import sys
        _C.setup_path()
        from phase12a import JL
        import phase9
        # phase9 order 4F 4P 2H 2G 2F 2P; 2D 11, 33, 13 -> ours
        name = {('4F',): 0, ('4P',): 1, ('2H',): 2, ('2G',): 3, ('2F',): 4, ('2P',): 5}
        out = {}
        for k, v9 in JL.items():
            v = []
            for key, _ in components(3):
                L, twoS, i, j = key
                t = f'{twoS + 1}{LCH[L]}'
                if t == '2D':
                    v.append(v9[6] if (i, j) == (0, 0) else v9[7] if (i, j) == (1, 1) else v9[8])
                else:
                    v.append(v9[phase9.T3.index(t)])
            out[k] = np.array(v)
        return out
    fn = {4: 'JL1986_table7_d4.csv', 5: 'JL1986_table8_d5.csv'}[N]
    rows = [r for r in open(os.path.join(JLDIR, fn)) if not r.startswith('#')]
    rd = list(csv.reader(rows))
    hdr = rd[0][2:]
    ent = {(r[0], r[1]): [_val(x) for x in r[2:]] for r in rd[1:]}
    out = {}
    for c, op in enumerate(hdr):
        v = []
        for key, _ in components(N):
            lab = comp_label(N, key).split('|')
            k = (lab[0], lab[-1])
            v.append(ent[k][c] if k in ent else 0.0)
        out[op] = np.array(v)
    missing = set(ent) - {tuple((comp_label(N, k).split('|') * 2)[:2]) if '|' not in comp_label(N, k)
                          else tuple(comp_label(N, k).split('|')) for k, _ in components(N)}
    assert not missing, missing
    return out


def jl_norms():
    rows = [r for r in csv.reader(l for l in open(os.path.join(JLDIR, 'JL1986_table9_norm.csv'))
                                   if not l.startswith('#'))]
    return {r[0]: {N: int(x) for N, x in enumerate(r[1:]) if x} for r in rows[1:]}


def jl_coeffs(N, x, table):
    """Coefficients of x on the (orthogonal, unnormalized) JL operators: c_i = <x,H_i>/<H_i,H_i>."""
    return {k: ip(N, x, h) / ip(N, h, h) for k, h in table.items()}
