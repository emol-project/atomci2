"""pyref.py -- brute-force reference for the stage-2 C port (S2-M0).

Everything here is built from Slater determinants and explicit Clebsch-
Gordan coefficients. It does not use CFPTBL or any ATOMCI code, so it can
check the C engine independently (atomci2 technical note, Sec. 7.2).

Conventions (atomci2 technical note, Sec. 3.1):
  * a spin orbital is (shell k, m, 2*mu); shells are numbered from 0 here;
  * b_{m mu} = a^dagger_{m mu}  (the creation tensor of the shell);
  * b~_{m mu} = (-1)^(l-m) (-1)^(1/2-mu) a_{-m,-mu}  (Sasaki's adjoint
    tensor, METECC eq. 9 with U = sqrt[j] (00|j m j m'));
  * coupling (A B)^(C)_c = sum <A a B b|C c> A_a B_b, Condon-Shortley,
    left factor first; spin and orbital parts couple separately;
  * reduced matrix elements in Edmonds' convention:
      <j m|T_q|j' m'> = (-1)^(j-m) (j k j'; -m q m') <j||T||j'>,
    applied to spin and orbit separately for double tensors.
Angular momenta of spin type are passed doubled (ts = 2S, tmu = 2mu);
orbital ones as integers.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------- symbols


@lru_cache(maxsize=None)
def _fact(n: int) -> float:
    return float(math.factorial(n))


@lru_cache(maxsize=None)
def cg2(tj1: int, tm1: int, tj2: int, tm2: int, tj: int, tm: int) -> float:
    """<j1 m1 j2 m2 | j m> with every argument doubled (Racah formula)."""
    if tm1 + tm2 != tm:
        return 0.0
    if not (abs(tj1 - tj2) <= tj <= tj1 + tj2) or (tj1 + tj2 + tj) % 2:
        return 0.0
    for tj_, tm_ in ((tj1, tm1), (tj2, tm2), (tj, tm)):
        if abs(tm_) > tj_ or (tj_ + tm_) % 2:
            return 0.0
    a = (tj1 + tj2 - tj) // 2
    b = (tj1 - tj2 + tj) // 2
    c = (-tj1 + tj2 + tj) // 2
    pre = math.sqrt((tj + 1) * _fact(a) * _fact(b) * _fact(c)
                    / _fact((tj1 + tj2 + tj) // 2 + 1))
    pre *= math.sqrt(_fact((tj1 + tm1) // 2) * _fact((tj1 - tm1) // 2)
                     * _fact((tj2 + tm2) // 2) * _fact((tj2 - tm2) // 2)
                     * _fact((tj + tm) // 2) * _fact((tj - tm) // 2))
    s = 0.0
    for k in range(0, a + 1):
        d = [k, a - k, (tj1 - tm1) // 2 - k, (tj2 + tm2) // 2 - k,
             (tj - tj2 + tm1) // 2 + k, (tj - tj1 - tm2) // 2 + k]
        if min(d) < 0:
            continue
        den = 1.0
        for x in d:
            den *= _fact(x)
        s += (-1) ** k / den
    return pre * s


def threej2(tj1, tm1, tj2, tm2, tj3, tm3) -> float:
    """3j symbol (j1 j2 j3; m1 m2 m3), doubled arguments."""
    if tm1 + tm2 + tm3 != 0:
        return 0.0
    ph = -1.0 if ((tj1 - tj2 - tm3) // 2) % 2 else 1.0
    return ph * cg2(tj1, tm1, tj2, tm2, tj3, -tm3) / math.sqrt(tj3 + 1)


# ---------------------------------------------------------------- Fock space


class Space:
    """Determinants of a set of configurations over the given shells.

    shells: list of l values; configs: list of occupation tuples.
    """

    def __init__(self, shells, configs):
        self.shells = list(shells)
        self.orb = {}  # (k, m, tmu) -> bit index
        for k, l in enumerate(self.shells):
            for m in range(-l, l + 1):
                for tmu in (1, -1):
                    self.orb[(k, m, tmu)] = len(self.orb)
        self.dets = []
        for occ in configs:
            per_shell = []
            for k, (l, n) in enumerate(zip(self.shells, occ)):
                idx = [self.orb[(k, m, t)] for m in range(-l, l + 1) for t in (1, -1)]
                per_shell.append([sum(1 << i for i in c) for c in itertools.combinations(idx, n)])
            for parts in itertools.product(*per_shell):
                self.dets.append(sum(parts))
        self.dets = sorted(set(self.dets))
        self.index = {d: i for i, d in enumerate(self.dets)}
        self.dim = len(self.dets)

    # elementary operators on a determinant: (sign, new det) or None
    @staticmethod
    def _create(det, i):
        if det >> i & 1:
            return None
        sign = -1 if bin(det & ((1 << i) - 1)).count("1") % 2 else 1
        return sign, det | (1 << i)

    @staticmethod
    def _annihilate(det, i):
        if not det >> i & 1:
            return None
        sign = -1 if bin(det & ((1 << i) - 1)).count("1") % 2 else 1
        return sign, det & ~(1 << i)

    def matrix(self, terms):
        """Matrix of sum_c coef * op_1 op_2 ... (op_last acts first).

        terms: list of (coef, [(is_creator, bit), ...]).
        States leaving the space are dropped (the space must be closed
        under the operator for that to be exact)."""
        M = np.zeros((self.dim, self.dim))
        for coef, ops in terms:
            if coef == 0.0:
                continue
            for j, det in enumerate(self.dets):
                s, d = coef, det
                for cre, bit in reversed(ops):
                    r = self._create(d, bit) if cre else self._annihilate(d, bit)
                    if r is None:
                        break
                    s, d = s * r[0], r[1]
                else:
                    i = self.index.get(d)
                    if i is not None:
                        M[i, j] += s
        return M

    # tensor components as lists of (coef, (is_creator, bit))
    def b(self, k, m, tmu):
        return [(1.0, (True, self.orb[(k, m, tmu)]))]

    def bt(self, k, m, tmu):
        l = self.shells[k]
        ph = (-1) ** (l - m) * (-1) ** ((1 - tmu) // 2)
        return [(float(ph), (False, self.orb[(k, -m, -tmu)]))]


# ---------------------------------------------------------------- coupled operators


def _components(space, k):
    """All (m, tmu) of shell k."""
    l = space.shells[k]
    return [(m, t) for m in range(-l, l + 1) for t in (1, -1)]


def one_body(space, a, b, ts, kl, tnu, q):
    """Component (nu, q) of (b_a b~_b)^(sigma, k): ts = 2 sigma, tnu = 2 nu."""
    la, lb = space.shells[a], space.shells[b]
    terms = []
    for (m1, t1) in _components(space, a):
        for (m2, t2) in _components(space, b):
            c = cg2(2 * la, 2 * m1, 2 * lb, 2 * m2, 2 * kl, 2 * q) * cg2(1, t1, 1, t2, ts, tnu)
            if c == 0.0:
                continue
            for (c1, o1), in [space.b(a, m1, t1)]:
                for (c2, o2), in [space.bt(b, m2, t2)]:
                    terms.append((c * c1 * c2, [o1, o2]))
    return terms


def one_body_tensor(space, a, b, ts, kl):
    """{(tnu, q): matrix} of (b_a b~_b)^(sigma, k)."""
    out = {}
    for tnu in range(-ts, ts + 1, 2):
        for q in range(-kl, kl + 1):
            out[(tnu, q)] = space.matrix(one_body(space, a, b, ts, kl, tnu, q))
    return out


def two_body_normal(space, a, b, c, d, p, q, tot):
    """{(tnu, mq): matrix} of N[((b_a b~_b)^p (b_c b~_d)^q)^tot];
    p, q, tot are (2 sigma, k) pairs."""
    out = {}
    tsT, kT = tot
    for tnu in range(-tsT, tsT + 1, 2):
        for mq in range(-kT, kT + 1):
            terms = []
            for t1 in range(-p[0], p[0] + 1, 2):
                for q1 in range(-p[1], p[1] + 1):
                    t2, q2 = tnu - t1, mq - q1
                    if abs(t2) > q[0] or abs(q2) > q[1]:
                        continue
                    cc = cg2(p[0], t1, q[0], t2, tsT, tnu) * cg2(2 * p[1], 2 * q1, 2 * q[1], 2 * q2, 2 * kT, 2 * mq)
                    if cc == 0.0:
                        continue
                    for c1, ops1 in one_body(space, a, b, p[0], p[1], t1, q1):
                        for c2, ops2 in one_body(space, c, d, q[0], q[1], t2, q2):
                            # N[b_a b~_b b_c b~_d] = - b_a b_c b~_b b~_d
                            terms.append((-cc * c1 * c2, [ops1[0], ops2[0], ops1[1], ops2[1]]))
            out[(tnu, mq)] = space.matrix(terms)
    return out


# ---------------------------------------------------------------- S, L and multiplets


def spin_orbit_generators(space):
    """Spherical components (index -1, 0, +1) of the total S and L."""
    S = {q: np.zeros((space.dim, space.dim)) for q in (-1, 0, 1)}
    L = {q: np.zeros((space.dim, space.dim)) for q in (-1, 0, 1)}
    for k, l in enumerate(space.shells):
        for m in range(-l, l + 1):
            for t in (1, -1):
                o = space.orb[(k, m, t)]
                S[0] += space.matrix([(t / 2.0, [(True, o), (False, o)])])
                L[0] += space.matrix([(float(m), [(True, o), (False, o)])])
            # S+ = a+_{m up} a_{m down};  S_{+1} = -S+/sqrt2,  S_{-1} = S-/sqrt2
            up, dn = space.orb[(k, m, 1)], space.orb[(k, m, -1)]
            sp = space.matrix([(1.0, [(True, up), (False, dn)])])
            S[1] += -sp / math.sqrt(2)
            S[-1] += sp.T / math.sqrt(2)
            if m < l:
                for t in (1, -1):
                    hi, lo = space.orb[(k, m + 1, t)], space.orb[(k, m, t)]
                    lp = math.sqrt(l * (l + 1) - m * (m + 1)) * space.matrix([(1.0, [(True, hi), (False, lo)])])
                    L[1] += -lp / math.sqrt(2)
                    L[-1] += lp.T / math.sqrt(2)
    return S, L


def casimir(V):
    """V^2 = sum_q (-1)^q V_q V_-q."""
    return V[0] @ V[0] - V[1] @ V[-1] - V[-1] @ V[1]


class Multiplets:
    """Orthonormal (S, L) multiplets of a space, each with its states at
    every (M_S, M_L), built from the stretched state by S- and L-
    (Condon-Shortley phases)."""

    def __init__(self, space):
        self.space = space
        S, L = spin_orbit_generators(space)
        self.S, self.L = S, L
        S2, L2 = casimir(S), casimir(L)
        Sz, Lz = S[0], L[0]
        Sminus, Lminus = math.sqrt(2) * S[-1], math.sqrt(2) * L[-1]
        dz_s = np.round(2 * np.diag(Sz)).astype(int)
        dz_l = np.round(np.diag(Lz)).astype(int)
        self.blocks = {}  # (2S, L) -> {(2MS, ML): matrix of column vectors}
        for tms in sorted(set(dz_s)):
            for ml in sorted(set(dz_l)):
                sel = np.where((dz_s == tms) & (dz_l == ml))[0]
                if len(sel) == 0 or tms < 0 or ml < 0:
                    continue
                # stretched states of multiplets with 2S = tms, L = ml
                P = np.zeros((space.dim, len(sel)))
                P[sel, np.arange(len(sel))] = 1.0
                X = P.T @ (S2 / 1.0 + 0.7071 * L2) @ P
                target = tms / 2 * (tms / 2 + 1) + 0.7071 * ml * (ml + 1)
                w, v = np.linalg.eigh(X)
                pick = np.abs(w - target) < 1e-8
                if not pick.any():
                    continue
                top = P @ v[:, pick]
                states = {}
                for k2 in range(0, tms + 1):
                    for kl in range(0, 2 * ml + 1):
                        vec = top.copy()
                        for _ in range(k2):
                            vec = Sminus @ vec
                        for _ in range(kl):
                            vec = Lminus @ vec
                        vec /= np.linalg.norm(vec, axis=0)
                        states[(tms - 2 * k2, ml - kl)] = vec
                self.blocks[(tms, ml)] = states

    def reduced(self, op, bra, ket, tsig, kl):
        """Edmonds reduced matrix <bra||op||ket> (rows: bra multiplets,
        columns: ket multiplets) of a double tensor op = {(tnu, q): matrix}
        of ranks (sigma, k) = (tsig/2, kl). bra/ket: (2S, L) keys."""
        (tS, L), (tSp, Lp) = bra, ket
        B, K = self.blocks[bra], self.blocks[ket]
        best = None
        for (tms_k, ml_k), vk in K.items():
            tnu, q = tS - tms_k, L - ml_k
            if abs(tnu) > tsig or abs(q) > kl or (tnu, q) not in op:
                continue
            f = threej2(tS, -tS, tsig, tnu, tSp, tms_k) * threej2(2 * L, -2 * L, 2 * kl, 2 * q, 2 * Lp, 2 * ml_k)
            if abs(f) > 1e-12 and (best is None or abs(f) > abs(best[0])):
                best = (f, tnu, q, vk)
        if best is None:
            return None
        f, tnu, q, vk = best
        # (-1)^(S-M_S) and (-1)^(L-M_L) are +1 for the stretched bra
        return (B[(tS, L)].T @ op[(tnu, q)] @ vk) / f
