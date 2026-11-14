"""Minimal l^N determinant machinery (d shell + one s/d'/g shell) for
checking Trees (1963) Fe III 3d6 fit against our T1/T2.  Self-contained, numpy only."""
import numpy as np, math, itertools
from functools import lru_cache

@lru_cache(None)
def w3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0: return 0.0
    if j3 < abs(j1-j2) or j3 > j1+j2: return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3: return 0.0
    f = math.factorial
    tri = f(j1+j2-j3)*f(j1-j2+j3)*f(-j1+j2+j3)/f(j1+j2+j3+1)
    pre = math.sqrt(tri*f(j1+m1)*f(j1-m1)*f(j2+m2)*f(j2-m2)*f(j3+m3)*f(j3-m3))
    s = 0.0
    for t in range(0, 100):
        d = [t, j3-j2+t+m1, j3-j1+t-m2, j1+j2-j3-t, j1-t-m1, j2-t+m2]
        if min(d) < 0:
            if d[3] < 0 or d[4] < 0 or d[5] < 0: break
            continue
        s += (-1)**t / np.prod([f(x) for x in d])
    return (-1)**(j1-j2-m3) * pre * s

@lru_cache(None)
def ck(l1, m1, l2, m2, k):
    return (-1)**m1 * math.sqrt((2*l1+1)*(2*l2+1)) * w3j(l1, k, l2, 0, 0, 0) * w3j(l1, k, l2, -m1, m1-m2, m2)

LVAL = {'s': 0, 'd': 2, 'x': 2, 'g': 4, 'i': 6}

class Space:
    def __init__(self, shells):
        # spin orbitals (shell, m, sigma); sigma = +1/-1 (as 2*ms)
        self.so = [(sh, m, sg) for sh in shells for m in range(-LVAL[sh], LVAL[sh]+1) for sg in (1, -1)]
        self.n = len(self.so)
        self.index = {s: i for i, s in enumerate(self.so)}
    def dets(self, occ):
        """occ: dict shell->number of electrons. returns list of bitmasks"""
        groups = []
        for sh, ne in occ.items():
            idx = [i for i, s in enumerate(self.so) if s[0] == sh]
            groups.append([sum(1 << i for i in c) for c in itertools.combinations(idx, ne)])
        out = []
        for combo in itertools.product(*groups):
            out.append(sum(combo))
        return sorted(out)

def sgn_ann(mask, i):
    return -1 if bin(mask & ((1 << i) - 1)).count('1') % 2 else 1

def apply2(mask, p, q, r, s):
    """a+_p a+_q a_s a_r |mask>; returns (sign, newmask) or None"""
    if not (mask >> r) & 1 or not (mask >> s) & 1: return None
    sg = sgn_ann(mask, r); m = mask ^ (1 << r)
    sg *= sgn_ann(m, s); m ^= (1 << s)
    if (m >> q) & 1: return None
    sg *= sgn_ann(m, q); m |= (1 << q)
    if (m >> p) & 1: return None
    sg *= sgn_ann(m, p); m |= (1 << p)
    return sg, m

def apply1(mask, p, q):
    if not (mask >> q) & 1: return None
    sg = sgn_ann(mask, q); m = mask ^ (1 << q)
    if (m >> p) & 1: return None
    sg *= sgn_ann(m, p); m |= (1 << p)
    return sg, m

def coulomb_table(sp, radial):
    """antisymmetrised <pq||rs> for p<q, r<s; radial(shells_tuple(a,b,c,d), k) -> R^k(ab;cd)"""
    so = sp.so
    def g(a, b, c, d):
        A, B, C, D = so[a], so[b], so[c], so[d]
        if A[2] != C[2] or B[2] != D[2] or A[1]+B[1] != C[1]+D[1]: return 0.0
        la, lb, lc, ld = (LVAL[x[0]] for x in (A, B, C, D))
        v = 0.0
        for k in range(0, 9):
            R = radial((A[0], B[0], C[0], D[0]), k)
            if R == 0: continue
            v += ck(la, A[1], lc, C[1], k) * ck(ld, D[1], lb, B[1], k) * R
        return v
    tab = {}
    n = sp.n
    for r, s in itertools.combinations(range(n), 2):
        lst = []
        for p, q in itertools.combinations(range(n), 2):
            val = g(p, q, r, s) - g(p, q, s, r)
            if abs(val) > 1e-14: lst.append((p, q, val))
        if lst: tab[(r, s)] = lst
    return tab

def build2(tab, kets, bras):
    bi = {m: i for i, m in enumerate(bras)}
    M = np.zeros((len(bras), len(kets)))
    for j, k in enumerate(kets):
        occ = [i for i in range(64) if (k >> i) & 1]
        for r, s in itertools.combinations(occ, 2):
            for p, q, val in tab.get((r, s), ()):
                res = apply2(k, p, q, r, s)
                if res is None: continue
                sg, m = res
                i = bi.get(m)
                if i is not None: M[i, j] += sg * val
    return M

def build1(h, kets, bras):
    """h: dict (p,q)->val"""
    bi = {m: i for i, m in enumerate(bras)}
    M = np.zeros((len(bras), len(kets)))
    for j, k in enumerate(kets):
        for (p, q), val in h.items():
            res = apply1(k, p, q)
            if res is None: continue
            sg, m = res
            i = bi.get(m)
            if i is not None: M[i, j] += sg * val
    return M

def onebody_ang(sp, shell='d'):
    """Lz, L+, Sz, S+ and l.s as dicts"""
    so = sp.so; l = LVAL[shell]
    Lz, Lp, Sz, Sp, ls = {}, {}, {}, {}, {}
    for i, (sh, m, sg) in enumerate(so):
        if sh != shell: continue
        Lz[(i, i)] = m; Sz[(i, i)] = sg/2
        if m < l:
            j = sp.index[(sh, m+1, sg)]
            Lp[(j, i)] = math.sqrt(l*(l+1) - m*(m+1))
        if sg == -1:
            j = sp.index[(sh, m, 1)]
            Sp[(j, i)] = 1.0
    # l.s = lz sz + 1/2(l+ s- + l- s+)
    for i, (sh, m, sg) in enumerate(so):
        if sh != shell: continue
        ls[(i, i)] = ls.get((i, i), 0) + m*sg/2
        # l+ s- : (m,+) -> (m+1,-)
        if sg == 1 and m < l:
            j = sp.index[(sh, m+1, -1)]
            ls[(j, i)] = ls.get((j, i), 0) + 0.5*math.sqrt(l*(l+1)-m*(m+1))
        if sg == -1 and m > -l:
            j = sp.index[(sh, m-1, 1)]
            ls[(j, i)] = ls.get((j, i), 0) + 0.5*math.sqrt(l*(l+1)-m*(m-1))
    return Lz, Lp, Sz, Sp, ls
