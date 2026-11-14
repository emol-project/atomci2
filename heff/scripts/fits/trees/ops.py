import numpy as np, itertools
from dn import *

def dN_ops(N):
    """returns dict with dets, two-body basis (5 lifted d2 term projectors), L2, S2, Q, F2, F4 ops, zeta op"""
    sp = Space(['d'])
    dets = sp.dets({'d': N})
    Lz, Lp, Sz, Sp, ls = onebody_ang(sp)
    def L2S2(ds):
        lz = build1(Lz, ds, ds); lp = build1(Lp, ds, ds)
        sz = build1(Sz, ds, ds); spl = build1(Sp, ds, ds)
        return lp.T@lp + lz@lz + lz, spl.T@spl + sz@sz + sz
    # d2 term projectors
    d2 = sp.dets({'d': 2})
    L2_2, S2_2 = L2S2(d2)
    X = L2_2 + 100*S2_2
    w, v = np.linalg.eigh(X)
    projs = {}
    for val in np.unique(np.round(w, 6)):
        V = v[:, abs(w-val) < 1e-6]
        projs[val] = V@V.T
    assert len(projs) == 5
    idx2 = {m: i for i, m in enumerate(d2)}
    ops2 = []
    labels = []
    for val, P in sorted(projs.items()):
        tab = {}
        for r, s in itertools.combinations(range(sp.n), 2):
            ket = (1 << r) | (1 << s)
            lst = []
            for p, q in itertools.combinations(range(sp.n), 2):
                bra = (1 << p) | (1 << q)
                x = P[idx2[bra], idx2[ket]]
                if abs(x) > 1e-14: lst.append((p, q, x))
            tab[(r, s)] = lst
        ops2.append(build2(tab, dets, dets))
        S = (-1+np.sqrt(1+4*(val//100)))/2 if False else None
        labels.append(val)
    L2, S2 = L2S2(dets)
    # Coulomb F^k ops
    def Fk(k):
        tab = coulomb_table(sp, lambda sh, kk: 1.0 if kk == k else 0.0)
        return build2(tab, dets, dets)
    F2, F4 = Fk(2), Fk(4)
    zeta = build1(ls, dets, dets)
    # pairing op = lifted 1S projector (val with L=0,S=0 -> X=0)
    Q = ops2[0] if abs(labels[0]) < 1e-6 else None
    return dict(sp=sp, dets=dets, ops2=ops2, labels=labels, L2=L2, S2=S2, F2=F2, F4=F4, zeta=zeta, Qpair=Q,
                Lz=build1(Lz, dets, dets), Sz=build1(Sz, dets, dets))

def closure(N, kind, rad):
    """constant-denominator 2nd-order operator  W = - P V Q V P  (Delta = 1) on d^N.
    kind: 'core' (3s->3d), 'g' (3d->g), 'x' (3d->d')  ; rad: dict k->R^k for the coupling"""
    if kind == 'core':
        sp = Space(['s', 'd']); P = sp.dets({'s': 2, 'd': N}); Qd = sp.dets({'s': 1, 'd': N+1}); other = 's'
    else:
        sp = Space(['d', kind]); P = sp.dets({'d': N, kind: 0}); Qd = sp.dets({'d': N-1, kind: 1}); other = kind
    def radial(sh, k):
        if sorted(sh).count('d') == 3 and sorted(sh).count(other) == 1:
            return rad.get(k, 0.0)
        return 0.0
    tab = coulomb_table(sp, radial)
    V = build2(tab, P, Qd)
    W = -V.T@V
    if kind == 'core':
        Pd = [m >> 2 for m in P]
    else:
        Pd = P
    ref = Space(['d']).dets({'d': N})
    order = [Pd.index(m) for m in ref]
    return W[np.ix_(order, order)]

def project_out(A, basis):
    """Frobenius projection of A off span(basis); returns residual and coefficients"""
    B = np.array([b.ravel() for b in basis]).T
    c, *_ = np.linalg.lstsq(B, A.ravel(), rcond=None)
    return A - (B@c).reshape(A.shape), c
