# quasispin (technical note, Sec. 3.3): quasi-spin generators of one shell in tensor form.
#   Q_+ = (sqrt[b]/2) (b b)^(0),  Q_- = -(sqrt[b]/2) (b~ b~)^(0),  Q_z = (N - (2l+1))/2,
# check [Q_+, Q_-] = 2 Q_z, [Q_z, Q_+-] = +-Q_+-, Q_- = Q_+^T, and [Q, S] = [Q, L] = 0.
import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
from pyref import Space, cg2, spin_orbit_generators  # noqa: E402

for l in range(0, 3):  # f (2^14 determinants) is too large for dense matrices
    sp = Space([l], [(n,) for n in range(0, 4 * l + 3)])
    db = 2 * (2 * l + 1)
    comps = [(m, t) for m in range(-l, l + 1) for t in (1, -1)]
    tp, tm = [], []
    for (m1, t1) in comps:
        m2, t2 = -m1, -t1
        c = cg2(2 * l, 2 * m1, 2 * l, 2 * m2, 0, 0) * cg2(1, t1, 1, t2, 0, 0)
        tp.append((math.sqrt(db) / 2 * c, [(True, sp.orb[(0, m1, t1)]), (True, sp.orb[(0, m2, t2)])]))
        # b~_{m mu} = (-1)^(l-m) (-1)^(1/2-mu) a_{-m,-mu}
        p1 = (-1) ** (l - m1) * (-1) ** ((1 - t1) // 2)
        p2 = (-1) ** (l - m2) * (-1) ** ((1 - t2) // 2)
        tm.append((-math.sqrt(db) / 2 * c * p1 * p2,
                   [(False, sp.orb[(0, -m1, -t1)]), (False, sp.orb[(0, -m2, -t2)])]))
    Qp, Qm = sp.matrix(tp), sp.matrix(tm)
    Nop = np.diag([bin(d).count("1") for d in sp.dets]).astype(float)
    Qz = (Nop - (2 * l + 1) * np.eye(sp.dim)) / 2
    S, L = spin_orbit_generators(sp)
    errs = [np.abs(Qp @ Qm - Qm @ Qp - 2 * Qz).max(), np.abs(Qz @ Qp - Qp @ Qz - Qp).max(),
            np.abs(Qm - Qp.T).max()]
    for V in list(S.values()) + list(L.values()):
        errs.append(np.abs(V @ Qp - Qp @ V).max())
    # B_{+1/2} = b, B_{-1/2} = -b~ as a standard quasi-spin doublet needs Q'_+ = -Q_+, Q'_- = -Q_-:
    # [Q'_+, B_{-1/2}] = B_{+1/2},  [Q'_-, B_{+1/2}] = B_{-1/2},  [Q_z, B_tau] = tau B_tau
    dbl = 0.0
    for (m, t) in comps:
        bm = sp.matrix([(1.0, [(True, sp.orb[(0, m, t)])])])
        ph_ = (-1) ** (l - m) * (-1) ** ((1 - t) // 2)
        btm = sp.matrix([(float(ph_), [(False, sp.orb[(0, -m, -t)])])])
        Bp, Bm = bm, -btm
        Qp2, Qm2 = -Qp, -Qm
        dbl = max(dbl, np.abs(Qp2 @ Bm - Bm @ Qp2 - Bp).max(), np.abs(Qm2 @ Bp - Bp @ Qm2 - Bm).max(),
                  np.abs(Qz @ Bp - Bp @ Qz - 0.5 * Bp).max(), np.abs(Qz @ Bm - Bm @ Qz + 0.5 * Bm).max())
    errs.append(dbl)
    print("doublet err", dbl, end="  ")
    print("l", l, "dim", sp.dim, "errs", [round(float(e), 3) for e in errs[:3]], "SL", max(errs[3:]))
