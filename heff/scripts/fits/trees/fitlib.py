import numpy as np
from ops import *
from dn import onebody_ang, build1

OBS = [  # term, J, E0 (Trees 1963 Table I)
 ('a5D',4,0),('a5D',3,436),('a5D',2,739),('a5D',1,932),('a5D',0,1027),
 ('a3P',2,19405),('a3P',1,20688),('a3P',0,21208),
 ('a3H',6,20051),('a3H',5,20301),('a3H',4,20482),
 ('a3F',4,21462),('a3F',3,21700),('a3F',2,21857),
 ('a3G',5,24559),('a3G',4,24941),('a3G',3,25142),
 ('a1I',6,30356),('a3D',1,30726),('a3D',2,30716),('a3D',3,30858),
 ('a1G',4,30886),('a1S',0,34812),('a1D',2,35804),('a1F',3,42897),
 ('b3P',0,49148),('b3P',1,49577),('b3P',2,50412),
 ('b3F',2,50185),('b3F',3,50295),('b3F',4,50276),('b1G',4,57222)]

class D6:
    def __init__(self):
        o = dN_ops(6); self.o = o
        sp, dets = o['sp'], o['dets']
        Lz, Lp, Sz, Sp, ls = onebody_ang(sp)
        lz, lp, sz, spp = (build1(x, dets, dets) for x in (Lz, Lp, Sz, Sp))
        LS = lz@sz + 0.5*(lp@spp.T + lp.T@spp)
        self.J2 = o['L2'] + o['S2'] + 2*LS
        self.Jz = lz + sz
        self.L2, self.S2 = o['L2'], o['S2']
        jz = np.round(np.diag(self.Jz), 6)
        self.blocks = {J: np.where(abs(jz-J) < 1e-6)[0] for J in range(0, 7)}
        # pairing (seniority) operator scaled to Racah Q(N,v)=(N-v)(12-N-v)/4
        P = o['Qpair']; ev = np.unique(np.round(np.linalg.eigvalsh(P), 6))
        # eigenvalues for v=4,2,0 should be in ratio 1:4:9
        self.Q = P*(1.0/ev[0] if ev[0] > 1e-8 else 1.0/ev[1])
        self.Qev = np.unique(np.round(np.linalg.eigvalsh(self.Q), 6))
        I = np.eye(len(dets))
        self.ops = {'A': I, 'B': 49*o['F2'], 'C': 7*o['F2'] + 12.6*o['F4'],
                    'zeta': o['zeta'], 'alpha': o['L2'], 'beta': self.Q}
    def levels(self, H):
        """returns dict J -> (energies ascending, vectors) for states with total J (taken from M_J=J block)"""
        out = {}
        for J, idx in self.blocks.items():
            h = H[np.ix_(idx, idx)]; w, v = np.linalg.eigh(h)
            j2 = np.einsum('ia,ij,ja->a', v, self.J2[np.ix_(idx, idx)], v)
            sel = abs(j2 - J*(J+1)) < 1e-4
            out[J] = (w[sel], v[:, sel], idx)
        return out
    def assign(self, lev, obs, ref):
        """ref: dict obs-index -> (J, reference vector in M_J=J block). returns list of (k) per obs"""
        ks = {}
        for J in range(7):
            rows = [i for i in range(len(obs)) if obs[i][1] == J]
            if not rows: continue
            w, v, idx = lev[J]
            if ref is None:
                rows = sorted(rows, key=lambda i: obs[i][2])
                for k, i in enumerate(rows): ks[i] = k
            else:
                used = set()
                for i in rows:
                    ov = abs(ref[obs[i][:2]]@v)
                    for k in np.argsort(-ov):
                        if k not in used: used.add(k); ks[i] = k; break
        return ks
    def fit(self, names, extra, p0, obs=OBS, iters=40, weights=None, ref=None, Es=None):
        """least squares on observed levels; extra: dict name->matrix"""
        ops = {**self.ops, **extra}; p = np.array(p0, float)
        if Es is None: Es = np.array([e for *_, e in obs], float)
        for it in range(iters):
            H = sum(pi*ops[n] for pi, n in zip(p, names))
            lev = self.levels(H)
            ks = self.assign(lev, obs, ref)
            calc = np.zeros(len(obs)); Jac = np.zeros((len(obs), len(names)))
            for J in range(7):
                rows = [i for i, (t, jj, e) in enumerate(obs) if jj == J]
                if not rows: continue
                w, v, idx = lev[J]
                for i in rows:
                    k = ks[i]
                    calc[i] = w[k]
                    for a, n in enumerate(names):
                        O = ops[n][np.ix_(idx, idx)]
                        Jac[i, a] = v[:, k]@O@v[:, k]
            r = Es - calc
            W = np.ones(len(obs)) if weights is None else weights
            dp, *_ = np.linalg.lstsq(Jac*np.sqrt(W)[:, None], r*np.sqrt(W), rcond=None)
            p += dp
            if np.max(abs(dp)) < 1e-9: break
        H = sum(pi*ops[n] for pi, n in zip(p, names))
        lev = self.levels(H)
        dof = len(obs) - len(names)
        s = np.sqrt(np.sum(W*r**2)/dof)
        cov = np.linalg.inv((Jac*W[:, None]).T@Jac)*s**2
        refv = {}
        for J in range(7):
            w, v, idx = lev[J]
            for i in range(len(obs)):
                if obs[i][1] == J: refv[obs[i][:2]] = v[:, ks[i]]
        return dict(refv=refv, p=p, err=np.sqrt(np.diag(cov)), resid=r, calc=calc, mean=s, Jac=Jac, lev=lev, cov=cov)
