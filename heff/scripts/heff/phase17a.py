#!/usr/bin/env python3
"""Phase 17 A: second-order H_eff on the stored H (results/<run>.npz), same P, orbitals and H as the exact H_eff.

Per term block (P = the m CSFs of 3s2 3p6 3d^N, Q = the rest; Hpp = PHP, V = PHQ, p_j / a_j = eigenpairs of Hpp, the
term-dependent P energies), every second-order variant is
    H2 = Hpp + (V X + X^T V^T) / 2,   X = Y A^T,   Y_qj = (a_j^T V)_q / D_jq
with the denominator D_jq = (P energy) - (Q energy):
    C2       closure: Ebar_P - Ebar_c(q)  (configuration averages of the CSF diagonal over all terms, weight (2L+1)(2S+1);
             term independent; == the 15j/15o 'closure')
    S2-cav   p_j - Ebar_c(q)              (term-dependent P energy, configuration-averaged Q energy)
    S2-diag  p_j - H_qq                   (term-dependent P energy, CSF diagonal of QHQ; Epstein-Nesbet-type H0)
    S2-full  (H_QQ - p_j) y_j = -V^T a_j  (full QHQ; == the 15o 'sylvester', not recomputed here: partially dressed)
C2, S2-cav and S2-diag are additive over Q; they are stored per Q configuration. For A3 the lowest QHQ eigenpairs
(eps_k, u_k) are computed (Davidson on QHQ) and their S2-full contributions
    dH_k = (g_k + g_k^T) / 2,  g_k = (V u_k) (w_k / (p - eps_k))^T A^T,  w_k = A^T V u_k
are stored, with the dominant configurations of u_k.
The levels are placed on the HF P block as in 15k analyze3: HF + (H2 - PHP).

Ion via env ION (vnd | crnd | mnnd). Older runs (15m/15n, standard basis, own orbitals of the 15 series): env A_RUNFMT
(run name with {t}), A_TAG (orbital tag of the vcm_d3 HF blocks), A_ION (label in the output name), e.g.
A_RUNFMT=v15n_rasK1_d3_{t}e_o14n3 A_TAG=o14n3 A_ION=v. Commands:
  compute <space> [nlow]   -> results/phase17/a_<ion>_<space>.npz (+ .json summary)"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase17 as P17  # noqa: E402  (ion setup, spaces, run names)

Q = P17.Q
K = Q.K
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase17')
METH = ('C2', 'S2-cav', 'S2-diag')
NO_QHQ = ('besth_sall_s4',)       # A3 uses the Step 8 space; the Step 10 space gets the additive variants only


def blocks(space):
    import phase13d as D
    import phase15j as J
    tl = D.term_list(3)
    fmt = os.environ.get('A_RUNFMT')
    names = [fmt.format(t=t) if fmt else K.rname_any(space, t) for L, twoS, m, t in tl]
    hfn = [f'vcm_d3_{t}e_{os.environ.get("A_TAG", K.TAG3)}' for L, twoS, m, t in tl]
    hf = {b['t']: b['U'].T @ b['H'][:b['m'], :b['m']] @ b['U'] for b in K.H.blocks(3, hfn)}
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    bl, acc = [], {}
    for (L, twoS, m, t), name in zip(tl, names):
        d = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
        lab = [str(x) for x in d['labels']]
        from atomci_heff.sparse import SymLower
        dg = SymLower.from_npz(d).d.copy()
        g = (2 * L + 1) * (twoS + 1)
        for i, la in enumerate(lab):
            s_, w = acc.get(la, (0.0, 0.0))
            acc[la] = (s_ + g * dg[i], w + g)
        bl.append(dict(t=t, L=L, twoS=twoS, m=m, name=name, lab=lab, U=np.array(maps[f'vcm_d3_{t}e'])))
        del d, dg
    eav = {k: s_ / w for k, (s_, w) in acc.items()}
    return bl, eav, hf, J


class QOp:
    """QHQ - c applied through H on [0; x] (P = first m rows), no copy of the Q block (as iterative.sylvester)."""
    def __init__(self, op, m, c):
        self.op, self.m, self.c = op, m, c
        self.n = op.n - m
        self.d = op.d[m:] - c

    def matvec(self, X):
        X = X.reshape(self.n, -1)
        Z = np.zeros((self.op.n, X.shape[1]))
        Z[self.m:] = X
        return self.op.matvec(Z)[self.m:] - self.c * X


def compute(space, nlow=20):
    from atomci_heff.sparse import SymLower, davidson
    t0 = time.time()
    if space in NO_QHQ:
        nlow = 0
    bl, eav, hf, J = blocks(space)
    p0 = bl[0]['lab'][0]
    cfgs = sorted(set(eav) - {p0})
    ci = {c: k for k, c in enumerate(cfgs)}
    out = {'cfgs': np.array(cfgs), 'eav': np.array([eav[c] for c in cfgs]), 'eav_P': eav[p0]}
    ion = os.environ.get('A_ION', Q.ION)
    summ = dict(space=space, ion=ion, blocks={})
    for b in bl:
        m, t, U = b['m'], b['t'], b['U']
        d = np.load(os.path.join(ROOT, 'results', b['name'] + '.npz'), allow_pickle=True)
        op = SymLower.from_npz(d)
        del d
        lab = b['lab']
        assert all(la == p0 for la in lab[:m]) and all(la != p0 for la in lab[m:]), b['name']
        n = op.n
        P, Qi = np.arange(m), np.arange(m, n)
        Hpp = op.sub(P).dense()
        V = op.block(P, Qi).toarray()
        dq = op.d[m:]
        cq = np.array([ci[la] for la in lab[m:]])
        p, A = np.linalg.eigh(Hpp)
        W = A.T @ V                                      # (m, nq): a_j^T V
        den = {'C2': np.broadcast_to(eav[p0] - out['eav'][cq], (m, len(cq))),
               'S2-cav': p[:, None] - out['eav'][cq][None, :],
               'S2-diag': p[:, None] - dq[None, :]}
        for meth, Dn in den.items():
            Y = (W / Dn)                                 # (m, nq) = Y^T
            # per configuration: sum over its CSFs of (V_q Y_q A^T + h.c.)/2
            G = np.zeros((len(cfgs), m, m))
            contrib = V[:, :, None] * Y.T[None, :, :]     # (m, nq, m): V_iq Y_qj
            np.add.at(G, cq, np.transpose(contrib, (1, 0, 2)))
            G = G @ A.T
            G = 0.5 * (G + np.transpose(G, (0, 2, 1)))
            out[f'{meth}|{t}'] = U.T @ G @ U
        out[f'php|{t}'] = U.T @ Hpp @ U
        out[f'hf|{t}'] = hf[t]
        out[f'p|{t}'] = p
        out[f'min_gap_diag|{t}'] = np.array([np.min(np.abs(p[j] - dq)) for j in range(m)])
        # lowest QHQ eigenpairs (A3 needs them only in the Step 8 space and k1fh)
        if nlow == 0:
            summ['blocks'][t] = dict(n=int(n), m=int(m), p=p.tolist(), min_gap_diag=out[f'min_gap_diag|{t}'].tolist())
            print(f'{space} {t} n {n} (no QHQ states) {time.time() - t0:.0f} s', flush=True)
            del op, V, W
            continue
        c = float(np.mean(p))
        sh = QOp(op, m, c)
        k = min(nlow, sh.n - 1)
        E, C, rn, it, nmv = davidson(sh, k, tol=1e-7, maxit=3000, max_sub=max(6 * k, 60))
        eps = E + c
        Vu = V @ C                                       # (m, k)
        w = A.T @ Vu                                     # (m, k)
        dH = np.zeros((k, m, m))
        for kk in range(k):
            g = np.outer(Vu[:, kk], (w[:, kk] / (p - eps[kk])) @ A.T)
            g = 0.5 * (g + g.T)
            dH[kk] = U.T @ g @ U
        # dominant configurations of each QHQ eigenvector
        wc = np.zeros((len(cfgs), k))
        np.add.at(wc, cq, C ** 2)
        top = [[(cfgs[i], float(wc[i, kk])) for i in np.argsort(-wc[:, kk])[:3]] for kk in range(k)]
        out[f'qhq_eps|{t}'], out[f'qhq_dH|{t}'], out[f'qhq_res|{t}'] = eps, dH, rn
        summ['blocks'][t] = dict(n=int(n), m=int(m), p=p.tolist(), eps=eps.tolist(), res_max=float(rn.max()),
                                 top=top, gap_min_eV=float((eps[0] - p.max()) * 27.211386),
                                 min_gap_diag=out[f'min_gap_diag|{t}'].tolist())
        print(f'{space} {t} n {n} QHQ eps0 - p_max {(eps[0] - p.max()):+.4f} Eh, res {rn.max():.1e}, '
              f'{time.time() - t0:.0f} s', flush=True)
        del op, sh, V, W
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'a_{ion}_{space}.npz'), **out)
    json.dump(summ, open(os.path.join(OUT, f'a_{ion}_{space}.json'), 'w'), indent=1)
    import resource
    print('done', space, f'{time.time() - t0:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        compute(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)


def load(ion, space):
    return np.load(os.path.join(OUT, f'a_{ion}_{space}.npz'), allow_pickle=True)


def params(by):
    """{t: m x m matrix (Eh, canonical basis)} -> decompose3 dict (A..beta cm-1, T1..T4 mEh)."""
    import phase13d as D
    tl = {t: (L, twoS) for L, twoS, m, t in D.term_list(3)}
    return K.decompose3(K.B.vec_from_blocks(3, {tl[t]: v for t, v in by.items()}))


def terms():
    import phase13d as D
    return [t for L, twoS, m, t in D.term_list(3)]


def total(a, meth, mask=None):
    """HF + sum of the per-configuration second-order increments (mask: boolean over a['cfgs'])."""
    by = {}
    for t in terms():
        G = a[f'{meth}|{t}']
        by[t] = a[f'hf|{t}'] + (G if mask is None else G[mask]).sum(axis=0)
    return params(by)


def incr(a, meth, mask):
    """Increment of the configurations in mask alone (no HF): linear part of the decomposition."""
    return params({t: a[f'{meth}|{t}'][mask].sum(axis=0) for t in terms()})


CLASSES = ['3s→3d (3s 3d^(N+1))', '3p²→3d² (3p⁴ 3d^(N+2))', 'other n = 3', 'inner hole (2s, 2p)', '3p→f (3p hole + f)',
           'virtual s', 'virtual p', 'virtual d', 'virtual f (no 3p hole)', 'virtual g', 'virtual h', 'rest (mixed l)']


def classify(label, N=3):
    import atomci_heff.approx as AP
    o = AP.parse_label(label)
    inner = (2 - o.get('1s', 0)) + (2 - o.get('2s', 0)) + (6 - o.get('2p', 0))
    hs, hp = 2 - o.get('3s', 0), 6 - o.get('3p', 0)
    vir = {k: q for k, q in o.items() if int(k[:-1]) >= 11}
    if inner:
        return 'inner hole (2s, 2p)'
    if not vir:
        if (hs, hp) == (1, 0):
            return '3s→3d (3s 3d^(N+1))'
        if (hs, hp) == (0, 2):
            return '3p²→3d² (3p⁴ 3d^(N+2))'
        return 'other n = 3'
    ls = {k[-1] for k in vir}
    if 'f' in ls and hp >= 1:
        return '3p→f (3p hole + f)'
    if len(ls) == 1:
        lv = ls.pop()
        return 'virtual f (no 3p hole)' if lv == 'f' else f'virtual {lv}'
    return 'rest (mixed l)'


def single3d(label, N=3):
    """3d -> v single: 3s2 3p6 3d^(N-1) v."""
    import atomci_heff.approx as AP
    o = AP.parse_label(label)
    vir = [k for k in o if int(k[:-1]) >= 11]
    return o.get('3s', 0) == 2 and o.get('3p', 0) == 6 and o.get('3d', 0) == N - 1 and sum(o[k] for k in vir) == 1


def class_table(a, q=('T1', 'T2')):
    cl = np.array([classify(str(c)) for c in a['cfgs']])
    rows = {}
    for c in CLASSES:
        mask = cl == c
        if mask.any():
            rows[c] = {m: incr(a, m, mask) for m in METH} | {'n': int(mask.sum())}
    s1 = np.array([single3d(str(c)) for c in a['cfgs']])
    rows['  of which 3d→v single'] = {m: incr(a, m, s1) for m in METH} | {'n': int(s1.sum())}
    return rows
