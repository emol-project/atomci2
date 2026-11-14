#!/usr/bin/env python3
"""Phase 13h: is the term-dependent four-body source the third-order connected term V T V?

Model per path (13f/13g model Hamiltonians, tape labels): H = H0 + H_PQ + sum_c lambda_c T_c, where
  H0      = E_av(P) on P, eps_c (exact configuration average) on each Q configuration, gaps frozen at N = 3
            (mu = 0, rho = 0) for the baseline;
  T_c     = term-dependent part of integral class c, applied on P and Q alike (within-configuration positions):
            C1  R^k(3d3d;3d3d), k > 0
            C2  direct R^k(aa;3d3d), k > 0, a = nd / ng            (core: none, F^k(3s,3d) has k = 0 only)
            C3  exchange R^k(a3d;a3d), all k (core: G^2(3s,3d))
            C6  same-shell virtual R^k(aa;aa), k > 0 (g: 11g^2; 3d path: 11d^2, 12d^2)
            C2v/C3v  direct / exchange between two virtual shells (3d path doubles)
            C4  all Q-Q elements between different Q configurations (positional)
          each minus its exact configuration average (analytic average coefficients); H_PQ fixed.
step1  slopes d r4 / d lambda_c at 0 (lambda 0.05, 0.1, 0.2, -0.1), additivity, lambda = 1 one class on.
step2  third order, des Cloizeaux (Hermitian) form with configuration-average denominators Delta_i (rho = 0):
         H3 = sum_ij H_Pi T_ij H_jP / (Delta_i Delta_j)  -  1/2 {sum_i H_Pi H_iP / Delta_i^2, T_PP}
       lifted (V0..V3 from d0..d3), four-body part r4(H3), per class; compared with the step-1 slopes.
step3  Q-insertion and folded parts of r4(H3) per class.
step4  prediction at lambda = 1: r4(base) + r4(H3, all classes) + r4(rho closure, 13g) vs exact.
Metrics: r4/|V3 t| and r4/|V3| (|V3| = full lifted three-body V3 in d^4).
-> results/phase13/phase13h.txt/.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff import eetape  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_csfs  # noqa: E402
import phase13b as B  # noqa: E402
import phase13c as C13  # noqa: E402
import phase13d as D  # noqa: E402
import phase13e as E  # noqa: E402
import phase13f as F  # noqa: E402
import phase13g as G  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
LET = 'spdfgh'
TK = B.TK
CLASSES = ['C1', 'C2', 'C3', 'C6', 'C2v', 'C3v', 'C4']


# ---------------------------------------------------------------- tapes

def tape_W_multi(run, keeps):
    """{name: {label: matrix}} for several CSF subsets of one tape, read once."""
    tape = os.path.join(ROOT, 'runs', run, 'fort.15')
    hd, cs = read_csfs(tape)
    names = hd['orbitals']
    tab = I.Addressing([LET.index(o[-1]) + 1 for o in names]).label_table()
    ntot = len(cs)
    allk = sorted(set().union(*[set(k) for k in keeps.values()]))
    pos = {nm: -np.ones(ntot, int) for nm in keeps}
    for nm, k in keeps.items():
        pos[nm][list(k)] = np.arange(len(k))
    W = {nm: defaultdict(lambda n=len(k): np.zeros((n, n))) for nm, k in keeps.items()}
    if len(allk) < ntot:
        import phase11c
        ent = ((i, j, wt, adr) for i, j, wt, adr in phase11c.read_pairs(tape, np.array(allk)))
    else:
        ent = ((i, j, wt, adr) for i, j, wt, adr, _ in eetape.read_ee(tape))
    for i, j, wt, adr in ent:
        labs = None
        for nm in keeps:
            pi, pj = pos[nm][i], pos[nm][j]
            if pi < 0 or pj < 0:
                continue
            if labs is None:
                labs = [D.canon(tab[int(a)], names) for a in adr]
            Wn = W[nm]
            for x, l in zip(wt, labs):
                Wn[l][pi, pj] += x
                if pi != pj:
                    Wn[l][pj, pi] += x
    return {nm: dict(w) for nm, w in W.items()}


# ---------------------------------------------------------------- classes

def label_class(nm, V):
    """(class, a, b, k, kind) for a valence two-electron label inside a configuration, or None."""
    if not nm.startswith('R'):
        return None
    k = int(nm[1:nm.index('(')])
    o = G.orbs_of(nm)
    if not all(x in V for x in o):
        return None
    p1, p2 = (o[0], o[1]), (o[2], o[3])
    if p1[0] == p1[1] and p2[0] == p2[1]:          # direct R^k(aa;bb)
        a, b = p1[0], p2[0]
        if k == 0:
            return None
        if a == b:
            return ('C1' if a == '3d' else 'C6'), a, b, k, 'same'
        return ('C2' if '3d' in (a, b) else 'C2v'), a, b, k, 'direct'
    if p1 == p2 and p1[0] != p1[1]:                 # exchange R^k(ab;ab)
        a, b = p1
        return ('C3' if '3d' in (a, b) else 'C3v'), a, b, k, 'exch'
    return ('X', None, None, k, 'other')


def avg_coef(cl, oc):
    """Configuration-average coefficient of a classified label for occupation oc."""
    c, a, b, k, kind = cl
    if kind == 'same':
        n, l = oc.get(a, 0), LET.index(a[-1])
        return -comb(n, 2) * (2 * l + 1) / (4 * l + 1) * G.w3j2(l, k, l)
    if kind == 'direct':
        return 0.0
    if kind == 'exch':
        return -oc.get(a, 0) * oc.get(b, 0) * 0.5 * G.w3j2(LET.index(a[-1]), k, LET.index(b[-1]))
    return 0.0


def decompose(blocks, V, lf):
    """Per block: T[c] (n x n, P and Q within-configuration parts, minus averages; C4 positional), eps per CSF.
    Checks: within-configuration constancy of H - sum T and E_av(P) consistency."""
    spread = defaultdict(list)
    for b in blocks:
        n = len(b['lab'])
        cfg = np.array([('P' if i in set(b['P']) else b['lab'][i]) for i in range(n)])
        same = cfg[:, None] == cfg[None, :]
        isQ = np.zeros(n, bool)
        isQ[b['Q']] = True
        T = {c: np.zeros((n, n)) for c in CLASSES}
        for nm, Wm in b['W'].items():
            cl = label_class(nm, V)
            if cl is None or cl[0] == 'X':
                continue
            x, _ = G.lval(nm, lf)
            M = x * np.where(same, Wm, 0.0)
            for i in range(n):
                oc = G.occ(b['lab'][i])
                M[i, i] -= x * avg_coef(cl, oc)
            T[cl[0]] += M
        T['C4'] = np.where(~same & isQ[:, None] & isQ[None, :], b['H'], 0.0)
        tot = sum(T.values())
        rest = b['H'] - tot
        b['T'] = T
        b['_restdiag'] = np.diag(rest).copy()
        b['_cfg'] = cfg
        for i in range(n):
            spread[(b['N'], cfg[i])].append(rest[i, i])
        offw = np.where(same, rest - np.diag(np.diag(rest)), 0.0)
        b['_offw'] = np.abs(offw).max()
    const = {k: float(np.mean(v)) for k, v in spread.items()}
    sp = max(max(v) - min(v) for v in spread.values())
    off = max(b['_offw'] for b in blocks)
    for b in blocks:
        b['eps_full'] = np.array([const[(b['N'], c)] for c in b['_cfg']])
    return const, dict(spread=sp, offw=off)


def ctype(c):
    return ' '.join(t for t in c.split() if not t.startswith('3d')) or 'P'


def base_eps(blocks, const, mu=0.0):
    """eps per CSF with gaps Delta_type(N) = Delta_type(3) + mu (Delta_type(N) - Delta_type(3))."""
    gap = defaultdict(dict)
    for (N, c), x in const.items():
        if c != 'P':
            gap[ctype(c)][N] = x - const[(N, 'P')]
    out = {}
    for b in blocks:
        N = b['N']
        e = []
        for c in b['_cfg']:
            if c == 'P':
                e.append(const[(N, 'P')])
            else:
                g = gap[ctype(c)]
                g3 = g[3] if 3 in g else g[min(g, key=lambda n: abs(n - 3))]   # nearest N if the type has no N = 3
                e.append(const[(N, 'P')] + g3 + mu * (g[N] - g3))
        out[id(b)] = np.array(e)
    return out, gap


def model_H(b, eps, lam):
    """H0 (eps diagonal) + H_PQ + sum_c lam[c] T_c."""
    n = len(b['lab'])
    H = np.diag(eps[id(b)])
    P, Q = b['P'], b['Q']
    H[np.ix_(P, Q)] = b['H'][np.ix_(P, Q)]
    H[np.ix_(Q, P)] = b['H'][np.ix_(Q, P)]
    for c, l in lam.items():
        if l:
            H = H + l * b['T'][c]
    return H


# ---------------------------------------------------------------- pipelines

def run(blocks, d0, Hfun):
    vecs = {} if d0 is None else {0: np.array([d0])}
    by = defaultdict(dict)
    info = {}
    for b in blocks:
        if b['N'] > 4 or (b['N'] == 0 and d0 is not None):
            continue
        Bk, inf = F.heff_block(b, Hfun(b))
        by[b['N']][(b['L'], b['twoS'])] = Bk
        info[(b['N'], b['t'])] = inf
    for N, bl in by.items():
        vecs[N] = B.vec_from_blocks(N, bl)
    m = C13.metrics(vecs)
    wP = min(x['wP'] for (N, t), x in info.items() if N == 4)
    below = sum(x['below'] for (N, t), x in info.items() if N == 4)
    return dict(r=m['r'], V3=m['V3'], V3t=m['V3t'], wP=wP, below=below, ratio=m['ratio_t'])


def r4_of(blocks, d0, mats):
    """Four-body residual vector of a linear 'H_eff contribution' given as P x P matrices per block (CSF basis)."""
    vecs = {} if d0 is None else {0: np.array([0.0])}
    by = defaultdict(dict)
    for b in blocks:
        if b['N'] > 4 or (b['N'] == 0 and d0 is not None):
            continue
        by[b['N']][(b['L'], b['twoS'])] = b['U'].T @ mats[id(b)] @ b['U']
    for N, bl in by.items():
        vecs[N] = B.vec_from_blocks(N, bl)
    V = B.extract(vecs)
    return vecs[4] - B.predict(V, 4)


def norms(x):
    _, nr = B.jl_split(4, x)
    return float(np.sqrt(S.ip(4, x, x))), float(np.sqrt(sum(nr[k] ** 2 for k in TK)))


def third_order(blocks, eps, c):
    """(Q-insertion, folded) P x P matrices per block for class c (rho = 0 denominators)."""
    Qi, Fo = {}, {}
    for b in blocks:
        P, Q = b['P'], b['Q']
        if not Q:
            Qi[id(b)] = Fo[id(b)] = np.zeros((len(P), len(P)))
            continue
        e = eps[id(b)]
        EP = e[P[0]]
        dl = e[Q] - EP
        HPQ = b['H'][np.ix_(P, Q)]
        T = b['T'][c]
        A = HPQ / dl
        Qi[id(b)] = A @ T[np.ix_(Q, Q)] @ A.T
        X2 = (HPQ / dl ** 2) @ HPQ.T
        TP = T[np.ix_(P, P)]
        Fo[id(b)] = -0.5 * (X2 @ TP + TP @ X2)
    return Qi, Fo


def closure_r4(blocks, d0, eps):
    mats = {}
    for b in blocks:
        P, Q = b['P'], b['Q']
        if not Q:
            mats[id(b)] = np.zeros((len(P), len(P)))
            continue
        e = eps[id(b)]
        dl = e[Q] - e[P[0]]
        HPQ = b['H'][np.ix_(P, Q)]
        mats[id(b)] = -(HPQ / dl) @ HPQ.T
    return r4_of(blocks, d0, mats)


# ---------------------------------------------------------------- paths

def build():
    t0 = time.time()
    paths = {}
    cb = F.core_blocks()
    for b in cb:
        b['W'] = G.tape_W(f'vcm_d{b["N"]}_{b["t"]}e')
        b['H0'] = b['H']
    paths['core'] = dict(blocks=cb, d0=None, V={'3s', '3d'}, lf=G.fit_labels(cb))
    data = {sp: {(N, t): D.load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)} for sp in D.SPACES}
    lf = E.label_fit(data)
    dgp, _, _ = E.exact_d_gaps(data['s6d'], lf)
    shift = {o: {N: dgp[o][3] - dgp[o][N] for N in dgp[o]} for o in dgp}
    for sp in D.SPACES:
        f, _ = D.split(data[sp])
        for kind, lab, sh in (('b', "b'", shift), ('a', 'a(13d)', None)):
            bl = F.singles_blocks(data[sp], f, kind, sh)
            for b in bl:
                b['W'] = data[sp][(b['N'], b['t'])]['W']
                b['H0'] = data[sp][(b['N'], b['t'])]['H']
            paths[f'{lab} {sp}'] = dict(blocks=bl, d0=B.ECORE, V={'3d'} | set(D.SPACES[sp]))
        lfs = G.fit_labels(paths[f"b' {sp}"]['blocks'])
        paths[f"b' {sp}"]['lf'] = paths[f'a(13d) {sp}']['lf'] = lfs
    print(f'core/singles built {time.time() - t0:.0f} s', flush=True)
    # g (3d + 11g) and the 13c d path "3d" (3d, 11d, 12d), d^1..d^4: QZP tapes read once for both subsets
    gb = [b for b in F.g_blocks() if b['N'] <= 4]
    orbs3d = {'3d', '11d', '12d'}
    db = []
    for N in range(1, 5):
        for L, twoS, m, t in D.term_list(N):
            run_ = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r3d'}[N].format(t)
            db.append(F.blk(N, t, *F.load_subset(run_, orbs3d if N < 4 else None)))
    gruns = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r1d1g'}
    for bg in gb:
        run_ = gruns[bg['N']].format(bg['t'])
        z = np.load(os.path.join(ROOT, 'results', run_ + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        keeps = {'g': [i for i, l in enumerate(lab) if set(B.cfg_orbitals(l)) <= {'3d', '11g'}]}
        if bg['N'] < 4:
            keeps['d'] = [i for i, l in enumerate(lab) if set(B.cfg_orbitals(l)) <= orbs3d]
        Wm = tape_W_multi(run_, keeps)
        bg['W'] = Wm['g']
        bg['H0'] = bg['H']
        if 'd' in Wm:
            bd = next(x for x in db if x['N'] == bg['N'] and x['t'] == bg['t'])
            bd['W'] = Wm['d']
            bd['H0'] = bd['H']
    for bd in db:
        if bd['N'] == 4:
            bd['W'] = G.tape_W(f'vq_v1p_{bd["t"]}e_ivo_r3d')
            bd['H0'] = bd['H']
    paths['g'] = dict(blocks=gb, d0=B.ECORE, V={'3d', '11g'}, lf=G.fit_labels(gb))
    paths['d 3d (13c)'] = dict(blocks=db, d0=B.ECORE, V=orbs3d, lf=G.fit_labels(db))
    print(f'all built {time.time() - t0:.0f} s', flush=True)
    return paths


def main():
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13h: is the term-dependent four-body source the third-order connected term V T V?')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    paths = build()
    a(f'\nBuild {time.time() - t0:.0f} s.')

    # ---------------- decomposition checks
    a('\n## Decomposition check (H = H0(exact eps) + H_PQ + sum_c T_c)')
    a('| path | label fit (Eh) | spread of H - sum T within a configuration (Eh) | max within-configuration off-diagonal '
      'remainder (Eh) | |eps_P - E_av(P)| (Eh) | classes present (max |T_c|, Eh) |')
    a('|---|---|---|---|---|---|')
    for nm, p in paths.items():
        const, chk = decompose(p['blocks'], p['V'], p['lf'])
        p['const'] = const
        avP = F.averages(p['blocks'])
        dP = max(abs(const[(N, 'P')] - avP[(N, 'P')]) for N in {b['N'] for b in p['blocks']})
        pres = {c: max(np.abs(b['T'][c]).max() for b in p['blocks']) for c in CLASSES}
        a(f'| {nm} | {p["lf"]["dev"]:.1e} | {chk["spread"]:.1e} | {chk["offw"]:.1e} | {dP:.1e} | ' +
          ', '.join(f'{c} {x:.3f}' for c, x in pres.items() if x > 1e-12) + ' |')
        p['classes'] = [c for c, x in pres.items() if x > 1e-12]
        p['eps0'], p['gaps'] = base_eps(p['blocks'], const, 0.0)
        p['eps1'], _ = base_eps(p['blocks'], const, 1.0)
    print(f'decomposed {time.time() - t0:.0f} s', flush=True)

    # ---------------- step 1
    a('\n## Step 1. Linear response by class (baseline: all term dependence off, rho = 0)')
    a('Slope s_c = (r4(lambda) - r4(0)) / lambda as a d^4 vector; |s| and its t-part, in units of the baseline |V3| and '
      '|V3 t| (so lambda x |s| is the r4/|V3| and r4/t increment). Linearity: |s(lambda) - s(0.05)| / |s(0.05)|.')
    res = {}
    for nm in ('core', 'g', "b' s3d", "b' s6d"):
        p = paths[nm]
        bl, d0 = p['blocks'], p['d0']
        base = run(bl, d0, lambda b: model_H(b, p['eps0'], {}))
        V3, V3t = base['V3'], base['V3t']
        a(f'\n### {nm}: baseline r4/|V3| = {norms(base["r"])[0] / V3:.4f}, r4/t = {base["ratio"]:.4f}, |V3| = '
          f'{1e3 * V3:.3f} mEh, |V3 t| = {1e3 * V3t:.3f} mEh, d4 min P {base["wP"]:.4f}')
        a('| class | |s|/|V3| | |s_t|/|V3 t| | lin. dev 0.1 / 0.2 / -0.1 | lambda=1 alone: r4/|V3| | r4/t | d4 min P | below |')
        a('|---|---|---|---|---|---|---|---|')
        sl = {}
        for c in p['classes']:
            s = {}
            for l in (0.05, 0.1, 0.2, -0.1):
                r = run(bl, d0, lambda b, l=l: model_H(b, p['eps0'], {c: l}))
                s[l] = (r['r'] - base['r']) / l
            r1 = run(bl, d0, lambda b: model_H(b, p['eps0'], {c: 1.0}))
            n0 = norms(s[0.05])
            dev = [np.sqrt(S.ip(4, s[l] - s[0.05], s[l] - s[0.05])) / max(n0[0], 1e-300) for l in (0.1, 0.2, -0.1)]
            sl[c] = dict(s=s[0.05], n=n0, dev=dev, r1=(norms(r1['r'])[0] / r1['V3'], r1['ratio'], r1['wP'], r1['below']))
            a(f'| {c} | {n0[0] / V3:.4f} | {n0[1] / V3t:.4f} | ' + ' / '.join(f'{x:.3f}' for x in dev) +
              f' | {sl[c]["r1"][0]:.4f} | {sl[c]["r1"][1]:.4f} | {r1["wP"]:.4f} | {r1["below"]} |')
        # additivity
        lam_all = {c: 0.05 for c in p['classes']}
        ra = run(bl, d0, lambda b: model_H(b, p['eps0'], lam_all))
        s_all = (ra['r'] - base['r']) / 0.05
        s_sum = sum(x['s'] for x in sl.values())
        dd = s_all - s_sum
        a(f'Additivity: |s_all| / |V3| = {norms(s_all)[0] / V3:.4f}, |sum s_c| / |V3| = {norms(s_sum)[0] / V3:.4f}, '
          f'|s_all - sum s_c| / |s_all| = {np.sqrt(S.ip(4, dd, dd)) / norms(s_all)[0]:.3f}')
        r1all = run(bl, d0, lambda b: model_H(b, p['eps0'], {c: 1.0 for c in p['classes']}))
        a(f'All classes at lambda = 1 (rho = 0): r4/|V3| = {norms(r1all["r"])[0] / r1all["V3"]:.4f}, '
          f'r4/t = {r1all["ratio"]:.4f}; linear extrapolation |r4(0) + sum s_c| / |V3| = '
          f'{norms(base["r"] + s_sum)[0] / V3:.4f}')
        res[nm] = dict(base=base, sl=sl, s_all=s_all, r1all=r1all)
        print(f'step1 {nm} {time.time() - t0:.0f} s', flush=True)

    # ---------------- step 2 / 3
    a('\n## Step 2/3. Third-order formula (des Cloizeaux / Hermitian, rho = 0 denominators) vs step-1 slopes')
    a('| path | class | |r4(H3)|/|V3| | t-part / |V3 t| | |slope|/|V3| | |r4(H3) - slope| / |slope| | angle (deg) | '
      '|Q-ins|/|V3| | |folded|/|V3| | |Q-ins + folded| / (|Q-ins| + |folded|) |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    s23 = {}
    for nm in ('core', 'g', "b' s3d", "b' s6d"):
        p = paths[nm]
        bl, d0 = p['blocks'], p['d0']
        V3, V3t = res[nm]['base']['V3'], res[nm]['base']['V3t']
        tot = 0.0
        for c in p['classes']:
            Qi, Fo = third_order(bl, p['eps0'], c)
            rq, rf = r4_of(bl, d0, Qi), r4_of(bl, d0, Fo)
            r3 = rq + rf
            tot = tot + r3
            s = res[nm]['sl'][c]['s']
            n3, ns = norms(r3), norms(s)
            dd = r3 - s
            cos = S.ip(4, r3, s) / max(n3[0] * ns[0], 1e-300)
            nq, nf = norms(rq)[0], norms(rf)[0]
            s23[f'{nm} {c}'] = dict(n3=n3, ns=ns, dev=np.sqrt(S.ip(4, dd, dd)) / max(ns[0], 1e-300), nq=nq, nf=nf)
            a(f'| {nm} | {c} | {n3[0] / V3:.4f} | {n3[1] / V3t:.4f} | {ns[0] / V3:.4f} | '
              f'{np.sqrt(S.ip(4, dd, dd)) / max(ns[0], 1e-300):.3f} | {np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} | '
              f'{nq / V3:.4f} | {nf / V3:.4f} | {n3[0] / max(nq + nf, 1e-300):.3f} |')
        res[nm]['r3tot'] = tot
    js['step23'] = s23

    # ---------------- step 4
    a('\n## Step 4. Prediction at lambda = 1: r4(base, rho = 0) + r4(H3, all classes) + r4(rho closure) vs exact')
    a('| path | exact r4/|V3| | pred r4/|V3| | |exact - pred|/|V3| | exact r4/t | pred r4/t | parts /|V3|: base, H3, rho | '
      'exact d4 min P |')
    a('|---|---|---|---|---|---|---|---|')
    s4 = {}
    for nm in ('core', 'g', "b' s3d", "b' s6d", 'a(13d) s3d', 'a(13d) s6d', 'd 3d (13c)'):
        p = paths[nm]
        bl, d0 = p['blocks'], p['d0']
        if nm not in res:
            base = run(bl, d0, lambda b: model_H(b, p['eps0'], {}))
            tot = 0.0
            for c in p['classes']:
                Qi, Fo = third_order(bl, p['eps0'], c)
                tot = tot + r4_of(bl, d0, Qi) + r4_of(bl, d0, Fo)
        else:
            base, tot = res[nm]['base'], res[nm]['r3tot']
        rr = closure_r4(bl, d0, p['eps1'])
        ex = run(bl, d0, lambda b: model_H(b, p['eps1'], {c: 1.0 for c in p['classes']}))
        chk = max(np.abs(model_H(b, p['eps1'], {c: 1.0 for c in p['classes']}) - b['H']).max() for b in bl)
        pred = base['r'] + tot + rr
        V3, V3t = ex['V3'], ex['V3t']
        dd = ex['r'] - pred
        ne, npd = norms(ex['r']), norms(pred)
        s4[nm] = dict(ex=ne[0] / V3, pred=npd[0] / V3, err=np.sqrt(S.ip(4, dd, dd)) / V3, ex_t=ex['ratio'],
                      pred_t=npd[0] / V3t, parts=[norms(base['r'])[0] / V3, norms(tot)[0] / V3, norms(rr)[0] / V3],
                      wP=ex['wP'], chk=chk)
        a(f'| {nm} | {ne[0] / V3:.4f} | {npd[0] / V3:.4f} | {s4[nm]["err"]:.4f} | {ex["ratio"]:.4f} | '
          f'{npd[0] / V3t:.4f} | ' + ' / '.join(f'{x:.4f}' for x in s4[nm]['parts']) + f' | {ex["wP"]:.4f} |')
        print(f'step4 {nm} {time.time() - t0:.0f} s', flush=True)
    a('Check: the lambda = 1 model with actual gaps equals the path Hamiltonian to ' +
      f'{max(x["chk"] for x in s4.values()):.1e} Eh.')
    js['step4'] = s4
    js['step1'] = {nm: {c: dict(n=x['n'], dev=x['dev'], r1=x['r1']) for c, x in r['sl'].items()} for nm, r in res.items()}
    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13h.json'), 'w'), indent=1,
              default=lambda x: float(x) if np.isscalar(x) else (x.tolist() if hasattr(x, 'tolist') else str(x)))
    open(os.path.join(OUT, 'phase13h.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
