#!/usr/bin/env python3
"""Phase 13g: test "r4/|V3 t| = linked denominator spread + rho" on the 13f model Hamiltonians (no ATOMCI runs).

step1  kappa = sigma_D / Dbar of the P-Q denominators E_beta - E_alpha (eigenstates of H_QQ and H_PP per (L,S)
       block, weights (2L+1)(2S+1) |<beta|H_QP|alpha>|^2), kappa2 = weighted std of Dbar/D, and kappa_in (the same,
       with Dbar taken per dominant Q configuration of beta). Computed for every 13f scan point (csf scheme:
       (i), (ii), (i+ii), (iv), (vi)); fit r4/t = a kappa + b rho.
step2  H_QQ = eps_c I + T_spec + T_link from the tape labels:
         VV     = two-electron labels with all four orbitals in the valence set (b': 3d, nd; core: 3s, 3d);
         spec   = VV labels R^k(3d3d;3d3d) (interaction among the 3d electrons: the parent multiplet);
         T_spec = spec - C(n_3d, 2) F_av(3d,3d);
         T_link = (VV - spec) - E_av(other valence pairs) + off-diagonal remainder (one-body / core, between
                  configurations);
         eps_c  = mean over the CSFs of c of (diag H_QQ - diag VV) + E_av(VV) (exact configuration average; the first
                  term must be constant within c: checked).
       Scans (vii) lambda x T_link, (viii) lambda x T_spec.
step3  all term dependence off (13f (vi), lambda = 0) and the gap slope scaled: Delta_i(N) = Delta_i(3) + mu delta_i
       (N - 3); closed-form leading term r4 = sum_k Delta^(4-k) g(k) X_k (finite differences of 1/Delta(N)).
step4  g path: exact configuration averages from the tapes (QZP d1..d3, r1d1g d4, r3d1g d5) and 13f (vi) redone.
-> results/phase13/phase13g.txt/.json
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

OUT = os.path.join(ROOT, 'results', 'phase13')
HART_EV = D.HART_EV
LAM = [0.0, 0.25, 0.5, 1.0, 2.0]
LET = 'spdfgh'


# ---------------------------------------------------------------- tapes and labels

def tape_W(run, keep=None):
    """{label: n x n coefficient matrix} of runs/<run>/fort.15, restricted to the CSF indices keep (in that order)."""
    tape = os.path.join(ROOT, 'runs', run, 'fort.15')
    hd, cs = read_csfs(tape)
    names = hd['orbitals']
    tab = I.Addressing([LET.index(o[-1]) + 1 for o in names]).label_table()
    ntot = len(cs)
    keep = list(range(ntot)) if keep is None else list(keep)
    pos = -np.ones(ntot, int)
    pos[keep] = np.arange(len(keep))
    n = len(keep)
    W = defaultdict(lambda: np.zeros((n, n)))
    if len(keep) < ntot:
        import phase11c
        ent = ((i, j, wt, adr, 0) for i, j, wt, adr in phase11c.read_pairs(tape, np.array(keep)))
    else:
        ent = eetape.read_ee(tape)
    for i, j, wt, adr, n1 in ent:
        pi, pj = pos[i], pos[j]
        if pi < 0 or pj < 0:
            continue
        for x, a in zip(wt, adr):
            nm = D.canon(tab[int(a)], names)
            W[nm][pi, pj] += x
            if pi != pj:
                W[nm][pj, pi] += x
    return dict(W)


def fit_labels(blocks):
    labs = sorted({nm for b in blocks for nm in b['W']})
    col = {nm: k for k, nm in enumerate(labs)}
    rows, rhs = [], []
    for b in blocks:
        iu = np.tril_indices(len(b['lab']))
        R = np.zeros((len(iu[0]), len(labs)))
        for nm, Wm in b['W'].items():
            R[:, col[nm]] = Wm[iu]
        rows.append(R)
        rhs.append(b['H0'][iu])
    R, y = np.vstack(rows), np.concatenate(rhs)
    g, *_ = np.linalg.lstsq(R, y, rcond=None)
    _, sv, vt = np.linalg.svd(R, full_matrices=False)
    null = vt[sv < sv[0] * 1e-10]
    dev = max(np.abs(sum(g[col[nm]] * Wm for nm, Wm in b['W'].items()) - b['H0']).max() for b in blocks)
    return dict(col=col, g=g, null=null, dev=dev)


def orbs_of(nm):
    return re.findall(r'\d+[spdfgh]', nm[nm.index('(') + 1:-1])


def lval(name, lf):
    """Fitted value of a canonical label and its null-space component (identifiability)."""
    if name not in lf['col']:
        return 0.0, 0.0
    k = lf['col'][name]
    return float(lf['g'][k]), float(np.abs(lf['null'][:, k]).max()) if len(lf['null']) else 0.0


def w3j2(a, k, b):
    from sympy.physics.wigner import wigner_3j
    return float(wigner_3j(a, k, b, 0, 0, 0) ** 2)


def rname(k, a, b, c, d):
    p1, p2 = tuple(sorted((a, b))), tuple(sorted((c, d)))
    p1, p2 = sorted((p1, p2))
    return f'R{k}({p1[0]}{p1[1]};{p2[0]}{p2[1]})'


def fav(a, b, lf, nul):
    """Average pair energy F_av(a,b) from label values (equivalent: a == b)."""
    la, lb = LET.index(a[-1]), LET.index(b[-1])
    if a == b:
        v, n0 = lval(rname(0, a, a, a, a), lf)
        nul.append(n0)
        for k in range(2, 2 * la + 1, 2):
            x, n0 = lval(rname(k, a, a, a, a), lf)
            nul.append(n0)
            v -= (2 * la + 1) / (4 * la + 1) * w3j2(la, k, la) * x
        return v
    v, n0 = lval(rname(0, a, a, b, b), lf)
    nul.append(n0)
    for k in range(abs(la - lb), la + lb + 1, 2):
        x, n0 = lval(rname(k, a, b, a, b), lf)
        nul.append(n0)
        v -= 0.5 * w3j2(la, k, lb) * x
    return v


def occ(label):
    return {o: int(q) for o, q in re.findall(r'(\d+[spdfgh])(\d+)', label)}


def eav_vv(oc, V, lf, nul, spec_only=False):
    """E_av of the valence-valence two-electron energy of occupation oc over shells V (spec_only: 3d-3d pairs)."""
    sh = [o for o in V if oc.get(o, 0)]
    e = 0.0
    for i, a in enumerate(sh):
        e += comb(oc[a], 2) * fav(a, a, lf, nul) if (not spec_only or a == '3d') else 0.0
        if spec_only:
            continue
        for b2 in sh[i + 1:]:
            e += oc[a] * oc[b2] * fav(a, b2, lf, nul)
    return e


def split_Q(blocks, V, lf):
    """Per block: eps (per Q CSF), T_spec, T_link on the Q x Q block; returns diagnostics."""
    nul = []
    spread = defaultdict(list)
    for b in blocks:
        Q = b['Q']
        if not Q:
            continue
        VV = np.zeros((len(Q), len(Q)))
        SP = np.zeros((len(Q), len(Q)))
        for nm, Wm in b['W'].items():
            if not nm.startswith('R'):
                continue
            o = orbs_of(nm)
            if all(x in V for x in o):
                x, n0 = lval(nm, lf)
                blk = x * Wm[np.ix_(Q, Q)]
                VV += blk
                if all(y == '3d' for y in o):
                    SP += blk
        HQ = b['H'][np.ix_(Q, Q)]
        rest = HQ - VV
        cfg = [b['lab'][i] for i in Q]
        eav_s = np.array([eav_vv(occ(c), V, lf, nul, spec_only=True) for c in cfg])
        eav_a = np.array([eav_vv(occ(c), V, lf, nul) for c in cfg])
        b['_rest_diag'] = np.diag(rest).copy()
        b['_cfg'] = cfg
        b['_eavs'], b['_eava'] = eav_s, eav_a
        b['_VV'], b['_SP'], b['_rest'] = VV, SP, rest
        for c, x in zip(cfg, np.diag(rest)):
            spread[(b['N'], c)].append(x)
    const = {k: float(np.mean(v)) for k, v in spread.items()}
    sp = max(max(v) - min(v) for v in spread.values())
    err = 0.0
    for b in blocks:
        Q = b['Q']
        if not Q:
            continue
        cfg = b['_cfg']
        D0 = np.array([const[(b['N'], c)] for c in cfg])
        b['eps'] = D0 + b['_eava']
        b['Tspec'] = b['_SP'] - np.diag(b['_eavs'])
        off = b['_rest'] - np.diag(np.diag(b['_rest']))
        b['Tlink'] = (b['_VV'] - b['_SP']) - np.diag(b['_eava'] - b['_eavs']) + off
        rec = np.diag(b['eps']) + b['Tspec'] + b['Tlink']
        err = max(err, np.abs(rec - b['H'][np.ix_(Q, Q)]).max())
    return dict(spread=sp, rec_err=err, null=max(nul) if nul else 0.0)


# ---------------------------------------------------------------- pipeline with an arbitrary H

def pipe(blocks, d0, Hfun, Nmax=5):
    vecs = {} if d0 is None else {0: np.array([d0])}
    info, by = {}, defaultdict(dict)
    for b in blocks:
        if b['N'] > Nmax or (b['N'] == 0 and d0 is not None):
            continue
        Bk, inf = F.heff_block(b, Hfun(b))
        by[b['N']][(b['L'], b['twoS'])] = Bk
        info[(b['N'], b['t'])] = inf
    for N, bl in by.items():
        vecs[N] = B.vec_from_blocks(N, bl)
    m = C13.metrics(vecs)
    tg = {}
    for N in range(2, Nmax + 1):
        it = [x for (NN, t), x in info.items() if NN == N]
        tg[N] = dict(wP=min(x['wP'] for x in it), amb=max(x['amb'] for x in it), below=sum(x['below'] for x in it),
                     gap=min(x['gap'] for x in it))
    flag = {N: tg[N]['wP'] < 0.9 or tg[N]['amb'] > 0.3 or tg[N]['below'] > 0 for N in tg}
    return dict(ratio=m['ratio_t'], r4=m['r4'], V3t=m['V3t'], V3=m['V3'], h4=m['h4'], tg=tg, flag=flag, V=m['V'],
                vecs=vecs)


def kappas(blocks, Hfun, N):
    """kappa, kappa2, kappa_in at N."""
    Ds, ws, cf = [], [], []
    for b in blocks:
        if b['N'] != N or not b['Q']:
            continue
        H = Hfun(b)
        P, Q = b['P'], b['Q']
        eP, UP = np.linalg.eigh(H[np.ix_(P, P)])
        eQ, UQ = np.linalg.eigh(H[np.ix_(Q, Q)])
        M = UQ.T @ H[np.ix_(Q, P)] @ UP
        cfg = np.array([b['lab'][i] for i in Q])
        dom = [cfg[np.argmax(UQ[:, k] ** 2)] for k in range(len(Q))]
        for k in range(len(Q)):
            for j in range(len(P)):
                Ds.append(eQ[k] - eP[j])
                ws.append(b['g'] * M[k, j] ** 2)
                cf.append(dom[k])
    D_, w = np.array(Ds), np.array(ws)
    w = w / w.sum()
    Db = float(w @ D_)
    sig = float(np.sqrt(w @ (D_ - Db) ** 2))
    k2 = float(np.sqrt(w @ (Db / D_ - w @ (Db / D_)) ** 2))
    cf = np.array(cf)
    var_in = 0.0
    for c in set(cf):
        m = cf == c
        wc = w[m]
        if wc.sum() == 0:
            continue
        dc = float(wc @ D_[m] / wc.sum())
        var_in += float(wc @ ((D_[m] - dc) / dc) ** 2)
    return dict(kappa=sig / Db, kappa2=k2, kappa_in=float(np.sqrt(var_in)), Dbar=Db)


def fmod(b, av, **kw):
    return F.modify(b, av, **kw)


def qmod(b, lamS=1.0, lamL=1.0):
    H = b['H'].copy()
    Q = b['Q']
    if Q:
        H[np.ix_(Q, Q)] = np.diag(b['eps']) + lamS * b['Tspec'] + lamL * b['Tlink']
    return H


# ---------------------------------------------------------------- paths

def build():
    paths = {}
    cb = F.core_blocks()
    for b in cb:
        b['W'] = tape_W(f'vcm_d{b["N"]}_{b["t"]}e')
        b['H0'] = b['H']
    lfc = fit_labels(cb)
    cgx, _ = E.exact_core_gaps(json.load(open(os.path.join(OUT, 'phase13b_b.json')))['params'])
    paths['core'] = dict(blocks=cb, d0=None, lf=lfc, V={'3s', '3d'}, rho=0.0275,
                         gap=lambda N, c: cgx[N] if c.startswith('3s1') else None, delta={'3s1': cgx[4] - cgx[3]},
                         D3={'3s1': cgx[3]})
    data = {sp: {(N, t): D.load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)} for sp in D.SPACES}
    lf = E.label_fit(data)
    dgp, _, _ = E.exact_d_gaps(data['s6d'], lf)
    shift = {o: {N: dgp[o][3] - dgp[o][N] for N in dgp[o]} for o in dgp}
    orb = lambda c: c.split()[0][:-1]
    for sp in D.SPACES:
        f, _ = D.split(data[sp])
        for kind, lab in (('b', "b'"), ('a', "a'")):
            bl = F.singles_blocks(data[sp], f, kind, shift)
            for b in bl:
                b['W'] = data[sp][(b['N'], b['t'])]['W']
                b['H0'] = data[sp][(b['N'], b['t'])]['H']
            paths[f'{lab} {sp}'] = dict(blocks=bl, d0=B.ECORE, V={'3d'} | set(D.SPACES[sp]), rho=0.0,
                                        gap=lambda N, c: dgp[orb(c)][3],
                                        delta={o: dgp[o][4] - dgp[o][3] for o in D.SPACES[sp]},
                                        D3={o: dgp[o][3] for o in D.SPACES[sp]})
        lfs = fit_labels(paths[f"b' {sp}"]['blocks'])
        paths[f"b' {sp}"]['lf'] = paths[f"a' {sp}"]['lf'] = lfs
    paths['g'] = dict(blocks=F.g_blocks(), d0=B.ECORE, gap='linear', rho=0.0007)
    for p in paths.values():
        p['av'] = F.averages(p['blocks'], p['gap'])
    return paths


def g_exact(p):
    """Exact configuration averages for the g path (3d + 11g) from tape labels; returns the new av and diagnostics."""
    runs = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r1d1g',
            5: 'vq_v0p_{}e_ivo_r3d1g'}
    orbs = {'3d', '11g'}
    for b in p['blocks']:
        z = np.load(os.path.join(ROOT, 'results', runs[b['N']].format(b['t']) + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        keep = [i for i, l in enumerate(lab) if set(B.cfg_orbitals(l)) <= orbs]
        b['W'] = tape_W(runs[b['N']].format(b['t']), keep)
        b['H0'] = b['H']
    lf = fit_labels(p['blocks'])
    diag = split_Q(p['blocks'], {'3d', '11g'}, lf)
    av = dict(F.averages(p['blocks']))
    for b in p['blocks']:
        for c, e in zip(b.get('_cfg', []), b.get('eps', [])):
            av[(b['N'], c)] = float(e)
    return av, dict(diag, dev=lf['dev'])


def main():
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13g: test "r4/|V3 t| = linked denominator spread + rho" (13f model Hamiltonians; no ATOMCI runs)')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    paths = build()
    a(f'\nBuild {time.time() - t0:.0f} s.')
    print(f'built {time.time() - t0:.0f} s', flush=True)

    # ---------------- step 2 split (needed for kappa labels too)
    a('\n## Step 2 check: H_QQ = eps_c I + T_spec + T_link from tape labels')
    a('| path | label fit max |H_model - H| (Eh) | max spread of diag(H_QQ - VV) within a configuration (Eh) | '
      'max |eps I + T_spec + T_link - H_QQ| (Eh) | max null-space component of the labels used |')
    a('|---|---|---|---|---|')
    sd = {}
    for nm in ('core', "b' s3d", "b' s6d"):
        p = paths[nm]
        d = split_Q(p['blocks'], p['V'], p['lf'])
        sd[nm] = d
        a(f'| {nm} | {p["lf"]["dev"]:.1e} | {d["spread"]:.1e} | {d["rec_err"]:.1e} | {d["null"]:.1e} |')
    js['split_check'] = sd

    # ---------------- step 1
    a('\n## Step 1. Denominator spread kappa at every 13f scan point (csf scheme) and r4/|V3 t|')
    scans = {'(i) T': lambda l: dict(lamT=l), '(ii) offdiag': lambda l: dict(lamC=l),
             '(i+ii)': lambda l: dict(lamT=l, lamC=l), '(iv) P': lambda l: dict(lamP=l),
             '(vi) all': lambda l: dict(lamP=l, lamT=l, lamC=l)}
    pts = []
    for nm in ('core', 'g', "b' s3d", "b' s6d", "a' s3d", "a' s6d"):
        p = paths[nm]
        for sn, fx in scans.items():
            for l in F.LAMS:
                kw = fx(l)
                Hf = lambda b, kw=kw, p=p: fmod(b, p['av'], **kw)
                r = pipe(p['blocks'], p['d0'], Hf)
                k3, k4 = kappas(p['blocks'], Hf, 3), kappas(p['blocks'], Hf, 4)
                pts.append(dict(path=nm, scan=sn, lam=l, ratio=r['ratio'], rho=p['rho'], k3=k3, k4=k4,
                                flag4=r['flag'][4]))
        print(f'step1 {nm} {time.time() - t0:.0f} s', flush=True)
    a('| path | scan | lambda | r4/t | kappa N3 / N4 | kappa2 N4 | kappa_in N3 / N4 | flagged d4 |')
    a('|---|---|---|---|---|---|---|---|')
    for q in pts:
        a(f'| {q["path"]} | {q["scan"]} | {q["lam"]:g} | {q["ratio"]:.4f} | {q["k3"]["kappa"]:.4f} / '
          f'{q["k4"]["kappa"]:.4f} | {q["k4"]["kappa2"]:.4f} | {q["k3"]["kappa_in"]:.4f} / {q["k4"]["kappa_in"]:.4f} | '
          f'{"yes" if q["flag4"] else ""} |')
    fitpts = [q for q in pts if not q['path'].startswith("a'") and not q['flag4']]
    a(f'\nFits r4/t = a x + b rho over {len(fitpts)} points (core, g, b\'; a\' excluded; flagged points excluded):')
    a('| x | a | b | rms residual | max |residual| | rms r4/t | R^2 |')
    a('|---|---|---|---|---|---|---|')
    y = np.array([q['ratio'] for q in fitpts])
    fits = {}
    for key, getx in (('kappa N4', lambda q: q['k4']['kappa']), ('kappa N3', lambda q: q['k3']['kappa']),
                      ('kappa2 N4', lambda q: q['k4']['kappa2']), ('kappa_in N4', lambda q: q['k4']['kappa_in']),
                      ('kappa_in N3', lambda q: q['k3']['kappa_in'])):
        X = np.array([[getx(q), q['rho']] for q in fitpts])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ c
        R2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
        fits[key] = dict(a=c[0], b=c[1], rms=float(np.sqrt(np.mean(res ** 2))), max=float(np.abs(res).max()), R2=R2)
        a(f'| {key} | {c[0]:+.4f} | {c[1]:+.4f} | {fits[key]["rms"]:.4f} | {fits[key]["max"]:.4f} | '
          f'{np.sqrt(np.mean(y ** 2)):.4f} | {R2:.3f} |')
    # V shape: where are the minima of r4/t and of kappa in scans (i), (ii)?
    a('\nV-shaped minima: lambda of min r4/t vs lambda of min kappa (N4) and kappa_in (N4) in scans (i), (ii), (i+ii):')
    a('| path | scan | argmin r4/t | argmin kappa | argmin kappa_in |')
    a('|---|---|---|---|---|')
    for nm in ('core', 'g', "b' s3d", "b' s6d"):
        for sn in ('(i) T', '(ii) offdiag', '(i+ii)'):
            qs = [q for q in pts if q['path'] == nm and q['scan'] == sn]
            am = lambda f: min(qs, key=f)['lam']
            a(f'| {nm} | {sn} | {am(lambda q: q["ratio"]):g} | {am(lambda q: q["k4"]["kappa"]):g} | '
              f'{am(lambda q: q["k4"]["kappa_in"]):g} |')
    js['step1'] = dict(points=pts, fits=fits)

    # ---------------- step 2 scans
    a('\n## Step 2. Spectator / linked scans (P term dependence and eps_c kept)')
    a('| path | scan | lambda | r4/t | |r4| (mEh) | |V3 t| (mEh) | kappa N4 | kappa_in N4 | d4 min P | d4 gap (eV) | '
      'd5 min P | flagged (d4, d5) |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|')
    s2 = {}
    for nm in ('core', "b' s3d", "b' s6d"):
        p = paths[nm]
        for sn, kwf in (('(vii) linked', lambda l: dict(lamL=l)), ('(viii) spectator', lambda l: dict(lamS=l))):
            rows = []
            for l in LAM:
                Hf = lambda b, kw=kwf(l): qmod(b, **kw)
                r = pipe(p['blocks'], p['d0'], Hf)
                k4 = kappas(p['blocks'], Hf, 4)
                rows.append(dict(lam=l, ratio=r['ratio'], r4=r['r4'], V3t=r['V3t'], k4=k4, tg=r['tg'], flag=r['flag']))
                a(f'| {nm} | {sn} | {l:g} | {r["ratio"]:.4f} | {1e3 * r["r4"]:.4f} | {1e3 * r["V3t"]:.4f} | '
                  f'{k4["kappa"]:.4f} | {k4["kappa_in"]:.4f} | {r["tg"][4]["wP"]:.4f} | {r["tg"][4]["gap"]:+.1f} | '
                  f'{r["tg"][5]["wP"]:.4f} | {"d4" if r["flag"][4] else ""}{" d5" if r["flag"][5] else ""} |')
            s2[f'{nm} {sn}'] = rows
        print(f'step2 {nm} {time.time() - t0:.0f} s', flush=True)
    js['step2'] = s2

    # ---------------- step 3
    a('\n## Step 3. Gap slope only: all term dependence off (13f (vi), lambda = 0), Delta_i(N) = Delta_i(3) + mu '
      'delta_i (N - 3)')
    a('Closed form (leading term, closure with term-free denominators): for g_i(N) = 1/Delta_i(N) and X_i = H_PQi H_QiP '
      'split into pure k-body parts X_ik, the d^4 residual is r4 = -sum_i sum_k [Delta^(4-k) g_i](k) X_ik^(4) '
      '(forward differences from N = k). With one configuration and Delta(N) = Delta0 (1 + rho (N-3)) the leading '
      'term is k = 3: r4 = rho/(1+rho) x lift(X_3/Delta0), i.e. r4/|V3 lifted| = rho/(1+rho) (c = 1 against the full '
      '3-body V3, not its t-part). Columns: actual r4/|V3 t| and r4/|V3 lifted|, and the closure-model value.')
    a('| path | mu | rho_eff (lowest Q) | r4/|V3 t| | r4/|V3 lifted| | rho_eff/(1+rho_eff) | closure model r4/|V3 lifted| | '
      'c = (r4/|V3|) / rho_eff | d4 min P | flagged (d4, d5) |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    s3 = {}
    for nm in ('core', "b' s3d", "b' s6d"):
        p = paths[nm]
        lowest = min(p['D3'], key=p['D3'].get)
        rho0 = abs(p['delta'][lowest] / p['D3'][lowest])
        for mu in (0.0, 0.5, 1.0, 2.0):
            av = dict(p['av'])
            for (N, c) in list(av):
                if c == 'P':
                    continue
                key = '3s1' if nm == 'core' else c.split()[0][:-1]
                av[(N, c)] = av[(N, 'P')] + p['D3'][key] + mu * p['delta'][key] * (N - 3)
            Hf = lambda b, av=av: fmod(b, av, lamP=0.0, lamT=0.0, lamC=0.0)
            r = pipe(p['blocks'], p['d0'], Hf)
            clo = closure_ratio(p, av)
            rho_e = mu * rho0
            s3[f'{nm} {mu}'] = dict(rho=rho_e, ratio=r['ratio'], rV3=r['r4'] / r['V3'], clo=clo, tg=r['tg'][4],
                                     flag=r['flag'])
            a(f'| {nm} | {mu:g} | {rho_e:.4f} | {r["ratio"]:.4f} | {r["r4"] / r["V3"]:.4f} | '
              f'{rho_e / (1 + rho_e):.4f} | {clo:.4f} | {(r["r4"] / r["V3"]) / rho_e if rho_e else float("nan"):.3f} | '
              f'{r["tg"][4]["wP"]:.4f} | {"d4" if r["flag"][4] else ""}{" d5" if r["flag"][5] else ""} |')
        print(f'step3 {nm} {time.time() - t0:.0f} s', flush=True)
    js['step3'] = s3

    # ---------------- step 4
    a('\n## Step 4. g path with exact configuration averages (tape labels)')
    try:
        p = paths['g']
        av_x, dg = g_exact(p)
        a(f'Label fit max |H_model - H| {dg["dev"]:.1e} Eh; spread of diag(H_QQ - VV) within a configuration '
          f'{dg["spread"]:.1e} Eh; reconstruction {dg["rec_err"]:.1e} Eh; max null-space component {dg["null"]:.1e}.')
        gaps = defaultdict(dict)
        for (N, c), x in av_x.items():
            if c != 'P':
                gaps[c][N] = x - av_x[(N, 'P')]
        a('| configuration | exact gap Delta(N), N = ... (Eh) | surrogate (13f) | delta | rho |')
        a('|---|---|---|---|---|')
        for c, gp in sorted(gaps.items(), key=lambda kv: min(kv[1].values())):
            Ns = sorted(gp)
            sur = [p['av'][(N, c)] - p['av'][(N, 'P')] for N in Ns]
            dl = (gp[Ns[-1]] - gp[Ns[0]]) / (Ns[-1] - Ns[0]) if len(Ns) > 1 else float('nan')
            g3 = gp.get(3, gp[Ns[0]])
            a(f'| {c} | ' + ', '.join(f'{N}: {gp[N]:.4f}' for N in Ns) + ' | ' + ', '.join(f'{x:.4f}' for x in sur) +
              f' | {dl:+.5f} | {abs(dl / g3):.4f} |')
        for lab, av in (('surrogate (13f)', p['av']), ('exact', av_x)):
            rr = []
            for l in (0.0, 1.0):
                Hf = lambda b, av=av, l=l: fmod(b, av, lamP=l, lamT=l, lamC=l)
                rr.append(pipe(p['blocks'], p['d0'], Hf)['ratio'])
            a(f'- {lab}: r4/t at (vi) lambda = 0: {rr[0]:.4f}; lambda = 1: {rr[1]:.4f}')
        js['step4'] = dict(gaps={c: g for c, g in gaps.items()}, diag=dg)
    except Exception as ex:   # report rather than abort
        a(f'Step 4 not completed: {ex!r}')
    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13g.json'), 'w'), indent=1,
              default=lambda x: float(x) if np.isscalar(x) else (x.tolist() if hasattr(x, 'tolist') else str(x)))
    open(os.path.join(OUT, 'phase13g.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


def closure_ratio(p, av):
    """r4/|V3 lifted| of the closure model H_eff(N) = E_P(N) - sum_c H_Pc H_cP / gap_c(N) (term-free denominators,
    H_PP replaced by E_av(P)), same Fock extraction."""
    vecs = {} if p['d0'] is None else {0: np.array([p['d0']])}
    by = defaultdict(dict)
    for b in p['blocks']:
        if b['N'] > 4 or (b['N'] == 0 and p['d0'] is not None):
            continue
        P, Q, N = b['P'], b['Q'], b['N']
        EP = av[(N, 'P')]
        A = EP * np.eye(len(P))
        if Q:
            gap = np.array([av[(N, b['lab'][i])] - EP for i in Q])
            HPQ = b['H'][np.ix_(P, Q)]
            A = A - (HPQ / gap) @ HPQ.T
        by[N][(b['L'], b['twoS'])] = b['U'].T @ A @ b['U']
    for N, bl in by.items():
        vecs[N] = B.vec_from_blocks(N, bl)
    m = C13.metrics(vecs)
    return m['r4'] / m['V3']


def post():
    """Follow-ups: step-1 fit restricted to lambda <= 1; identifiability of the core labels used; small mu.
    -> results/phase13/phase13g_post.txt"""
    w = []
    a = w.append
    js = json.load(open(os.path.join(OUT, 'phase13g.json')))
    pts = [q for q in js['step1']['points'] if not q['path'].startswith("a'") and not q['flag4']]
    a('# Phase 13g follow-ups')
    for lmax in (1.0, 2.0):
        sel = [q for q in pts if q['lam'] <= lmax]
        y = np.array([q['ratio'] for q in sel])
        a(f'\nStep-1 fits over lambda <= {lmax:g} ({len(sel)} points):')
        a('| x | a | b | rms residual | rms r4/t | R^2 |')
        a('|---|---|---|---|---|---|')
        for key, getx in (('kappa N4', lambda q: q['k4']['kappa']), ('kappa2 N4', lambda q: q['k4']['kappa2']),
                          ('kappa_in N4', lambda q: q['k4']['kappa_in']), ('kappa_in N3', lambda q: q['k3']['kappa_in'])):
            X = np.array([[getx(q), q['rho']] for q in sel])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ c
            a(f'| {key} | {c[0]:+.4f} | {c[1]:+.4f} | {np.sqrt(np.mean(res ** 2)):.4f} | {np.sqrt(np.mean(y ** 2)):.4f} | '
              f'{1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2):.3f} |')
    paths = build()
    lf = paths['core']['lf']
    a('\nCore label identifiability (null-space component) and values:')
    for nm in ('R0(3d3d;3d3d)', 'R2(3d3d;3d3d)', 'R4(3d3d;3d3d)', 'R0(3d3d;3s3s)', 'R2(3d3s;3d3s)', 'R0(3s3s;3s3s)'):
        v, n0 = lval(nm, lf)
        a(f'- {nm}: {v:+.6f} Eh, null component {n0:.1e}')
    nul = []
    a(f'- F_av(3d,3d) (core fit) = {fav("3d", "3d", lf, nul):.6f} Eh (13e: 0.707039); F_av(3s,3d) = '
      f'{fav("3s", "3d", lf, nul):.6f} Eh')
    a('\nSmall-mu gap-slope scan (all term dependence off):')
    a('| path | mu | rho_eff | r4/|V3 t| | r4/|V3 lifted| | rho/(1+rho) | closure model | c = (r4/|V3|)/rho |')
    a('|---|---|---|---|---|---|---|---|')
    for nm in ('core', "b' s3d", "b' s6d"):
        p = paths[nm]
        lowest = min(p['D3'], key=p['D3'].get)
        rho0 = abs(p['delta'][lowest] / p['D3'][lowest])
        for mu in (0.05, 0.1, 0.2):
            av = dict(p['av'])
            for (N, c) in list(av):
                if c != 'P':
                    key = '3s1' if nm == 'core' else c.split()[0][:-1]
                    av[(N, c)] = av[(N, 'P')] + p['D3'][key] + mu * p['delta'][key] * (N - 3)
            r = pipe(p['blocks'], p['d0'], lambda b, av=av: fmod(b, av, lamP=0.0, lamT=0.0, lamC=0.0))
            re_ = mu * rho0
            a(f'| {nm} | {mu:g} | {re_:.4f} | {r["ratio"]:.4f} | {r["r4"] / r["V3"]:.4f} | {re_ / (1 + re_):.4f} | '
              f'{closure_ratio(p, av):.4f} | {(r["r4"] / r["V3"]) / re_:.3f} |')
    open(os.path.join(OUT, 'phase13g_post.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    post() if sys.argv[1:] == ['post'] else main()
