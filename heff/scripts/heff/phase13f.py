#!/usr/bin/env python3
"""Phase 13f: what sets the size of the residual four-body term r4/|V3 t|?

Paths (one dense H per (N, L, S) block, CSF labels = configurations):
  core   3s2 3d^N + 3s 3d^(N+1) (vcm_* runs), N = 0..9 (d0..d5 used)
  g      3d + 11g full CI: QZP d1..d3 and r3d1g d5 exports restricted to {3d, 11g}, r1d1g d4
  b'     singles s3d / s6d, M-only coupling, Q diagonal frozen at the N = 3 averaged gaps (13e)
  a'     the same with A + M coupling (reference)
  d(13c) 3d, 5d1g, 6d2g full CI (measurements only, N = 3, 4; sparse)

Q-side split, per N: eps_c = exact configuration average of c (core: 13e formula; b'/a': E_av(P) + Delta_i(3);
a: E_av(P) + Delta_i(N)); g: linear-in-N fit of the P-coupled CSF-diagonal gaps (surrogate); 13c full-CI d spaces
(eta only): P-coupled CSF-diagonal means. E_P = E_av(P) (complete). Pass 1 (phase13f_pass1_pcoupled.txt) used
P-coupled means everywhere and is superseded (their N dependence is erratic in the core, up to 0.06 Eh).
  gap_c   = eps_c - E_P
  T_diag  = diag(H_QQ) - eps_c                       (term-dependent part of the Q diagonal, CSF basis)
  W       = off-diagonal H_QQ inside one configuration
  C       = H_QQ blocks between different Q configurations
Scaling schemes (lambda multiplies the ingredient, everything else fixed):
  csf  (literal):        (i) T_diag;        (ii) W and C (all off-diagonal H_QQ in the CSF basis)
  cfg  (basis-free):     (i) T_diag and W;  (ii) C only
  (iii) gaps: gap_c -> gap_c / lambda.  (i+ii) both (i) and (ii).
-> results/phase13/phase13f.txt/.json
"""
import json
import os
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
from atomci_heff import heff  # noqa: E402
import phase13b as B  # noqa: E402
import phase13c as C13  # noqa: E402
import phase13d as D  # noqa: E402
import phase13e as E  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
HART_EV = D.HART_EV
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


# ---------------------------------------------------------------- data

def load_subset(run, orbs):
    """Dense H and labels of the CSFs of results/<run>.npz whose orbitals lie in orbs (no full densification)."""
    z = np.load(os.path.join(ROOT, 'results', run + '.npz'), allow_pickle=True)
    lab = [str(x) for x in z['labels']]
    keep = [i for i, l in enumerate(lab) if orbs is None or set(B.cfg_orbitals(l)) <= orbs]
    pos = -np.ones(len(lab), int)
    pos[keep] = np.arange(len(keep))
    if 'H' in z:
        H = z['H'][np.ix_(keep, keep)]
    else:
        r, c, v = z['H_rows'], z['H_cols'], z['H_vals']
        m = (pos[r] >= 0) & (pos[c] >= 0)
        H = np.zeros((len(keep), len(keep)))
        H[pos[r[m]], pos[c[m]]] = v[m]
        H = H + np.tril(H, -1).T
    return H, [lab[i] for i in keep]


def blk(N, t, H, lab, P=None):
    L, twoS = B.term_key(N, t)
    P = P if P is not None else [i for i, l in enumerate(lab) if l == f'3d{N}']
    return dict(N=N, t=t, L=L, twoS=twoS, g=(2 * L + 1) * (twoS + 1), H=H, lab=lab, P=P,
                Q=[i for i in range(len(lab)) if i not in set(P)], U=B.core_map(N, t),
                m=len(dict(((a, b), c) for a, b, c in S.terms(N))[(L, twoS)]))


def core_blocks():
    data, _ = C13.core_data()
    out = []
    for name, d in data.items():
        t = f'{d["twoS"] + 1}{S.LCH[d["L"]]}'
        _, lab = B.load_run(name)
        b = blk(d['N'], t, d['H'], lab, P=d['P'])
        b['U'] = d['U']
        out.append(b)
    return out


def g_blocks():
    runs = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r1d1g',
            5: 'vq_v0p_{}e_ivo_r3d1g'}
    orbs = {'3d', '11g'}
    return [blk(N, t, *load_subset(runs[N].format(t), orbs)) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)]


def dfull_blocks(sp):
    """13c d-path spaces, N = 3 (QZP subset) and 4 (own export)."""
    orbs = C13.B.space_orbitals(C13.VSPACES[sp])
    out = []
    for L, twoS, m, t in D.term_list(3):
        out.append(blk(3, t, *load_subset(f'vq_v2p_{t}e_ivo', orbs)))
    for L, twoS, m, t in D.term_list(4):
        out.append(blk(4, t, *load_subset(f'vq_v1p_{t}e_ivo_r{sp}', None)))
    return out


def singles_blocks(data, f, kind, shift):
    out = []
    for (N, t), d in data.items():
        H = E.frozen_H(d, f, kind, shift) if shift else D.variant_H(d, f, kind)
        out.append(blk(N, t, H, d['lab']))
    return out


# ---------------------------------------------------------------- split and modification

def averages(blocks, gap=None):
    """eps per (N, configuration): P-coupled weighted CSF-diagonal means; then, if gap is given, the Q entries are
    replaced by E_av(P) + gap(N, cfg) (exact configuration averages), or by a linear-in-N fit of the P-coupled gaps per
    configuration (gap == 'linear')."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for b in blocks:
        dg = np.diag(b['H'])
        for i, l in enumerate(b['lab']):
            k = 'P' if i in set(b['P']) else l
            acc[(b['N'], k)][0] += b['g'] * dg[i]
            acc[(b['N'], k)][1] += b['g']
    av = {k: v[0] / v[1] for k, v in acc.items()}
    if gap is None:
        return av
    out = dict(av)
    if gap == 'linear':
        byc = defaultdict(dict)
        for (N, c), x in av.items():
            if c != 'P':
                byc[c][N] = x - av[(N, 'P')]
        for c, gp in byc.items():
            if len(gp) >= 2:
                Ns = np.array(sorted(gp))
                k1, k0 = np.polyfit(Ns, [gp[N] for N in Ns], 1)
                for N in Ns:
                    out[(N, c)] = av[(N, 'P')] + k0 + k1 * N
        return out
    for (N, c) in av:
        if c != 'P':
            x = gap(N, c)
            if x is not None:
                out[(N, c)] = av[(N, 'P')] + x
    return out


def build_paths(data_only=False):
    """All paths with their blocks, d0 and the exact / surrogate constant part of the Q diagonal."""
    paths = {}
    cgx, _ = E.exact_core_gaps(json.load(open(os.path.join(OUT, 'phase13b_b.json')))['params'])
    paths['core 3s->3d'] = dict(blocks=core_blocks(), d0=None, gap=lambda N, c: cgx[N] if c.startswith('3s1') else None)
    paths['g 3d+11g'] = dict(blocks=g_blocks(), d0=B.ECORE, gap='linear')
    data = {sp: {(N, t): D.load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)} for sp in D.SPACES}
    lf = E.label_fit(data)
    dgp, _, _ = E.exact_d_gaps(data['s6d'], lf)
    shift = {o: {N: dgp[o][3] - dgp[o][N] for N in dgp[o]} for o in dgp}
    orb = lambda c: c.split()[0][:-1]
    for sp in D.SPACES:
        f, _ = D.split(data[sp])
        for kind, lab in (('b', "b'"), ('a', "a'")):
            paths[f"{lab} {sp}"] = dict(blocks=singles_blocks(data[sp], f, kind, shift), d0=B.ECORE,
                                        gap=lambda N, c: dgp[orb(c)][3])
        paths[f'a {sp} (13d)'] = dict(blocks=singles_blocks(data[sp], f, 'a', None), d0=B.ECORE,
                                      gap=lambda N, c: dgp[orb(c)][N])
    for p in paths.values():
        p['av'] = averages(p['blocks'], p['gap'])
    return paths


def modify(b, av, lamT=1.0, lamC=1.0, lamG=1.0, scheme='csf', lamP=1.0, lamPQ=1.0):
    H = b['H'].copy()
    Q = b['Q']
    N, EP = b['N'], av[(b['N'], 'P')]
    if lamP != 1.0:
        P = b['P']
        H[np.ix_(P, P)] = EP * np.eye(len(P)) + lamP * (H[np.ix_(P, P)] - EP * np.eye(len(P)))
    if not Q:
        return H
    if lamPQ != 1.0:
        H[np.ix_(b['P'], Q)] *= lamPQ
        H[np.ix_(Q, b['P'])] *= lamPQ
    cfg = np.array([b['lab'][i] for i in Q])
    eps = np.array([av[(N, c)] for c in cfg])
    HQ = H[np.ix_(Q, Q)]
    same = cfg[:, None] == cfg[None, :]
    off = ~np.eye(len(Q), dtype=bool)
    lamW = lamC if scheme == 'csf' else lamT
    new = np.where(same & off, lamW * HQ, np.where(off, lamC * HQ, 0.0))
    new[np.diag_indices(len(Q))] = EP + (eps - EP) / lamG + lamT * (np.diag(HQ) - eps)
    H[np.ix_(Q, Q)] = new
    return H


# ---------------------------------------------------------------- measurements (step 1)

def eta(blocks, Ns=(3, 4), av=None):
    av = av or averages(blocks)
    out = {}
    for N in Ns:
        bs = [b for b in blocks if b['N'] == N]
        EP = av[(N, 'P')]
        gaps = {k[1]: v - EP for k, v in av.items() if k[0] == N and k[1] != 'P'}
        cmin = min(gaps, key=gaps.get)
        D0 = gaps[cmin]
        eP, wP = [], []
        tq, wq, cnt = [], [], 0.0
        W2 = C2 = PQ2 = 0.0
        specW = specC = specO = 0.0
        tspread = 0.0
        for b in bs:
            H, P, Q = b['H'], b['P'], b['Q']
            e = np.linalg.eigvalsh(H[np.ix_(P, P)])
            eP += e.tolist()
            wP += [b['g']] * len(e)
            if not Q:
                continue
            cfg = np.array([b['lab'][i] for i in Q])
            dq = np.diag(H)[Q] - np.array([av[(N, c)] for c in cfg])
            tq += dq.tolist()
            wq += [b['g']] * len(Q)
            for c in set(cfg):
                x = dq[cfg == c]
                tspread = max(tspread, x.max() - x.min())
            HQ = H[np.ix_(Q, Q)].copy()
            np.fill_diagonal(HQ, 0.0)
            same = cfg[:, None] == cfg[None, :]
            Wm, Cm = np.where(same, HQ, 0.0), np.where(same, 0.0, HQ)
            W2 += b['g'] * np.sum(Wm ** 2)
            C2 += b['g'] * np.sum(Cm ** 2)
            PQ2 += b['g'] * np.sum(H[np.ix_(P, Q)] ** 2)
            cnt += b['g'] * len(Q)
            if len(Q) <= 4000:
                specW = max(specW, np.abs(np.linalg.eigvalsh(Wm)).max())
                specC = max(specC, np.abs(np.linalg.eigvalsh(Cm)).max())
                specO = max(specO, np.abs(np.linalg.eigvalsh(HQ)).max())
            else:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh
                for M, nm in ((Wm, 'W'), (Cm, 'C'), (HQ, 'O')):
                    x = abs(eigsh(csr_matrix(M), k=1, which='LM', return_eigenvectors=False)[0])
                    if nm == 'W':
                        specW = max(specW, x)
                    elif nm == 'C':
                        specC = max(specC, x)
                    else:
                        specO = max(specO, x)
        eP, wP, tq, wq = map(np.array, (eP, wP, tq, wq))
        nP = float(np.sum(wP))
        out[N] = dict(D0=D0, cmin=cmin, ncfg=len(gaps),
                      etaP_spread=(eP.max() - eP.min()) / D0,
                      etaP_rms=float(np.sqrt(np.sum(wP * (eP - EP) ** 2) / nP)) / D0,
                      etaQ_rms=float(np.sqrt(np.sum(wq * tq ** 2) / np.sum(wq))) / D0 if len(tq) else 0.0,
                      etaQ_spread=tspread / D0,
                      etaQQ_W_rms=float(np.sqrt(W2 / cnt)) / D0 if cnt else 0.0,
                      etaQQ_C_rms=float(np.sqrt(C2 / cnt)) / D0 if cnt else 0.0,
                      etaQQ_spec=specO / D0, etaQQ_W_spec=specW / D0, etaQQ_C_spec=specC / D0,
                      etaPQ_rms=float(np.sqrt(PQ2 / nP)) / D0)
    return out


# ---------------------------------------------------------------- pipeline

def heff_block(b, H):
    P = b['P']
    if not b['Q']:
        return b['U'].T @ H[np.ix_(P, P)] @ b['U'], dict(wP=1.0, amb=0.0, below=0, gap=np.inf)
    e, Cv, wP, _ = heff.spectrum(H, P)
    st, amb = heff.select_states(wP, b['m'])
    other = [k for k in range(len(e)) if k not in st]
    r = heff.exact_heff(H, P, st) if other else dict(hermitian=H)
    return b['U'].T @ r['hermitian'] @ b['U'], dict(
        wP=float(wP[st].min()), amb=float(amb), below=int(sum(1 for k in other if e[k] < e[st].max())),
        gap=float((e[other[0]] - e[st].max()) * HART_EV) if other else np.inf)


def pipeline(blocks, av, d0=None, Nmax=5, **mod):
    vecs = {} if d0 is None else {0: np.array([d0])}
    info = {}
    by = defaultdict(dict)
    for b in blocks:
        if b['N'] > Nmax or (b['N'] == 0 and d0 is not None):
            continue
        Bk, inf = heff_block(b, modify(b, av, **mod))
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
    flag4 = tg[4]['wP'] < 0.9 or tg[4]['amb'] > 0.3 or tg[4]['below'] > 0
    flag = {N: tg[N]['wP'] < 0.9 or tg[N]['amb'] > 0.3 or tg[N]['below'] > 0 for N in tg}
    return dict(r4=m['r4'], V3t=m['V3t'], ratio=m['ratio_t'], h4=m['h4'], e=m['e'], t=m['t'], f=m['f'], tg=tg,
                flag4=flag4, flag=flag)


def slopes(lams, ys, y0):
    """Local log-log exponents of y(lambda) - y(0) between successive positive lambdas."""
    out = []
    pts = [(l, y - y0) for l, y in zip(lams, ys) if l > 0]
    for (l1, a), (l2, b) in zip(pts, pts[1:]):
        out.append(float(np.log(abs(b) / abs(a)) / np.log(l2 / l1)) if a and b and a * b > 0 else np.nan)
    return out


def main():
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13f: what sets r4/|V3 t|? Resolvent-shape ratios and controlled scaling (no ATOMCI runs)')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip().replace('\n', '\n'))

    # ---------------- build paths
    paths = build_paths()
    print(f'paths built {time.time() - t0:.0f} s', flush=True)

    # ---------------- step 1
    rho = {'core 3s->3d': 0.0275, 'g 3d+11g': 0.0007}
    meas = {}
    for nm, p in paths.items():
        r = pipeline(p['blocks'], p['av'], p['d0'])
        meas[nm] = dict(eta=eta(p['blocks'], av=p['av']), r=r)
    for sp in ('3d', '5d1g', '6d2g'):
        bl = dfull_blocks(sp)
        meas[f'd {sp} (13c full CI)'] = dict(eta=eta(bl), r=None)
        print(f'{sp} eta done {time.time() - t0:.0f} s', flush=True)
    c13 = json.load(open(os.path.join(OUT, 'phase13c_valence.json')))['spaces']
    a('\n## Step 1. Resolvent-shape ratios (Delta0 = lowest Q-configuration gap gap_c; all ratios / Delta0)')
    a('eta_P: P term energies (eigenvalues of H_PP), max-min spread and (2L+1)(2S+1)-weighted rms about E_av(P). '
      'eta_Q: T_diag, weighted rms over Q CSFs and the largest max-min within one configuration. eta_QQ: off-diagonal '
      'H_QQ, weighted rms row norm sqrt(sum g |X|_F^2 / sum g n_Q) for W (inside a configuration) and C (between '
      'configurations), and the largest spectral norm |offdiag H_QQ|_2 over blocks. eta_PQ: weighted rms of H_PQ per '
      'P state. rho from 13e (0 for frozen diagonals).')
    a('| path | N | Delta0 (Eh) [config] | eta_P spread / rms | eta_Q rms / spread | eta_QQ W rms | eta_QQ C rms | '
      'eta_QQ spec | eta_PQ rms | rho | r4/|V3 t| |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    known = {'d 3d (13c full CI)': c13['3d']['m']['ratio_t'], 'd 5d1g (13c full CI)': c13['5d1g']['m']['ratio_t'],
             'd 6d2g (13c full CI)': c13['6d2g']['m']['ratio_t']}
    for nm, x in meas.items():
        rr = x['r']['ratio'] if x['r'] else known[nm]
        rh = rho.get(nm, 0.0 if "'" in nm else 0.77)
        for N, e in x['eta'].items():
            a(f'| {nm} | {N} | {e["D0"]:.4f} [{e["cmin"]}] | {e["etaP_spread"]:.4f} / {e["etaP_rms"]:.4f} | '
              f'{e["etaQ_rms"]:.4f} / {e["etaQ_spread"]:.4f} | {e["etaQQ_W_rms"]:.4f} | {e["etaQQ_C_rms"]:.4f} | '
              f'{e["etaQQ_spec"]:.4f} | {e["etaPQ_rms"]:.4f} | {rh:.4f} | {rr:.4f} |')
    js['step1'] = {nm: dict(eta=x['eta'], ratio=(x['r']['ratio'] if x['r'] else known[nm])) for nm, x in meas.items()}
    print(f'step 1 done {time.time() - t0:.0f} s', flush=True)

    # ---------------- step 2 / 3
    scans = {}
    targets = ["core 3s->3d", "b' s3d", "b' s6d", 'g 3d+11g']
    facs = {'(i) T': lambda l: dict(lamT=l), '(ii) offdiag/C': lambda l: dict(lamC=l),
            '(i+ii)': lambda l: dict(lamT=l, lamC=l), '(iii) gaps/lambda': lambda l: dict(lamG=l)}
    for nm in targets:
        p = paths[nm]
        av = p['av']
        for scheme in ('csf', 'cfg'):
            for fn, fx in facs.items():
                if fn.startswith('(iii)') and scheme == 'cfg':
                    continue
                rows = []
                for l in LAMS:
                    if fn.startswith('(iii)') and l == 0:
                        continue
                    r = pipeline(p['blocks'], av, p['d0'], scheme=scheme, **fx(l))
                    rows.append((l, r))
                scans[(nm, scheme, fn)] = rows
        print(f'scan {nm} {time.time() - t0:.0f} s', flush=True)

    a('\n## Step 2/3. Controlled scaling: r4/|V3 t| vs lambda (one ingredient scaled, all else fixed)')
    a('Per point: r4/|V3 t| [|r4| mEh, |V3 t| mEh]; d^4 min P weight; lowest non-target - highest target gap (eV); '
      '"!" = d^4 flagged (P < 0.9, amb > 0.3 or roots below a target). d^5 flags in the json.')
    a('Exponents: local log-log slopes of y(lambda) - y(0) between successive lambdas (y = r4/|V3 t|; for (iii) '
      'y(0) := 0, the infinite-gap limit).')
    jsc = {}
    for (nm, scheme, fn), rows in scans.items():
        a(f'\n### {nm}, scheme {scheme}, {fn}')
        a('| lambda | r4/|V3 t| | |r4| | |V3 t| | h4 | |e| / |t| / |f| | d4 min P | d4 gap (eV) | d5 flagged |')
        a('|---|---|---|---|---|---|---|---|---|')
        for l, r in rows:
            a(f'| {l:g} | {r["ratio"]:.4f}{" !" if r["flag4"] else ""} | {1e3 * r["r4"]:.4f} | {1e3 * r["V3t"]:.4f} | '
              f'{1e3 * r["h4"]:+.4f} | {1e3 * r["e"]:.3f} / {1e3 * r["t"]:.3f} / {1e3 * r["f"]:.3f} | '
              f'{r["tg"][4]["wP"]:.4f} | {r["tg"][4]["gap"]:+.1f} | {"yes" if r["flag"][5] else "no"} |')
        lams = [l for l, _ in rows]
        ys = [r['ratio'] for _, r in rows]
        y0 = ys[0] if lams[0] == 0 else 0.0
        sl = slopes(lams, ys, y0)
        a('local exponents: ' + ', '.join(f'{x:.2f}' for x in sl) + f'  (y(0) = {y0:.4f})')
        jsc[f'{nm} | {scheme} | {fn}'] = dict(rows=[(l, {k: v for k, v in r.items()}) for l, r in rows], y0=y0,
                                              slopes=sl)
    js['scans'] = jsc
    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13f.json'), 'w'), indent=1,
              default=lambda x: float(x) if np.isscalar(x) else (x.tolist() if hasattr(x, 'tolist') else str(x)))
    open(os.path.join(OUT, 'phase13f.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


def extra():
    """Extra scalings (not in the request): (iv) P term spread, (v) H_PQ, (vi) all term dependence (P, T_diag, all
    off-diagonal H_QQ) together. Scheme csf. -> results/phase13/phase13f_extra.txt/.json"""
    t0 = time.time()
    w = []
    a = w.append
    paths = {k: v for k, v in build_paths().items() if k in ("core 3s->3d", 'g 3d+11g', "b' s3d", "b' s6d")}
    facs = {'(iv) P term spread': lambda l: dict(lamP=l), '(v) H_PQ': lambda l: dict(lamPQ=l),
            '(vi) all term dependence (P, T, offdiag QQ)': lambda l: dict(lamP=l, lamT=l, lamC=l)}
    a('# Phase 13f extra scalings (scheme csf; same conventions as phase13f.txt)')
    js = {}
    for nm, p in paths.items():
        av = p['av']
        for fn, fx in facs.items():
            rows = []
            for l in LAMS:
                if fn.startswith('(v)') and l == 0:
                    continue
                rows.append((l, pipeline(p['blocks'], av, p['d0'], **fx(l))))
            a(f'\n### {nm}, {fn}')
            a('| lambda | r4/|V3 t| | |r4| | |V3 t| | h4 | |e| / |t| / |f| | d4 min P | d4 gap (eV) | d5 flagged |')
            a('|---|---|---|---|---|---|---|---|---|')
            for l, r in rows:
                a(f'| {l:g} | {r["ratio"]:.4f}{" !" if r["flag4"] else ""} | {1e3 * r["r4"]:.4f} | {1e3 * r["V3t"]:.4f} | '
                  f'{1e3 * r["h4"]:+.4f} | {1e3 * r["e"]:.3f} / {1e3 * r["t"]:.3f} / {1e3 * r["f"]:.3f} | '
                  f'{r["tg"][4]["wP"]:.4f} | {r["tg"][4]["gap"]:+.1f} | {"yes" if r["flag"][5] else "no"} |')
            lams = [l for l, _ in rows]
            ys = [r['ratio'] for _, r in rows]
            y0 = ys[0] if lams[0] == 0 else 0.0
            sl = slopes(lams, ys, y0)
            a('local exponents: ' + ', '.join(f'{x:.2f}' for x in sl) + f'  (y(0) = {y0:.4f})')
            js[f'{nm} | {fn}'] = dict(rows=rows, slopes=sl, y0=y0)
        print(f'{nm} {time.time() - t0:.0f} s', flush=True)
    a(f'\nWall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13f_extra.json'), 'w'), indent=1,
              default=lambda x: float(x) if np.isscalar(x) else (x.tolist() if hasattr(x, 'tolist') else str(x)))
    open(os.path.join(OUT, 'phase13f_extra.txt'), 'w').write('\n'.join(w) + '\n')


if __name__ == '__main__':
    extra() if sys.argv[1:] == ['extra'] else main()
