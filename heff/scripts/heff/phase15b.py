#!/usr/bin/env python3
"""Phase 15b: isoelectronic d3 (Ti+ .. Fe5+) and d4 (Ti .. Fe4+) series, Z = 22-26: what sets the size of t.

Own configuration-average orbitals per ion (15a recipe); decks = the V own-sector decks with z and basis changed
(tags <el>15b<N>; V uses the 14b runs, Cr the 15a runs). Spaces s3d, s6d, 3d, core, g.
Per ion and space: exact H_eff (P weight, gap, intruders), |t|, |f|; HF 3d <r>, F2, F4; Delta(11d), Delta(12d)
(14a configuration-average formula at the own N); the core gap Delta_core = E_av(3s 3p6 3d^(N+1)) - E_av(3s2 3p6
3d^N) with the own orbitals frozen; ||V_PQ||_F^2 = sum_blocks g ||V||_F^2 and
Delta_eff = sum g |V_qp|^2 / sum g |V_qp|^2 / (e_q - e_p) in the eigenbases of H_PP and H_QQ (g = (2L+1)(2S+1));
(c) and (c2) f errors and Bloch convergence (14d).
usage: phase15b.py orb | prep | run | analyze
-> results/phase15/phase15b_orb.json, phase15b.txt/.json, orbsets_15b.npz
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase14a as A14  # noqa: E402
import phase14a_orb as O  # noqa: E402
import phase14b as P14  # noqa: E402
import phase14d as D14  # noqa: E402
import phase15a as F15  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
EL = {'Ti': (22, 'ti.bas'), 'V': (23, 'v_qzp.bas'), 'Cr': (24, 'cr.bas'), 'Mn': (25, 'mn.bas'), 'Fe': (26, 'fe.bas')}
NEW = ('Ti', 'Mn', 'Fe')
NS = (3, 4)
QP = {3: (3, 3), 4: (4, 6)}
SPACES = ['s3d', 's6d', '3d', 'core', 'g']
NBF = F15.NBF


TAGV = {3: 'o14n3', 4: 'o14n4'}


def tag_of(el, N):
    if el == 'V':
        return TAGV[N]
    if el == 'Cr':
        return f'cr15n{N}'
    return f'{el.lower()}15b{N}'


def charge(el, N):
    return EL[el][0] - 18 - N


def eav(bs, C, occ):
    """Average-of-configuration energy for {shell: q}; shells 1s 2s 3s 2p 3p 3d (columns of C)."""
    col = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '2p': (1, 0), '3p': (1, 1), '3d': (2, 0)}
    sh = [(n, col[n][0], bs.P(col[n][0], C[col[n][0]][:, col[n][1]]), q, C[col[n][0]][:, col[n][1]])
          for n, q in occ.items() if q > 0]
    E = 0.0
    for i, (na, la, Pa, qa, ca) in enumerate(sh):
        E += qa * ca @ bs.h[la] @ ca
        E += qa * (qa - 1) / 2 * (bs.R(Pa, Pa, Pa, Pa, 0) - (2 * la + 1) / (4 * la + 1) *
                                  sum(O.w3(la, k, la) * bs.R(Pa, Pa, Pa, Pa, k) for k in range(2, 2 * la + 1, 2)))
        for nb, lb, Pb, qb, cb in sh[i + 1:]:
            E += qa * qb * (bs.R(Pa, Pa, Pb, Pb, 0) -
                            0.5 * sum(O.w3(la, k, lb) * bs.R(Pa, Pb, Pa, Pb, k) for k in O.exch_ks(la, lb)))
    return float(E)


def info(bs, C, N):
    d = F15.diag_info(bs, C, N)
    base = {'1s': 2, '2s': 2, '2p': 6, '3p': 6}
    d['Dcore'] = eav(bs, C, {**base, '3s': 1, '3d': N + 1}) - eav(bs, C, {**base, '3s': 2, '3d': N})
    d['D'] = {o: float(v) for o, v in d['D'].items()}
    d['rho'] = {o: float(v) for o, v in d['rho'].items()}
    return d


def orb():
    t0 = time.time()
    js, sets = {}, {}
    zv = np.load(os.path.join(ROOT, 'results', 'phase14', 'orbsets_14b.npz'))
    zc = np.load(os.path.join(OUT, 'orbsets_15a.npz'))
    C0 = O.load_original()
    for el, (Z, bas) in EL.items():
        bs = F15.ElBasis(bas, Z)
        for N in NS:
            q, pair = QP[N]
            if el == 'V':
                C = {l: (zv[f'{N}|{l}'] if l < 3 else C0[l]) for l in range(len(NBF))}
                E = O.energy(bs, C, q, pair)
            elif el == 'Cr':
                C = {l: zc[f'{N}|{l}'] for l in range(len(NBF))}
                E = O.energy(bs, C, q, pair)
            else:
                r = O.scf(bs, q, pair)
                Cv, _ = A14.virtuals(bs, r['C'], A14.own_coef(q, pair))
                C = {0: Cv[0], 1: Cv[1], 2: Cv[2]}
                for l in range(3, len(NBF)):
                    C[l] = O.gen_eig(bs.h[l], bs.S[l])[1]
                E = r['E']
                sets[(el, N)] = C
            js[f'{el} {N}'] = dict(E=float(E), q=charge(el, N), **info(bs, C, N))
            print(f'{el} N={N} q={charge(el, N)} E {E:.6f} {time.time() - t0:.0f} s', flush=True)
    np.savez(os.path.join(OUT, 'orbsets_15b.npz'), **{f'{el}|{N}|{l}': C[l] for (el, N), C in sets.items() for l in C})
    json.dump(js, open(os.path.join(OUT, 'phase15b_orb.json'), 'w'), indent=1)


def prep():
    z = np.load(os.path.join(OUT, 'orbsets_15b.npz'))
    n = 0
    for el in NEW:
        Z, bas = EL[el]
        for N in NS:
            t = tag_of(el, N)
            F15.write_be_no({l: z[f'{el}|{N}|{l}'] for l in range(len(NBF))}, t)
            for src, dst in P14.run_names(N, t):
                os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
                deck = open(os.path.join(ROOT, 'inputs', src, 'be.deck')).read()
                deck = deck.replace(src, dst) if src.startswith('vcm') else deck.replace('_ivo_', f'_{t}_')
                assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1, src
                deck = deck.replace('z=23,', f'z={Z},').replace('FILE=v_qzp.bas', f'FILE={bas}')
                open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
                shutil.copy(os.path.join(ROOT, 'inputs', src, 'be.csf'), os.path.join(ROOT, 'inputs', dst))
                n += 1
    print(f'{n} decks')


def run():
    t0 = time.time()
    for el in NEW:
        for N in NS:
            t = tag_of(el, N)
            for src, dst in P14.run_names(N, t):
                if os.path.isdir(os.path.join(ROOT, 'runs', dst)):
                    continue
                subprocess.run([sys.executable, _C.script('run_atomci.py'), dst,
                                '--vectors-from', f'vq_orb_{t}'], check=True, stdout=subprocess.DEVNULL)
            print(f'{t} runs {time.time() - t0:.0f} s', flush=True)


def coupling(sp, N, t):
    V2, den = 0.0, 0.0
    for b in P14.sector_blocks(sp, N, t):
        P, Q, H = b['P'], b['Q'], b['H']
        if not Q:
            continue
        g = (2 * b['L'] + 1) * (b['twoS'] + 1)
        a, A = np.linalg.eigh(H[np.ix_(P, P)])
        e, Bq = np.linalg.eigh(H[np.ix_(Q, Q)])
        Vt = A.T @ H[np.ix_(P, Q)] @ Bq
        V2 += g * float(np.sum(Vt ** 2))
        den += g * float(np.sum(Vt ** 2 / (e[None, :] - a[:, None])))
    return V2, V2 / den


def analyze():
    t0 = time.time()
    orbj = json.load(open(os.path.join(OUT, 'phase15b_orb.json')))
    res = {}
    for el in EL:
        for N in NS:
            t = tag_of(el, N)
            P14.OWN.clear()
            P14.OWN.update({N: t})
            for sp in SPACES:
                x, dg = P14.sector(sp, N, t)
                dec = P14.decompose(N, x)
                V2, Deff = coupling(sp, N, t)
                s = D14.sector(sp, N)
                res[f'{el} {N} {sp}'] = dict(diag=dg, n=dec['n'], V2=V2, Deff=Deff, iters=s['iters'], conv=s['conv'],
                                             conv_dev=s['conv_dev'],
                                             c=s['pred']['c']['f'] if N == 4 else s['pred']['c']['t'],
                                             c2=s['pred']['c2sym']['f'] if N == 4 else s['pred']['c2sym']['t'])
            print(f'{el} {N} {time.time() - t0:.0f} s', flush=True)
    P14.OWN.clear()
    P14.OWN.update({2: 'o14n2', 3: 'o14n3', 4: 'o14n4', 5: 'o14d'})
    json.dump(res, open(os.path.join(OUT, 'phase15b.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    w = ['# Phase 15b: isoelectronic d3 / d4 series (Z = 22-26)']
    ion = lambda el, N: f'{el}{"" if charge(el, N) == 0 else (str(charge(el, N)) if charge(el, N) > 1 else "") + "+"}'  # noqa: E731
    # gaps and Q1 fit
    w.append('\n## Gaps (Eh) and HF 3d quantities')
    w.append('| series | ion | q | 3d <r> | F2 | F4 | Delta(11d) | Delta(12d) | Delta_core (3s->3d, config avg) | '
             'exact lowest core gap (eV) |')
    w.append('|---|---|---|---|---|---|---|---|---|---|')
    fits = {}
    for N in NS:
        qs, d11, dc = [], [], []
        for el in EL:
            o = orbj[f'{el} {N}']
            r = res[f'{el} {N} core']['diag']
            w.append(f'| d{N} | {ion(el, N)} | {o["q"]} | {o["r"]:.4f} | {o["F2"]:.5f} | {o["F4"]:.5f} | '
                     f'{o["D"]["11d"]:.4f} | {o["D"]["12d"]:.4f} | {o["Dcore"]:.4f} | {r["gap"]:.1f} |')
            qs.append(o['q'])
            d11.append(o['D']['11d'])
            dc.append(o['Dcore'])
        x = (np.array(qs) + 1.0) ** 2
        b, a = np.polyfit(x, d11, 1)
        resid = np.array(d11) - (a + b * x)
        rng = max(d11) - min(d11)
        fits[N] = dict(a=a, b=b, resid=resid.tolist(), rel=float(np.abs(resid).max() / rng),
                       core_change=float((max(dc) - min(dc)) / min(dc)))
        w.append(f'- d{N} fit Delta(11d) = a + b (q+1)^2: a = {a:.4f}, b = {b:.5f} Eh; residuals ' +
                 ' / '.join(f'{v:+.4f}' for v in resid) + f' Eh; max |resid| / range = {fits[N]["rel"]:.3f}. '
                 f'Core gap change across the series: {100 * fits[N]["core_change"]:.1f}%.')
        # alternative: linear in q
        b1, a1 = np.polyfit(qs, d11, 1)
        r1 = np.array(d11) - (a1 + b1 * np.array(qs))
        w.append(f'  (for comparison, linear in q: max |resid| / range = {np.abs(r1).max() / rng:.3f})')
    # per space table
    w.append('\n## Per ion and space: P, gap (eV), below; |t|, |f| (mEh); ||V||_F^2 (Eh^2); Delta_eff (Eh); '
             'R = |t| / (||V||^2 / Delta_eff); Bloch iterations; (c), (c2) f rel (d4) or t rel (d3)')
    w.append('| series | space | ion | P, gap, below | abs(t) | abs(f) | V^2 | Delta_eff | R (x1e3) | Bloch | (c) | (c2) |')
    w.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for N in NS:
        for sp in SPACES:
            for el in EL:
                r = res[f'{el} {N} {sp}']
                d, n = r['diag'], r['n']
                R = n['t'] / (r['V2'] / r['Deff'])
                r['R'] = R
                w.append(f'| d{N} | {sp} | {ion(el, N)} | {d["wP"]:.3f}, {d["gap"]:.1f}, {d["below"]}'
                         f'{" F" if d["flag"] else ""} | {1e3 * n["t"]:.4f} | {1e3 * n["f"]:.4f} | {r["V2"]:.4f} | '
                         f'{r["Deff"]:.4f} | {1e3 * R:.3f} | {r["iters"]}{"" if r["conv"] else " NC"} | '
                         f'{r["c"]["rel"]:.4f} | {r["c2"]["rel"]:.4f} |')
    # Q2 and Q3 summaries
    w.append('\n## R = |t| / (||V||^2 / Delta_eff) along each series (x1e3; well-posed ions only), spread = max/min - 1')
    w.append('| series | space | ' + ' | '.join(EL) + ' | spread | |t| monotone in q |')
    w.append('|---|---|' + '---|' * len(EL) + '---|---|')
    summ = {}
    for N in NS:
        for sp in SPACES:
            rows = [(el, res[f'{el} {N} {sp}']) for el in EL]
            ok = [(el, r) for el, r in rows if not r['diag']['flag']]
            Rs = [r['R'] for el, r in ok]
            ts = [r['n']['t'] for el, r in ok]
            spread = max(Rs) / min(Rs) - 1
            dif = np.diff(ts)
            mono = 'decreasing' if (dif < 0).all() else 'increasing' if (dif > 0).all() else 'not monotone'
            summ[f'{N} {sp}'] = dict(spread=spread, mono=mono, used=[el for el, r in ok])
            w.append(f'| d{N} | {sp} | ' + ' | '.join(
                (f'{1e3 * r["R"]:.3f}' + (' F' if r['diag']['flag'] else '')) for el, r in rows) +
                f' | {spread:.3f} | {mono} (over {",".join(el for el, r in ok)}) |')
    json.dump(dict(fits=fits, summ=summ), open(os.path.join(OUT, 'phase15b_fits.json'), 'w'), indent=1)
    w.append(f'\nWall time {time.time() - t0:.0f} s.')
    open(os.path.join(OUT, 'phase15b.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    {'orb': orb, 'prep': prep, 'run': run, 'analyze': analyze}[sys.argv[1]]()
