#!/usr/bin/env python3
"""Phase 14b: N-adapted orbitals (one configuration-average HF set per N) and within-sector many-body content.

Own sets: N = 2 (new, q = 2, pair = 1), 3 (14a d3), 4 (14a a), 5 (14a d; its runs are the 14a o14d runs).
Reference: set o (vq_orb, V2+ 3d3 4F) at the same N. Virtual d by the 14a recipe.
usage: phase14b.py prep | run | analyze
Spaces per N: s3d, s6d (13d singles), 3d (13c d path, reduced _r3d runs), core (vcm, 3s -> 3d), g (_r1d1g, {3d, 11g}).
Within-sector content of the exact des Cloizeaux H_eff of d^N (seniority basis, component metric of dshell.ip):
  operators e0 (identity), e1..e4 (pure 2-body), t1..t4 (pure 3-body) of 13b lifted to d^N, plus f1..f12 (JL d^4/d^5
  tables) for N = 4, 5. The lifts equal the JL tables to 5e-13 (N = 3..5). Sets are complete: d^2 5 components
  (e0..e4; t = 0 identically), d^3 9 (e0..e4, t1..t4), d^4 / d^5 21 (all 21 JL operators). Coefficients by a Gram
  solve; norms |e'| (e1..e4), |t|, |f| = norms of the corresponding parts (the parts are orthogonal; JL operators are
  mutually orthogonal in this metric, so these are the Table-9-normalized coefficient norms).
-> results/phase14/phase14b.txt/.json
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff.reader import read_vectors, records  # noqa: E402
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase13f as F  # noqa: E402
import phase14a as A14  # noqa: E402
import phase14a_orb as O  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase14')
OWN = {2: 'o14n2', 3: 'o14n3', 4: 'o14n4', 5: 'o14d'}
QP = {2: (2, 1), 3: (3, 3), 4: (4, 6), 5: (5, 10)}
SPACES = ['s3d', 's6d', '3d', 'core', 'g']
HART_EV = 27.211386


def write_set(C, tag):
    C0 = O.load_original()
    new = {l: (C[l] if l < 3 else C0[l]) for l in range(len(O.NBF))}
    rec = records(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'))
    nl = len(O.NBF)
    work = os.path.join(ROOT, 'runs', f'vq_orb_{tag}')
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, 'be.NO'), 'wb') as fo:
        for r in rec[:1 + nl + 1]:
            fo.write(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)))
        for l in range(nl):
            r = struct.pack('<i', l) + new[l].T.astype('<f8').tobytes()
            fo.write(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)))
    chk = read_vectors(os.path.join(work, 'be.NO'), O.NBF)
    assert all(np.allclose(chk[1][l], new[l]) for l in range(nl))
    return new


def run_names(N, tag):
    out = []
    for L, twoS, m, t in D.term_list(N):
        for sp in ('s3d', 's6d'):
            out.append((f'vq_{D.ION[N]}_{t}e_ivo_{sp}', f'vq_{D.ION[N]}_{t}e_{tag}_{sp}'))
        for sp in ('r3d', 'r1d1g'):
            out.append((f'vq_{D.ION[N]}_{t}e_ivo_{sp}', f'vq_{D.ION[N]}_{t}e_{tag}_{sp}'))
        out.append((f'vcm_d{N}_{t}e', f'vcm_d{N}_{t}e_{tag}'))
    return out


def prep():
    bs = O.Basis()
    z = np.load(os.path.join(OUT, 'orbsets.npz'))
    sets = {3: 'd3', 4: 'a (d4)', 5: 'd (d5)'}
    res = O.scf(bs, *QP[2])
    C2, _ = A14.virtuals(bs, res['C'], A14.own_coef(*QP[2]))
    allC = {2: C2}
    for N, nm in sets.items():
        allC[N] = {l: z[f'{nm}|{l}'] for l in range(3)}
    np.savez(os.path.join(OUT, 'orbsets_14b.npz'), **{f'{N}|{l}': C[l] for N, C in allC.items() for l in range(3)})
    for N in (2, 3, 4):
        tag = OWN[N]
        new = write_set(allC[N], tag)
        ecore = O.energy(bs, new, 0, 0)
        json.dump(dict(N=N, ecore=ecore), open(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'set.json'), 'w'))
        for src, dst in run_names(N, tag):
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            deck = open(os.path.join(ROOT, 'inputs', src, 'be.deck')).read()
            deck = deck.replace(src, dst) if src.startswith('vcm') else deck.replace('_ivo_', f'_{tag}_')
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            shutil.copy(os.path.join(ROOT, 'inputs', src, 'be.csf'), os.path.join(ROOT, 'inputs', dst))
        print(f'N={N} {tag}: E_core {ecore:.8f}, d2 SCF E {res["E"]:.8f}' if N == 2 else f'N={N} {tag}', flush=True)


def run():
    t0 = time.time()
    for N in (2, 3, 4):
        tag = OWN[N]
        for src, dst in run_names(N, tag):
            if os.path.isdir(os.path.join(ROOT, 'runs', dst)):
                continue
            subprocess.run([sys.executable, _C.script('run_atomci.py'), dst,
                            '--vectors-from', f'vq_orb_{tag}'], check=True, stdout=subprocess.DEVNULL)
        print(f'{tag} runs {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- analysis

def sector_blocks(sp, N, tag):
    js = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))
    out = []
    for L, twoS, m, t in D.term_list(N):
        if sp == 'core':
            base = f'vcm_d{N}_{t}e'
            Hm, lab = B.load_run(base if tag == 'ivo' else f'{base}_{tag}')
            b = dict(N=N, t=t, L=L, twoS=twoS, H=Hm, lab=lab, U=np.array(js['maps'][base]), m=m,
                     P=[i for i, l in enumerate(lab) if l.startswith('3s2')],
                     Q=[i for i, l in enumerate(lab) if not l.startswith('3s2')])
        else:
            run_ = f'vq_{D.ION[N]}_{t}e_{tag}_{ {"3d": "r3d", "g": "r1d1g"}.get(sp, sp)}'
            b = F.blk(N, t, *F.load_subset(run_, None))
        out.append(b)
    return out


def sector(sp, N, tag):
    blocks, info = {}, []
    for b in sector_blocks(sp, N, tag):
        Bk, inf = F.heff_block(b, b['H'])
        blocks[(b['L'], b['twoS'])] = Bk
        info.append(inf)
    x = B.vec_from_blocks(N, blocks)
    diag = dict(wP=min(i['wP'] for i in info), gap=min(i['gap'] for i in info), below=sum(i['below'] for i in info),
                amb=max(i['amb'] for i in info))
    diag['flag'] = bool(diag['wP'] < 0.9 or diag['below'] > 0 or diag['amb'] > 0.3)
    return x, diag


_OPS = None


def basis_ops(N):
    global _OPS
    if _OPS is None:
        _OPS, _, _ = B.ops_fock()
    names, vecs = [], []
    for k in B.EK + B.TK:
        M, n = _OPS[k]
        if n <= N:
            names.append(k)
            vecs.append(S.to_vec(N, S.lift(M, n, N)))
    if N >= 4:
        for k, v in S.jl_table(N).items():
            if k.startswith('f'):
                names.append(k)
                vecs.append(v)
    return names, vecs


def decompose(N, x):
    names, vecs = basis_ops(N)
    G = np.array([[S.ip(N, p, q) for q in vecs] for p in vecs])
    rank = np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max())
    c = np.linalg.solve(G, [S.ip(N, p, x) for p in vecs])
    part = {g: sum(ci * v for ci, v, nm in zip(c, vecs, names) if nm in ks)
            for g, ks in (('e0', ['e0']), ('e', B.EK[1:]), ('t', B.TK),
                          ('f', [n for n in names if n.startswith('f')]))}
    nrm = {g: (float(np.sqrt(S.ip(N, v, v))) if not np.isscalar(v) else 0.0) for g, v in part.items()}
    res = x - sum(ci * v for ci, v in zip(c, vecs))
    td = float(np.sqrt(nrm['e'] ** 2 + nrm['t'] ** 2 + nrm['f'] ** 2))
    return dict(c=dict(zip(names, c.tolist())), n=nrm, td=td, rank=int(rank), dim=len(x),
                resid=float(np.sqrt(abs(S.ip(N, res, res)))))


def ut4(c):
    """Coefficient of the t-part along u = 3 t1 + 2 t2 and t4 (least squares in the d^3 metric of the Fock t_i) and
    the relative norm outside span{u, t4}."""
    T3 = S.jl_table(3)
    t = sum(c[k] * T3[k] for k in B.TK)
    u = 3 * T3['t1'] + 2 * T3['t2']
    M = np.array([[S.ip(3, p, q) for q in (u, T3['t4'])] for p in (u, T3['t4'])])
    a = np.linalg.solve(M, [S.ip(3, p, t) for p in (u, T3['t4'])])
    r = t - a[0] * u - a[1] * T3['t4']
    return float(a[0]), float(a[1]), float(np.sqrt(S.ip(3, r, r) / max(S.ip(3, t, t), 1e-300)))


def kappa(bs, C0, CN):
    """3d(N) and virtual d(N) in the set-o d basis; generator K = log(U), U = C0^T S CN (columns sign-aligned)."""
    from scipy.linalg import logm
    U = C0[2].T @ bs.S[2] @ CN[2]
    sg = np.sign(np.diag(U))
    sg[sg == 0] = 1
    U = U * sg
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    K = np.real(logm(U))
    return dict(c3d=U[:, 0].tolist(), kap=K[1:, 0].tolist(), nk=float(np.linalg.norm(K[1:, 0])),
                nvv=float(np.linalg.norm(K[1:, 1:]) / np.sqrt(2)), anti=float(np.abs(K + K.T).max()),
                ov_virt=[float(abs(U[i, i])) for i in range(1, 6)])


def analyze():
    t0 = time.time()
    js = {}
    w = []
    a = w.append
    a('# Phase 14b: N-adapted orbitals; within-sector many-body content')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    bs = O.Basis()
    C0 = O.load_original()
    z = np.load(os.path.join(OUT, 'orbsets_14b.npz'))
    own = {N: {l: z[f'{N}|{l}'] for l in range(3)} for N in range(2, 6)}
    for N in range(2, 6):
        for l in range(3, len(O.NBF)):
            own[N][l] = C0[l]

    # ---- step 1: integrals (A residual, Delta) and exact diagnostics
    a('\n## Step 1. Well-posedness: Brillouin residual, Delta_i at N, exact P+Q diagnostics')
    a('Integrals (frozen core of the set): max|f_N(i)| over 11d-15d (mEh, A at N), Delta_i(N) for 11d / 12d / 13d '
      '(Eh), rho_i (from the set\'s own linear Delta_i).')
    a('| N | set | max abs(f_N) (mEh) | Delta(N) 11d / 12d / 13d | rho 11d / 12d / 13d |')
    a('|---|---|---|---|---|')
    s1 = {}
    for N in range(2, 6):
        for nm, C in (('own', own[N]), ('o', C0)):
            g = A14.gaps(bs, C)
            fm = max(abs(g[o]['f'][N]) for o in A14.VIRT)
            s1[f'{N} {nm}'] = dict(fmax=1e3 * fm, D={o: g[o]['D'][N] for o in A14.VIRT},
                                   rho={o: g[o]['rho'] for o in A14.VIRT})
            a(f'| {N} | {nm} | {1e3 * fm:.4f} | ' + ' / '.join(f'{g[o]["D"][N]:.4f}' for o in A14.VIRT[:3]) + ' | ' +
              ' / '.join(f'{g[o]["rho"]:.3f}' for o in A14.VIRT[:3]) + ' |')
    js['step1_int'] = s1
    print(f'step1 integrals {time.time() - t0:.0f} s', flush=True)

    res = {}
    for sp in SPACES:
        for N in range(2, 6):
            for nm, tag in (('own', OWN[N]), ('o', 'ivo')):
                x, dg = sector(sp, N, tag)
                res[(sp, N, nm)] = dict(x=x, diag=dg, dec=decompose(N, x))
        print(f'{sp} {time.time() - t0:.0f} s', flush=True)

    a('\nExact P+Q (des Cloizeaux) per space: min P weight, lowest Q-P gap (eV), states below targets, F = flagged '
      '(P < 0.9, states below, or ambiguous).')
    a('| space | N | own: P, gap, below | o: P, gap, below |')
    a('|---|---|---|---|')
    for sp in SPACES:
        for N in range(2, 6):
            cells = []
            for nm in ('own', 'o'):
                d = res[(sp, N, nm)]['diag']
                cells.append(f'{d["wP"]:.3f}, {d["gap"]:.1f}, {d["below"]}{" F" if d["flag"] else ""}')
            a(f'| {sp} | {N} | ' + ' | '.join(cells) + ' |')

    # ---- step 2
    a('\n## Step 2. Within-sector content, N = 4 and 5 (mEh; component metric)')
    a('|e\'| = e1..e4 part, |t|, |f|; TD = term-dependent part (all but e0). Basis completeness: rank / dim; '
      'residual outside the basis.')
    a('| space | N | set | abs(e\') | abs(t) | abs(f) | abs(f)/abs(t) | abs(f)/TD | rank/dim | resid |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    s2 = {}
    for sp in SPACES:
        for N in (4, 5):
            for nm in ('own', 'o'):
                d = res[(sp, N, nm)]['dec']
                n = d['n']
                s2[f'{sp} {N} {nm}'] = dict(e=n['e'], t=n['t'], f=n['f'], ft=n['f'] / n['t'], ftd=n['f'] / d['td'],
                                            rank=d['rank'], dim=d['dim'], resid=d['resid'],
                                            flag=res[(sp, N, nm)]['diag']['flag'])
                x = s2[f'{sp} {N} {nm}']
                a(f'| {sp} | {N} | {nm}{" (F)" if x["flag"] else ""} | {1e3 * n["e"]:.3f} | {1e3 * n["t"]:.4f} | '
                  f'{1e3 * n["f"]:.4f} | {x["ft"]:.3f} | {x["ftd"]:.4f} | {d["rank"]}/{d["dim"]} | '
                  f'{d["resid"]:.1e} |')
    js['step2'] = s2
    a('\nCompleteness in d^2 / d^3 (rank / dim, residual): ' + '; '.join(
        f'{sp} N={N} {res[(sp, N, "own")]["dec"]["rank"]}/{res[(sp, N, "own")]["dec"]["dim"]} '
        f'{res[(sp, N, "own")]["dec"]["resid"]:.0e}' for sp in ('s6d',) for N in (2, 3)))

    # relation to the lifted r4 of set o (13 style, from 13h/14a outputs)
    a('\nSet o: within-sector |f| (d^4) vs the 13-style lifted r4 (13h / 14a exact values, d^4): the lifts of V0..V3 '
      'have no f component in d^4, so |f(H_eff)| = |f-part of r4| exactly; r4 also has e and t parts.')
    a('| space | 13-style r4/abs(V3) (o) | abs(V3) (mEh) | abs(r4) (mEh) | within-sector abs(f) (mEh) | abs(f)/abs(r4) |')
    a('|---|---|---|---|---|---|')
    import phase13c as C13
    rel = {}
    for sp in SPACES:
        vecs = {0: np.array([B.ECORE])} if sp != 'core' else {}
        for N in range(1 if sp != 'core' else 0, 5):
            if (sp, N, 'o') in res:
                vecs[N] = res[(sp, N, 'o')]['x']
            else:
                vecs[N] = sector(sp, N, 'ivo')[0]
        m = C13.metrics(vecs)
        fN = res[(sp, 4, 'o')]['dec']['n']['f']
        rel[sp] = dict(r4V3=m['r4'] / m['V3'], V3=m['V3'], r4=m['r4'], f=fN, f_from_r4=m['f'])
        a(f'| {sp} | {m["r4"] / m["V3"]:.4f} | {1e3 * m["V3"]:.3f} | {1e3 * m["r4"]:.3f} | {1e3 * fN:.4f} | '
          f'{fN / m["r4"]:.3f} (f-part of r4: {1e3 * m["f"]:.4f}) |')
    js['rel'] = rel

    # ---- step 3
    a('\n## Step 3. Coefficients on the Fock-space operators, N = 2..5 (Eh; e0 in Eh, others in mEh)')
    a('u, t4: t-part along u = 3t1 + 2t2 and t4; "out" = relative norm of the t-part outside span{u, t4}. '
      'D1, D2 = first / second differences; lin = rms residual of a linear fit in N (e: N = 2..5, t: N = 3..5).')
    s3 = {}
    for sp in SPACES:
        for nm in ('own', 'o'):
            rows = {N: res[(sp, N, nm)]['dec']['c'] for N in range(2, 6)}
            for N in range(3, 6):
                rows[N]['u'], rows[N]['t4s'], rows[N]['out'] = ut4(rows[N])
            keys = B.EK + B.TK + ['u', 't4s']
            tab = {}
            for k in keys:
                Ns = [N for N in range(2, 6) if k in rows[N]]
                v = np.array([rows[N][k] for N in Ns])
                d1 = np.diff(v)
                d2 = np.diff(v, 2)
                p = np.polyfit(Ns, v, 1)
                lin = float(np.sqrt(np.mean((np.polyval(p, Ns) - v) ** 2)))
                tab[k] = dict(Ns=Ns, v=v.tolist(), d1=d1.tolist(), d2=d2.tolist(), slope=float(p[0]), lin=lin)
            s3[f'{sp} {nm}'] = dict(tab=tab, out={N: rows[N]['out'] for N in range(3, 6)},
                                    flag={N: res[(sp, N, nm)]['diag']['flag'] for N in range(2, 6)})
            sc = lambda k: 1.0 if k == 'e0' else 1e3  # noqa: E731
            a(f'\n### {sp}, {nm} (flagged N: {[N for N in range(2, 6) if res[(sp, N, nm)]["diag"]["flag"]]}; '
              f'out of span: ' + ', '.join(f'N={N} {rows[N]["out"]:.3f}' for N in range(3, 6)) + ')')
            a('| coef | values (N = 2..5 or 3..5) | D1 | D2 | slope | lin rms | lin rms / abs(slope) |')
            a('|---|---|---|---|---|---|---|')
            for k in keys:
                x = tab[k]
                f = sc(k)
                a(f'| {k if k != "t4s" else "t4 (u,t4 fit)"} | ' + ' / '.join(f'{f * v:.4f}' for v in x['v']) + ' | ' +
                  ' / '.join(f'{f * v:.4f}' for v in x['d1']) + ' | ' + ' / '.join(f'{f * v:.4f}' for v in x['d2']) +
                  f' | {f * x["slope"]:.4f} | {f * x["lin"]:.4f} | {x["lin"] / max(abs(x["slope"]), 1e-300):.3f} |')
    js['step3'] = s3

    # ---- step 4
    a('\n## Step 4. Own 3d(N) relative to set o: U = C_o^T S C_N, K = log U (antisymmetric)')
    a('| N | <3d(o)|3d(N)> | coefficients on 11d..15d (o) | kappa(3d) = K[virt, 3d] | abs(kappa) | '
      'abs(K virt-virt) | virtual overlaps abs(<nd(o)|nd(N)>) |')
    a('|---|---|---|---|---|---|---|')
    s4 = {}
    for N in range(2, 6):
        k = kappa(bs, C0, own[N])
        s4[N] = k
        a(f'| {N} | {k["c3d"][0]:.6f} | ' + ' '.join(f'{v:+.4f}' for v in k['c3d'][1:]) + ' | ' +
          ' '.join(f'{v:+.4f}' for v in k['kap']) + f' | {k["nk"]:.4f} | {k["nvv"]:.4f} | ' +
          ' '.join(f'{v:.3f}' for v in k['ov_virt']) + ' |')
    kv = np.array([s4[N]['kap'] for N in range(2, 6)])
    d1 = np.diff(kv, axis=0)
    d2 = np.diff(kv, 2, axis=0)
    a('First differences |kappa(N+1) - kappa(N)|: ' + ' / '.join(f'{np.linalg.norm(x):.4f}' for x in d1) +
      '; second differences: ' + ' / '.join(f'{np.linalg.norm(x):.4f}' for x in d2) +
      f'; max antisymmetry error of log U {max(s4[N]["anti"] for N in s4):.1e}.')
    js['step4'] = s4
    a(f'\nWall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase14b.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, 'phase14b.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    {'prep': prep, 'run': run, 'analyze': analyze}[sys.argv[1]]()
