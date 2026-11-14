#!/usr/bin/env python3
"""Phase 17 item 6 (1 Nov brief): projection vs (1b)-model representation of our spin-free H_eff.

One determinant basis for everything (atomci_heff.dshell, d^N of the 10 d spin-orbitals):
  model (the (1b) operators): A (identity), B, C, alpha L(L+1), beta (two-body Racah Q), T1, T2 (the pure
    three-body O1, O2 of d3, Frobenius norm 1 in the d3 metric, coefficients in mEh; lifted for d6), zeta (l.s);
  beyond the model: d3: O3, O4 (the rest of the three-body space, orthogonal to the model in the d3 metric);
    d6: O3, O4 lifted and the twelve pure four-body JL operators f1..f12 of d4, lifted (phase15h.basis6).
Our H_eff (Eh component vector in the canonical seniority basis) -> determinant matrix (cm-1). d3: rebuilt from the
stored 9 projection coefficients (A..beta, T1..T4); d6: from the exact blocks dumped by phase17q11_dump.py.
Three representations of the same H_eff:
  proj   the complete orthogonal projection (stored; recomputed here as a check),
  terms  LS fit of all spin-free term energies (uniform weights), A B C alpha beta T1 T2 (the '16-term' fit for d6),
  levels the (1b) fit itself on pseudo-levels: H_eff + zeta (the free (1b) zeta of the ion) -> the J levels of the NIST
         set (same J and order as in the measured fit), fitted with the (1b) model (8 parameters incl. zeta).
Commands: check (the (1b) NIST fits with these operators against phase15u_bfit), d3, absorb, d6, chain, refit"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import cfp_l, dshell as S  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase17')
R15 = os.path.join(ROOT, 'results', 'phase15')
EH_CM, MEH_CM = 219474.6313705, 219.4746         # phase15h.EH_CM; phase15k.MEH_CM
D2 = {'B': [14, 7, -3, -8, 4], 'C': [7, 0, 2, 0, 2], 'alpha': [0, 2, 6, 12, 20], 'beta': [5, 0, 0, 0, 0]}
MODEL = ['A', 'B', 'C', 'alpha', 'beta', 'T1', 'T2']
NM1B = ['A', 'B', 'C', 'zeta', 'alpha', 'beta', 'T1', 'T2']


def o_axes3():
    """d3 component vectors O1..O4 (orthonormal in the d3 metric; O1, O2 = the 15c axes, O3, O4 = jl t3, t1
    orthogonalised; phase15k.op_basis)."""
    import phase15h as H
    O1, O2 = H.o_axes()
    T3 = S.jl_table(3)
    ax = [O1, O2]
    for k in ('t3', 't1'):
        v = np.array(T3[k], float)
        for a in ax:
            v = v - S.ip(3, v, a) * a
        ax.append(v / np.sqrt(S.ip(3, v, v)))
    return ax


class DS:
    """Operators on the d^N determinants (cm-1 per unit parameter; T* in mEh) and level / term extraction."""

    def __init__(self, N):
        self.N = N
        SO = cfp_l.spin_orbitals(2)
        lzs = np.diag([m for m, s in SO]).astype(float)
        szs = np.diag([s / 2 for m, s in SO])
        ls = lzs @ szs
        for p, (m, s) in enumerate(SO):
            for q, (m2, s2) in enumerate(SO):
                if m == m2 + 1 and s == -1 and s2 == 1:          # l+ s-
                    ls[p, q] += 0.5 * np.sqrt(6 - m2 * (m2 + 1))
                if m2 == m + 1 and s2 == -1 and s == 1:          # l- s+
                    ls[p, q] += 0.5 * np.sqrt(6 - m * (m + 1))
        ob = lambda f: S._one_body(N, f)                          # noqa: E731
        self.lz, self.sz = ob(lzs), ob(szs)
        lp, sp = S.ladders(N)
        self.L2 = lp.T @ lp + self.lz @ self.lz + self.lz
        self.S2 = sp.T @ sp + self.sz @ self.sz + self.sz
        self.J2 = self.L2 + self.S2 + 2 * (self.lz @ self.sz + 0.5 * (lp @ sp.T + lp.T @ sp))
        jz2 = np.round(2 * np.diag(self.lz + self.sz)).astype(int)
        self.blocks = {j2: np.where(jz2 == j2)[0] for j2 in sorted(set(jz2)) if j2 >= 0}
        n = len(S.dets(N))
        up = lambda v, k: S.lift(S.from_vec(k, v), k, N)          # noqa: E731
        ax = o_axes3()
        self.ops = {'A': np.eye(n), 'zeta': ob(ls)}
        for k, c in D2.items():
            self.ops[k] = up(np.array(c, float), 2)
        for i, a in enumerate(ax):
            self.ops[f'T{i + 1}'] = up(a, 3) * MEH_CM
        self.extra = ['T3', 'T4']
        if N >= 4:
            for k, v in S.jl_table(4).items():
                if k.startswith('f'):
                    self.ops[k] = up(np.array(v, float), 4)
                    self.extra.append(k)
        X = self.ops['zeta']
        assert np.abs(X - X.T).max() < 1e-12

    def vec(self, X):
        return S.to_vec(self.N, X)

    def mat(self, x):
        return S.from_vec(self.N, x)

    def decompose(self, X):
        """X (det matrix, cm-1) on model + extra (the complete spin-free set): {name: coefficient}, residual."""
        names = MODEL + self.extra
        V = [self.vec(self.ops[k]) for k in names]
        x = self.vec(X)
        G = np.array([[S.ip(self.N, p, q) for q in V] for p in V])
        c = np.linalg.solve(G, [S.ip(self.N, p, x) for p in V])
        r = x - sum(ci * v for ci, v in zip(c, V))
        return dict(zip(names, map(float, c))), float(np.sqrt(abs(S.ip(self.N, r, r)))), int(np.linalg.matrix_rank(G))

    def levels(self, H):
        out = {}
        for j2, idx in self.blocks.items():
            w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
            jj = np.einsum('ia,ij,ja->a', v, self.J2[np.ix_(idx, idx)], v)
            sel = abs(jj - j2 / 2 * (j2 / 2 + 1)) < 1e-4
            out[j2] = (w[sel], v[:, sel], idx)
        return out

    def terms(self, H):
        out = {}
        dl, ds = np.round(np.diag(self.lz), 8), np.round(2 * np.diag(self.sz)).astype(int)
        for L in range(9):
            for s2 in range(self.N % 2, self.N + 1, 2):
                idx = np.where((abs(dl - L) < 1e-8) & (ds == s2))[0]
                if not len(idx):
                    continue
                w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
                l2 = np.einsum('ia,ij,ja->a', v, self.L2[np.ix_(idx, idx)], v)
                ss = np.einsum('ia,ij,ja->a', v, self.S2[np.ix_(idx, idx)], v)
                sel = (abs(l2 - L * (L + 1)) < 1e-4) & (abs(ss - s2 / 2 * (s2 / 2 + 1)) < 1e-4)
                if sel.any():
                    out[(s2 + 1, L)] = (w[sel], v[:, sel], idx)
        return out


_DS = {}


def ds(N):
    if N not in _DS:
        _DS[N] = DS(N)
    return _DS[N]


def model_H(d, p):
    return sum(v * d.ops[k] for k, v in p.items())


def fit(d, obs, get, names, fixed, p0):
    import phase15u_crfit as CF
    f = CF.lsq(d, names, fixed, obs, get, p0, iters=200)
    f['mean_error'] = f['rms']
    return f


def nist_obs(ion):
    import phase15u_bfit as BF
    rec = next(x for x in BF.IONS if x[0] == ion)
    lev = BF.nist(rec[2], rec[3])
    byj = {}
    for x in sorted(lev, key=lambda x: x[3]):
        byj.setdefault(x[2], []).append(x)
    return [(j2, k, x[3]) for j2, xs in sorted(byj.items()) for k, x in enumerate(xs)], rec[3]


def bfit_free(ion):
    """The (1b) fit as phase15u_bfit / phase17_bfix do it (its own operators)."""
    import phase15u_bfit as BF
    rec = next(x for x in BF.IONS if x[0] == ion)
    N = rec[3]
    d = BF.D(N)
    lev = BF.nist(rec[2], N)
    e0 = np.mean([x[3] for x in lev])
    base = [e0, 74000.0, 46000.0, 300.0, 50.0, -300.0]
    nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
    if N == 3:
        f1a = BF.fit(d, lev, ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T'], {}, [*base, 0.0])
        p0 = [f1a['p'][n] for n in nm[:6]] + [BF.t1t2(f1a)[0], 0.0]
    else:
        p0 = [*base, 9.0, 3.0]
    return BF.fit(d, lev, nm, {}, p0)


def to_bc(p):
    """(1b) F2, F4 parametrisation -> ours (B, C; A shifts by the F0-like part, irrelevant)."""
    q = dict(p)
    q['B'], q['C'] = p['F2'] / 49 - 5 * p['F4'] / 441, 35 * p['F4'] / 441
    return {k: q[k] for k in NM1B}


def fit_levels(ion, X, zeta, p0, fixed=None):
    """(1b) fit of the pseudo-levels of X + zeta l.s on the NIST level set of the ion (same J, same order)."""
    obs0, N = nist_obs(ion)
    d = ds(N)
    lev = d.levels(X + zeta * d.ops['zeta'])
    obs = [(j2, k, float(lev[j2][0][k])) for j2, k, _ in obs0]
    fixed = fixed or {}
    names = [n for n in NM1B if n not in fixed]
    return fit(d, obs, d.levels, names, fixed, [p0[n] for n in names])


def fit_terms(N, X, p0):
    d = ds(N)
    tm = d.terms(X)
    obs = [(key, k, float(e)) for key, (w, v, idx) in sorted(tm.items()) for k, e in enumerate(w)]
    return fit(d, obs, d.terms, MODEL, {}, [p0[n] for n in MODEL])


# ---------------------------------------------------------------- commands

def check():
    """The (1b) NIST fits redone with the DS operators: must give the phase15u_bfit values (same model)."""
    res = {}
    for ion in ('cr', 'v', 'mn', 'fe'):
        b = bfit_free(ion)
        p0 = to_bc(b['p'])
        obs, N = nist_obs(ion)
        d = ds(N)
        f = fit(d, obs, d.levels, NM1B, {}, [p0[n] for n in NM1B])
        res[ion] = dict(bfit={k: b['p'][k] for k in ('T1', 'T2', 'zeta', 'alpha', 'beta')}, bfit_rms=b['rms'],
                        ds={k: f['p'][k] for k in ('T1', 'T2', 'zeta', 'alpha', 'beta', 'B', 'C')}, ds_rms=f['rms'],
                        dT1=f['p']['T1'] - b['p']['T1'], dT2=f['p']['T2'] - b['p']['T2'])
        print(ion, 'bfit T1 T2 %.4f %.4f rms %.2f | DS T1 T2 %.4f %.4f rms %.2f' % (
            b['p']['T1'], b['p']['T2'], b['rms'], f['p']['T1'], f['p']['T2'], f['rms']), flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q11_check.json'), 'w'), indent=1)


ZETA = {'cr': 276.8, 'v': 171.3, 'mn': 412.7, 'fe': 421.4}     # free (1b) zeta (phase17_bfix), cm-1
D3RUNS = {   # (ion, label): (json file, key) of stored d3 projections (exact)
    ('v', 'Step 10'): ('phase15k_u_vnd_besth_sall_s4', 'besth_sall_s4'),
    ('cr', 'Step 10'): ('phase15k_u_crnd_besth_sall_s4', 'besth_sall_s4'),
    ('mn', 'Step 10'): ('phase15k_u_mnnd_besth_sall_s4', 'besth_sall_s4'),
    ('cr', 'k1fh'): ('phase15k_t_crnd_k1fh', 'k1fh'), ('cr', 'k1fh_n3'): ('phase15k_u_crnd_k1fh_n3', 'k1fh_n3'),
    ('cr', 'k1fh_sall'): ('phase15k_17_crnd_k1fh_sall', 'k1fh_sall'), ('cr', 'u3c1'): ('phase15k_u_crnd_u3c1', 'u3c1'),
    ('cr', 'besth'): ('phase15k_t_crnd_besth', 'besth'), ('cr', 'Step 8'): ('phase15k_u_crnd_besth_sall', 'besth_sall'),
    ('cr', 'fam'): ('phase15k_17_crnd_fam_d5f2g2h1', 'fam_d5f2g2h1')}


def x_from_proj(p):
    """d3 det matrix (cm-1) from the stored projection coefficients (A..beta cm-1, T1..T4 mEh)."""
    d = ds(3)
    return model_H(d, {k: p[k] for k in ('A', 'B', 'C', 'alpha', 'beta')}) + \
        sum(p[f'T{i}'] * d.ops[f'T{i}'] for i in range(1, 5))


def three(ion, X, proj=None):
    """proj / terms / levels representation of one H_eff (det matrix, cm-1)."""
    N = 6 if ion == 'fe' else 3
    d = ds(N)
    c, res, rank = d.decompose(X)
    p0 = dict(c, zeta=ZETA[ion])
    ft = fit_terms(N, X, p0)
    fl = fit_levels(ion, X, ZETA[ion], p0)
    out = dict(proj=dict(c, resid=res, rank=rank),
               terms={**ft['p'], 'rms': ft['rms'], 'n': ft['n']}, levels={**fl['p'], 'rms': fl['rms'], 'n': fl['n']})
    if proj:
        out['stored_proj'] = {k: proj[k] for k in ('T1', 'T2')}
    return out


def d3():
    res = {}
    for (ion, lab), (f, k) in D3RUNS.items():
        p = json.load(open(os.path.join(R15, f + '.json')))[k]['exact']['proj']
        X = x_from_proj(p)
        r = three(ion, X, p)
        res[f'{ion}:{lab}'] = r
        pr, t, l = r['proj'], r['terms'], r['levels']
        print(f'{ion:2s} {lab:9s} proj {pr["T1"]:8.4f} {pr["T2"]:8.4f} (stored {p["T1"]:8.4f} {p["T2"]:8.4f}; T3 {pr["T3"]:+.3f}'
              f' T4 {pr["T4"]:+.3f}; resid {pr["resid"]:.1e}) | terms {t["T1"]:8.4f} {t["T2"]:8.4f} rms {t["rms"]:.2f}'
              f' | levels {l["T1"]:8.4f} {l["T2"]:8.4f} rms {l["rms"]:.2f}', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q11_d3.json'), 'w'), indent=1)


def absorb():
    """Linear response of the (1b) fit (levels: NIST set of the ion; terms: all terms) to each operator outside the
    model, at the free (1b) point: d(model parameter) / d(coefficient of the extra operator)."""
    res = {}
    for ion in ('cr', 'v', 'mn', 'fe'):
        N = 6 if ion == 'fe' else 3
        d = ds(N)
        p = to_bc(bfit_free(ion)['p'])
        X0 = model_H(d, {k: p[k] for k in MODEL})
        f0l, f0t = fit_levels(ion, X0, p['zeta'], p), fit_terms(N, X0, p)
        out = {}
        for k in d.extra:
            eps = 0.05 if k.startswith("T") else 1.0             # mEh for T3, T4; cm-1 (JL units) for f1..f12 (linear regime)
            X = X0 + eps * d.ops[k]
            fl, ft = fit_levels(ion, X, p['zeta'], p), fit_terms(N, X, p)
            out[k] = dict(levels={q: (fl['p'][q] - f0l['p'][q]) / eps for q in NM1B if q != 'A'},
                          terms={q: (ft['p'][q] - f0t['p'][q]) / eps for q in MODEL if q != 'A'},
                          d_rms_levels=(fl['rms'] - f0l['rms']) / eps)
        res[ion] = out
        print(ion, {k: {q: round(v, 3) for q, v in o['levels'].items() if q in ('T1', 'T2', 'B', 'C', 'beta')}
                    for k, o in out.items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q11_absorb.json'), 'w'), indent=1)


def x_from_dump(tag, space, key='exact'):
    """d6 det matrix (cm-1) of a dumped H_eff (phase17q11_dump: blocks in the canonical seniority basis, Eh)."""
    import phase13b as B
    import phase13d as D
    f = os.path.join(OUT, 'heff_dump', f'q11_{tag}_{space}_{space}_iter.npz')
    if not os.path.exists(f):       # dumped by queue 2 itself (HEFF_DUMP; paper items 3-5)
        f = os.path.join(OUT, 'heff_dump', f'17_{tag}_{space}_{space}_iter.npz')
    z = np.load(f)
    by = {(L, twoS): z[f'{key}|{t}'] for L, twoS, m, t in D.term_list(6)}
    return S.from_vec(6, B.vec_from_blocks(6, by) * EH_CM)


def x_from_q18(sub):
    """d6 det matrix (cm-1) of one exact sub-space of the item 11 famiq dump (heff_dump/q18_feti1_famiq.npz)."""
    import phase13b as B
    import phase13d as D
    z = np.load(os.path.join(OUT, 'heff_dump', 'q18_feti1_famiq.npz'))
    by = {(L, twoS): z[f'{sub}|{t}'] for L, twoS, m, t in D.term_list(6)}
    return S.from_vec(6, B.vec_from_blocks(6, by) * EH_CM)


def rep(src):
    """('d3', json, key, ion) | ('d6', tag, space, ion) -> the three representations (cached on disk)."""
    cp = os.path.join(OUT, 'phase17q11_rep.json')
    cache = json.load(open(cp)) if os.path.exists(cp) else {}
    k = '|'.join(src)
    if k not in cache:
        kind, a, b, ion = src
        if kind == 'd3':
            pr = json.load(open(os.path.join(R15, a + '.json')))[b]['exact']['proj']
            r = three(ion, x_from_proj(pr), pr)
        elif kind == 'q18':     # item 11: exact sub-space a of the famiq dump (phase17q18.py)
            r = three(ion, x_from_q18(a), None)
            st = json.load(open(os.path.join(OUT, 'phase17q18.json')))['fe'][a]
            r['stored_proj'] = dict(T1=st['T1'], T2=st['T2'])
        else:
            r = three(ion, x_from_dump(a, b), None)
            st = json.load(open(os.path.join(R15, f'phase15j_17_{a}_{b}.json')))[b]['exact']
            r['stored_proj'] = dict(T1=st['proj_T1'], T2=st['proj_T2'], **{f'fit16_{q}': st['16'][q] for q in ('T1', 'T2')})
        cache[k] = r
        json.dump(cache, open(cp, 'w'), indent=1)
    return cache[k]


def d6():
    for tag, sp in [('feop', 'rasK1r'), ('fedk', 'rasK1r')] + [('fend', x) for x in ('fam_d5f2g2h1', 'k1fh', 'k1fh_n3',
                                                                                       'u3c1', 'k1fh_sall')] + \
            [('feti1', 'k1fhi'), ('feti5', 'k1fhi')]:
        if not os.path.exists(os.path.join(OUT, 'heff_dump', f'q11_{tag}_{sp}_{sp}_iter.npz')):
            print(tag, sp, 'no dump yet')
            continue
        r = rep(('d6', tag, sp, 'fe'))
        pr, t, l, st = r['proj'], r['terms'], r['levels'], r['stored_proj']
        print(f'{tag:5s} {sp:13s} proj {pr["T1"]:8.4f} {pr["T2"]:8.4f} (stored {st["T1"]:8.4f} {st["T2"]:8.4f}; resid '
              f'{pr["resid"]:.1e} rank {pr["rank"]}) | terms {t["T1"]:8.4f} {t["T2"]:8.4f} (stored 16-term '
              f'{st["fit16_T1"]:8.4f} {st["fit16_T2"]:8.4f}) rms {t["rms"]:.1f} | levels {l["T1"]:8.4f} {l["T2"]:8.4f} rms '
              f'{l["rms"]:.1f}', flush=True)


def _d3(f, k, ion='cr'):
    return ('d3', f, k, ion)


SRC = dict(
    base={'V²⁺': _d3('phase15k_u_vnd_besth_sall_s4', 'besth_sall_s4', 'v'),
          'Cr³⁺': _d3('phase15k_u_crnd_besth_sall_s4', 'besth_sall_s4'),
          'Mn⁴⁺': _d3('phase15k_u_mnnd_besth_sall_s4', 'besth_sall_s4', 'mn'),
          'Fe²⁺': ('d6', 'fend', 'fam_d5f2g2h1', 'fe')},
    k1fh=_d3('phase15k_t_crnd_k1fh', 'k1fh'), v3d3=_d3('phase15k_17_crnd_k1fh_v3d3', 'k1fh_v3d3'),
    t=_d3('phase15k_17_crt2_k1fh_t', 'k1fh_t'), s2=_d3('phase15k_17_crt2_k1fh_2s_t', 'k1fh_2s_t'),
    p2=_d3('phase15k_17_crt2_k1fh_2p_h_t', 'k1fh_2p_h_t'),
    P={b: (_d3(f'phase15k_17_{b}_k1f_tnf', 'k1f_tnf'), _d3(f'phase15k_17_{b}_k1f_2p_h_tnf', 'k1f_2p_h_tnf'))
       for b in ('crt2', 'crt3', 'crt2g')},
    i=_d3('phase15k_17_crti5_k1fhi', 'k1fhi'), i1=_d3('phase15k_17_crti1_k1fhi', 'k1fhi'),
    k=_d3('phase15k_17_crtk1_k1fhik', 'k1fhik'),
    dk={'V²⁺': (_d3('phase15k_17_vop_k1fhr', 'k1fhr', 'v'), _d3('phase15k_17_vdk_k1fhr', 'k1fhr', 'v')),
        'Cr³⁺': (_d3('phase15k_17_crop_k1fhr', 'k1fhr'), _d3('phase15k_17_crdk_k1fhr', 'k1fhr')),
        'Mn⁴⁺': (_d3('phase15k_17_mnop_k1fhr', 'k1fhr', 'mn'), _d3('phase15k_17_mndk_k1fhr', 'k1fhr', 'mn')),
        'Fe²⁺': (('d6', 'feop', 'rasK1r', 'fe'), ('d6', 'fedk', 'rasK1r', 'fe'))},
    n3=_d3('phase15k_u_crnd_k1fh_n3', 'k1fh_n3'), u3c1=_d3('phase15k_u_crnd_u3c1', 'u3c1'),
    sall=_d3('phase15k_17_crnd_k1fh_sall', 'k1fh_sall'), besth=_d3('phase15k_t_crnd_besth', 'besth'),
    s8=_d3('phase15k_u_crnd_besth_sall', 'besth_sall'), cfam=_d3('phase15k_17_crnd_fam_d5f2g2h1', 'fam_d5f2g2h1'),
    fe={k: ('d6', 'fend', k, 'fe') for k in ('k1fh', 'k1fh_n3', 'u3c1', 'k1fh_sall')},
    # item 3 (1 Nov): each ion's own i pair (<el>nd k1fh, <el>ti5 k1fhi) and d3 hole pair (k1fh_n3, u3c1)
    own_i={'V²⁺': (_d3('phase15k_17_vnd_k1fh', 'k1fh', 'v'), _d3('phase15k_17_vti5_k1fhi', 'k1fhi', 'v')),
           'Mn⁴⁺': (_d3('phase15k_17_mnnd_k1fh', 'k1fh', 'mn'), _d3('phase15k_17_mnti5_k1fhi', 'k1fhi', 'mn')),
           'Fe²⁺': (('d6', 'fend', 'k1fh', 'fe'), ('d6', 'feti5', 'k1fhi', 'fe'))},
    # item 3 (2 Nov): Fe2+ own 2p in P'' (fet26, no virtual g) and the Cr3+ P'' calibration pair (crt23)
    fe2p=(('d6', 'fet2', 'k1f_tnfg0', 'fe'), ('d6', 'fet2', 'k1f_2p_h_tnfg0', 'fe')),
    cr2p=(_d3('phase15k_17_crt2_k1f_tnfg0', 'k1f_tnfg0'), _d3('phase15k_17_crt2_k1f_2p_h_tnfg0', 'k1f_2p_h_tnfg0')),
    own_hole={'V²⁺': (_d3('phase15k_17_vnd_k1fh_n3', 'k1fh_n3', 'v'), _d3('phase15k_17_vnd_u3c1', 'u3c1', 'v')),
              'Mn⁴⁺': (_d3('phase15k_17_mnnd_k1fh_n3', 'k1fh_n3', 'mn'), _d3('phase15k_17_mnnd_u3c1', 'u3c1', 'mn'))})
# item 11 (4 Nov): Fe2+ n = 3 and the first i measured in fam - 12f (famiq sub-spaces), the second i from RAS k = 1 + h
_q = lambda sub: ('q18', sub, 'famiq', 'fe')       # noqa: E731
SRC['fe_large'] = dict(n3=(_q('X-11i,h2'), _q('X-11i')), i1=(_q('X-11i,h2'), _q('X-h2')),
                       i2=(('d6', 'feti1', 'k1fhi', 'fe'), ('d6', 'feti5', 'k1fhi', 'fe')))
TAIL = 0.09 / 0.077            # l >= 7: one k (-0.077 %) + geometric tail = -0.09 % (Final table, item 1c)
Q = ('T1', 'T2', 'B', 'C', 'beta')


def chain(R='levels'):
    """The Final-table chain in representation R (proj | terms | levels): same items, same uncertainty model."""
    import phase17_final as PF
    g = lambda src: rep(src)[R]                                   # noqa: E731

    def inc(a, b):           # (dT1 %, dT2, dB, dC, dbeta) of a over b
        a, b = g(a), g(b)
        return dict(T1=100 * (a['T1'] / b['T1'] - 1), **{q: a[q] - b[q] for q in Q[1:]})

    def add(*xs, w=None):
        w = w or [1] * len(xs)
        return {q: sum(c * x[q] for c, x in zip(w, xs)) for q in Q}

    P = {b: inc(o, c) for b, (c, o) in SRC['P'].items()}
    carry = {'three virtual electrons': inc(SRC['v3d3'], SRC['k1fh']), '2s open': inc(SRC['s2'], SRC['t']),
             '2p open, tight d, f (+ 2nd pair, g)': add(inc(SRC['p2'], SRC['t']), P['crt3'], P['crt2'], P['crt2g'],
                                                        P['crt2'], w=[1, 1, -1, 1, -1]),
             'i functions': inc(SRC['i'], SRC['k1fh']),
             'l ≥ 7': {q: TAIL * v for q, v in inc(SRC['k'], SRC['i1']).items()}}
    hole = {q: g(SRC['u3c1'])[q] - g(SRC['n3'])[q] for q in Q}
    out = dict(carry=carry, hole=hole, ions={})
    for ion in PF.IONS:
        try:
            b = g(SRC['base'][ion])
            dk = inc(SRC['dk'][ion][1], SRC['dk'][ion][0])
            ci = dict(carry)
            own = {'i functions'} if ion in SRC['own_i'] else set()
            if own:
                ci['i functions'] = inc(SRC['own_i'][ion][1], SRC['own_i'][ion][0])
            if ion == 'Fe²⁺':       # item 11: first i in fam - 12f + second i in RAS k = 1 + h
                FL = {k: inc(v[1], v[0]) for k, v in SRC['fe_large'].items()}
                ci['i functions'] = add(FL['i1'], FL['i2'])
            if ion == 'Fe²⁺':       # own 2p: P'' T1 % x Cr3+ (% k1fh_t / % P''), raw T2, + the Cr3+ 2nd pair and g
                nm2 = '2p open, tight d, f (+ 2nd pair, g)'
                fp, cp, ck = inc(SRC['fe2p'][1], SRC['fe2p'][0]), inc(SRC['cr2p'][1], SRC['cr2p'][0]), inc(SRC['p2'], SRC['t'])
                xt = add(P['crt3'], P['crt2'], P['crt2g'], P['crt2'], w=[1, -1, 1, -1])
                ci[nm2] = dict(T1=fp['T1'] * ck['T1'] / cp['T1'] + xt['T1'], T2=fp['T2'],
                               **{q: None for q in Q[2:]})
                own = own | {nm2}
            ho, uion = hole, PF.U_ION[ion]
            if ion in SRC['own_hole']:
                ho = {q: g(SRC['own_hole'][ion][1])[q] - g(SRC['own_hole'][ion][0])[q] for q in Q}
                uion = 0.0
        except (FileNotFoundError, KeyError) as e:
            out['ions'][ion] = dict(missing=repr(e))
            continue
        d3 = ion in PF.D3
        bud = []
        if d3:
            T1h, T2 = b['T1'] + ho['T1'], b['T2'] + ho['T2']
            BCb = {q: b[q] + ho[q] for q in Q[2:]}
        else:
            try:
                fe = {k: g(v) for k, v in SRC['fe'].items()}
            except FileNotFoundError as e:
                out['ions'][ion] = dict(missing=repr(e))
                continue
            pc = lambda x, y: 100 * (x['T1'] / y['T1'] - 1)           # noqa: E731
            cr = {k: g(SRC[k]) for k in ('k1fh', 'n3', 'sall', 'besth', 's8', 'cfam')}
            cr['s10'] = g(SRC['base']['Cr³⁺'])
            items = {}
            for nm, (fa, fb), (ca, cb), (la, lb) in (('n = 3 configurations', ('k1fh_n3', 'k1fh'), ('n3', 'k1fh'), ('s10', 's8')),
                                                     ('all virtual s', ('k1fh_sall', 'k1fh'), ('sall', 'k1fh'), ('s8', 'besth'))):
                raw1, raw2 = pc(fe[fa], fe[fb]), fe[fa]['T2'] - fe[fb]['T2']
                tf1 = pc(cr[la], cr[lb]) / pc(cr[ca], cr[cb])
                tf2 = (cr[la]['T2'] - cr[lb]['T2']) / (cr[ca]['T2'] - cr[cb]['T2'])
                a1, a2 = raw1 * tf1, raw2 * tf2
                items[nm] = dict(raw1=raw1, raw2=raw2, tf1=tf1, tf2=tf2, d1=a1, d2=a2,
                                 u1=np.hypot(PF.U_TRANSFER * a1, 0.5 * (a1 - raw1) if abs(tf1 - 1) > 0.2 else 0.0),
                                 u2=np.hypot(PF.U_TRANSFER * a2, 0.5 * (a2 - raw2) if abs(tf2 - 1) > 0.2 else 0.0))
            a1, a2 = pc(cr['besth'], cr['cfam']), cr['besth']['T2'] - cr['cfam']['T2']
            f = np.hypot(PF.U_TRANSFER, PF.U_ION['Fe²⁺'])
            x = FL['n3']             # item 11: n = 3 measured in fam - 12f, no transfer factor
            items['n = 3 configurations'] = dict(raw1=x['T1'], raw2=x['T2'], tf1=1.0, tf2=1.0, d1=x['T1'], d2=x['T2'],
                                                 u1=PF.U_LARGE * abs(x['T1']), u2=PF.U_LARGE * abs(x['T2']),
                                                 ras=items['n = 3 configurations'])
            items['fam → besth (f, h)'] = dict(raw1=a1, raw2=a2, tf1=1.0, tf2=1.0, d1=a1, d2=a2, u1=f * abs(a1), u2=f * abs(a2))
            hfe = dict(d1=fe['u3c1']['T1'] - fe['k1fh_n3']['T1'], d2=fe['u3c1']['T2'] - fe['k1fh_n3']['T2'])
            T1s = b['T1'] * (1 + sum(x['d1'] for x in items.values()) / 100)
            T1h, T2 = T1s + hfe['d1'], b['T2'] + sum(x['d2'] for x in items.values()) + hfe['d2']
            for nm, x in items.items():
                bud.append((nm, x['d1'], x['d2'], x['u1'] / 100 * b['T1'], x['u2']))
            tf = 0.0 if R == 'proj' else PF.U_TRANSFER
            bud.append(('hole remainder', hfe['d1'], hfe['d2'], float(np.hypot(PF.U_HOLE[0], tf * hfe['d1'])),
                        float(np.hypot(PF.U_HOLE[1], tf * hfe['d2']))))
            out['fe_items'] = dict(items, hole=hfe)
        pct = sum(c['T1'] for c in ci.values()) + dk['T1']
        T1 = T1h * (1 + pct / 100)
        T2 += sum(c['T2'] for c in ci.values()) + dk['T2']
        for nm, c in ci.items():
            fct = PF.U_TRANSFER if nm in own else np.hypot(PF.U_TRANSFER, PF.U_ION[ion])
            u1 = np.hypot(fct * abs(c['T1']), PF.U_FIX.get(nm, (0, 0))[0]) / 100 * T1h
            u2 = np.hypot(fct * abs(c['T2']), PF.U_FIX.get(nm, (0, 0))[1])
            if ion == 'Fe²⁺' and nm == 'i functions':     # item 11: as phase17_final
                u1 = np.hypot(PF.U_LARGE * FL['i1']['T1'], PF.U_TRANSFER * FL['i2']['T1']) / 100 * T1h
                u2 = np.hypot(PF.U_LARGE * FL['i1']['T2'], PF.U_TRANSFER * FL['i2']['T2'])
            bud.append((nm, c['T1'], c['T2'], u1, u2))
        bud.append(('DK3 (per ion)', dk['T1'], dk['T2'], PF.U_TRANSFER * abs(dk['T1']) / 100 * T1h, PF.U_TRANSFER * abs(dk['T2'])))
        if d3:
            # fit representations: the hole carries a large T3 / T4 change (u3c1 vs k1fh_n3: +0.45 / -0.42 mEh) whose
            # transfer to the large space is untested, so it also gets the +-10 % transfer term of the other items
            tf = 0.0 if R == 'proj' else PF.U_TRANSFER
            bud.append(('hole remainder', ho['T1'], ho['T2'],
                        float(np.sqrt(PF.U_HOLE[0] ** 2 + (uion ** 2 + tf ** 2) * ho['T1'] ** 2)),
                        float(np.sqrt(PF.U_HOLE[1] ** 2 + (uion ** 2 + tf ** 2) * ho['T2'] ** 2))))
        u1 = float(np.sqrt(sum(x[3] ** 2 for x in bud)))
        u2 = float(np.sqrt(sum(x[4] ** 2 for x in bud)))
        m = PF.MEAS[ion]
        r = dict(base={q: b[q] for q in Q}, T1=T1, T2=T2, u1=u1, u2=u2, pct=pct, dk=dk, budget=bud, ratio=T1 / m[0],
                 dev1=(T1 - m[0]) / m[1], dev1c=(T1 - m[0]) / np.hypot(m[1], u1), dev2=(T2 - m[2]) / m[3],
                 dev2c=(T2 - m[2]) / np.hypot(m[3], u2))
        if d3:
            for q in Q[2:]:
                r[q] = BCb[q] + sum(c[q] for c in ci.values()) + dk[q]
        r.update(i=ci['i functions'], hole=ho)
        out['ions'][ion] = r
    return out


def chains():
    res = {R: chain(R) for R in ('proj', 'terms', 'levels')}
    json.dump(res, open(os.path.join(OUT, 'phase17q11_chain.json'), 'w'), indent=1, default=float)
    for R, c in res.items():
        print(f'== {R}: carried', {k: round(v['T1'], 3) for k, v in c['carry'].items()}, 'hole', round(c['hole']['T1'], 3),
              round(c['hole']['T2'], 3))
        for ion, r in c['ions'].items():
            if 'missing' in r:
                print('  ', ion, 'missing', r['missing'][:80])
                continue
            print(f"   {ion}: base {r['base']['T1']:.3f} / {r['base']['T2']:.3f} -> T1 {r['T1']:.3f} ± {r['u1']:.3f} "
                  f"({r['ratio']:.3f}, {r['dev1']:+.1f}σ) T2 {r['T2']:.3f} ± {r['u2']:.3f} ({r['dev2']:+.1f}σ) DK {r['dk']['T1']:+.2f} %"
                  + (f" B {r['B']:.1f} C {r['C']:.1f} beta {r['beta']:+.1f}" if 'B' in r else ''))


def refit():
    """The (1b) NIST refit with T1, T2 fixed at the corrected values of each representation (phase17_bfix logic);
    F-test on the two constraints, and on each alone."""
    import phase15u_bfit as BF
    from scipy.stats import f as fdist
    ch = json.load(open(os.path.join(OUT, 'phase17q11_chain.json')))
    lab = {'cr': 'Cr³⁺', 'v': 'V²⁺', 'mn': 'Mn⁴⁺', 'fe': 'Fe²⁺'}
    res = {}
    for ion, _, fn, N in BF.IONS:
        d = BF.D(N)
        lev = BF.nist(fn, N)
        f0 = bfit_free(ion)
        c0, n = float(np.sum(np.square(f0['resid']))), f0['n']
        nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
        out = dict(free=dict(T1=f0['p']['T1'], T2=f0['p']['T2'], mean_error=f0['rms']))
        for R in ('proj', 'terms', 'levels'):
            r = ch[R]['ions'].get(lab[ion], {})
            if 'T1' not in r:
                continue
            for k, fx in (('both', dict(T1=r['T1'], T2=r['T2'])), ('T1', dict(T1=r['T1'])), ('T2', dict(T2=r['T2']))):
                free = [q for q in nm if q not in fx]
                f = BF.fit(d, lev, free, fx, [f0['p'][q] for q in free])
                c = float(np.sum(np.square(f['resid'])))
                q = len(fx)
                F = (c - c0) / q / (c0 / (n - 8))
                out[f'{R}:{k}'] = dict(fixed=fx, mean_error=f['rms'], F=F, p=float(fdist.sf(F, q, n - 8)),
                                       beta=f['p']['beta'] - f0['p']['beta'], zeta=f['p']['zeta'] - f0['p']['zeta'],
                                       **{f'free_{x}': f['p'][x] for x in ('T1', 'T2') if x in f['p']})
        res[lab[ion]] = out
        print(lab[ion], {k: (round(v['mean_error'], 1), round(v.get('p', 1), 4)) for k, v in out.items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q11_refit.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'check': check, 'd3': d3, 'absorb': absorb, 'd6': d6, 'chain': chains, 'refit': refit}[sys.argv[1]]()
