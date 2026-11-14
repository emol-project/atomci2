#!/usr/bin/env python3
"""Phase 19 (12 Nov): diagnostics for draft v7. Read-only on every committed input; writes only results/phase19/.

  fits   jobs 1-3: level residual tables of the V III, Fe III, Co IV fits (same code path as the paper: phase17q11.fit /
         phase15u_crfit.lsq on phase17q11.DS; F tests as phase17q11.refit and phase18_co.tests), the T1,T2-only test
         for all six ions (phase17q11.refit logic, BF.D), Co IV with the Hansen f vector fixed (phase17q20.fits logic)
  nist   job 4: configurations in data/levels/NIST_ASD_levels_{Co_IV,Fe_VI}.txt (+ Fe III, V III for P1/P2)
  pos    job 5: 3d^(N-1) ns roots in the stored H of the fam (Fe2+, Co3+) and Step 8 (V2+) spaces
         (estimate first: pos est; then pos run <run> ...)
Inputs: results/phase18/phase18_{fe5,fe2,co}.json (paper values, as phase18_tables.py), data/levels/*.txt,
results/<run>.npz (job 5 only)."""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase19')
R18 = os.path.join(ROOT, 'results', 'phase18')
R17 = os.path.join(ROOT, 'results', 'phase17')
LET = 'SPDFGHIKLMN'


def j(*p):
    return json.load(open(os.path.join(*p)))


# ------------------------------------------------------------------ NIST parsing (all configurations)

def nist_all(fn):
    """Every row with a level: [(config, term, 2J, E, leading)] with the Config / Term columns carried forward."""
    cfg = term = None
    col = None
    out = []
    for line in open(os.path.join(_C.REFDATA, 'levels', f'NIST_ASD_levels_{fn}.txt')):
        c = [x.strip() for x in line.split('|')]
        if c[0].startswith('Config'):
            col = {k: next(i for i, h in enumerate(c) if h.startswith(k)) for k in ('Term', 'J', 'Level', 'Leading')}
            continue
        if col is None or len(c) <= col['Level'] or (c[0] and set(c[0]) <= set('-')):
            continue
        if c[0]:
            cfg = c[0]
        if c[col['Term']]:
            term = c[col['Term']]
        lv = c[col['Level']].strip('[]?() ')
        if not lv or not c[col['J']]:
            continue
        try:
            E = float(lv)
        except ValueError:
            continue
        jj = c[col['J']].split(',')[0].split('/')
        try:
            j2 = int(jj[0]) * (1 if len(jj) == 2 else 2)
        except ValueError:
            continue
        out.append((cfg, term, j2, E, c[col['Leading']] if len(c) > col['Leading'] else ''))
    return out


def jstr(j2):
    return f'{j2}/2' if j2 % 2 else str(j2 // 2)


def is_dn(cfg, N):
    return cfg == f'3d{N}'


def is_dn1s(cfg, N, n=None):
    """3d^(N-1) ns (parent term in parentheses allowed)."""
    m = re.fullmatch(rf'3d{N - 1}(\.\([^)]*\))?\.(\d+)s', cfg)
    return bool(m) and (n is None or int(m.group(2)) == n)


def nist():
    res = {}
    for fn, N in (('Co_IV', 6), ('Fe_VI', 3), ('Fe_III', 6), ('V_III', 3)):
        rows = nist_all(fn)
        cfgs = {}
        for c, t, j2, E, _ in rows:
            cfgs.setdefault(c, []).append(E)
        dn = [r for r in rows if is_dn(r[0], N)]
        s4 = [r for r in rows if is_dn1s(r[0], N, 4)]
        r = dict(n_rows=len(rows), configs={c: dict(n=len(v), Emin=min(v), Emax=max(v)) for c, v in cfgs.items()},
                 n_dN=len(dn), highest_dN=max(dn, key=lambda x: x[3]) if dn else None,
                 lowest_dN1_4s=min(s4, key=lambda x: x[3]) if s4 else None, dN1_4s=s4)
        res[fn] = r
        print(f'== {fn}: {len(rows)} levels, {len(cfgs)} configurations; 3d{N} {len(dn)} levels, highest {r["highest_dN"]}; '
              f'lowest 3d{N - 1}4s {r["lowest_dN1_4s"]}')
    return res


# ------------------------------------------------------------------ fits

def paper_values():
    """T1, T2 (proj, (1b) repr.) and the extra sets exactly as phase18_tables.py reads them."""
    fe5, fe2, co = j(R18, 'phase18_fe5.json'), j(R18, 'phase18_fe2.json'), j(R18, 'phase18_co.json')
    T = {}
    for lab, ion in (('V²⁺', 'v'), ('Cr³⁺', 'cr'), ('Mn⁴⁺', 'mn')):
        T[ion] = {R: dict(T1=fe5['series'][k][lab]['T1'], T2=fe5['series'][k][lab]['T2']) for R, k in (('proj', 'proj'), ('levels', 'levels'))}
    T['fe5'] = {R: dict(T1=fe5['chain'][R]['T1'], T2=fe5['chain'][R]['T2']) for R in ('proj', 'levels')}
    X = {}
    import phase17q12 as Q12
    for ion, d in (('fe', fe2), ('co', co)):
        T[ion] = {R: dict(T1=d['final'][R]['final']['T1'], T2=d['final'][R]['final']['T2']) for R in ('proj', 'levels')}
        X[ion] = dict(corrected={k: d['final']['proj']['final'][k] for k in Q12.EXTRA},
                      fam={k: d['fam_sets']['fam']['proj'][k] for k in Q12.EXTRA})
    return T, X, dict(fe2=fe2, co=co, fe5=fe5)


def labels(ion, fn, N):
    """NIST (term, leading) per obs entry of phase17q11.nist_obs (sorted by J, then energy)."""
    import phase15u_bfit as BF
    lev = BF.nist(fn, N)
    byj = {}
    for x in sorted(lev, key=lambda x: x[3]):
        byj.setdefault(x[2], []).append(x)
    return [(x[0], x[1]) for j2, xs in sorted(byj.items()) for x in xs]


def near_s(ion_fn, N, j2, E):
    """Nearest NIST 3d^(N-1) ns level of the same J (any n): (dE = E_s - E, config, term, E_s)."""
    rows = [r for r in nist_all(ion_fn) if is_dn1s(r[0], N) and r[2] == j2]
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(r[3] - E))
    return dict(dE=r[3] - E, cfg=r[0], term=r[1], E=r[3])


def table(ion, fn, N, obs, f, lab):
    rows = []
    for (j2, k, E), r, (t, lead) in zip(obs, f['resid'], lab):
        rows.append(dict(cfg=f'3d{N}', term=t, J=jstr(j2), j2=j2, E_obs=E, E_calc=E - r, resid=r, lead=lead,
                         near_4s=near_s(fn, N, j2, E)))
    return sorted(rows, key=lambda x: -abs(x['resid']))


def fits():
    from scipy.stats import f as fdist
    import phase15u_bfit as BF
    import phase17q11 as Q
    import phase17q12 as Q12
    import phase17q20 as Q20
    import phase18_nist  # noqa: F401  (registers fe5, co)
    T, X, D = paper_values()
    FS = [f'f{i}' for i in range(1, 13)]
    res = dict(job1={}, job2={}, job3={})

    def nested(c, c0, q, dof):
        F = (c - c0) / q / (c0 / dof)
        return float(F), float(fdist.sf(F, q, dof))

    # ---------------- job 1: residual tables
    for ion, fn, N, sets in (('v', 'V_III', 3, None), ('fe', 'Fe_III', 6, X['fe']), ('co', 'Co_IV', 6, X['co'])):
        obs, N_ = Q.nist_obs(ion)
        assert N_ == N
        d = Q.ds(N)
        p0 = Q.to_bc(Q.bfit_free(ion)['p'])
        n = len(obs)
        lab = labels(ion, fn, N)
        assert len(lab) == n

        def run(names, fixed):
            f = Q.fit(d, obs, d.levels, names, fixed, [p0.get(q, 1.0) for q in names])
            f['order_ok'] = Q12._order_ok(f, obs)
            return f, float(np.sum(np.square(f['resid'])))
        f0, c0 = run(Q.NM1B, {})
        out = {}
        out['free (1b)'] = dict(fit=f0, p=None, ptest='reference', table=table(ion, fn, N, obs, f0, lab))
        free12 = [q for q in Q.NM1B if q not in ('T1', 'T2')]
        for R in ('proj', 'levels'):
            fx = T[ion][R]
            f1, c1 = run(free12, fx)
            F, p = nested(c1, c0, 2, n - 8)
            out[f'T1,T2 fixed ({R})'] = dict(fit=f1, fixed=fx, F=F, p=p, ptest='F(2, n-8) vs free (1b)', table=table(ion, fn, N, obs, f1, lab))
        if sets:
            for sn in ('fam', 'corrected'):
                Xs = sets[sn]
                fX, cX = run(Q.NM1B, dict(Xs))
                d.ops['Xab'] = sum(c * d.ops[k] for k, c in Xs.items())
                fl, cl = run(Q.NM1B + ['Xab'], {})
                del d.ops['Xab']
                dl = n - len(Q.NM1B) - 1
                F1, p1 = nested(cX, cl, 1, dl)
                F0, p0_ = nested(c0, cl, 1, dl)
                out[f'T3,T4,f fixed ({sn})'] = dict(fit=fX, fixed='T3,T4,f1..f12 ' + sn, lam=fl['p']['Xab'], lam_err=fl['err']['Xab'],
                                                    F=F1, p=p1, p_lam0=p0_, ptest='p(lam = 1), F(1, n-9) vs lam free',
                                                    table=table(ion, fn, N, obs, fX, lab))
                for R in ('proj', 'levels'):
                    f2, c2 = run(free12, dict(Xs, **T[ion][R]))
                    F2, p2 = nested(c2, cX, 2, n - 8)
                    out[f'T3,T4,f ({sn}) + T1,T2 ({R}) fixed'] = dict(fit=f2, F=F2, p=p2, ptest='F(2, n-8) vs X fixed (phase18_co p_12_given_X)',
                                                                      table=table(ion, fn, N, obs, f2, lab))
        res['job1'][ion] = out
        for k, v in out.items():
            print(f"{ion:3s} {k:42s} n {n} mean dev {v['fit']['rms']:.2f} order_ok {v['fit']['order_ok']} p {v.get('p')} "
                  f"largest {v['table'][0]['term']} J={v['table'][0]['J']} {v['table'][0]['E_obs']:.0f} {v['table'][0]['resid']:+.1f}", flush=True)

    # consistency with the committed values (phase18_co / fe2 refit_final; same code path)
    chk = {}
    for ion, dd in (('fe', D['fe2']), ('co', D['co'])):
        rf = dd['refit_final']
        o = res['job1'][ion]
        chk[ion] = dict(free=(o['free (1b)']['fit']['rms'], rf['proj']['free']['mean_error']),
                        fam=(o['T3,T4,f fixed (fam)']['fit']['rms'], rf['proj']['fam']['mean_error']),
                        corrected=(o['T3,T4,f fixed (corrected)']['fit']['rms'], rf['proj']['final']['mean_error']),
                        lam_fam=(o['T3,T4,f fixed (fam)']['lam'], rf['proj']['fam']['lam']),
                        fixX12_proj_fam=(o['T3,T4,f (fam) + T1,T2 (proj) fixed']['fit']['rms'], rf['proj']['fam'].get('fixX12_mean_error')),
                        fixX12_proj_corr=(o['T3,T4,f (corrected) + T1,T2 (proj) fixed']['fit']['rms'], rf['proj']['final'].get('fixX12_mean_error')),
                        p12_proj_corr=(o['T3,T4,f (corrected) + T1,T2 (proj) fixed']['p'], rf['proj']['final'].get('p_12_given_X')))
    res['check_vs_committed'] = chk
    print('check vs committed:', json.dumps(chk, default=float))

    # ---------------- job 2: T1,T2-only fixed test, six ions, phase17q11.refit logic (BF.D, F2/F4 parametrization)
    q11 = j(R17, 'phase17q11_refit.json')
    fe5r = D['fe5']['refit']
    for ion, fn, N, lab in (('v', 'V_III', 3, 'V²⁺'), ('cr', 'Cr_IV', 3, 'Cr³⁺'), ('mn', 'Mn_V', 3, 'Mn⁴⁺'),
                            ('fe5', 'Fe_VI', 3, 'Fe⁵⁺'), ('fe', 'Fe_III', 6, 'Fe²⁺'), ('co', 'Co_IV', 6, 'Co³⁺')):
        d = BF.D(N)
        lev = BF.nist(fn, N)
        f0 = Q.bfit_free(ion)
        c0, n = float(np.sum(np.square(f0['resid']))), f0['n']
        nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
        out = dict(n=n, free=dict(mean_error=f0['rms'], T1=f0['p']['T1'], T1_err=f0['err']['T1'], T2=f0['p']['T2'], T2_err=f0['err']['T2']))
        for R in ('proj', 'levels'):
            fx = T[ion][R]
            free = [q for q in nm if q not in fx]
            f = BF.fit(d, lev, free, fx, [f0['p'][q] for q in free])
            c = float(np.sum(np.square(f['resid'])))
            F, p = nested(c, c0, 2, n - 8)
            out[R] = dict(fixed=fx, mean_error=f['rms'], F=F, p=p, order_ok=bool(f['order_ok']))
        # committed p of the same test
        if lab in q11:
            out['committed'] = {R: dict(p=q11[lab].get(f'{R}:both', {}).get('p'), fixed=q11[lab].get(f'{R}:both', {}).get('fixed'))
                                for R in ('proj', 'levels')}
        elif ion == 'fe5':
            out['committed'] = {R: dict(p=fe5r.get(f'{R}:both', {}).get('p'), fixed=fe5r.get(f'{R}:both', {}).get('fixed')) for R in ('proj', 'levels')}
        res['job2'][lab] = out
        print(f"T1,T2 fixed {lab}: n {n} free {f0['rms']:.2f}; proj {out['proj']['mean_error']:.2f} p {out['proj']['p']:.3g}; "
              f"(1b) repr. {out['levels']['mean_error']:.2f} p {out['levels']['p']:.3g}; committed {out.get('committed')}", flush=True)

    # ---------------- job 3: Co IV with the Hansen f vector fixed (phase17q20.fits logic; also rerun for Fe III)
    A = Q20.jlA()
    for ion, hf, Xion in (('fe', {k: float(Q20.H_FE[k][0]) for k in FS}, X['fe']),
                          ('co', {int(k[1:]) if False else k: float(v[0]) for k, v in D['co']['hansen_f'].items()}, X['co'])):
        obs, N = Q.nist_obs(ion)
        d = Q.ds(N)
        p0 = Q.to_bc(Q.bfit_free(ion)['p'])
        x0 = [p0[n] for n in Q.NM1B]
        Xh = {k: hf[k] / np.sqrt(A[k]) for k in FS}
        fam_f = {k: Xion['fam'][k] for k in FS}
        o = {}
        o['free'] = Q20.wlsq(d, Q.NM1B, {}, obs, x0)
        o['Hansen f fixed (T3 = T4 = 0)'] = Q20.wlsq(d, Q.NM1B, Xh, obs, x0)
        o['fam f fixed (T3 = T4 = 0)'] = Q20.wlsq(d, Q.NM1B, fam_f, obs, x0)
        o['fam T3,T4,f fixed'] = Q20.wlsq(d, Q.NM1B, dict(Xion['fam']), obs, x0)
        lam = {}
        for nmx, XX in (('Hansen f', Xh), ('fam f', fam_f)):
            d.ops['Xab'] = sum(c * d.ops[k] for k, c in XX.items())
            fl = Q20.wlsq(d, Q.NM1B + ['Xab'], {}, obs, x0 + [1.0])
            del d.ops['Xab']
            lam[nmx] = dict(lam=fl['p']['Xab'], err=fl['err']['Xab'], mean=fl['mean'])
        o['lam'] = lam
        # batches as phase17q20.fits: six large f free, the rest held at theirs (T3 = T4 = 0: no Co IV p_t, p_t' on disk;
        # for Fe III their p_t, p_t' as in phase17q20) or ours (fam)
        LARGE = ['f1', 'f3', 'f4', 'f5', 'f8', 'f12']
        ref = d.levels(sum(v * d.ops[k] for k, v in {**o['Hansen f fixed (T3 = T4 = 0)']['p'], **Xh}.items()))
        if ion == 'fe':
            th = {**Xh, 'T4': Q20.H_P['fe'][2][0] * Q20.MEH_PER_P, 'T3': Q20.H_P['fe'][3][0] * Q20.MEH_PER_P}
        else:
            th = {**Xh, 'T3': 0.0, 'T4': 0.0}
        st_th = {**o['Hansen f fixed (T3 = T4 = 0)']['p'], **th}
        st_ou = {**o['fam f fixed (T3 = T4 = 0)']['p'], **Xion['fam']}
        bt = {}
        for key, held, x0b in (('large free, rest theirs', {k: v for k, v in th.items() if k not in LARGE}, st_th),
                               ('large free, rest ours (fam)', {k: Xion['fam'][k] for k in Q12.EXTRA if k not in LARGE}, st_ou),
                               ('f4, f12 free, rest theirs', {k: v for k, v in th.items() if k not in ('f4', 'f12')}, st_th),
                               ('f4, f12 free, rest ours (fam)', {k: Xion['fam'][k] for k in Q12.EXTRA if k not in ('f4', 'f12')}, st_ou)):
            free = LARGE if key.startswith('large') else ['f4', 'f12']
            names = Q.NM1B + free
            f = Q20.tr_fit(d, names, held, obs, {k: x0b[k] for k in names}, None, ref)
            f['fnorm'] = {k: f['p'][k] * np.sqrt(A[k]) for k in free}
            f['fnorm_err'] = {k: f['err'][k] * np.sqrt(A[k]) for k in free}
            bt[key] = f
        o['batch'] = bt
        res['job3'][ion] = o
        print(f"== {ion}: free {o['free']['mean']:.2f}; Hansen f fixed {o['Hansen f fixed (T3 = T4 = 0)']['mean']:.2f} "
              f"(order {o['Hansen f fixed (T3 = T4 = 0)']['order_ok']}); fam f fixed {o['fam f fixed (T3 = T4 = 0)']['mean']:.2f}; "
              f"fam T3,T4,f {o['fam T3,T4,f fixed']['mean']:.2f}; lam {lam}")
        for k, f in bt.items():
            print(f"   {k:32s} mean {f['mean']:.2f} " + ' '.join(f"{q} {f['fnorm'][q]:+.0f}({f['fnorm_err'][q]:.0f})" for q in f['fnorm']))
    return res


def single():
    """job 6: T1 alone, T2 alone, both fixed at the corrected computed values (proj, (1b) repr.), F(q, n-8) against the
    fit with T1, T2 free. Plain (1b): phase17q11.refit logic (BF.D). Fe III, Co IV also with T3, T4, f1..f12 fixed
    (fam and corrected sets; phase18_co.tests logic on phase17q11.DS, reference = that set fixed, T1, T2 free)."""
    from scipy.stats import f as fdist
    import phase15u_bfit as BF
    import phase17q11 as Q
    import phase17q12 as Q12
    import phase18_nist  # noqa: F401
    T, X, D = paper_values()
    res = {}
    cases = (('a: T1 fixed', ('T1',)), ('b: T2 fixed', ('T2',)), ('c: both fixed', ('T1', 'T2')))

    def block(run, f0, n, ion):
        c0 = float(np.sum(np.square(f0['resid'])))
        out = dict(ref=dict(mean_error=f0['rms'], T1=f0['p']['T1'], T1_err=f0['err']['T1'], T2=f0['p']['T2'], T2_err=f0['err']['T2']))
        for R in ('proj', 'levels'):
            for cn, ks in cases:
                fx = {k: T[ion][R][k] for k in ks}
                f = run(fx)
                c = float(np.sum(np.square(f['resid'])))
                F = (c - c0) / len(ks) / (c0 / (n - 8))
                fr = {k: (f['p'][k], f['err'][k]) for k in ('T1', 'T2') if k in f['p']}
                out[f'{R}|{cn}'] = dict(fixed=fx, mean_error=f['rms'], F=F, p=float(fdist.sf(F, len(ks), n - 8)), free=fr)
        return out
    nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
    for ion, fn, N, lab in (('v', 'V_III', 3, 'V III'), ('cr', 'Cr_IV', 3, 'Cr IV'), ('mn', 'Mn_V', 3, 'Mn V'),
                            ('fe5', 'Fe_VI', 3, 'Fe VI'), ('fe', 'Fe_III', 6, 'Fe III'), ('co', 'Co_IV', 6, 'Co IV')):
        d = BF.D(N)
        lev = BF.nist(fn, N)
        f0 = Q.bfit_free(ion)

        def run(fx, d=d, lev=lev, f0=f0):
            free = [q for q in nm if q not in fx]
            return BF.fit(d, lev, free, fx, [f0['p'][q] for q in free])
        res[f'{lab} | (1b)'] = block(run, f0, f0['n'], ion)
    for ion, lab in (('fe', 'Fe III'), ('co', 'Co IV')):
        obs, N = Q.nist_obs(ion)
        d = Q.ds(N)
        p0 = Q.to_bc(Q.bfit_free(ion)['p'])
        for sn in ('fam', 'corrected'):
            Xs = X[ion][sn]

            def run(fx, d=d, obs=obs, Xs=Xs):
                free = [q for q in Q.NM1B if q not in fx]
                return Q.fit(d, obs, d.levels, free, dict(Xs, **fx), [p0[q] for q in free])
            f0 = Q.fit(d, obs, d.levels, Q.NM1B, dict(Xs), [p0[q] for q in Q.NM1B])
            res[f'{lab} | T3,T4,f fixed ({sn})'] = block(run, f0, len(obs), ion)
    for k, v in res.items():
        print(k, f"ref {v['ref']['mean_error']:.2f}", ' '.join(f"{kk}: {x['mean_error']:.2f} p {x['p']:.3g}" for kk, x in v.items() if kk != 'ref'), flush=True)
    return res


def strip(x):
    """json-safe, drop big arrays (cov, levels)."""
    if isinstance(x, dict):
        return {str(k): strip(v) for k, v in x.items() if not (k in ('cov', 'levels', 'sv') and isinstance(v, list))}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ------------------------------------------------------------------ job 5

RUNS = {'fe': ('v15u_fam_d5f2g2h1_d6_{t}e_fend6', 6, ['5D']), 'co': ('v15u_fam_d5f2g2h1_d6_{t}e_cond6', 6, ['5D']),
        'v': ('v15u_besth_sall_d3_{t}e_vnd3', 3, ['4F', '2D'])}
WANT = {'fe': ['7S', '5S', '5D'], 'co': ['7S', '5S', '5D'], 'v': ['4F', '2D']}
EH_CM = 219474.6313705


def classify(labels, N):
    """per CSF: 'P' (3d^N, closed 1s2 2s2 3s2 2p6 3p6), 'ns:<orb>' (3d^(N-1) ns1 with the same core), else 'other'."""
    core = '1s2 2s2 3s2 2p6 3p6'
    out = []
    for lab in labels:
        t = lab.split()
        rest = [x for x in t if x not in core.split()]
        ok_core = all(x in t for x in core.split())
        if ok_core and rest == [f'3d{N}']:
            out.append('P')
        elif ok_core and len(rest) == 2 and f'3d{N - 1}' in rest:
            o = [x for x in rest if x != f'3d{N - 1}'][0]
            out.append(f'ns:{o[:-1]}' if re.fullmatch(r'\d+s1', o) else 'other')
        else:
            out.append('other')
    return np.array(out)


def pos(args):
    from atomci_heff.sparse import SymLower, davidson
    import shutil
    mode = args[0]
    todo = []
    for ion, (fmt, N, ts) in RUNS.items():
        for t in WANT[ion]:
            name = fmt.format(t=t)
            p = os.path.join(ROOT, 'results', name + '.npz')
            todo.append((ion, t, name, N, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else 0))
    if mode == 'est':
        for x in todo:
            print(x)
        return
    names = args[1:]
    res = json.load(open(os.path.join(OUT, 'phase19_pos.json'))) if os.path.exists(os.path.join(OUT, 'phase19_pos.json')) else {}
    for ion, t, name, N, ex, sz in todo:
        if names and name not in names:
            continue
        if not ex:
            res[f'{ion}:{t}'] = dict(run=name, status='no stored H (term block not in the space)')
            continue
        free = shutil.disk_usage('/data').free / 2 ** 30
        assert free > 150, f'/data free {free:.0f} GB < 150 GB'
        t0 = time.time()
        z = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
        op = SymLower.from_npz(z)
        lab = [str(x) for x in z['labels']]
        cls = classify(lab, N)
        P = np.where(cls == 'P')[0]
        nsorb = sorted({c for c in cls if c.startswith('ns:')}, key=lambda c: -int(c[3:-1]))
        c = float(op.d[P].mean())
        s = SymLower.__new__(SymLower)
        s.L, s.LT, s.d, s.n = op.L, op.LT, op.d - c, op.n
        k = 8
        X0 = None
        found = None
        while True:
            q = list(np.argsort(s.d)[:k])
            X0n = np.zeros((op.n, k))
            X0n[q, np.arange(k)] = 1.0
            if X0 is not None:
                X0n[:, :X0.shape[1]] = X0
            E, C, rn, it, nmv = davidson(s, k, X0=X0n, tol=1e-8, maxit=4000, max_sub=max(6 * k, 60))
            assert rn.max() < 1e-8, rn.max()
            w = {'P': (C[P] ** 2).sum(axis=0)}
            for o in nsorb:
                w[o] = (C[cls == o] ** 2).sum(axis=0)
            wns = sum(w[o] for o in nsorb)
            woth = 1 - w['P'] - wns
            cand = [a for a in range(k) if w['P'][a] < 0.5 and wns[a] > max(w['P'][a], woth[a])]
            if cand:
                found = cand[0]
                break
            X0 = C
            k *= 2
            assert k <= 128, 'no 3d^(N-1) ns root among 64 roots'
            assert time.time() - t0 < 7200, 'block > 2 h'
        e = (E - E[0]) * EH_CM
        roots = [dict(i=a, E_cm=float(e[a]), wP=float(w['P'][a]), w_ns={o[3:]: float(w[o][a]) for o in nsorb},
                      w_other=float(woth[a])) for a in range(k)]
        res[f'{ion}:{t}'] = dict(run=name, N_csf=int(op.n), nP=int(len(P)), k=k, found=found, E_found_cm=float(e[found]),
                                 E_abs_Eh=[float(x + c) for x in E],
                                 roots=roots, wall_s=time.time() - t0, matvecs=int(nmv))
        print(f'{ion}:{t} {name}: N {op.n} P {len(P)} k {k} found root {found} at {e[found]:.0f} cm-1 above the lowest root of the '
              f'block; wP {w["P"][found]:.3f} ns {wns[found]:.3f} ({time.time() - t0:.0f} s)', flush=True)
        json.dump(res, open(os.path.join(OUT, 'phase19_pos.json'), 'w'), indent=1)
        del z, op, s


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'nist':
        r = nist()
        json.dump(strip(r), open(os.path.join(OUT, 'phase19_nist.json'), 'w'), indent=1)
    elif cmd == 'fits':
        r = fits()
        json.dump(strip(r), open(os.path.join(OUT, 'phase19_fits.json'), 'w'), indent=1)
    elif cmd == 'single':
        r = single()
        json.dump(strip(r), open(os.path.join(OUT, 'phase19_single.json'), 'w'), indent=1)
    elif cmd == 'pos':
        pos(sys.argv[2:])
