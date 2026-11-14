#!/usr/bin/env python3
"""Phase 21 jobs B, E, F (no ATOMCI, no stored-H runs; seconds to minutes). Writes only results/phase21/.

  B  fits on LS term energies, six spectra. From the free (1b) level fit (phase17q11.fit on phase17q11.DS, the paper's
     path), every model J level is assigned to an LS term of the zeta = 0 model (H0 = H_fit - zeta l.s) by maximum
     overlap within each J (Hungarian); the zeta shift of a term is its (2J+1)-weighted model centroid minus its H0
     energy. Observed NIST levels go to the model level they are fitted to (equal calculated energy), so each NIST term
     label maps to one model term; zeta-corrected term energy = observed (2J+1) centroid - zeta shift. Terms with every
     J observed only. Fit: A (E_av), B, C (F2 = 49 B + 7 C, F4 = 441 C / 35), alpha, beta, T1, T2, unit weights
     (phase15u_crfit.lsq on DS.terms; mean deviation = sqrt(sum r^2 / dof) as for the level fits). Tests: nested F
     against the free term fit. d6 also: T3, T4, f fixed (fam, corrected), lambda, p(lambda = 0), p(lambda = 1),
     Hansen vector fixed (f only, and f + p_t, p_t'). Level-fit numbers side by side from phase19_fits.json and
     phase19_single.json (same code path).
  E  parameter table, six ions
  F  term energies of Cr3+ and Fe2+: computed (base H_eff of the paper: Cr3+ Step 10, Fe2+ fam; projection = H_eff
     itself, (1b) representation = the (1b) fit of its pseudo-levels with zeta then set to 0) vs zeta-corrected
     observed (B)
-> phase21b.json, phase21b_tables.md"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase21')
R15, R17, R18, R19 = (os.path.join(ROOT, 'results', p) for p in ('phase15', 'phase17', 'phase18', 'phase19'))
P_MEH = 1.83813
LET = 'SPDFGHIKLMN'
SIX = [('v', 'V III', 'V_III', 3, 'V²⁺'), ('cr', 'Cr IV', 'Cr_IV', 3, 'Cr³⁺'), ('mn', 'Mn V', 'Mn_V', 3, 'Mn⁴⁺'),
       ('fe5', 'Fe VI', 'Fe_VI', 3, 'Fe⁵⁺'), ('fe', 'Fe III', 'Fe_III', 6, 'Fe²⁺'), ('co', 'Co IV', 'Co_IV', 6, 'Co³⁺')]
HP = {'fe': (0.013, -0.013), 'co': (0.027, -0.014)}      # Hansen et al. 1988 Table 1 p_t, p_t' (T4, T3 = x 1.83813)


def j(*p):
    return json.load(open(os.path.join(*p)))


def tname(key, k, nk):
    m, L = key
    return f'{m}{LET[L]}' + (f'({k + 1})' if nk > 1 else '')


def term_assign(d, H, Hz):
    """{j2: [(term key, k) per level of Hz in that J, energy order]}, H0 term energies {(key, k): E}."""
    from scipy.optimize import linear_sum_assignment
    tm = d.terms(H)
    E0 = {(key, k): float(e) for key, (w, v, idx) in tm.items() for k, e in enumerate(w)}
    eps = 1e-7          # H0 is J-degenerate within a term: eps J^2 makes its eigenvectors J-pure
    lz, l0 = d.levels(Hz), d.levels(H + eps * d.J2)
    out = {}
    for j2 in lz:
        wz, vz, idx = lz[j2]
        w0, v0, idx0 = l0[j2]
        assert list(idx) == list(idx0) and len(w0) == len(wz), (j2, len(w0), len(wz))
        # identify each H0 J-state with a term by energy (H0 is block diagonal in LS; distinct term energies)
        lab0 = []
        for e in w0 - eps * (j2 / 2) * (j2 / 2 + 1):
            kk = min(E0, key=lambda t: abs(E0[t] - e))
            assert abs(E0[kk] - e) < 1e-6, (j2, e, E0[kk])
            lab0.append(kk)
        ov = np.abs(vz.T @ v0) ** 2
        r, c = linear_sum_assignment(-ov)
        assert list(r) == list(range(len(wz)))
        out[j2] = [(lab0[c[i]], float(ov[i, c[i]])) for i in range(len(wz))]
    return out, E0


def nested(c, c0, q, dof):
    from scipy.stats import f as fdist
    F = (c - c0) / q / (c0 / dof)
    return float(F), float(fdist.sf(F, q, dof))


def paper():
    import phase19 as P19
    T, X, D = P19.paper_values()
    return T, X, D


def zeta_terms(ion, fn, N):
    """observed and zeta-corrected term energies from the free (1b) level fit."""
    import phase17q11 as Q
    import phase19 as P19
    obs, N_ = Q.nist_obs(ion)
    d = Q.ds(N)
    p0 = Q.to_bc(Q.bfit_free(ion)['p'])
    f0 = Q.fit(d, obs, d.levels, Q.NM1B, {}, [p0[q] for q in Q.NM1B])
    p = f0['p']
    Hz = Q.model_H(d, p)
    H0 = Hz - p['zeta'] * d.ops['zeta']
    asg, E0 = term_assign(d, H0, Hz)
    lz = d.levels(Hz)
    # model centroids
    acc = {}
    for j2, lst in asg.items():
        for i, (t, w) in enumerate(lst):
            a = acc.setdefault(t, [0.0, 0.0, []])
            a[0] += (j2 + 1) * float(lz[j2][0][i])
            a[1] += j2 + 1
            a[2].append((j2, round(w, 4)))
    shift = {t: a[0] / a[1] - E0[t] for t, a in acc.items()}
    minw = min(w for lst in asg.values() for _, w in lst)
    # observed levels -> model level (calc = E - resid equals a model level of that J)
    lab = P19.labels(ion, fn, N)
    byterm = {}
    for (j2, k, E), r, (tl, lead) in zip(obs, f0['resid'], lab):
        calc = E - r
        i = int(np.argmin(abs(lz[j2][0] - calc)))
        assert abs(lz[j2][0][i] - calc) < 1e-6
        t = asg[j2][i][0]
        b = byterm.setdefault(tl, dict(levels=[], terms=set()))
        b['levels'].append((j2, E))
        b['terms'].add(t)
    rows = {}
    excluded = []
    for tl, b in byterm.items():
        assert len(b['terms']) == 1, (ion, tl, b['terms'])
        t = b['terms'].pop()
        js = sorted(j2 for j2, _ in b['levels'])
        need = sorted(j2 for j2, lst in asg.items() for (tt, w) in lst if tt == t)
        if js != need:
            excluded.append(dict(label=tl, J_obs=js, J_all=need))
            continue
        c = sum((j2 + 1) * E for j2, E in b['levels']) / sum(j2 + 1 for j2, _ in b['levels'])
        rows[t] = dict(label=tl, obs_centroid=c, zeta_shift=shift[t], corrected=c - shift[t], J=js)
    missing = [tname(t[0], t[1], sum(1 for x in E0 if x[0] == t[0])) for t in E0 if t not in rows]
    return dict(fit=f0, p=p, rows=rows, excluded=excluded, missing_terms=missing, min_overlap=minw, E0=E0, shift=shift)


def term_fits(ion, N, z, T, X):
    import phase17q11 as Q
    import phase17q20 as Q20
    d = Q.ds(N)
    obs = [(t[0], t[1], r['corrected']) for t, r in sorted(z['rows'].items())]
    n = len(obs)
    p0 = z['p']
    names = Q.MODEL

    def run(nm, fixed, extra=None):
        x0 = [p0.get(q, 1.0) if q != 'Xab' else 1.0 for q in nm]
        f = Q.fit(d, obs, d.terms, nm, fixed, x0)
        return f, float(np.sum(np.square(f['resid'])))
    f0, c0 = run(names, {})
    dof0 = n - len(names)
    out = dict(n=n, dof=dof0, free=dict(mean=f0['rms'], p=f0['p'], err=f0['err'], resid=f0['resid']))
    for R in ('proj', 'levels'):
        fx = T[ion][R]
        for tag, fixed in (('both', dict(T1=fx['T1'], T2=fx['T2'])), ('T1', dict(T1=fx['T1'])), ('T2', dict(T2=fx['T2']))):
            nm = [q for q in names if q not in fixed]
            f, c = run(nm, fixed)
            F, pv = nested(c, c0, len(fixed), dof0)
            out[f'{R}:{tag}'] = dict(fixed=fixed, mean=f['rms'], F=F, p=pv, free={q: (f['p'][q], f['err'][q]) for q in ('T1', 'T2') if q in nm})
    if N == 6:
        FS = [f'f{i}' for i in range(1, 13)]
        A = Q20.jlA()
        hf = ({k: float(Q20.H_FE[k][0]) for k in FS} if ion == 'fe' else
              {k: float(v[0]) for k, v in j(R18, 'phase18_co.json')['hansen_f'].items()})
        Xh = {k: hf[k] / np.sqrt(A[k]) for k in FS}
        pt, ptp = HP[ion]
        sets = dict(fam=dict(X[ion]['fam']), corrected=dict(X[ion]['corrected']), hansen_f=Xh,
                    hansen_f_pt=dict(Xh, T4=pt * P_MEH, T3=ptp * P_MEH))
        dl = n - len(names) - 1
        for sn, Xs in sets.items():
            fX, cX = run(names, dict(Xs))
            d.ops['Xab'] = sum(c * d.ops[k] for k, c in Xs.items())
            fl, cl = run(names + ['Xab'], {})
            del d.ops['Xab']
            F1, p1 = nested(cX, cl, 1, dl)
            F0, pl0 = nested(c0, cl, 1, dl)
            out[f'X:{sn}'] = dict(mean=fX['rms'], gain=f0['rms'] / fX['rms'], lam=fl['p']['Xab'], lam_err=fl['err']['Xab'],
                                  lam_mean=fl['rms'], p_lam0=pl0, p_lam1=p1, T1=(fX['p']['T1'], fX['err']['T1']),
                                  T2=(fX['p']['T2'], fX['err']['T2']))
    return out


def level_side(ion, lab, fits19, single19, ionname):
    o = {}
    j2 = fits19['job2'][lab]
    o['free_mean'] = j2['free']['mean_error']
    o['n'] = j2['n']
    o['T1'], o['T2'] = (j2['free']['T1'], j2['free']['T1_err']), (j2['free']['T2'], j2['free']['T2_err'])
    for R in ('proj', 'levels'):
        o[f'{R}:both'] = j2[R]['p']
        s = single19.get(f'{ionname} | (1b)', {})
        for tag, key in (('T1', 'a: T1 fixed'), ('T2', 'b: T2 fixed')):
            if f'{R}|{key}' in s:
                o[f'{R}:{tag}'] = s[f'{R}|{key}']['p']
    if ion in ('fe', 'co'):
        j1 = fits19['job1'][ion]
        o['free_mean_BF'] = o['free_mean']
        o['free_mean'] = j1['free (1b)']['fit']['rms']      # same DS path as the X rows below (job 1)
        o['X'] = {sn: dict(mean=j1[f'T3,T4,f fixed ({sn})']['fit']['rms'], lam=j1[f'T3,T4,f fixed ({sn})']['lam'],
                           lam_err=j1[f'T3,T4,f fixed ({sn})']['lam_err'], p_lam0=j1[f'T3,T4,f fixed ({sn})']['p_lam0'],
                           p_lam1=j1[f'T3,T4,f fixed ({sn})']['p']) for sn in ('fam', 'corrected')}
        o['X']['hansen_f'] = dict(mean=fits19['job3'][ion]['Hansen f fixed (T3 = T4 = 0)']['mean'],
                                  lam=fits19['job3'][ion]['lam']['Hansen f']['lam'], lam_err=fits19['job3'][ion]['lam']['Hansen f']['err'])
        o['level_free_mean_wlsq'] = fits19['job3'][ion]['free']['mean']
    return o


def jobB():
    import phase18_nist  # noqa: F401
    T, X, D = paper()
    fits19, single19 = j(R19, 'phase19_fits.json'), j(R19, 'phase19_single.json')
    res = {}
    for ion, name, fn, N, lab in SIX:
        z = zeta_terms(ion, fn, N)
        tf = term_fits(ion, N, z, T, X)
        res[ion] = dict(name=name, label=lab, N=N,
                        terms={tname(t[0], t[1], sum(1 for x in z['E0'] if x[0] == t[0])): r for t, r in sorted(z['rows'].items())},
                        excluded=z['excluded'], missing_terms=z['missing_terms'], min_overlap=z['min_overlap'],
                        level_free=dict(mean=z['fit']['rms'], p=z['p']), term_fit=tf,
                        level=level_side(ion, lab, fits19, single19, name))
        f = tf['free']
        print(f"{name}: {tf['n']} terms (dof {tf['dof']}), excluded {[x['label'] for x in z['excluded']]}, missing {z['missing_terms']}, "
              f"min overlap {z['min_overlap']:.3f}; mean {f['mean']:.2f}, T1 {f['p']['T1']:.3f} ± {f['err']['T1']:.3f}, "
              f"T2 {f['p']['T2']:.3f} ± {f['err']['T2']:.3f} | p both proj {tf['proj:both']['p']:.3g} levels {tf['levels:both']['p']:.3g}; "
              f"T1 {tf['proj:T1']['p']:.3g}/{tf['levels:T1']['p']:.3g}; T2 {tf['proj:T2']['p']:.3g}/{tf['levels:T2']['p']:.3g}", flush=True)
        for k, v in tf.items():
            if k.startswith('X:'):
                print(f"   {k:16s} mean {v['mean']:.2f} (gain {v['gain']:.2f}) lam {v['lam']:.3f} ± {v['lam_err']:.3f} "
                      f"p(lam=0) {v['p_lam0']:.3g} p(lam=1) {v['p_lam1']:.3g}", flush=True)
    return res


# ------------------------------------------------------------------ E
def jobE(B):
    import phase17q11 as Q
    import phase18_nist  # noqa: F401
    import phase18_tables as PT
    fe5, fe2, co = j(R18, 'phase18_fe5.json'), j(R18, 'phase18_fe2.json'), j(R18, 'phase18_co.json')
    fin = PT.finals()
    rep = j(R17, 'phase17q11_rep.json')
    base_src = {'V²⁺': 'd3|phase15k_u_vnd_besth_sall_s4|besth_sall_s4|v', 'Cr³⁺': 'd3|phase15k_u_crnd_besth_sall_s4|besth_sall_s4|cr',
                'Mn⁴⁺': 'd3|phase15k_u_mnnd_besth_sall_s4|besth_sall_s4|mn'}
    res = {}
    hs = {'V²⁺': (4.49, 1.05), 'Cr³⁺': (6.29, .81), 'Mn⁴⁺': (7.44, .71), 'Fe⁵⁺': (8.35, .65)}
    for ion, name, fn, N, lab in SIX:
        r = {}
        if lab in base_src:
            b = rep[base_src[lab]]
            r['base_space'] = 'Step 10 (besth_sall_s4)'
            r['proj_base'], r['levels_base'] = b['proj'], b['levels']
        elif lab == 'Fe⁵⁺':
            r['base_space'] = 'Step 10 (besth_sall_s4)'
            r['proj_base'], r['levels_base'] = fe5['reps']['s10']['proj'], fe5['reps']['s10']['levels']
        else:
            dd = fe2 if lab == 'Fe²⁺' else co
            r['base_space'] = 'fam_d5f2g2h1'
            r['proj_base'] = dd['final']['proj']['base']
            r['levels_base'] = dd['final']['levels']['base']
            r['proj_corr'] = dd['final']['proj']['final']
            r['levels_corr'] = dd['final']['levels']['final']
        if lab == 'Fe⁵⁺':
            r['proj_corr'] = {k: fe5['chain']['proj'][k] for k in ('T1', 'T2', 'B', 'C', 'beta')}
            r['levels_corr'] = {k: fe5['chain']['levels'][k] for k in ('T1', 'T2') if k in fe5['chain']['levels']}
        r['final'] = {k: fin[lab][k] for k in ('proj', 'levels', 'meas')}
        f = Q.bfit_free(ion)
        r['fit'] = dict(p=f['p'], err=f['err'], mean=f['rms'], n=f['n'])
        if lab in hs:
            r['hansen'] = dict(T1=hs[lab][0] * P_MEH, T2=hs[lab][1] * P_MEH)
        else:
            r['hansen'] = dict(T1=fin[lab]['H'][0], T2=fin[lab]['H'][1])
        res[lab] = r
    return res


# ------------------------------------------------------------------ F
def jobF(B):
    import phase17q11 as Q
    out = {}
    for ion, lab, N in (('cr', 'Cr³⁺', 3), ('fe', 'Fe²⁺', 6)):
        d = Q.ds(N)
        if N == 3:
            pr = j(R15, 'phase15k_u_crnd_besth_sall_s4.json')['besth_sall_s4']['exact']['proj']
            Xm = Q.x_from_proj(pr)
            space = 'Step 10 (besth_sall_s4), exact H_eff'
        else:
            Xm = Q.x_from_dump('fend', 'fam_d5f2g2h1')
            space = 'fam_d5f2g2h1, exact H_eff'
        c, res_, rank = d.decompose(Xm)
        p0 = dict(c, zeta=Q.ZETA[ion])
        fl = Q.fit_levels(ion, Xm, Q.ZETA[ion], p0)
        H1b = Q.model_H(d, {k: v for k, v in fl['p'].items() if k != 'zeta'})
        tp, t1 = d.terms(Xm), d.terms(H1b)
        rows = B[ion]['terms']
        names = {}
        for key, (w, v, idx) in tp.items():
            for k in range(len(w)):
                names[(key, k)] = tname(key, k, len(w))
        g = '4F' if N == 3 else '5D'
        gk = next(t for t in names if names[t] == g)
        Ep = {names[t]: float(tp[t[0]][0][t[1]] - tp[gk[0]][0][gk[1]]) for t in names}
        E1 = {names[t]: float(t1[t[0]][0][t[1]] - t1[gk[0]][0][gk[1]]) for t in names}
        og = rows[g]['corrected']
        tab = []
        for nm in sorted(names.values(), key=lambda x: Ep[x]):
            o = rows.get(nm)
            eo = o['corrected'] - og if o else None
            tab.append(dict(term=nm, label=o['label'] if o else None, proj=Ep[nm], rep1b=E1[nm], obs=eo,
                            d_proj=(Ep[nm] - eo) if o else None, d_1b=(E1[nm] - eo) if o else None))
        out[lab] = dict(space=space, rows=tab, rep1b_fit=dict(p=fl['p'], rms=fl['rms']),
                        rms_proj=float(np.sqrt(np.mean([x['d_proj'] ** 2 for x in tab if x['obs'] is not None]))),
                        rms_1b=float(np.sqrt(np.mean([x['d_1b'] ** 2 for x in tab if x['obs'] is not None]))))
        for x in tab:
            print(f"{lab} {x['term']:6s} {x['label'] or '-':6s} proj {x['proj']:9.1f} (1b) {x['rep1b']:9.1f} obs "
                  + (f"{x['obs']:9.1f} d {x['d_proj']:+7.1f} {x['d_1b']:+7.1f}" if x['obs'] is not None else '-'))
    return out


if __name__ == '__main__':
    B = jobB()
    E = jobE(B)
    F = jobF(B)
    json.dump(dict(B=B, E=E, F=F), open(os.path.join(OUT, 'phase21b.json'), 'w'), indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))
