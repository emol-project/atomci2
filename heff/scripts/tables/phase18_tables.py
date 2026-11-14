#!/usr/bin/env python3
"""Phase 18 (10 Nov, item 3): paper tables for the six ions on one footing -> results/phase18/paper_tables.md.
Reads only committed result files (no H): phase17_final.json / phase17q11_chain.json (V2+, Cr3+, Mn4+ proj and (1b)-model
levels), phase18_fe5.json (Fe5+ chain, d3 (1b) refits, Hansen, Uylings Table 14.5), phase18_fe2.json (Fe2+ on the Co3+
correction list, phase18_co.py fefinal / ferefit2), phase18_co.json (Co3+), phase17a_tables.json, c_* / crule_* /
a_* (second order and Q1_rule), q1diag_<ion>_s8.json (Q1_rule + pf2), phase17q20.json (Fe2+ Hansen T, paper f set).
Normalizations as the paper: T in mEh (Uylings O2', O2'', Tr O^2 = 1 in d3), f in cm-1 (f x sqrt A(f), Tr f^2 = 1 in d4);
Hansen p converted with 1.83813 mEh per unit."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
R15, R17, R18 = (os.path.join(ROOT, 'results', p) for p in ('phase15', 'phase17', 'phase18'))
P = 1.83813


def j(*p):
    return json.load(open(_C.need(os.path.join(*p))))


def opt(*p):
    """heff: formerly optional (a missing file gave 'pending'); every input is now required."""
    return j(*p)


D3 = ['V²⁺', 'Cr³⁺', 'Mn⁴⁺']
IONS = D3 + ['Fe⁵⁺', 'Fe²⁺', 'Co³⁺']
TAG = {'V²⁺': 'vnd', 'Cr³⁺': 'crnd', 'Mn⁴⁺': 'mnnd', 'Fe⁵⁺': 'fe5nd'}
FS = [f'f{i}' for i in range(1, 13)]


def finals():
    fe5, fe2, co = j(R18, 'phase18_fe5.json'), j(R18, 'phase18_fe2.json'), j(R18, 'phase18_co.json')
    q20 = j(R17, 'phase17q20.json')['compare']['T']
    S1 = fe5['refit']['series_1b']
    hs = {'V²⁺': (4.49, 1.05), 'Cr³⁺': (6.29, .81), 'Mn⁴⁺': (7.44, .71), 'Fe⁵⁺': (8.35, .65)}
    r = {}
    for i in D3:
        r[i] = dict(proj=fe5['series']['proj'][i], levels=fe5['series']['levels'][i], meas=S1[i],
                    H=(hs[i][0] * P, hs[i][1] * P))
    ch = fe5['chain']
    r['Fe⁵⁺'] = dict(proj=ch['proj'], levels=ch['levels'], meas=S1['Fe⁵⁺'], H=(hs['Fe⁵⁺'][0] * P, hs['Fe⁵⁺'][1] * P))
    for i, d, H in (('Fe²⁺', fe2, (q20['fe']['H_mEh'][0][0] * -1, q20['fe']['H_mEh'][1][0])), ('Co³⁺', co, (6.17 * P, 1.74 * P))):
        f = d['final']
        fr = d['refit_final']['proj']['free']
        r[i] = dict(proj=dict(T1=f['proj']['final']['T1'], T2=f['proj']['final']['T2'], u1=f['proj']['u1'], u2=f['proj']['u2']),
                    levels=dict(T1=f['levels']['final']['T1'], T2=f['levels']['final']['T2'], u1=f['levels']['u1'], u2=f['levels']['u2']),
                    meas=dict(T1=fr['T1'], u1=fr['T1_err'], T2=fr['T2'], u2=fr['T2_err'], mean_error=fr['mean_error']), H=H)
    return r


def t_final(r):
    L = ['## 1. Final T1, T2 (mEh)', '',
         'proj = projection of the corrected H_eff; (1b) repr. = the corrected H_eff fitted with the (1b) model on the '
         'ion\'s NIST J levels; measured = our (1b) fit of the NIST levels; Hansen = Hansen, Raassen, Uylings & Lister 1988 '
         '(× 1.83813; their errors are batch-dependent). d³ corrected values: Step 10 chain; Fe²⁺, Co³⁺: the same d⁶ chain '
         '(phase18_co.py; Fe²⁺ reproduces phase17_final 9.482 / 3.626).', '',
         '| ion | T1 proj | T1 (1b) repr. | T1 measured (1b) | T1 Hansen | proj / (1b) | proj / Hansen | T2 proj | T2 (1b) repr. | T2 measured (1b) | T2 Hansen |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for i in IONS:
        x = r[i]
        p, lv, m, H = x['proj'], x['levels'], x['meas'], x['H']
        L.append(f"| {i} | {p['T1']:.3f} ± {p['u1']:.3f} | {lv['T1']:.3f} ± {lv['u1']:.3f} | {m['T1']:.3f} ± {m['u1']:.3f} | {H[0]:.3f} | "
                 f"{p['T1'] / m['T1']:.3f} | {p['T1'] / H[0]:.3f} | {p['T2']:.3f} ± {p['u2']:.3f} | {lv['T2']:.3f} ± {lv['u2']:.3f} | "
                 f"{m['T2']:.3f} ± {m['u2']:.3f} | {H[1]:.3f} |")
    return L


def t_methods():
    A17 = j(R17, 'phase17a_tables.json')['A1']
    fe5 = j(R18, 'phase18_fe5.json')
    L = ['', '## 2. Second order and Q1_rule against exact (ΔT1 %, ΔT2 mEh)', '',
         'd³: Step 8 space (`besth_sall`); d⁶: `fam_d5f2g2h1`. Closure = C2; S2-diag = second order with diagonal Q '
         'denominators; Q1_rule = n = 3 + 3p→f singles + 3d→v singles + same-l 3d² doubles in Q1, balanced; '
         '+ pf2 = Q1_rule with the pf2 class (3p hole + f + a second excitation) also in Q1.', '',
         '| ion (space) | exact T1 / T2 | closure | S2-full | S2-diag | Q1_rule (Q1 fraction) | Q1_rule + pf2 (Q1 fraction) |',
         '|---|---|---|---|---|---|---|']

    def c(v, ex):
        return f"{100 * (v['T1'] / ex['T1'] - 1):+.2f} % / {v['T2'] - ex['T2']:+.3f}" if v else 'pending'
    # Fe5+ variants: only in its generated tables (phase18_fe5.py tables), rows '| Fe5+ | <variant> | T1 | T2 | ...'
    fe5v = {}
    for x in open(_C.need(os.path.join(R18, 'phase18_fe5_tables.md'))):
        c_ = [y.strip() for y in x.split('|')]
        if len(c_) > 4 and c_[1] == 'Fe5+' and c_[2] in ('C2', 'S2-full', 'S2-diag', 'exact'):
            fe5v[c_[2]] = dict(T1=float(c_[3]), T2=float(c_[4]))
    for i in D3 + ['Fe⁵⁺']:
        t = TAG[i]
        a = fe5v if i == 'Fe⁵⁺' else A17[f'{t}:besth_sall']
        q1 = opt(R18, f'q1diag_{t}_s8.json')          # Q1_rule ('') and Q1_rule + pf2, same code for the four d3 ions
        rule = (q1['sets'][''] if q1 else None) or j(R18 if i == 'Fe⁵⁺' else R17, f'crule_{t}_s8.json')['sets']['R']
        pf = q1['sets'].get('pf2') if q1 else None
        ex = a['exact']
        L.append(f"| {i} (S8) | {ex['T1']:.3f} / {ex['T2']:.3f} | {c(a['C2'], ex)} | {c(a['S2-full'], ex)} | {c(a['S2-diag'], ex)} | "
                 f"{c(rule, ex)} ({100 * rule['frac']:.2f} %) | " + (f"{c(pf, ex)} ({100 * pf['frac']:.2f} %)" if pf else 'pending') + ' |')
    # d6
    p = j(R15, 'phase15j_17_fend_fam_d5f2g2h1.json')['fam_d5f2g2h1']
    ex = dict(T1=p['exact']['proj_T1'], T2=p['exact']['proj_T2'])
    s2 = j(R17, 'a_fend_fam_d5f2g2h1.json')['S2-diag']
    rule = j(R17, 'crule_fend_fam.json')['sets']['R']
    tt = lambda k: dict(T1=p[k]['proj_T1'], T2=p[k]['proj_T2'])  # noqa: E731
    L.append(f"| Fe²⁺ (fam) | {ex['T1']:.3f} / {ex['T2']:.3f} | {c(tt('closure'), ex)} | {c(tt('sylvester'), ex)} | {c(s2, ex)} | "
             f"{c(rule, ex)} ({100 * rule['frac']:.2f} %) | – |")
    p = j(R15, 'phase15j_18_cond_fam_d5f2g2h1.json')['fam_d5f2g2h1']
    ex = dict(T1=p['exact']['proj_T1'], T2=p['exact']['proj_T2'])
    a = opt(R17, 'a_cond_fam_d5f2g2h1.json')
    s2 = a['S2-diag'] if a else None
    rq = j(R18, 'q18_cond_fam.json')['sets']['R']
    co = j(R18, 'phase18_co.json')['fam']['R']['proj']
    L.append(f"| Co³⁺ (fam) | {ex['T1']:.3f} / {ex['T2']:.3f} | {c(tt('closure'), ex)} | {c(tt('sylvester'), ex)} | {c(s2, ex)} | "
             f"{c(co, ex)} ({100 * rq['frac']:.2f} %) | – |")
    return L


def t_fits():
    L = ['', '## 3. Four-body fits on the NIST levels, Fe²⁺ (Fe III, 33 levels) and Co³⁺ (Co IV)', '',
         'Fixed extras = our T3, T4, f1..f12 (proj); λ = one common scale of the fixed extras fitted with the (1b) '
         'parameters; p(λ = 0), p(λ = 1) from F tests. fam = largest-space projection; corrected = the d⁶ chain; '
         'without 11d–13d = the fam sub-space without the three diffuse virtual d (exact H_eff of that sub-space).', '',
         '| ion | extras | mean error | factor | λ | p(λ=0) | p(λ=1) | T1 fit | T2 fit | mean error, T1 T2 also fixed (proj) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    fe2, co = j(R18, 'phase18_fe2.json'), j(R18, 'phase18_co.json')
    rows = {'Fe²⁺': [('free', None), ('corrected T3+T4+f', 'final'), ('corrected f only', 'final f only'),
                     ('corrected T3+T4 only', 'final T3T4 only'), ('fam T3+T4+f', 'fam'), ('fam f only', 'fam f only'),
                     ('fam T3+T4 only', 'fam T3T4 only'), ('fam without 11d–13d', 'fam X-11d,12d,13d')],
            'Co³⁺': [('free', None), ('corrected T3+T4+f', 'final'), ('corrected f only', 'final f only'),
                     ('corrected T3+T4 only', 'final T3T4 only'), ('fam T3+T4+f', 'fam'), ('fam f only', 'fam f only'),
                     ('fam T3+T4 only', 'fam T3T4 only'), ('fam without 11d–13d', 'fam X-11d,12d,13d')]}
    for ion, d in (('Fe²⁺', fe2), ('Co³⁺', co)):
        rf = d['refit_final']['proj']
        for lab, k in rows[ion]:
            if k is None:
                v = rf['free']
                L.append(f"| {ion} | none (free (1b)) | {v['mean_error']:.2f} | 1 | | | | {v['T1']:.3f} ± {v['T1_err']:.3f} | {v['T2']:.3f} ± {v['T2_err']:.3f} | |")
                continue
            v = rf[k]
            L.append(f"| {ion} | {lab} | {v['mean_error']:.2f} | {v['gain']:.2f} | {v['lam']:.3f} ± {v['lam_err']:.3f} | {v['p_lam0']:.2g} | "
                     f"{v['p_lam1']:.2g} | {v['T1']:.3f} ± {v['T1_err']:.3f} | {v['T2']:.3f} ± {v['T2_err']:.3f} | "
                     + (f"{v['fixX12_mean_error']:.2f} (p {v['p_12_given_X']:.1g})" if 'fixX12_mean_error' in v else '') + ' |')
    return L


def t_f():
    import phase17q20 as Q20
    A = Q20.jlA()
    fe2, co = j(R18, 'phase18_fe2.json'), j(R18, 'phase18_co.json')
    paper = j(R17, 'phase17q20.json')['compare']['f']
    HC = co['hansen_f']
    L = ['', '## 4. f1..f12 (cm⁻¹, Tr f² = 1 in d⁴), corrected proj, with Hansen et al. 1988', '',
         'Corrected = the common d⁶ chain (the same correction list for Fe²⁺ and Co³⁺, 2p item included; phase18_co.py). '
         'Hansen errors in parentheses (batch-dependent).', '',
         '| f | Fe²⁺ fam | Fe²⁺ corrected | Hansen Fe III | Co³⁺ fam | Co³⁺ corrected | Hansen Co IV |',
         '|---|---|---|---|---|---|---|']
    fb, ff = fe2['final']['proj']['base'], fe2['final']['proj']['final']
    cb, cf = co['final']['proj']['base'], co['final']['proj']['final']
    dmax = 0.0
    for k in FS:
        s = np.sqrt(A[k])
        h, e = Q20.H_FE[k]
        dmax = max(dmax, abs(round(ff[k] * s) - round(paper[k]['norm']['fe_own'])))
        L.append(f"| {k} | {fb[k] * s:+.0f} | {ff[k] * s:+.0f} | {h:+d} ({e}) | "
                 f"{cb[k] * s:+.0f} | {cf[k] * s:+.0f} | {HC[k][0]:+d} ({HC[k][1]}) |")
    vf, vc = fe2['final']['proj']['vs_hansen'], co['final']['proj']['vs_hansen']
    pt = j(R17, 'phase17q20.json')['compare']
    L += ['', f"Against Hansen (whole vector): Fe²⁺ corrected cos {vf['cos']:.3f}, slope {vf['slope']:.3f}; "
          f"Co³⁺ corrected cos {vc['cos']:.3f}, slope {vc['slope']:.3f}.",
          '', f"T3, T4 corrected (proj, mEh): Fe²⁺ {ff['T3']:+.3f}, {ff['T4']:+.3f}; Co³⁺ {cf['T3']:+.3f}, {cf['T4']:+.3f}.",
          '', f"Note: the paper's earlier Fe²⁺ corrected values (Phase 17 `fe_own`: fam + the Fe²⁺ RAS increments, no 2p "
          f"item) differ from the common-list values by <= {dmax:.0f} cm⁻¹ in every f (cos vs Hansen "
          f"{pt['vec']['fe_own']['cos']:.3f}; T3, T4 {pt['T']['fe']['ours']['corr'][2]:+.3f}, {pt['T']['fe']['ours']['corr'][3]:+.3f})."]
    return L


def t_uylings():
    src = open(_C.need(os.path.join(R18, 'phase18_fe5_tables.md'))).read().split('\n')
    i = next(n for n, x in enumerate(src) if x.startswith('## Uylings (2021) Table 14.5'))
    k = next((n for n, x in enumerate(src) if n > i and x.startswith('## ')), len(src))
    sec = src[i:k]
    sec[0] = '## 5. Fe⁵⁺: Uylings (2021) Table 14.5, his normalization (cm⁻¹), Fe VI'
    tail = [x for x in src if x.startswith('fitted (Uylings)')]
    return [''] + sec + ([''] + tail if tail and tail[0] not in sec else [])


def main():
    r = finals()
    L = ['# Paper tables, six ions on one footing (Phase 18; scripts/phase18_tables.py)', '',
         'T in mEh (Uylings O2\', O2\'\', Tr O² = 1 in d³), f in cm⁻¹ (Tr f² = 1 in d⁴). Generated from committed result files only.', '']
    L += t_final(r) + t_methods() + t_fits() + t_f() + t_uylings()
    open(os.path.join(R18, 'paper_tables.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
