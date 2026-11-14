#!/usr/bin/env python3
"""Phase 21 tables from the phase21 result files only -> results/phase21/phase21_tables.md"""
import json
import os

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
OUT = os.path.join(_C.root(), 'results', 'phase21')  # inputs and output (was this script's directory)
P_MEH = 1.83813


def j(f):
    p = os.path.join(OUT, f)
    return json.load(open(_C.need(p)))


def f3(x, n=3):
    return '–' if x is None else f'{x:.{n}f}'


def pv(x):
    return '–' if x is None else (f'{x:.2g}' if x >= 1e-3 else f'{x:.1e}')


def secA():
    L = ['## A. Three 3s3p holes with one in 3s (RAS k = 1 + h; sub-matrices of the stored u3c1 H)', '',
         'k1fh = n_hole <= 2; n3a = + 3s 3p⁴ 3d^(N+3); Av2 = + every 3s 3p⁴ 3d^(N+1) v v′; A = + every 3s¹ 3p⁴ configuration '
         'with <= 2 electrons in V and excitation level <= 2 from R2 = 3s 3p⁶ 3d^(N+1) (= n3a + 3s 3p⁴ 3d^(N+2) v + Av2); '
         'h3 = + every 3-hole configuration (A + the 3s⁰ 3p⁵ ones) = stored k1fh_h3; n3all = + the three pure n = 3 '
         'configurations = stored k1fh_n3; u3c1 = the stored space. T in mEh (projection, vcm HF placement for d³).', '']
    for ion, lab in (('cr', 'Cr³⁺'), ('v', 'V²⁺'), ('fe', 'Fe²⁺')):
        d = j(f'phase21a_{ion}.json')
        if not d or 'heff' not in d:
            L += [f'{lab}: skipped (> 3 h): stopped after 5 of 11 blocks (1S 3P 1D 3D 5D, 6569 s; spaces k1fh n3a Av2 A); '
                  'n^1.6 extrapolation of the rest ~5.4 h in total (phase21a_heff_fe.log)' if ion == 'fe' else f'{lab}: pending', '']
            continue
        cf, H = d['cfg'], d['heff']
        L += [f'### {lab}', '', f"Configurations added to k1fh: {cf['n_configs_added']}", '',
              '| block | ' + ' | '.join(cf['blocks'][next(iter(cf['blocks']))]) + ' |',
              '|---|' + '---|' * len(cf['blocks'][next(iter(cf['blocks']))])]
        for t, v in cf['blocks'].items():
            L.append(f'| {t} | ' + ' | '.join(str(x) for x in v.values()) + ' |')
        b = H['k1fh']['proj']
        L += ['', '| space | T1 | T2 | ΔT1 vs k1fh (mEh) | ΔT1 (%) | ΔT2 | min wP |', '|---|---|---|---|---|---|---|']
        for s, h in H.items():
            p = h['proj']
            L.append(f"| {s} | {p['T1']:.4f} | {p['T2']:.4f} | {p['T1'] - b['T1']:+.4f} | {100 * (p['T1'] / b['T1'] - 1):+.2f} | "
                     f"{p['T2'] - b['T2']:+.4f} | {h['min_wP']:.4f} |")
        L.append('')
    return L


def secB(B):
    L = ['## B. Fits on zeta-corrected LS term energies (unit weights; A, B, C, α, β, T1, T2)', '',
         'Zeta shift of a term = its (2J+1)-weighted centroid in the free (1b) level-fit model minus the same model at '
         'ζ = 0 (levels assigned to terms by maximum overlap); corrected = observed (2J+1) centroid − shift. '
         'Mean deviation = sqrt(Σr² / dof) (as the level fits). p: nested F against the free term fit '
         '(a) T1, T2 fixed, (b) T1 alone, (c) T2 alone, at the computed values (proj / (1b) repr.). Level fits '
         '(phase19_fits.json, phase19_single.json; same code path) in brackets.', '',
         '| spectrum | terms | dof | excluded / missing | mean dev. (level) | T1 ± err (level) | T2 ± err (level) | (a) p proj / (1b) (level) | (b) T1 p proj / (1b) (level) | (c) T2 p proj / (1b) (level) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for ion, r in B.items():
        tf, lv = r['term_fit'], r['level']
        f = tf['free']
        L.append(f"| {r['name']} | {tf['n']} | {tf['dof']} | {', '.join(x['label'] for x in r['excluded']) or 'none'} / "
                 f"{', '.join(r['missing_terms']) or 'none'} | {f['mean']:.2f} ({lv['free_mean']:.2f}) | "
                 f"{f['p']['T1']:.3f} ± {f['err']['T1']:.3f} ({lv['T1'][0]:.3f} ± {lv['T1'][1]:.3f}) | "
                 f"{f['p']['T2']:.3f} ± {f['err']['T2']:.3f} ({lv['T2'][0]:.3f} ± {lv['T2'][1]:.3f}) | "
                 f"{pv(tf['proj:both']['p'])} / {pv(tf['levels:both']['p'])} ({pv(lv['proj:both'])} / {pv(lv['levels:both'])}) | "
                 f"{pv(tf['proj:T1']['p'])} / {pv(tf['levels:T1']['p'])} ({pv(lv.get('proj:T1'))} / {pv(lv.get('levels:T1'))}) | "
                 f"{pv(tf['proj:T2']['p'])} / {pv(tf['levels:T2']['p'])} ({pv(lv.get('proj:T2'))} / {pv(lv.get('levels:T2'))}) |")
    L += ['', '### Fe III, Co IV: four-body vector fixed (term fit; level fit in brackets)', '',
          '| spectrum | vector fixed | mean dev. | gain vs free | λ ± err | p(λ = 0) | p(λ = 1) |', '|---|---|---|---|---|---|---|']
    for ion in ('fe', 'co'):
        r = B[ion]
        tf, lv = r['term_fit'], r['level']
        L.append(f"| {r['name']} | none (free) | {tf['free']['mean']:.2f} ({lv['free_mean']:.2f}) | 1 | | | |")
        for sn, lab in (('fam', 'T3, T4, f (fam)'), ('corrected', 'T3, T4, f (corrected)'), ('hansen_f', 'Hansen f (T3 = T4 = 0)'),
                        ('hansen_f_pt', "Hansen f + p_t, p_t′")):
            v = tf[f'X:{sn}']
            x = lv['X'].get(sn, {})
            lm = f" ({x['mean']:.2f})" if 'mean' in x else ''
            ll = f" ({x['lam']:.3f} ± {x['lam_err']:.3f})" if 'lam' in x else ''
            l0 = f" ({pv(x['p_lam0'])})" if 'p_lam0' in x else ''
            l1 = f" ({pv(x['p_lam1'])})" if 'p_lam1' in x else ''
            L.append(f"| {r['name']} | {lab} | {v['mean']:.2f}{lm} | {v['gain']:.2f} | {v['lam']:.3f} ± {v['lam_err']:.3f}{ll} | "
                     f"{pv(v['p_lam0'])}{l0} | {pv(v['p_lam1'])}{l1} |")
    L += ['', 'Level-fit gains are on the free (1b) level fit (Fe III 33 levels mean 33-47; phase19 job 1/3).', '',
          '### Zeta-corrected term energies (cm⁻¹; observed centroid, zeta shift, corrected)', '']
    for ion, r in B.items():
        L.append(f"- {r['name']}: " + '; '.join(f"{t} ({x['label']}) {x['obs_centroid']:.1f} − ({x['zeta_shift']:+.1f}) = {x['corrected']:.1f}"
                                            for t, x in r['terms'].items()) + f" (min overlap {r['min_overlap']:.3f})")
    return L


def secE(E):
    L = ['', '## E. Parameter table (cm⁻¹; T in mEh, Uylings O2′, O2″ with Tr O² = 1 in d³)', '',
         'proj / (1b) repr.: T1, T2 = the paper\'s corrected values (Table 1); B, C, α, β of the base H_eff (Step 10 for d³, '
         'fam for d⁶; corrected for Fe²⁺, Co³⁺ and B, C, β for Fe⁵⁺ where the chain carries them). F2 = 49 B + 7 C, '
         'F4 = 441 C / 35. Fit = our free (1b) fit of the NIST levels (± 1σ). Hansen = Hansen, Raassen, Uylings & Lister 1988 '
         '(T only on disk). Trees T = T1 / 1.83813 (cm⁻¹); T1^U = −T1 / 1.83813 (Uylings 2021 normalization, the sign '
         'convention of phase18_fe5.py); T2^U = T2 / 1.83813 (same norm; the sign of Uylings\' T2 not checked on disk).', '',
         '| ion | source | F2 | F4 | B | C | α | β | T1 | T2 | Trees T | T1^U | T2^U |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lab, r in E.items():
        def row(src, B_, C_, a, b, t1, t2, F2=None, F4=None, e=None):
            F2 = F2 if F2 is not None else (49 * B_ + 7 * C_ if B_ is not None and C_ is not None else None)
            F4 = F4 if F4 is not None else (441 * C_ / 35 if C_ is not None else None)
            g = lambda x, n=1: '–' if x is None else f'{x:.{n}f}'   # noqa: E731
            ge = lambda k, x, n=1: g(x, n) + (f' ± {e[k]:.{n}f}' if e and k in e and x is not None else '')  # noqa: E731
            L.append(f"| {lab} | {src} | {ge('F2', F2, 0)} | {ge('F4', F4, 0)} | {g(B_)} | {g(C_)} | {ge('alpha', a, 2)} | "
                     f"{ge('beta', b)} | {ge('T1', t1, 3)} | {ge('T2', t2, 3)} | {g(t1 / P_MEH if t1 else None, 3)} | "
                     f"{g(-t1 / P_MEH if t1 else None, 3)} | {g(t2 / P_MEH if t2 is not None else None, 3)} |")
        pb, lb, fi = r['proj_base'], r['levels_base'], r['final']
        pc = r.get('proj_corr', {})
        lc = r.get('levels_corr', {})
        row(f"proj ({r['base_space']}; T corrected)", pc.get('B', pb['B']), pc.get('C', pb['C']), pc.get('alpha', pb['alpha']),
            pc.get('beta', pb['beta']), fi['proj']['T1'], fi['proj']['T2'])
        row(f"(1b) repr. ({r['base_space']}; T corrected)", lc.get('B', lb['B']), lc.get('C', lb['C']), lc.get('alpha', lb['alpha']),
            lc.get('beta', lb['beta']), fi['levels']['T1'], fi['levels']['T2'])
        p, e = r['fit']['p'], r['fit']['err']
        Bf, Cf = p['F2'] / 49 - 5 * p['F4'] / 441, 35 * p['F4'] / 441
        row(f"fit (n {r['fit']['n']}, mean {r['fit']['mean']:.1f}; ζ {p['zeta']:.1f} ± {e['zeta']:.1f})", Bf, Cf, p['alpha'], p['beta'],
            p['T1'], p['T2'], F2=p['F2'], F4=p['F4'], e=e)
        row('Hansen et al. 1988', None, None, None, None, r['hansen']['T1'], r['hansen']['T2'])
    return L


def secF(F):
    L = ['', '## F. LS term energies, computed vs zeta-corrected observed (cm⁻¹, relative to the ground term)', '']
    for lab, r in F.items():
        L += [f"### {lab} ({r['space']}; (1b) repr.: its level fit with ζ then set to 0)", '',
              '| term | NIST | projection | (1b) repr. | observed (ζ-corr.) | proj − obs | (1b) − obs |', '|---|---|---|---|---|---|---|']
        sg = lambda v: '–' if v is None else f'{v:+.1f}'     # noqa: E731
        for x in r['rows']:
            L.append(f"| {x['term']} | {x['label'] or '–'} | {x['proj']:.1f} | {x['rep1b']:.1f} | {f3(x['obs'], 1)} | "
                     f"{sg(x['d_proj'])} | {sg(x['d_1b'])} |")
        L += ['', f"rms (observed terms): projection {r['rms_proj']:.1f}, (1b) repr. {r['rms_1b']:.1f}", '']
    return L


def secC(C):
    L = ['## C. Basis tests (Fe⁵⁺, Cr³⁺)', '']
    if not C:
        return L + ['pending', '']
    if 'scan' in C:
        L += ['Space choice (stored Cr³⁺ fam_d5f2g2h1 H, sub-matrices; Step 8 T2 = 1.714):', '',
              '| space | virtual l dropped | largest block | T1 | T2 | T2 vs Step 8 |', '|---|---|---|---|---|---|']
        for s, v in C['scan'].items():
            L.append(f"| {s} | {v['drop'] or 'none'} | {v['max']} | {v['T1']:.4f} | {v['T2']:.4f} | {100 * v['T2_vs_step8']:+.1f} % |")
        L.append('')
    if 'heff' in C:
        sc = C.get('scan', {}).get('famnsphf')
        ref = {'fe5': C['heff'].get('fe5nd:famnsphf', {}).get('proj'), 'cr': dict(T1=sc['T1'], T2=sc['T2']) if sc else None}
        lab = {'fe5nd': 'plain (fe.without-diffuse)', 'fe5td': '+ tight d (2.5 × the tightest)', 'fe5qd': '+ diffuse (fe.bas)',
               'fe5dc': 'd decontracted (open d), 19d/20d dropped', 'crtd': '+ tight d (2.5 × the tightest)',
               'crqd': '+ diffuse (cr.bas)', 'crdc': 'd decontracted (open d), 19d/20d dropped'}
        L += ['Space famnsphf = fam virtuals without s, p, f, h (all virtual d, 11g, 12g); fnf8 = famnsphf without the two '
              'tightest virtual d of the open-d basis (19d ⟨r⟩ 0.31, 20d 0.13 a0 for Fe⁵⁺), the fallback after the full open-d '
              'famnsphf 2F (50511 CSFs) and 2G (48782) runs hit the 3 h limit (10813 s; skipped). Cr³⁺ plain = the scan '
              'famnsphf sub-matrix of the stored fam H (same orbitals). ΔT vs plain in mEh.', '',
              '| ion:space | basis | largest block | T1 | T2 | ΔT1 | ΔT1 (%) | ΔT2 | min wP |', '|---|---|---|---|---|---|---|---|---|']
        if sc:
            L.append(f"| crnd:famnsphf (scan) | plain (cr.without-diffuse) | {sc['max']} | {sc['T1']:.4f} | {sc['T2']:.4f} | | | | |")
        for k, v in C['heff'].items():
            ion = k.split(':')[0]
            r = ref['fe5' if ion.startswith('fe5') else 'cr']
            p = v['proj']
            mx = max(i['n'] for i in v['info'].values())
            dd = (f"{p['T1'] - r['T1']:+.5f} | {100 * (p['T1'] / r['T1'] - 1):+.3f} | {p['T2'] - r['T2']:+.5f}"
                  if r and k != 'fe5nd:famnsphf' else ' | | ')
            L.append(f"| {k} | {lab.get(ion, '')} | {mx} | {p['T1']:.4f} | {p['T2']:.4f} | {dd} | {v['min_wP']:.4f} |")
        L.append('')
    return L


def secAdd(Ed, A):
    L = ['## Addendum A1–A3 (Cr³⁺, V²⁺; RAS k = 1 + h, sub-matrices of the stored u3c1 H)', '']
    if A:
        L += ['### A1. Job-A class: configurations and double counting', '',
              'Class = the 25 configurations 3s¹3p⁴3d⁶, 3s¹3p⁴3d⁵ v (3), 3s¹3p⁴3d⁴ v v′ (21) (same list for Cr³⁺ and V²⁺, '
              'V = the k1fh virtuals). 3s¹3p⁴3d⁶ is one of the three pure n = 3 configurations (Step 10, k1fh_n3); the other '
              '24 are all in the Table S1 hole remainder (u3c1 − k1fh_n3, 60 configurations; the remaining 36 are 3s⁰3p⁵ '
              '3d⁵ v, 3s⁰3p⁵3d⁴ v v′, the 4-hole 3s⁰3p⁴3d⁶ v, 3s¹3p³3d⁶ v, 3s²3p²3d⁶ v, and 3s²3p³3d⁵ v). No job-A '
              'configuration is outside Step 10 + the hole item: the class is already counted once in the paper.', '']
    if not Ed:
        return L + ['A3 / roots: pending', '']
    if 'root' in Ed:
        L += ['### A1. Lowest 3s-hole root (R2 = 3s 3p⁶ 3d^(N+1) dominated; root-homing Davidson) vs the exact P roots', '',
              '| ion | term | wR2 (k1fh / A) | gap k1fh (cm⁻¹) | gap A (cm⁻¹) | narrowing (cm⁻¹) | narrowing (Eh) | (%) |',
              '|---|---|---|---|---|---|---|---|']
        for ion, R in Ed['root'].items():
            for t, r in R.items():
                for i, nw in enumerate(r['narrow_cm']):
                    gk, ga = r['k1fh']['gap_cm'][i], r['A']['gap_cm'][i]
                    L.append(f"| {ion} | {t}{'' if len(r['narrow_cm']) == 1 else f' ({i + 1})'} | {r['k1fh']['wR2']:.3f} / "
                             f"{r['A']['wR2']:.3f} | {gk:.0f} | {ga:.0f} | {nw:+.0f} | {nw / 219474.63:+.4f} | {100 * nw / gk:+.2f} |")
        L.append('')
    if 'a3' in Ed:
        a = Ed['a3']
        L += ['### A3. Cr³⁺: the class treated approximately (errors vs (i) exact in A; mEh)', '',
              '| variant | T1 | T2 | err T1 | err T2 | ΔT1 recovered (of exact +%.4f) |' % a['dT1_exact'], '|---|---|---|---|---|---|']
        nm = {'exact:A': '(i) exact, A', 'exact:k1fh': 'exact, k1fh (class absent)',
              'S2-diag': '(ii) k1fh exact + class at second order, S2-diag (additive)', 'C2': '(ii) same, C2 (closure)',
              'bal:k1fhQ1': '(ii) balanced: Q1 = Q(k1fh), Q2 = class (D = p_j − H_qq)', 'R:k1fh': 'Q1_rule in k1fh (class absent)',
              'R:A': 'Q1_rule in A (class in Q2 except 3s¹3p⁴3d⁶, which is n3)', 'R+class:A': '(iii) Q1_rule + class'}
        rk = {'S2-diag': 'S2-diag', 'C2': 'C2', 'bal:k1fhQ1': 'bal:k1fhQ1', 'R+class:A': 'R+class vs R:k1fh', 'R:A': 'R:A vs R:k1fh'}
        for v in ('exact:A', 'exact:k1fh', 'S2-diag', 'C2', 'bal:k1fhQ1', 'R:k1fh', 'R:A', 'R+class:A'):
            x = a[v]
            rc = a['recovery'].get(rk.get(v, ''), None)
            rr = f"{rc['dT1']:+.4f} ({100 * rc['frac']:.0f} %)" if rc else ''
            L.append(f"| {nm[v]} | {x['T1']:.4f} | {x['T2']:.4f} | {x['err_T1']:+.4f} | {x['err_T2']:+.4f} | {rr} |")
        L += ['', 'The additive second-order increment of the class is exactly zero: every class configuration is a triple '
              'excitation from P (3s → 3d with 3p² → 3d²), so H(P, class) = 0. The class acts through Q1 (R2 and the rest of '
              'k1fh), which the balanced form dresses (H(Q1, Q2) D⁻¹ H(Q2, Q1)).', '']
    return L


def secD(D):
    L = ['## D. Cr³⁺ Step 8 T1 by excitation class (mEh)', '']
    if not D:
        return L + ['pending', '']
    c = D['classes']
    ORDER = ['3s→3d', '2s→3d', '3d→v single', '3p²→3d²', '3p→f', '3s3p double', 'others']
    ex = D.get('exact')
    L += ['| class | configurations | closure (C2) | S2-diag | exact, cumulative (in this order) | exact, remove-one |', '|---|---|---|---|---|---|']
    for k in ORDER:
        cu = f"{ex['cumulative'][k]:+.3f}" if ex else 'pending'
        rm = f"{ex['remove_one'][k]:+.3f}" if ex else 'pending'
        L.append(f"| {k} | {c['n_cfg'][k]} | {c['second']['C2']['rows'][k]['T1']:+.3f} | {c['second']['S2-diag']['rows'][k]['T1']:+.3f} | {cu} | {rm} |")
    tot_ex = f"{ex['variants']['full']['T1']:.3f}" if ex else 'pending'
    L.append(f"| total (HF P block {c['hf']['T1']:+.3f}) | {sum(c['n_cfg'].values())} | {c['second']['C2']['total']['T1']:.3f} | "
             f"{c['second']['S2-diag']['total']['T1']:.3f} | {tot_ex} | |")
    L += ['', "'others' split (phase17a classes), C2 / S2-diag: " + '; '.join(
        f"{k} ({v['n']}) {v['T1']:+.3f} / {c['others_split']['S2-diag'][k]['T1']:+.3f}" for k, v in c['others_split']['C2'].items()), '']
    return L


if __name__ == '__main__':
    b = j('phase21b.json')
    L = ['# Phase 21 tables (generated by phase21_report.py from results/phase21/*.json)', '']
    L += secA() + secB(b['B']) + secE(b['E']) + secF(b['F']) + [''] + secC(j('phase21c.json')) + secD(j('phase21d.json'))
    L += secAdd(j('phase21e.json'), j('phase21a_cr.json'))
    open(os.path.join(OUT, 'phase21_tables.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
