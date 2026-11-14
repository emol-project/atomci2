#!/usr/bin/env python3
"""Phase 17 A1-A3 tables from results/phase17/a_<ion>_<space>.npz (phase17a.py compute), the stored exact and
S2-full ('sylvester') H_eff of 15t/15u (results/phase15/phase15k_<tag>.json) and the B fit 1b (phase15u_bfit.json).
-> results/phase17/phase17a_tables.md (+ .json)
usage: phase17a_tables.py"""
import json
import os
import sys

import numpy as np

os.environ.setdefault('ION', 'crnd')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase17a as A  # noqa: E402

ROOT = A.ROOT
P15 = os.path.join(ROOT, 'results', 'phase15')
OUT = A.OUT
LAB = {'vnd': 'V²⁺', 'crnd': 'Cr³⁺', 'mnnd': 'Mn⁴⁺'}
ORB = {'vnd': 'vnd3', 'crnd': 'crnd3', 'mnnd': 'mnnd3'}
TAG = {('vnd', 'besth_sall'): 'u_vnd_besth_sall', ('crnd', 'besth_sall'): 'u_crnd_besth_sall',
       ('mnnd', 'besth_sall'): 'u_mnnd_besth_sall', ('vnd', 'besth_sall_s4'): 'u_vnd_besth_sall_s4',
       ('crnd', 'besth_sall_s4'): 'u_crnd_besth_sall_s4', ('mnnd', 'besth_sall_s4'): 'u_mnnd_besth_sall_s4',
       ('crnd', 'k1fh'): 't_crnd_k1fh'}
RUN = {'besth_sall': 'v15u_besth_sall_d3_<t>e_{o}', 'besth_sall_s4': 'v15u_besth_sall_s4_d3_<t>e_{o}',
       'k1fh': 'v15t_k1fh_d3_<t>e_{o}'}
SPN = {'besth_sall': 'Step 8 (best + all h + all s)', 'besth_sall_s4': 'Step 10 (Step 8 + pure n = 3)',
       'k1fh': 'RAS k = 1 + all h'}
DEN = {'C2': 'Ē_P − Ē_c (configuration averages)', 'S2-cav': 'p_j(term) − Ē_c', 'S2-diag': 'p_j(term) − H_qq',
       'S2-full': '(p_j(term) − QHQ)⁻¹, full QHQ', 'exact': 'des Cloizeaux, all orders'}
Q5 = ('T1', 'T2', 'B', 'C', 'beta')
EH_EV = 27.211386


def stored(tag):
    d = json.load(open(os.path.join(P15, f'phase15k_{tag}.json')))
    v = d[next(iter(d))]
    return {'exact': v['exact']['proj'], 'S2-full': v['sylvester']['proj']}


def meas():
    b = json.load(open(os.path.join(P15, 'phase15u_bfit.json')))
    return {'vnd': b['v']['fits']['1b']['p']['T1'], 'crnd': b['cr']['fits']['1b']['p']['T1'],
            'mnnd': b['mn']['fits']['1b']['p']['T1'], 'fe': b['fe']['fits']['1b']['p']['T1']}


def have(ion, sp):
    return bool(_C.need(os.path.join(OUT, f'a_{ion}_{sp}.npz')))     # heff: required (was: skip the row)


def a1(M, R):
    L = ['## A1: second-order variants against exact (T1, T2 mEh; B, C, β cm⁻¹)', '',
         '| ion | space | run / orbitals | variant | denominators | T1 | T2 | B | C | β | T1 / exact | T1 / meas. (1b) |',
         '|' + '---|' * 12]
    for sp in ('besth_sall', 'besth_sall_s4', 'k1fh'):
        for ion in ('vnd', 'crnd', 'mnnd'):
            if (ion, sp) not in TAG or not have(ion, sp):
                continue
            a = A.load(ion, sp)
            st = stored(TAG[(ion, sp)])
            v = {m: A.total(a, m) for m in A.METH} | st
            R.setdefault('A1', {})[f'{ion}:{sp}'] = {m: {q: v[m][q] for q in Q5} for m in v}
            for m in ('C2', 'S2-cav', 'S2-diag', 'S2-full', 'exact'):
                x = v[m]
                L.append(f'| {LAB[ion]} | {SPN[sp]} | `{RUN[sp].format(o=ORB[ion])}` / {ORB[ion]} | {m} | {DEN[m]} | '
                         f'{x["T1"]:.3f} | {x["T2"]:+.3f} | {x["B"]:.1f} | {x["C"]:.1f} | {x["beta"]:+.1f} | '
                         f'{x["T1"] / v["exact"]["T1"]:.3f} | {x["T1"] / M[ion]:.3f} |')
    fe = json.load(open(os.path.join(OUT, 'a_fe_rasK1.json')))
    st = json.load(open(os.path.join(P15, 'phase15j_o_fe.json')))['rasK1']
    vf = {m: (fe[m]['T1'], fe[m]['T2']) for m in A.METH}
    vf['S2-full'] = (st['sylvester']['proj_T1'], st['sylvester']['proj_T2'])
    vf['exact'] = (st['exact']['proj_T1'], st['exact']['proj_T2'])
    R['A1']['fe:rasK1'] = vf
    for m in ('C2', 'S2-cav', 'S2-diag', 'S2-full', 'exact'):
        t1, t2 = vf[m]
        L.append(f'| Fe²⁺ | RAS k = 1 (15n; largest stored Fe²⁺ space, block 3F 11,491) | `v15n_rasK1_d6_<t>e_fe15h6` / '
                 f'fe15h6 (standard basis) | {m} | {DEN[m]} | {t1:.3f} | {t2:+.3f} | | | | {t1 / vf["exact"][0]:.3f} | '
                 f'{t1 / M["fe"]:.3f} |')
    L += ['', 'Fe²⁺: d⁶ projection of 15j (T1, T2 only; B, C, β come from the 16-term fit and are not given for the new '
          'variants). Measured (1b) T1: ' + ', '.join(f'{LAB.get(k, "Fe²⁺")} {v:.3f}' for k, v in M.items()) + '.', '']
    return L


def a2(R):
    L = ['## A2: second-order T1, T2 by Q class (additive variants; Step 8 space, and Cr³⁺ RAS k = 1 + h, Fe²⁺ '
         'RAS k = 1)', '']
    cases = [(ion, 'besth_sall') for ion in ('vnd', 'crnd', 'mnnd')] + [('crnd', 'k1fh')]
    for ion, sp in cases:
        if not have(ion, sp):
            continue
        a = A.load(ion, sp)
        rows = A.class_table(a)
        R.setdefault('A2', {})[f'{ion}:{sp}'] = {c: {m: [v[m]['T1'], v[m]['T2']] for m in A.METH} | {'n': v['n']}
                                                 for c, v in rows.items()}
        st = stored(TAG[(ion, sp)])
        L += [f'**{LAB[ion]}, {SPN[sp]}** (T1 / T2 increments, mEh)', '',
              '| Q class | configurations | C2 | S2-cav | S2-diag |', '|---|---|---|---|---|']
        for c, v in rows.items():
            L.append(f'| {c} | {v["n"]} | ' + ' | '.join(f'{v[m]["T1"]:+.3f} / {v[m]["T2"]:+.3f}' for m in A.METH) + ' |')
        tot = {m: A.total(a, m) for m in A.METH}
        L.append('| **total (HF + Σ)** | | ' + ' | '.join(f'{tot[m]["T1"]:.3f} / {tot[m]["T2"]:+.3f}' for m in A.METH) +
                 ' |')
        L += ['', f'S2-full {st["S2-full"]["T1"]:.3f} / {st["S2-full"]["T2"]:+.3f}; exact {st["exact"]["T1"]:.3f} / '
              f'{st["exact"]["T2"]:+.3f}.', '']
    fe = json.load(open(os.path.join(OUT, 'a_fe_rasK1.json')))
    L += ['**Fe²⁺, RAS k = 1 (15n)** (T1 / T2 increments, mEh)', '', '| Q class | configurations | C2 | S2-cav | S2-diag |',
          '|---|---|---|---|---|']
    for c in A.CLASSES:
        if c in fe['C2']['classes']:
            L.append(f'| {c} | {fe["C2"]["classes"][c][2]} | ' +
                     ' | '.join(f'{fe[m]["classes"][c][0]:+.3f} / {fe[m]["classes"][c][1]:+.3f}' for m in A.METH) + ' |')
    L.append('')
    return L


def a3(R, ngrp=15, cuts=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Per term: lowest QHQ eigenvalue against the P energies. Groups: the computed lowest QHQ states of all terms,
    grouped by their dominant configuration (a single-term contribution maps onto T1 with a large coefficient and
    cancels between terms, so only sets over all terms are reported). Drops: all states with |p_j - eps_k| < cut, and
    the lowest n states of every term. S2-diag: the configurations with the smallest |p_j - H_qq|."""
    L = ['## A3: smallest denominators and near-intruders (Step 8 space; Cr³⁺ RAS k = 1 + h for reference)', '']
    for ion, sp in (('vnd', 'besth_sall'), ('crnd', 'besth_sall'), ('mnnd', 'besth_sall'), ('crnd', 'k1fh')):
        if not have(ion, sp):
            continue
        a = A.load(ion, sp)
        js = json.load(open(os.path.join(OUT, f'a_{ion}_{sp}.json')))
        st = stored(TAG[(ion, sp)])
        terms = A.terms()
        z = {t: np.zeros_like(a[f'hf|{t}']) for t in terms}
        states = []
        for t in terms:
            p, eps, dH = a[f'p|{t}'], a[f'qhq_eps|{t}'], a[f'qhq_dH|{t}']
            for k in range(len(eps)):
                states.append(dict(t=t, k=k, eps_minus_p=float(eps[k] - p.max()), gap=float(np.min(np.abs(p - eps[k]))),
                                   dE=float(1e3 * np.trace(dH[k])), top=js['blocks'][t]['top'][k]))

        def dT(sel):
            by = {t: z[t].copy() for t in terms}
            for s_ in sel:
                by[s_['t']] = by[s_['t']] + a[f'qhq_dH|{s_["t"]}'][s_['k']]
            x = A.params(by)
            return x['T1'], x['T2']
        disc = st['exact']['T1'] - st['S2-full']['T1']
        nk = len(a['qhq_eps|2P'])
        L += [f'**{LAB[ion]}, {SPN[sp]}**: S2-full T1 {st["S2-full"]["T1"]:.3f}, exact {st["exact"]["T1"]:.3f} '
              f'(exact − S2-full {disc:+.3f}). Lowest {nk} QHQ eigenstates per term computed.', '',
              '| term | p_j (Eh) | lowest QHQ ε₀ − max p_j (Eh) | (eV) | dominant configurations of ε₀ | smallest \\|p_j − H_qq\\| (Eh) |',
              '|---|---|---|---|---|---|']
        for t in terms:
            p, eps = a[f'p|{t}'], a[f'qhq_eps|{t}']
            cf = '; '.join(f'{short(c)} {w:.2f}' for c, w in js['blocks'][t]['top'][0][:2])
            L.append(f'| {t} | {", ".join(f"{x:.4f}" for x in p)} | {eps[0] - p.max():+.4f} | {(eps[0] - p.max()) * EH_EV:+.2f} | '
                     f'{cf} | {float(np.min(a[f"min_gap_diag|{t}"])):.4f} |')
        grp = {}
        for s_ in states:
            grp.setdefault(s_['top'][0][0], []).append(s_)
        g = sorted(grp.items(), key=lambda kv: min(x['gap'] for x in kv[1]))
        L += ['', 'Groups of computed QHQ states by dominant configuration, smallest gap first (ΔT1, ΔT2 = their S2-full '
              'contribution, summed over all terms):', '',
              '| # | dominant configuration | states (terms) | smallest gap (Eh) | ΔT1 | ΔT2 |', '|---|---|---|---|---|---|']
        rows = []
        for i, (c, ss) in enumerate(g[:ngrp]):
            t1, t2 = dT(ss)
            rows.append(dict(cfg=c, n=len(ss), gap=min(x['gap'] for x in ss), T1=t1, T2=t2))
            L.append(f'| {i + 1} | {short(c)} | {len(ss)} ({len({x["t"] for x in ss})}) | {min(x["gap"] for x in ss):.4f} | '
                     f'{t1:+.3f} | {t2:+.3f} |')
        drops = {}
        for cut in cuts:
            sel = [x for x in states if x['gap'] < cut]
            drops[f'gap<{cut}'] = (len(sel), *dT(sel))
        for n in (1, 2, 3, 5, 10, nk):
            sel = [x for x in states if x['k'] < n]
            drops[f'lowest {n}/term'] = (len(sel), *dT(sel))
        L += ['', '| dropped from S2-full | states | ΔT1 removed | T1 after | ΔT2 removed | T2 after |', '|---|---|---|---|---|---|']
        for k, (n, t1, t2) in drops.items():
            L.append(f'| {k} | {n} | {t1:+.3f} | {st["S2-full"]["T1"] - t1:.3f} | {t2:+.3f} | {st["S2-full"]["T2"] - t2:+.3f} |')
        # S2-diag: configurations with the smallest CSF-diagonal gap (per configuration, min over its CSFs is not stored;
        # rank by the configuration average against the P energies instead)
        cfgs = [str(c) for c in a['cfgs']]
        pmin = min(float(a[f'p|{t}'].min()) for t in terms)
        order = np.argsort(np.asarray(a['eav']) - pmin)[:8]
        L += ['', 'S2-diag / C2: the 8 Q configurations with the lowest configuration average Ē_c (above the lowest p_j) '
              'and their own increments:', '', '| configuration | Ē_c − min p_j (Eh) | C2 ΔT1 | S2-diag ΔT1 | S2-diag ΔT2 |',
              '|---|---|---|---|---|']
        for i in order:
            mk = np.zeros(len(cfgs), bool)
            mk[i] = True
            x2, xd = A.incr(a, 'C2', mk), A.incr(a, 'S2-diag', mk)
            L.append(f'| {short(cfgs[i])} | {float(a["eav"][i]) - pmin:+.4f} | {x2["T1"]:+.3f} | {xd["T1"]:+.3f} | {xd["T2"]:+.3f} |')
        L.append('')
        R.setdefault('A3', {})[f'{ion}:{sp}'] = dict(disc=disc, groups=rows, drops=drops,
                                                     gap0={t: float(a[f'qhq_eps|{t}'][0] - a[f'p|{t}'].max()) for t in terms})
    return L


def short(c):
    """'11f1 1s2 2s2 3s2 2p6 3p6 3d2' -> '3d2 11f' (outside 1s2 2s2 2p6 3s2 3p6)."""
    full = {'1s': 2, '2s': 2, '2p': 6, '3s': 2, '3p': 6}
    out = []
    for tok in sorted(c.split(), key=lambda x: (int(x[:-2]) if x[:-2].isdigit() else 99, x)):
        i = len(tok) - 1
        while tok[i].isdigit():
            i -= 1
        o, q = tok[:i + 1], int(tok[i + 1:])
        if full.get(o) == q:
            continue
        out.append(f'{o}{q if q > 1 else ""}' if o in full or o == '3d' else f'{o}{q if q > 1 else ""}')
    return ' '.join(out)


def main():
    M = meas()
    R = {}
    L = ['Phase 17 A tables (scripts/phase17a_tables.py).', ''] + a1(M, R) + a2(R) + a3(R)
    open(os.path.join(OUT, 'phase17a_tables.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(R, open(os.path.join(OUT, 'phase17a_tables.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
