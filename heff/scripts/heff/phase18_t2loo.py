#!/usr/bin/env python3
"""Phase 18 item 2 (10 Nov): T2 of the d6 fam space by exact leave-one-out of each virtual orbital, Fe2+ vs Co3+.
Contribution of an orbital (set) = T2(fam) - T2(fam without it) (exact sub-space H_eff, phase17c_scan 'X-<orb>').
Reads q18_<ion>_fam_loo.json (phase18_loo.sh), the fam exact (q13_fend_fam / q18_cond_fam 'X-'), the famiq n = 3 and
first-i items, the d6 chain (phase18_fe2.json, phase18_co.json) and the four-body fits. -> stdout + phase18_t2loo.json"""
import json
import os

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
R17, R18 = (os.path.join(ROOT, 'results', p) for p in ('phase17', 'phase18'))
j = lambda *p: json.load(open(os.path.join(*p)))  # noqa: E731
ION = {'Fe²⁺': ('fend', j(R17, 'q13_fend_fam.json'), j(R17, 'q18_feti1_famiq.json'), j(R18, 'phase18_fe2.json')),
       'Co³⁺': ('cond', j(R18, 'q18_cond_fam.json'), j(R18, 'q18_coti1_famiq.json'), j(R18, 'phase18_co.json'))}
ROWS = [('16s', 'X-16s'), ('15p', 'X-15p'), ('11d', 'X-11d'), ('12d', 'X-12d'), ('13d', 'X-13d'), ('14d', 'X-14d'),
        ('11d+12d+13d (diffuse d)', 'X-11d,12d,13d'), ('11f', 'X-11f'), ('12f', 'X-12f'), ('11f+12f', 'X-11f,12f'),
        ('11g', 'X-11g'), ('12g', 'X-12g'), ('11g+12g', 'X-11g,12g'), ('11h', 'X-11h')]


def main():
    out = {}
    for ion, (t, fam, fq, ch) in ION.items():
        loo = j(R18, f'q18_{t}_fam_loo.json')['sets']
        x0 = fam['sets']['X-']
        r = dict(fam=dict(T1=x0['T1'], T2=x0['T2']), rows={})
        for nm, s in ROWS:
            v = loo.get(s) or fam['sets'].get(s)
            r['rows'][nm] = dict(dT2=x0['T2'] - v['T2'], dT1pct=100 * (x0['T1'] / v['T1'] - 1), frac=1 - v['frac'])
        q = fq['sets']
        r['n3'] = q['X-11i']['T2'] - q['X-11i,h2']['T2']
        r['i1'] = q['X-h2']['T2'] - q['X-11i,h2']['T2']
        p = ch['final']['proj']
        r['chain'] = {b['item']: b['dT2'] for b in p['budget']}
        r['final'] = dict(T2=p['final']['T2'], u2=p['u2'], T1=p['final']['T1'])
        rf = ch['refit_final']['proj']
        r['fit'] = {k: dict(T2=rf[k]['T2'], u=rf[k]['T2_err']) for k in ('free', 'fam', 'final')}
        out[ion] = r
    print('| removed | Fe²⁺ ΔT2 (mEh) | Co³⁺ ΔT2 (mEh) | Fe²⁺ ΔT1 % | Co³⁺ ΔT1 % |')
    print('|---|---|---|---|---|')
    f, c = out['Fe²⁺'], out['Co³⁺']
    for nm, _ in ROWS:
        a, b = f['rows'][nm], c['rows'][nm]
        print(f"| {nm} | {a['dT2']:+.3f} | {b['dT2']:+.3f} | {a['dT1pct']:+.2f} | {b['dT1pct']:+.2f} |")
    print(f"| fam T2 (exact) | {f['fam']['T2']:.3f} | {c['fam']['T2']:.3f} | | |")
    for ion, r in out.items():
        print(ion, 'n3', round(r['n3'], 4), 'first i', round(r['i1'], 4), 'chain sum', round(sum(r['chain'].values()), 4),
              'final', r['final'], 'fit', r['fit'])
    json.dump(out, open(os.path.join(R18, 'phase18_t2loo.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
