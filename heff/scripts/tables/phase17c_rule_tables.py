#!/usr/bin/env python3
"""Tables of the Phase 17 rule-based Q1 (results/phase17/crule_<run>.json) -> results/phase17/phase17c_rule_tables.md.
Columns: dim(P+Q1), fraction, T1/ex - 1, T2 - ex, B/ex, C/ex, beta - ex; leave-one-out rows also dT1 (%), dT2 against
Q1_rule. Exact references: the full stored-H values of each space (17 report)."""
import json
import os

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase17')
RUNS = {  # run: (label, exact T1, T2, B, C, beta)
    'vnd_s8': ('V²⁺ Step 8', 8.695, 1.786, 820.0, 3003.1, -24.5),
    'crnd_s8': ('Cr³⁺ Step 8', 12.103, 1.714, 997.7, 3702.1, 79.5),
    'mnnd_s8': ('Mn⁴⁺ Step 8', 14.317, 1.656, 1159.9, 4327.9, 153.7),
    'crnd_fam': ('Cr³⁺ fam', 'phase15k_17_crnd_fam_d5f2g2h1.json', 'fam_d5f2g2h1'),   # read in exact()
    'fend_fam': ('Fe²⁺ fam', 9.853, 3.786, None, None, None),
    'crti1_s8i': ('Cr³⁺ Step 8 + one i (Q4-i)', 12.028, 1.510, 997.2, 3685.1, 88.4),
}


def exact(ex):
    """(label, T1, T2, B, C, beta); a (label, json, space) entry is read from results/phase15/<json> exact.proj,
    rounded as the other entries (3 decimals in mEh, 1 in cm-1)."""
    if not isinstance(ex[1], str):
        return ex
    p = json.load(open(_C.need(os.path.join(ROOT, 'results', 'phase15', ex[1]))))[ex[2]]['exact']['proj']
    return (ex[0], round(p['T1'], 3), round(p['T2'], 3), round(p['B'], 1), round(p['C'], 1), round(p['beta'], 1))


def row(v, ex, ref=None):
    _, T1, T2, B, C, be = ex
    out = [f'{v["n1"]:,}', f'{100 * v["frac"]:.2f} %', f'{100 * (v["T1"] / T1 - 1):+.2f} %', f'{v["T2"] - T2:+.3f}']
    out += [f'{v["B"] / B:.4f}', f'{v["C"] / C:.4f}', f'{v["beta"] - be:+.1f}'] if B else ['–'] * 3
    if ref:
        out += [f'{100 * (v["T1"] - ref["T1"]) / T1:+.2f} %', f'{v["T2"] - ref["T2"]:+.3f}']
    return out


def main():
    lines = []
    for run, ex in RUNS.items():
        ex = exact(ex)
        p = os.path.join(OUT, f'crule_{run}.json')
        _C.need(p)                                     # heff: required (was: skip the run)
        r = json.load(open(p))
        s = r['sets']
        nsub = {k: len(v) for k, v in r['rule'].items()}
        lines += [f'### {ex[0]} (full space {r["ntot"]:,} CSFs; {r["wall_s"] / 60:.0f} min, '
                  f'{r["maxrss_MB"] / 1024:.1f} GB)', '',
                  '| Q1 | configurations | dim(P⊕Q1) | fraction | T1/ex − 1 | T2 − ex | B/ex | C/ex | β − ex | '
                  'ΔT1 vs rule | ΔT2 vs rule |', '|' + '---|' * 11]
        R = s['R']
        lines.append('| ' + ' | '.join(['**Q1_rule**', str(R['ncfg'])] + row(R, ex) + ['', '']) + ' |')
        for k, v in s.items():
            if k == 'R':
                continue
            sub = k[2:]
            if not nsub.get(sub):
                lines.append(f'| − {sub} | (none in this space) |' + ' |' * 9)
                continue
            lines.append('| ' + ' | '.join([f'− {sub} ({nsub[sub]} cfg)', str(v['ncfg'])] + row(v, ex, R)) + ' |')
        lines.append('')
    open(os.path.join(OUT, 'phase17c_rule_tables.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
