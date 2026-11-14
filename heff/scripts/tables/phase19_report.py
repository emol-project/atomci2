#!/usr/bin/env python3
"""Phase 19 report: results/phase19/phase19_{fits,nist,pos}.json -> results/phase19/phase19_tables.md (plain-text tables)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
HERE = os.path.join(_C.root(), 'results', 'phase19')  # inputs and outputs (was this script's directory)
import phase19 as P  # noqa: E402

EH_CM = 219474.6313705


def L(*a):
    return list(a)


def fit_tables(F):
    out = []
    for ion, lab in (('v', 'V III (3d3, 19 levels)'), ('fe', 'Fe III (3d6, 33 levels)'), ('co', 'Co IV (3d6, 33 levels)')):
        for k, v in F['job1'][ion].items():
            f = v['fit']
            pp = '' if v.get('p') is None else f"p {v['p']:.3g} ({v['ptest']})"
            extra = f", lambda {v['lam']:.3f} +- {v['lam_err']:.3f}, p(lambda=0) {v['p_lam0']:.2g}" if 'lam' in v else ''
            fx = v.get('fixed')
            fxs = (' fixed ' + ', '.join(f'{a} {b:.3f}' for a, b in fx.items())) if isinstance(fx, dict) else ''
            sep = ', ' + pp if pp else ''
            out += ['', f'### {lab}: {k}{fxs}', '', f"mean deviation {f['rms']:.2f} cm-1 (dof {f['dof']}){sep}{extra}, order ok {f['order_ok']}", '',
                    '```', f"{'config':6s} {'term':7s} {'J':>4s} {'E_obs':>10s} {'E_calc':>10s} {'obs-calc':>9s}   nearest 3d^(N-1)ns, same J (E_s - E_obs)"]
            for x in v['table']:
                n = x['near_4s']
                ns = f"{n['dE']:+8.0f}  {n['cfg']} {n['term']}" if n else 'none'
                out.append(f"{x['cfg']:6s} {x['term']:7s} {x['J']:>4s} {x['E_obs']:10.1f} {x['E_calc']:10.1f} {x['resid']:+9.1f}   {ns}")
            out.append('```')
    return out


def job2(F):
    out = ['', '## Job 2: T1, T2 fixed at the computed corrected values, rest of (1b) refitted (F(2, n-8) vs free (1b))', '', '```',
           f"{'ion':6s} {'n':>3s} {'free':>7s} | {'T1 proj':>8s} {'T2 proj':>8s} {'mean':>7s} {'p':>9s} | {'T1 (1b)':>8s} {'T2 (1b)':>8s} {'mean':>7s} {'p':>9s} | committed p (proj, (1b))"]
    for lab, v in F['job2'].items():
        c = v.get('committed')
        cs = f"{c['proj']['p']:.3g}, {c['levels']['p']:.3g}" if c else 'none (new)'
        out.append(f"{lab:6s} {v['n']:3d} {v['free']['mean_error']:7.2f} | {v['proj']['fixed']['T1']:8.3f} {v['proj']['fixed']['T2']:8.3f} "
                   f"{v['proj']['mean_error']:7.2f} {v['proj']['p']:9.3g} | {v['levels']['fixed']['T1']:8.3f} {v['levels']['fixed']['T2']:8.3f} "
                   f"{v['levels']['mean_error']:7.2f} {v['levels']['p']:9.3g} | {cs}")
    return out + ['```']


def job6(G):
    out = ['', '## Job 6: T1 alone / T2 alone / both fixed at the computed corrected values; F(q, n-8) vs the same fit with T1, T2 free',
           '', 'free = the refitted T1 or T2 (mEh) with its error; ref = mean deviation with T1, T2 free.', '', '```',
           f"{'ion | block':34s} {'R':6s} {'ref':>6s} | {'(a) T1 fixed: mean   p      T2 free':38s} | {'(b) T2 fixed: mean   p      T1 free':38s} | {'(c) both: mean   p':20s}"]
    for blk, v in G.items():
        for R in ('proj', 'levels'):
            a, b, c = (v[f'{R}|{k}'] for k in ('a: T1 fixed', 'b: T2 fixed', 'c: both fixed'))
            fa = a['free']['T2']
            fb = b['free']['T1']
            out.append(f"{blk:34s} {R if R == 'proj' else '(1b)':6s} {v['ref']['mean_error']:6.2f} | "
                       f"T1 {a['fixed']['T1']:6.3f} {a['mean_error']:6.2f} {a['p']:8.2g}  {fa[0]:6.3f} +- {fa[1]:.3f} | "
                       f"T2 {b['fixed']['T2']:6.3f} {b['mean_error']:6.2f} {b['p']:8.2g}  {fb[0]:6.3f} +- {fb[1]:.3f} | "
                       f"{c['mean_error']:6.2f} {c['p']:8.2g}")
    return out + ['```']


def job3(F):
    out = ['', '## Job 3: Hansen et al. 1988 four-body vector fixed (phase17q20.fits logic; Fe III rerun as the check)', '']
    for ion, lab in (('fe', 'Fe III'), ('co', 'Co IV')):
        o = F['job3'][ion]
        out += [f'### {lab}', '', '```']
        for k in ('free', 'Hansen f fixed (T3 = T4 = 0)', 'fam f fixed (T3 = T4 = 0)', 'fam T3,T4,f fixed'):
            f = o[k]
            out.append(f"{k:32s} mean {f['mean']:6.2f}  T1 {f['p']['T1']:.3f} +- {f['err']['T1']:.3f}  T2 {f['p']['T2']:.3f} +- {f['err']['T2']:.3f}  order {f['order_ok']}")
        for k, v in o['lam'].items():
            out.append(f"lambda on {k:22s} {v['lam']:.3f} +- {v['err']:.3f}  (mean {v['mean']:.2f})")
        for k, f in o['batch'].items():
            out.append(f"{k:32s} mean {f['mean']:6.2f}  " + '  '.join(f"{q} {f['fnorm'][q]:+.0f} ({f['fnorm_err'][q]:.0f})" for q in f['fnorm']))
        out.append('```')
    return out


def job4(N):
    out = ['', '## Job 4: NIST files', '']
    for fn, n in (('Co_IV', 6), ('Fe_VI', 3), ('Fe_III', 6), ('V_III', 3)):
        r = N[fn]
        hi, lo = r['highest_dN'], r['lowest_dN1_4s']
        out += [f'### {fn}: {r["n_rows"]} levels, {len(r["configs"])} configurations ({r["n_dN"]} levels of 3d{n})', '',
                f"highest 3d{n} level: {hi[1]} J={P.jstr(hi[2])} {hi[3]:.1f} cm-1; lowest 3d{n - 1}4s: {lo[0]} {lo[1]} J={P.jstr(lo[2])} {lo[3]:.1f} cm-1", '', '```']
        for c, v in sorted(r['configs'].items(), key=lambda x: x[1]['Emin']):
            out.append(f"{c:32s} {v['n']:4d} levels  {v['Emin']:10.1f} - {v['Emax']:10.1f}")
        out.append('```')
    return out


def centroid(rows):
    w = sum(r[2] + 1 for r in rows)
    return sum((r[2] + 1) * r[3] for r in rows) / w


def nist_terms():
    """term centroids (2J+1) used for P5."""
    fe = P.nist_all('Fe_III')
    v = P.nist_all('V_III')
    pick = lambda rows, cfg, term: [r for r in rows if r[0] == cfg and r[1] == term]  # noqa: E731
    g_fe = centroid(pick(fe, '3d6', '5D'))
    g_v = centroid(pick(v, '3d3', 'a 4F'))
    out = {'fe': dict(ground=g_fe, **{k: centroid(pick(fe, c, t)) - g_fe for k, c, t in
                                     (('7S', '3d5.(6S).4s', '7S'), ('5S', '3d5.(6S).4s', '5S'), ('5D', '3d5.(4D).4s', '5D'))}),
           'v': dict(ground=g_v, **{k: centroid(pick(v, c, t)) - g_v for k, c, t in
                                   (('4F', '3d2.(3F).4s', 'b 4F'), ('2D', '3d2.(1D).4s', 'c 2D'))})}
    out['v']['2D_terms_available'] = sorted({r[1] for r in v if r[0] == '3d2.(1D).4s'})
    return out


def job5(S):
    out = ['', '## Job 5: 3d^(N-1) ns roots in the stored H (block Davidson, residual < 1e-8)', '']
    if not S:
        return out + ['not run']
    gs = {}
    for key, v in S.items():
        if 'E_abs_Eh' in v and key in ('fe:5D', 'co:5D', 'v:4F'):
            gs[key.split(':')[0]] = v['E_abs_Eh'][0]
    nt = nist_terms()
    for key, v in S.items():
        if 'roots' not in v:
            out += [f"{key}: {v['status']} ({v['run']})"]
            continue
        ion = key.split(':')[0]
        E0 = gs[ion]
        out += ['', f"### {key} ({v['run']}, {v['N_csf']} CSFs, {v['nP']} P CSF; {v['k']} roots, {v['wall_s']:.0f} s)", '', '```',
                f"{'root':>4s} {'E - E_gs (cm-1)':>16s} {'wP':>7s} " + ' '.join(f"{o:>7s}" for o in v['roots'][0]['w_ns']) + f" {'other':>7s}"]
        for x, Ea in zip(v['roots'], v['E_abs_Eh']):
            out.append(f"{x['i']:4d} {(Ea - E0) * EH_CM:16.0f} {x['wP']:7.4f} " + ' '.join(f"{w:7.4f}" for w in x['w_ns'].values()) +
                       f" {x['w_other']:7.4f}" + ('   <- first 3d^(N-1)ns root' if x['i'] == v['found'] else ''))
        out.append('```')
        t = key.split(':')[1]
        if t in nt[ion]:
            Ef = (v['E_abs_Eh'][v['found']] - E0) * EH_CM
            out.append(f"NIST term centroid of 3d^(N-1)4s {t} above the ground-term centroid: {nt[ion][t]:.0f} cm-1; computed - NIST {Ef - nt[ion][t]:+.0f} cm-1")
    return out


def main():
    F = json.load(open(_C.need(os.path.join(HERE, 'phase19_fits.json'))))
    N = json.load(open(_C.need(os.path.join(HERE, 'phase19_nist.json'))))
    sp = os.path.join(HERE, 'phase19_pos.json')
    S = json.load(open(_C.need(sp)))
    out = ['# Phase 19 tables (generated by phase19_report.py)', '', '## Job 1: level residuals (cm-1), sorted by |obs - calc|']
    G = json.load(open(_C.need(os.path.join(HERE, 'phase19_single.json'))))
    out += fit_tables(F) + job2(F) + job3(F) + job4(N) + job5(S) + job6(G)
    open(os.path.join(HERE, 'phase19_tables.md'), 'w').write('\n'.join(out) + '\n')
    print(json.dumps(nist_terms(), indent=1))


if __name__ == '__main__':
    main()
