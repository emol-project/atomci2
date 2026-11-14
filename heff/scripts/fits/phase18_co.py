#!/usr/bin/env python3
"""Phase 18 Step 2: Co3+ (Co IV 3d6) fits on the NIST levels, same code path as Fe2+ (phase17q11 / phase17q12).
  free    (1b) fit (A B C zeta alpha beta T1 T2 free)
  refit   T3, T4, f1..f12 fixed at ab initio values, lambda scale, F tests (needs the fam H_eff of cond)
-> results/phase18/phase18_co.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15u_bfit as BF  # noqa: E402
import phase17q11 as Q  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase18')
JS = os.path.join(OUT, 'phase18_co.json')
FN = 'Co_IV'
HANSEN = dict(p3p=6.17, p3pp=1.74, zeta=592.7, f={1: 103, 2: 6, 3: 175, 4: -150, 5: 47, 6: 33, 7: -13, 8: 261, 9: -36, 10: -30, 11: -5, 12: -16})
import phase18_nist  # noqa: E402,F401  (registers co, Co IV parser)


def free():
    obs, N = Q.nist_obs('co')
    lev = BF.nist(FN, N)
    f = Q.bfit_free('co')
    p = Q.to_bc(f['p'])
    n = len(lev)
    print('n levels', n, 'N', N, 'mean error', f.get('rms'))
    print({k: round(float(v), 3) for k, v in p.items()})
    print('err', {k: round(float(v), 3) for k, v in f['err'].items()})
    print('Hansen T1, T2 mEh', 6.17 * 1.83813, 1.74 * 1.83813, 'zeta', 592.7)
    print('largest residuals', sorted([(round(float(r), 1), o[0], o[1], round(o[2], 1)) for r, o in zip(f['resid'], obs)], key=lambda x: -abs(x[0]))[:8] if len(f['resid']) == len(obs) else 'n/a')
    old = json.load(open(JS)) if os.path.exists(JS) else {}
    old['free'] = dict(n=n, p={k: float(v) for k, v in p.items()}, err={k: float(v) for k, v in f['err'].items()}, mean_error=float(f['rms']))
    json.dump(old, open(JS, 'w'), indent=1)


HF = dict(zip([f'f{i}' for i in range(1, 13)], [(103, 30), (6, 10), (175, 9), (-150, 8), (47, 5), (33, 5), (-13, 8), (261, 7),
                                                 (-36, 5), (-30, 5), (-5, 8), (-16, 8)]))   # Hansen et al. 1988 Table 2, Co IV d6 (cm-1, Tr f^2 = 1 in d4)
SETS = ['X-', 'X-11d,12d,13d', 'R']
DUMP = os.path.join(OUT, 'heff_dump', 'q18_cond_fam.npz')


def three6(X, ion='co'):
    """proj / terms / levels of a d6 H_eff (det matrix, cm-1); as phase17q11.three with N = 6, free (1b) zeta of ion."""
    d = Q.ds(6)
    c, res, rank = d.decompose(X)
    if ion not in Q.ZETA:
        Q.ZETA[ion] = float(Q.bfit_free(ion)['p']['zeta'])
    p0 = dict(c, zeta=Q.ZETA[ion])
    ft = Q.fit_terms(6, X, p0)
    fl = Q.fit_levels(ion, X, Q.ZETA[ion], p0)
    return dict(proj=dict(c, resid=res, rank=rank), terms={**ft['p'], 'rms': ft['rms'], 'n': ft['n']},
                levels={**fl['p'], 'rms': fl['rms'], 'n': fl['n']})


def x_of(z, s):
    import phase13b as B
    import phase13d as D
    from atomci_heff import dshell as SH
    by = {(L, twoS): z[f'{s}|{t}'] for L, twoS, m, t in D.term_list(6)}
    return SH.from_vec(6, B.vec_from_blocks(6, by) * Q.EH_CM)


def ana():
    import phase17q12 as Q12
    import phase17q20 as Q20
    z = np.load(DUMP)
    sc = json.load(open(os.path.join(OUT, 'q18_cond_fam.json')))['sets']
    st = json.load(open(os.path.join(ROOT, 'results', 'phase15', 'phase15j_18_cond_fam_d5f2g2h1.json')))['fam_d5f2g2h1']['exact']
    A = Q20.jlA()
    res = {}
    for s in SETS:
        r = three6(x_of(z, s))
        r['frac'] = sc[s]['frac']
        res[s] = r
        p = r['proj']
        print(f"{s:15s} proj T1 {p['T1']:.3f} T2 {p['T2']:.3f} T3 {p['T3']:+.3f} T4 {p['T4']:+.3f} B {p['B']:.1f} C {p['C']:.1f} beta {p['beta']:+.1f} "
              f"(resid {p['resid']:.1e}) | terms {r['terms']['T1']:.3f} {r['terms']['T2']:.3f} | levels {r['levels']['T1']:.3f} {r['levels']['T2']:.3f} rms {r['levels']['rms']:.1f}")
        print('   f (Tr f^2 = 1 in d4) ' + ' '.join(f"{k} {p[k] * np.sqrt(A[k]):+.0f}" for k in HF))
    print('X- check against the stored exact: T1', st['proj_T1'], 'T2', st['proj_T2'])
    h = np.array([HF[k][0] for k in HF], float)
    for s in SETS:
        o = np.array([res[s]['proj'][k] * np.sqrt(A[k]) for k in HF])
        res[s]['vs_hansen'] = dict(cos=float(o @ h / np.linalg.norm(o) / np.linalg.norm(h)), slope=float(o @ h / (h @ h)),
                                   norm=float(np.linalg.norm(o)), norm_H=float(np.linalg.norm(h)))
        print(s, 'f vs Hansen', res[s]['vs_hansen'])
    old = json.load(open(JS)) if os.path.exists(JS) else {}
    old['fam'] = res
    old['hansen_f'] = HF
    old['jl_sqrtA'] = {k: float(np.sqrt(A[k])) for k in HF}
    json.dump(old, open(JS, 'w'), indent=1, default=float)


def tests(sets, final=None, ion='co'):
    """NIST fit (Co IV, or Fe III with ion fe) with each extra-operator set fixed (fixX), scaled (lam), and with T1, T2
    also fixed (fixX12); phase17q12.fe_tests / fit logic."""
    from scipy.stats import f as fdist
    obs, N = Q.nist_obs(ion)
    d = Q.ds(N)
    p0 = Q.to_bc(Q.bfit_free(ion)['p'])
    n = len(obs)

    def run(names, fixed):
        f = Q.fit(d, obs, d.levels, names, fixed, [p0.get(q, 1.0) for q in names])
        return f, float(np.sum(np.square(f['resid'])))
    f0, c0 = run(Q.NM1B, {})
    out = {'free': dict(mean_error=f0['rms'], T1=f0['p']['T1'], T2=f0['p']['T2'], T1_err=f0['err']['T1'], T2_err=f0['err']['T2'], n=n)}
    print(f"free: mean error {f0['rms']:.2f}, T1 {f0['p']['T1']:.3f} ± {f0['err']['T1']:.3f}, T2 {f0['p']['T2']:.3f} ± {f0['err']['T2']:.3f}")
    for key, X in sets.items():
        d.ops['Xab'] = sum(c * d.ops[k] for k, c in X.items())
        f1, c1 = run(Q.NM1B, dict(X))
        fl, cl = run(Q.NM1B + ['Xab'], {})
        del d.ops['Xab']
        dl = n - len(Q.NM1B) - 1
        F = lambda c, q: (c - cl) / q / (cl / dl)                    # noqa: E731
        r = dict(mean_error=f1['rms'], T1=f1['p']['T1'], T1_err=f1['err']['T1'], T2=f1['p']['T2'], T2_err=f1['err']['T2'],
                 lam=fl['p']['Xab'], lam_err=fl['err']['Xab'], lam_mean_error=fl['rms'],
                 p_lam0=float(fdist.sf(F(c0, 1), 1, dl)), p_lam1=float(fdist.sf(F(c1, 1), 1, dl)),
                 gain=f0['rms'] / f1['rms'], B=f1['p'].get('B'), beta=f1['p']['beta'])
        if final:
            f2, c2 = run([q for q in Q.NM1B if q not in ('T1', 'T2')], dict(X, **final))
            d1 = n - len(Q.NM1B)
            Ft = (c2 - c1) / 2 / (c1 / d1)
            r.update(fixX12_mean_error=f2['rms'], F_12_given_X=Ft, p_12_given_X=float(fdist.sf(Ft, 2, d1)))
        out[key] = r
        print(f"{key:22s} mean error {f0['rms']:.2f} -> {r['mean_error']:.2f} (factor {r['gain']:.2f}), lam {r['lam']:.3f} ± {r['lam_err']:.3f} "
              f"(p lam=0 {r['p_lam0']:.3g}, lam=1 {r['p_lam1']:.3g}), T1 {r['T1']:.3f} ± {r['T1_err']:.3f}, T2 {r['T2']:.3f} ± {r['T2_err']:.3f}"
              + (f"; T1, T2 also fixed at {final}: mean error {r['fixX12_mean_error']:.2f}, p {r['p_12_given_X']:.3g}" if final else ''), flush=True)
    return out


def refit():
    import phase17q12 as Q12
    d = json.load(open(JS))
    sets = {}
    for s in ('X-', 'X-11d,12d,13d'):
        p = d['fam'][s]['proj']
        sets[f'fam {s}'] = {k: p[k] for k in Q12.EXTRA}
        sets[f'fam {s} T3T4 only'] = {k: p[k] for k in ('T3', 'T4')}
        sets[f'fam {s} f only'] = {k: p[k] for k in Q12.EXTRA if k.startswith('f')}
    fin = d.get('final', {}).get('proj')
    out = tests(sets, dict(T1=fin['T1'], T2=fin['T2']) if fin else dict(T1=d['fam']['X-']['proj']['T1'], T2=d['fam']['X-']['proj']['T2']))
    d['refit'] = out
    json.dump(d, open(JS, 'w'), indent=1, default=float)


HD = os.path.join(OUT, 'heff_dump')
JS_FE = os.path.join(OUT, 'phase18_fe2.json')      # Fe2+ on the Co3+ chain (fefinal, ferefit2)
FAMIQ = os.path.join(HD, 'q18_coti1_famiq.npz')         # phase18_cofamiq.sh (sets as phase17q18 for Fe2+)


HD17 = os.path.join(ROOT, 'results', 'phase17', 'heff_dump')


def dump_file(ion, sp):
    """exact H_eff blocks of a chain space: Co3+ the queue's own dump (18_*) or phase18_codump.sh (q18_*); Fe2+ the
    Phase 17 dumps (phase17q11_dump.py q11_*, queue 17_*)."""
    for d, pre in ((HD, '18'), (HD, 'q18'), (HD17, 'q11'), (HD17, '17')):
        f = os.path.join(d, f'{pre}_{ion}_{sp}_{sp}_iter.npz')
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f'no H_eff dump for {ion} {sp}')


def crep(src, lev='co'):
    """('sp', ion, space) | ('set', npz, set) -> three6 (cached in phase18_co_rep.json; Fe2+ keys prefixed 'fe|')."""
    cp = os.path.join(OUT, 'phase18_co_rep.json')
    cache = json.load(open(cp)) if os.path.exists(cp) else {}
    ks = tuple(os.path.relpath(x, ROOT) if os.path.isabs(x) else x for x in src)    # heff: no absolute paths in keys
    k = '|'.join(ks) if lev == 'co' else '|'.join((lev,) + ks)
    if k not in cache:
        if src[0] == 'sp':
            z, s = np.load(dump_file(src[1], src[2])), 'exact'
        else:
            z, s = np.load(src[1]), src[2]
        cache[k] = three6(x_of(z, s), lev)
        json.dump(cache, open(cp, 'w'), indent=1, default=float)
    return cache[k]


CFG = {'co': dict(base=('set', os.path.join(OUT, 'heff_dump', 'q18_cond_fam.npz'), 'X-'),
                 famiq=os.path.join(OUT, 'heff_dump', 'q18_coti1_famiq.npz'),
                 nd='cond', t2='cot2', ti1='coti1', ti5='coti5', op='coop', dk='codk'),
       'fe': dict(base=('sp', 'fend', 'fam_d5f2g2h1'), famiq=os.path.join(HD17, 'q18_feti1_famiq.npz'),
                  nd='fend', t2='fet2', ti1='feti1', ti5='feti5', op='feop', dk='fedk')}


def chain(R, ion='co'):
    """Co3+ final chain in representation R (proj: every operator incl. T3, T4, f1..f12; terms / levels: the (1b) model
    operators), the Fe2+ chain of phase17q11.chain with Co3+'s own pairs:
      n = 3          coti1 famiq X-11i,h2 -> X-11i (in fam - 12f, as Fe2+ item 11); without the famiq dump the RAS
                     pair cond k1fh -> k1fh_n3 (flagged)
      all virtual s  cond k1fh -> k1fh_sall, T1 / T2 x the Cr3+ transfer factors (as Fe2+), the rest raw
      fam -> besth   the Cr3+ increment (T1 %, T2, B, C, beta; as Fe2+)
      hole           cond k1fh_n3 -> u3c1 (own)
      carried d3     three virtual electrons, 2s open, l >= 7 (the Cr3+ values of phase17q11 'carry')
      2p             own cot2 k1f_tnfg0 -> k1f_2p_h_tnfg0, T1 % x Cr3+ (k1fh_t % / P'' %) + the Cr3+ 2nd pair and g
                     (as Fe2+), T2 and the rest raw
      i functions    famiq X-11i,h2 -> X-h2 (first i) + coti1 -> coti5 k1fhi (second i); without famiq the k1fhi pair only
      DK3            coop -> codk rasK1r (own)
    T1: own %-items on the base, + hole (absolute), then x (1 + carried % / 100) as phase17q11; other operators additive.
    Uncertainties (T1, T2) as phase17_final: own items U_TRANSFER, carried hypot(U_TRANSFER, U_ION Fe2+ 0.30), famiq items
    U_LARGE, the s transfer as Fe2+, hole U_HOLE; d3-only items carry nothing for T3, T4, f."""
    import phase17_final as PF
    C = CFG[ion]
    g = lambda src: crep(src, ion)[R]                               # noqa: E731
    base = g(C['base'])
    ks = [k for k in base if k not in ('resid', 'rank', 'rms', 'n', 'A', 'zeta')]
    sp = lambda ion, s: ('sp', ion, s)                              # noqa: E731
    fq = lambda s: ('set', C['famiq'], s)                                # noqa: E731

    def inc(a, b):          # a - b on every operator; T1 also as % of b
        a, b = g(a), g(b)
        r = {k: a[k] - b[k] for k in ks}
        r['T1%'] = 100 * (a['T1'] / b['T1'] - 1)
        return r
    qc = json.load(open(os.path.join(ROOT, 'results', 'phase17', 'phase17q11_chain.json')))
    qc = qc[R if R in qc else 'proj']
    fei = qc['fe_items']
    items, flags = {}, []
    have_fq = os.path.exists(C['famiq'])
    if have_fq:
        items['n = 3 configurations'] = (inc(fq('X-11i'), fq('X-11i,h2')), 'large')   # with n3 - without
    else:
        items['n = 3 configurations'] = (inc(sp(C['nd'], 'k1fh_n3'), sp(C['nd'], 'k1fh')), 'own')
        flags.append('n = 3 from the RAS pair (no famiq dump)')
    s = inc(sp(C['nd'], 'k1fh_sall'), sp(C['nd'], 'k1fh'))
    raw = dict(s)
    s['T1%'] *= fei['all virtual s']['tf1']
    s['T2'] *= fei['all virtual s']['tf2']
    items['all virtual s'] = (s, 'stf', raw)
    fb = {k: 0.0 for k in ks}
    fb['T1%'], fb['T2'] = fei['fam → besth (f, h)']['d1'], fei['fam → besth (f, h)']['d2']
    items['fam → besth (f, h)'] = (fb, 'carried')
    hole = inc(sp(C['nd'], 'u3c1'), sp(C['nd'], 'k1fh_n3'))
    # 2p: as the Fe2+ rule in phase17q11.chain, with the Cr3+ calibration (d3 reps, cached by phase17q11)
    gq = lambda src: Q.rep(src)['proj' if R == 'proj' else R]      # noqa: E731
    pc = lambda a, b: 100 * (gq(a)['T1'] / gq(b)['T1'] - 1)       # noqa: E731
    S = Q.SRC
    cp, ck = pc(S['cr2p'][1], S['cr2p'][0]), pc(S['p2'], S['t'])
    xt = sum(w * pc(S['P'][b][1], S['P'][b][0]) for b, w in (('crt3', 1), ('crt2', -1), ('crt2g', 1), ('crt2', -1)))
    p = inc(sp(C['t2'], 'k1f_2p_h_tnfg0'), sp(C['t2'], 'k1f_tnfg0'))
    p2raw = p['T1%']
    p['T1%'] = p2raw * ck / cp + xt
    items['2p open, tight d, f (+ 2nd pair, g)'] = (p, 'own')
    i2 = inc(sp(C['ti5'], 'k1fhi'), sp(C['ti1'], 'k1fhi'))
    if have_fq:
        i1 = inc(fq('X-h2'), fq('X-11i,h2'))                     # with first i - without
        items['i functions'] = ({k: i1[k] + i2[k] for k in i2}, 'i', i1, i2)
    else:
        items['i functions'] = (i2, 'own')
        flags.append('i functions: second i only (no famiq dump)')
    for nm in ('three virtual electrons', '2s open', 'l ≥ 7'):
        c = {k: 0.0 for k in ks}
        for q, v in qc['carry'][nm].items():
            if v is not None:
                c['T1%' if q == 'T1' else q] = v
        items[nm] = (c, 'carried')
    items['DK3 (per ion)'] = (inc(sp(C['dk'], 'rasK1r'), sp(C['op'], 'rasK1r')), 'own')
    first = ('n = 3 configurations', 'all virtual s', 'fam → besth (f, h)')
    T1s = base['T1'] * (1 + sum(items[n][0]['T1%'] for n in first) / 100)
    T1h = T1s + hole['T1']
    pct = sum(v[0]['T1%'] for n, v in items.items() if n not in first)
    fin = {k: base[k] + hole[k] + sum(v[0][k] for v in items.values()) for k in ks}
    fin['T1'] = T1h * (1 + pct / 100)
    bud = []
    fct = {'own': PF.U_TRANSFER, 'carried': np.hypot(PF.U_TRANSFER, PF.U_ION['Fe²⁺']), 'large': PF.U_LARGE}
    for nm, v in items.items():
        d, kind = v[0], v[1]
        if kind == 'stf':
            r0 = v[2]
            u1 = np.hypot(PF.U_TRANSFER * d['T1%'], 0.5 * (d['T1%'] - r0['T1%']))
            u2 = np.hypot(PF.U_TRANSFER * d['T2'], 0.5 * (d['T2'] - r0['T2']))
        elif kind == 'i':
            u1 = np.hypot(PF.U_LARGE * v[2]['T1%'], PF.U_TRANSFER * v[3]['T1%'])
            u2 = np.hypot(PF.U_LARGE * v[2]['T2'], PF.U_TRANSFER * v[3]['T2'])
        else:
            u1, u2 = fct[kind] * abs(d['T1%']), fct[kind] * abs(d['T2'])
            if nm in PF.U_FIX:
                u1, u2 = np.hypot(u1, PF.U_FIX[nm][0]), np.hypot(u2, PF.U_FIX[nm][1])
        ref = T1h if nm not in first else base['T1']
        bud.append(dict(item=nm, kind=kind, dT1pct=d['T1%'], dT2=d['T2'], u1=float(abs(u1) / 100 * ref), u2=float(abs(u2)),
                        **{f'd{k}': d[k] for k in ks if k not in ('T1', 'T2')}))
    tf = 0.0 if R == 'proj' else PF.U_TRANSFER
    bud.append(dict(item='hole remainder', kind='own', dT1=hole['T1'], dT2=hole['T2'],
                    u1=float(np.hypot(PF.U_HOLE[0], tf * hole['T1'])), u2=float(np.hypot(PF.U_HOLE[1], tf * hole['T2'])),
                    **{f'd{k}': hole[k] for k in ks if k not in ('T1', 'T2')}))
    u1 = float(np.sqrt(sum(b['u1'] ** 2 for b in bud)))
    u2 = float(np.sqrt(sum(b['u2'] ** 2 for b in bud)))
    return dict(base={k: base[k] for k in ks}, final=fin, u1=u1, u2=u2, T1s=T1s, T1h=T1h, pct=pct, budget=bud,
                flags=flags, p2=dict(raw=p2raw, ck=ck, cp=cp, xt=xt))


def final(ion='co'):
    import phase17q20 as Q20
    A = Q20.jlA()
    js, hf = (JS, HF) if ion == 'co' else (JS_FE, Q20.H_FE)
    d = json.load(open(js)) if os.path.exists(js) else {}
    d['final'] = {}
    for R in ('proj', 'terms', 'levels'):
        c = chain(R, ion)
        d['final'][R] = c
        f = c['final']
        print(f"== {R}: base T1 {c['base']['T1']:.3f} T2 {c['base']['T2']:.3f} -> T1 {f['T1']:.3f} ± {c['u1']:.3f}  T2 {f['T2']:.3f} ± {c['u2']:.3f}"
              + (f"  T3 {f['T3']:+.3f} T4 {f['T4']:+.3f}" if 'T3' in f else '') + f"  B {f['B']:.1f} C {f['C']:.1f} beta {f['beta']:+.1f}"
              + (f"  flags {c['flags']}" if c['flags'] else ''))
        for b in c['budget']:
            print(f"   {b['item']:38s} T1 {b.get('dT1pct', float('nan')):+7.3f} % {b.get('dT1', float('nan')):+.3f}  T2 {b['dT2']:+.4f}  "
                  f"u {b['u1']:.3f} / {b['u2']:.4f}" + (f"  T3 {b['dT3']:+.3f} T4 {b['dT4']:+.3f}" if 'dT3' in b else ''))
        if R == 'proj':
            print('   f final (Tr f^2 = 1 in d4) ' + ' '.join(f"{k} {f[k] * np.sqrt(A[k]):+.0f}" for k in hf)
                  + '\n   f base                     ' + ' '.join(f"{k} {c['base'][k] * np.sqrt(A[k]):+.0f}" for k in hf))
            h = np.array([hf[k][0] for k in hf], float)
            o = np.array([f[k] * np.sqrt(A[k]) for k in hf])
            c['vs_hansen'] = dict(cos=float(o @ h / np.linalg.norm(o) / np.linalg.norm(h)), slope=float(o @ h / (h @ h)))
            print('   final f vs Hansen', c['vs_hansen'])
    json.dump(d, open(js, 'w'), indent=1, default=float)


def refit2(ion='co'):
    """four-body refit with the corrected (final proj) extras; T1, T2 fixed at the final proj and levels values.
    Also the fam extras (all / T3T4 / f) and fam without 11d-13d (Fe2+ q13_fend_fam, Co3+ q18_cond_fam X-11d,12d,13d)."""
    import phase17q12 as Q12
    js = JS if ion == 'co' else JS_FE
    d = json.load(open(js))
    f = d['final']['proj']['final']
    sets = {'final': {k: f[k] for k in Q12.EXTRA}, 'final T3T4 only': {k: f[k] for k in ('T3', 'T4')},
            'final f only': {k: f[k] for k in Q12.EXTRA if k.startswith('f')}}
    nod = os.path.join(HD17, 'q13_fend_fam.npz') if ion == 'fe' else DUMP
    for nm, src in (('fam', CFG[ion]['base']), ('fam X-11d,12d,13d', ('set', nod, 'X-11d,12d,13d'))):
        p = crep(src, ion)['proj']
        sets[nm] = {k: p[k] for k in Q12.EXTRA}
        sets[nm + ' T3T4 only'] = {k: p[k] for k in ('T3', 'T4')}
        sets[nm + ' f only'] = {k: p[k] for k in Q12.EXTRA if k.startswith('f')}
        d.setdefault('fam_sets', {})[nm] = crep(src, ion)
    out = {}
    for R in ('proj', 'levels'):
        fr = d['final'][R]['final']
        print(f'-- T1, T2 fixed at final {R}: {fr["T1"]:.3f}, {fr["T2"]:.3f}')
        out[R] = tests(sets, dict(T1=fr['T1'], T2=fr['T2']), ion)
    d['refit_final'] = out
    json.dump(d, open(js, 'w'), indent=1, default=float)


if __name__ == '__main__':
    {'free': free, 'ana': ana, 'refit': refit, 'final': final, 'refit2': refit2,
     'fefinal': lambda: final('fe'), 'ferefit2': lambda: refit2('fe')}[sys.argv[1]]()
