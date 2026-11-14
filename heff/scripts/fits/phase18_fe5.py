#!/usr/bin/env python3
"""Phase 18 Step 1: Fe5+ (Fe VI 3d3) Final chain, in the same representations as the other d3 ions (phase17q11.chain),
from the stored results; second-order class table; Fe VI measured references.
  final    projection / 16-term / levels chain (Step 10 + own hole, own i, own 2p, own DK3 + Cr3+ carried items);
           levels: all 19 J levels of d3 with zeta 594.1 (Hansen et al. 1988 Table 3; no Fe VI NIST file on disk)
  second   second-order (closure) T1, T2 by class and k k', against Uylings, Complex Atoms (2021) Table 14.5
  refit    (1b) NIST fit and F-tests when data/levels/NIST_ASD_levels_Fe_VI.txt exists
-> results/phase18/phase18_fe5.json, phase18_fe5_tables.md"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase18')
R15, R17 = os.path.join(ROOT, 'results', 'phase15'), os.path.join(ROOT, 'results', 'phase17')
JS = os.path.join(OUT, 'phase18_fe5.json')
P_MEH = 1.83813                                # 1 unit of Hansen's p / Trees' T on the O3'' norm, mEh (403.42286 / 219.4746)
HANSEN = dict(T1=8.35 * P_MEH, u1=0.02 * P_MEH, T2=0.65 * P_MEH, u2=0.02 * P_MEH, zeta=594.1)
UYL_FIT = 8.452 * P_MEH                         # Uylings (2021) Table 14.5, fitted T1 of Fe VI, sign as ours (T1 > 0)
U_ION_FE5 = 0.30                               # Cr3+ item carried over two charges (phase17_final: 0.15 one, 0.30 Fe2+)
Q = ('T1', 'T2', 'B', 'C', 'beta')
NM2P = '2p open, tight d, f (+ 2nd pair, g)'


def load(tag, sp):
    return json.load(open(os.path.join(R15, f'phase15k_18_{tag}_{sp}.json')))[sp]


def hansen_series():
    """Hansen Table 1, mEh (same factor for all d3 ions)."""
    return {'V²⁺': (4.49, .02, 1.05, .02), 'Cr³⁺': (6.29, .02, .81, .02), 'Mn⁴⁺': (7.44, .02, .71, .02), 'Fe⁵⁺': (8.35, .02, .65, .02)}


def rep_all():
    """proj / terms / levels of the Fe5+ runs (cached in JS): same code as phase17q11.three, ion fe5.
    With the Fe VI NIST file: its 19 levels and the free (1b) zeta (as the other ions); without it: the J list of the
    exact H + Hansen's zeta."""
    import phase17q11 as Q11
    if os.path.exists(os.path.join(_C.REFDATA, 'levels', 'NIST_ASD_levels_Fe_VI.txt')):
        import phase18_nist  # noqa: F401
        Q11.ZETA['fe5'] = float(Q11.bfit_free('fe5')['p']['zeta'])
    else:
        nist0 = Q11.nist_obs

        def nist_fe5(ion):
            if ion != 'fe5':
                return nist0(ion)
            d = Q11.ds(3)
            # the J list needs zeta != 0: at zeta = 0 the J levels collapse onto the 8 terms
            lev = d.levels(Q11.x_from_proj(load('fe5nd', 'besth_sall')['exact']['proj']) + HANSEN['zeta'] * d.ops['zeta'])
            return [(int(j2), k, 0.0) for j2 in sorted(lev) for k in range(len(lev[j2][0]))], 3
        Q11.nist_obs = nist_fe5
        Q11.ZETA['fe5'] = HANSEN['zeta']
    src = dict(s8=('fe5nd', 'besth_sall'), s10=('fe5nd', 'besth_sall_s4'), k1fh=('fe5nd', 'k1fh'), n3=('fe5nd', 'k1fh_n3'),
               u3c1=('fe5nd', 'u3c1'), i5=('fe5ti5', 'k1fhi'), op=('fe5op', 'k1fhr'), dk=('fe5dk', 'k1fhr'),
               p0=('fe5t2', 'k1f_tnfg0'), p1=('fe5t2', 'k1f_2p_h_tnfg0'))
    out = {}
    for k, (tag, sp) in src.items():
        p = load(tag, sp)['exact']['proj']
        out[k] = Q11.three('fe5', Q11.x_from_proj(p), p)
    return out


def chain(reps, R, ch):
    """As phase17q11.chain for a d3 ion with own hole, own i, own DK3 and own 2p (phase17_final item 3 rule: P'' % x the Cr3+
    calibration ratio % k1fh_t / % P'' + the Cr3+ second-pair and g increments; dT2, dB, dC, dbeta raw P'')."""
    import phase17_final as PF
    import phase17q11 as Q11
    g = lambda k: reps[k][R]                                  # noqa: E731

    def inc(a, b):
        return dict(T1=100 * (a['T1'] / b['T1'] - 1), **{q: a[q] - b[q] for q in Q[1:]})
    C = ch[R]['carry']
    gc = lambda s: Q11.rep(s)[R]                               # noqa: E731  (Cr3+ cached values only)
    cp, ck = inc(gc(Q11.SRC['cr2p'][1]), gc(Q11.SRC['cr2p'][0])), inc(gc(Q11.SRC['p2']), gc(Q11.SRC['t']))
    Pi = {b: inc(gc(o), gc(c)) for b, (c, o) in Q11.SRC['P'].items()}
    xt1 = Pi['crt3']['T1'] - Pi['crt2']['T1'] + Pi['crt2g']['T1'] - Pi['crt2']['T1']
    fp = inc(g('p1'), g('p0'))
    p2 = dict(fp, T1=fp['T1'] * ck['T1'] / cp['T1'] + xt1)
    ci = {k: dict(C[k]) for k in ('three virtual electrons', '2s open', 'l ≥ 7')}
    ci[NM2P], ci['i functions'] = p2, inc(g('i5'), g('k1fh'))
    own = {NM2P, 'i functions'}
    ho = {q: g('u3c1')[q] - g('n3')[q] for q in Q}
    dk = inc(g('dk'), g('op'))
    b = g('s10')
    T1h = b['T1'] + ho['T1']
    pct = sum(c['T1'] for c in ci.values()) + dk['T1']
    T1 = T1h * (1 + pct / 100)
    T2 = b['T2'] + ho['T2'] + sum(c['T2'] for c in ci.values()) + dk['T2']
    bud = []
    for nm, c in ci.items():
        f = PF.U_TRANSFER if nm in own else math.hypot(PF.U_TRANSFER, U_ION_FE5)
        fx = PF.U_FIX.get(nm, (0, 0))
        bud.append((nm, c['T1'], c['T2'], math.hypot(f * abs(c['T1']), fx[0]) / 100 * T1h, math.hypot(f * abs(c['T2']), fx[1])))
    bud.append(('DK3 (per ion)', dk['T1'], dk['T2'], PF.U_TRANSFER * abs(dk['T1']) / 100 * T1h, PF.U_TRANSFER * abs(dk['T2'])))
    tf = 0.0 if R == 'proj' else PF.U_TRANSFER
    bud.append(('hole remainder', ho['T1'], ho['T2'], math.hypot(PF.U_HOLE[0], tf * ho['T1']), math.hypot(PF.U_HOLE[1], tf * ho['T2'])))
    r = dict(base={q: b[q] for q in Q}, T1=T1, T2=T2, T1h=T1h, pct=pct, budget=bud, dk=dk, hole=ho, i=ci['i functions'], p2=p2,
             p2_raw=fp, cal=ck['T1'] / cp['T1'], u1=math.sqrt(sum(x[3] ** 2 for x in bud)), u2=math.sqrt(sum(x[4] ** 2 for x in bud)))
    for q in Q[2:]:
        r[q] = b[q] + ho[q] + sum(c[q] for c in ci.values()) + dk[q]
    return r


def final():
    ch = json.load(open(os.path.join(R17, 'phase17q11_chain.json')))
    reps = rep_all()
    res = dict(reps=reps, chain={R: chain(reps, R, ch) for R in ('proj', 'terms', 'levels')})
    res['series'] = {R: {ion: {k: ch[R]['ions'][ion][k] for k in ('T1', 'T2', 'u1', 'u2')} for ion in ('V²⁺', 'Cr³⁺', 'Mn⁴⁺')} for R in ch}
    res['hansen'], res['uylings_fit_T1'] = HANSEN, UYL_FIT
    old = json.load(open(JS)) if os.path.exists(JS) else {}
    old.update(res)
    json.dump(old, open(JS, 'w'), indent=1, default=float)
    for R, r in res['chain'].items():
        print(f"{R}: base {r['base']['T1']:.3f} {r['base']['T2']:.3f} -> T1 {r['T1']:.3f} ± {r['u1']:.3f} (Hansen {r['T1'] / HANSEN['T1']:.3f}, "
              f"Uylings {r['T1'] / UYL_FIT:.3f}) T2 {r['T2']:.3f} ± {r['u2']:.3f} (Hansen {HANSEN['T2']:.3f}) | B {r['B']:.1f} C {r['C']:.1f} beta {r['beta']:+.1f} | "
              f"pct {r['pct']:+.2f} own2p {r['p2']['T1']:+.3f} (raw {r['p2_raw']['T1']:+.3f}, cal {r['cal']:.3f}) i {r['i']['T1']:+.3f}/{r['i']['T2']:+.3f} "
              f"dk {r['dk']['T1']:+.3f} hole {r['hole']['T1']:+.3f}/{r['hole']['T2']:+.3f}")
        s = res['series'][R]
        print('   series T2', {k: round(v['T2'], 3) for k, v in s.items()}, 'Fe5+', round(r['T2'], 3), 'Mn-Fe5 drop', round(s['Mn⁴⁺']['T2'] - r['T2'], 3),
              '| T1', {k: round(v['T1'], 3) for k, v in s.items()}, round(r['T1'], 3))


# Unit-integral closure vectors A_kk' (FINDINGS.md, Phase 11d; Delta = 1 Eh; 4F 4P 2H 2G 2F 2P; 2D 11, 33, 13)
_R21 = math.sqrt(21)
ANG = {'d': {'22': (-3/98, 1/14, 115/4802, -265/14406, 53/4802, 73/4802, 0, -106/2401, -_R21/98),
             '24': (-10/1029, 10/441, -50/21609, 800/64827, -160/21609, -1310/21609, 0, 640/21609, -10*_R21/3087),
             '44': (275/18522, -275/7938, -2375/388962, -1375/1166886, 275/388962, 11275/388962, 0, -550/194481, 275*_R21/55566)},
       'g': {'22': (48/1715, -16/245, -68/2401, 206/7203, -206/12005, -676/12005, 0, 824/12005, 16*_R21/1715),
             '24': (-50/1029, 50/441, 6850/237699, -800/64827, 160/21609, -790/21609, 0, -640/21609, -50*_R21/3087),
             '44': (1625/101871, -1625/43659, -8125/23532201, -81250/6417873, 16250/2139291, 154375/2139291, 0, -65000/2139291, 1625*_R21/305613)}}
UYL = {'total second order': -9.727, 'fitted': -8.452, 's->3d': -12.067, '3d->s': 0.209, '3d->d 22': -0.198, '3d->d 24': 0.405, '3d->d 44': -0.129,
       '3d->g 22': 2.391, '3d->g 24': 0.710, '3d->g 44': -1.107, '3d->i 44': 0.037}


def _blocks(v):
    a = np.array
    return {'4F': a([[v[0]]]), '4P': a([[v[1]]]), '2H': a([[v[2]]]), '2G': a([[v[3]]]), '2F': a([[v[4]]]), '2P': a([[v[5]]]),
            '2D': a([[v[6], v[8]], [v[8], v[7]]])}


def second():
    os.environ['ION'] = 'fe5nd'
    os.environ.setdefault('HEFF_SOLVER', 'iter')
    import atomci_heff.approx as AP
    import gen_v
    import phase15f as F
    import phase17a as A
    from atomci_heff import integrals as I  # noqa: F401
    from atomci_heff.reader import read_vectors
    a = A.load('fe5nd', 'besth_sall')
    cfgs = [str(c) for c in a['cfgs']]
    eav = np.asarray(a['eav'], float)
    eP = float(a['eav_P'])
    bs = F.Bas('fe.without-diffuse.bas', 26)
    C = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb_fe5nd3', 'be.NO'), bs.nbf)[0]
    P3d = bs.P(2, C[2][:, 0])
    LS = 'spdfghi'

    def parse(c):
        o = AP.parse_label(c)
        return o, [k for k in o if int(k[:-1]) >= 11]

    def is_single(o, vir):
        return o.get('3s', 0) == 2 and o.get('3p', 0) == 6 and o.get('3d', 0) == 2 and sum(o[k] for k in vir) == 1 and o.get('2s', 2) == 2 and o.get('2p', 6) == 6
    rows, per = {}, {}

    def add(name, mask):
        x = A.incr(a, 'C2', mask)
        rows[name] = dict(T1=x['T1'], T2=x['T2'], n=int(mask.sum()))
    m = np.zeros(len(cfgs), bool)
    masks = {k: m.copy() for k in ('s->3d', '2s->3d', '3d->s', '3d->d', '3d->g', '3d->other single')}
    for i, c in enumerate(cfgs):
        o, vir = parse(c)
        if not vir and o.get('3s') == 1 and o.get('3p') == 6 and o.get('3d') == 4 and o.get('2s', 2) == 2:
            masks['s->3d'][i] = True
        elif not vir and o.get('2s', 2) == 1 and o.get('3d') == 4:
            masks['2s->3d'][i] = True
        elif is_single(o, vir):
            let = vir[0][-1]
            masks[{'s': '3d->s', 'd': '3d->d', 'g': '3d->g'}.get(let, '3d->other single')][i] = True
    for k, v in masks.items():
        if v.any():
            add(k, v)
    tot = A.incr(a, 'C2', np.ones(len(cfgs), bool))
    rows['all configurations'] = dict(T1=tot['T1'], T2=tot['T2'], n=len(cfgs))
    used = np.logical_or.reduce(list(masks.values()))
    x = A.incr(a, 'C2', ~used)
    rows['everything else'] = dict(T1=x['T1'], T2=x['T2'], n=int((~used).sum()))
    # k k' split of d and g, per orbital; the sum must reproduce the stored per-configuration C2 increment
    kk = {}
    for let in 'dg':
        L = LS.index(let)
        acc = {q: 0.0 for q in ('22', '24', '44')}
        acct2 = dict(acc)
        for i in np.where(masks[f'3d->{let}'])[0]:
            _, vir = parse(cfgs[i])
            o = vir[0]
            Pv = bs.P(L, C[L][:, gen_v.vec(o) - 1])
            r2, r4 = bs.R(P3d, P3d, P3d, Pv, 2), bs.R(P3d, P3d, P3d, Pv, 4)
            D = eav[i] - eP
            pieces = {'22': r2 * r2 / D, '24': r2 * r4 / D, '44': r4 * r4 / D}
            sum_t1, sum_t2 = 0.0, 0.0
            for q, w in pieces.items():
                d = A.params(_blocks(tuple(w * y for y in ANG[let][q])))
                acc[q] += d['T1']
                acct2[q] += d['T2']
                sum_t1 += d['T1']
                sum_t2 += d['T2']
            one = np.zeros(len(cfgs), bool)
            one[i] = True
            st = A.incr(a, 'C2', one)
            per[o] = dict(R2=r2, R4=r4, Delta=D, sum_T1=sum_t1, stored_T1=st['T1'], sum_T2=sum_t2, stored_T2=st['T2'])
        kk[let] = dict(T1=acc, T2=acct2)
    res = dict(rows=rows, kk=kk, per_orbital=per, closure_total_T1=rows['all configurations']['T1'])
    pi = os.path.join(R17, 'a_fe5ti5_k1fhi.npz')
    if os.path.exists(pi):      # 3d -> i: k = 4 only, so 44 by construction; run in the k1fhi space with the two i (fe5ti5)
        ai = np.load(pi, allow_pickle=True)
        ci = [str(c) for c in ai['cfgs']]
        mi = np.array([(lambda o, v: is_single(o, v) and v[0][-1] == 'i')(*parse(c)) for c in ci])
        x = A.incr(ai, 'C2', mi)
        res['rows']['3d->i'] = dict(T1=x['T1'], T2=x['T2'], n=int(mi.sum()), space='fe5ti5 k1fhi')
    # his normalization: T [cm-1] = -T1 [mEh] / 1.83813
    res['uylings'] = UYL
    res['conversion'] = 'T1 [mEh] = 1.83813 |T| [cm-1], sign: his T1 = -(our T1)'
    old = json.load(open(JS)) if os.path.exists(JS) else {}
    old['second'] = res
    json.dump(old, open(JS, 'w'), indent=1, default=float)
    his = lambda t: -t / P_MEH                     # noqa: E731
    for k, v in res['rows'].items():
        print(f"{k:22s} n {v['n']:4d} T1 {v['T1']:+9.4f} mEh  T2 {v['T2']:+8.4f}  his norm {his(v['T1']):+8.3f}")
    for let, v in kk.items():
        print(let, 'T1 22/24/44', {q: round(x, 4) for q, x in v['T1'].items()}, 'his norm', {q: round(his(x), 3) for q, x in v['T1'].items()})
    print('max |sum - stored| per orbital T1, T2:', max(abs(p['sum_T1'] - p['stored_T1']) for p in per.values()), max(abs(p['sum_T2'] - p['stored_T2']) for p in per.values()))
    print({o: (round(p['R2'], 4), round(p['R4'], 4), round(p['Delta'], 3), round(p['stored_T1'], 4)) for o, p in per.items()})


def tables():
    """Step 1 tables -> results/phase18/phase18_fe5_tables.md (needs final and second in JS)."""
    os.environ['ION'] = 'fe5nd'
    os.environ.setdefault('HEFF_SOLVER', 'iter')
    import phase17a as A
    d = json.load(open(JS))
    a = A.load('fe5nd', 'besth_sall')
    pk = load('fe5nd', 'besth_sall')
    ex = pk['exact']['proj']
    syl = pk['sylvester']['proj']
    A17 = json.load(open(os.path.join(R17, 'phase17a_tables.json')))['A1']
    out = ['# Phase 18 Step 1: Fe5+ (Fe VI 3d3) tables (scripts/phase18_fe5.py tables)', '']
    out += ['## Second-order variants against exact, Step 8 space (T1, T2 mEh; B, C, beta cm-1)', '',
            '| ion | variant | T1 | T2 | B | C | beta | T1/exact |', '|---|---|---|---|---|---|---|---|']
    row = {m: A.total(a, m) for m in A.METH} | {'S2-full': syl, 'exact': ex}
    for ion, r in (('V2+', A17['vnd:besth_sall']), ('Cr3+', A17['crnd:besth_sall']), ('Mn4+', A17['mnnd:besth_sall']), ('Fe5+', row)):
        for m in ('C2', 'S2-cav', 'S2-diag', 'S2-full', 'exact'):
            x = r[m]
            out.append(f"| {ion} | {m} | {x['T1']:.3f} | {x['T2']:.3f} | {x['B']:.1f} | {x['C']:.1f} | {x['beta']:+.1f} | {x['T1'] / r['exact']['T1']:.3f} |")
    cl = A.class_table(a)
    out += ['', '## Second-order T1 / T2 by Q class, Fe5+ Step 8 (mEh)', '', '| Q class | configurations | C2 | S2-cav | S2-diag |', '|---|---|---|---|---|']
    for k, v in cl.items():
        out.append(f"| {k} | {v['n']} | " + ' | '.join(f"{v[m]['T1']:+.3f} / {v[m]['T2']:+.3f}" for m in A.METH) + ' |')
    out.append('| **total (HF + sum)** | | ' + ' | '.join(f"{row[m]['T1']:.3f} / {row[m]['T2']:+.3f}" for m in A.METH) + ' |')
    c = json.load(open(os.path.join(R17, 'c_fe5nd_besth_sall.json')))['variants']
    rule = json.load(open(os.path.join(OUT, 'crule_fe5nd_s8.json')))['sets']['R']
    out += ['', '## Methods against exact, Fe5+ Step 8 (balanced; dT1 %, dT2 mEh, dbeta cm-1, Q1 fraction)', '',
            '| Q1 level | T1 | T2 | dT1 | dT2 | dbeta | fraction |', '|---|---|---|---|---|---|---|']
    methods = [('closure (C2)', row['C2'], 0), ('S2-full', syl, 0), ('S2-diag', row['S2-diag'], 0), ('C-a', c['C-a']['balanced'], c['C-a']['frac']),
               ('C-b', c['C-b']['balanced'], c['C-b']['frac']), ('Q1_rule', rule, rule['frac'])]
    for k, v, fr in methods:
        out.append(f"| {k} | {v['T1']:.3f} | {v['T2']:.3f} | {100 * (v['T1'] / ex['T1'] - 1):+.2f} % | {v['T2'] - ex['T2']:+.3f} | {v['beta'] - ex['beta']:+.1f} | {100 * fr:.2f} % |")
    ch = d['chain']
    out += ['', '## Corrections, Fe5+ projection (from Step 10 besth_sall_s4)', '', '| item | dT1 | dT2 (mEh) | u(T1) | u(T2) |', '|---|---|---|---|---|']
    for nm, d1, d2, u1, u2 in ch['proj']['budget']:
        unit = 'mEh' if nm == 'hole remainder' else '%'
        out.append(f'| {nm} | {d1:+.3f} {unit} | {d2:+.4f} | {u1:.3f} | {u2:.3f} |')
    H = d['hansen']
    u = d['uylings_fit_T1']
    out += ['', '## Final Fe5+', '', '| representation | T1 | T1/Hansen | T1/Uylings | T2 | T2 - Hansen | B | C | beta |', '|---|---|---|---|---|---|---|---|---|']
    for rp in ('proj', 'terms', 'levels'):
        x = ch[rp]
        out.append(f"| {rp} | {x['T1']:.3f} ± {x['u1']:.3f} | {x['T1'] / H['T1']:.3f} | {x['T1'] / u:.3f} | {x['T2']:.3f} ± {x['u2']:.3f} | {x['T2'] - H['T2']:+.3f} | {x['B']:.1f} | {x['C']:.1f} | {x['beta']:+.1f} |")
    out.append(f"| Hansen 1988 (conventional p) | {H['T1']:.3f} ± {H['u1']:.3f} | | | {H['T2']:.3f} ± {H['u2']:.3f} | | | | |")
    out.append(f'| Uylings 2021 (fitted) | {u:.3f} | | | | | | | |')
    rf = d.get('refit')
    if rf:
        m = rf['free']
        out.append(f"| (1b) fit of NIST Fe VI ({m['n']} levels, mean error {m['mean_error']:.1f}) | {m['p']['T1']:.3f} ± {m['err']['T1']:.3f} | | | "
                   f"{m['p']['T2']:.3f} ± {m['err']['T2']:.3f} | | {m['p']['B']:.1f} | {m['p']['C']:.1f} | {m['p']['beta']:+.1f} |")
        out += ['', f"(1b) zeta {m['p']['zeta']:.1f} ± {m['err']['zeta']:.1f} cm-1 (Hansen 594.1). Computed - (1b): "
                + '; '.join(f"{rp} T1 {ch[rp]['T1'] / m['p']['T1']:.3f} x, T2 {ch[rp]['T2'] - m['p']['T2']:+.3f} "
                            f"({(ch[rp]['T2'] - m['p']['T2']) / math.hypot(ch[rp]['u2'], m['err']['T2']):+.1f} sigma comb.)" for rp in ('proj', 'levels')),
                '', '### Fe VI (1b) refit with computed T1, T2 fixed (F test on the constraints, 19 - 8 = 11 dof)', '',
                '| fixed at | representation | mean error | F | p | free T1 / T2 |', '|---|---|---|---|---|---|',
                f"| none (free) | | {m['mean_error']:.2f} | | | {m['p']['T1']:.3f} / {m['p']['T2']:.3f} |"]
        for rp in ('proj', 'terms', 'levels'):
            for k in ('both', 'T1', 'T2'):
                v = rf[f'{rp}:{k}']
                fr = ' / '.join(f"{v[x]:.3f}" if x in v else '(fixed)' for x in ('free_T1', 'free_T2'))
                out.append(f"| {k} | {rp} | {v['mean_error']:.2f} | {v['F']:.2f} | {v['p']:.3f} | {fr} |")
    else:
        out.append('| (1b) fit of NIST Fe VI | pending: no Fe VI levels file | | | | | | | |')
    hs = hansen_series()
    out += ['', '## d3 series (mEh)', '', '| ion | T1 proj | T1 levels | (1b) T1 | Hansen T1 | T1 proj/(1b) | T1 proj/Hansen | T2 proj | T2 terms | T2 levels | (1b) T2 | Hansen T2 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    S1 = (rf or {}).get('series_1b', {})
    fe5 = {rp: dict(T1=ch[rp]['T1'], T2=ch[rp]['T2']) for rp in ('proj', 'terms', 'levels')}

    def val(rp, i, q):
        return fe5[rp][q] if i == 'Fe⁵⁺' else d['series'][rp][i][q]
    for i in ('V²⁺', 'Cr³⁺', 'Mn⁴⁺', 'Fe⁵⁺'):
        h = hs[i]
        m1 = S1.get(i, {})
        b1 = f"{m1['T1']:.3f} ± {m1['u1']:.3f}" if m1 else '–'
        b2 = f"{m1['T2']:.3f} ± {m1['u2']:.3f}" if m1 else '–'
        r1 = f"{val('proj', i, 'T1') / m1['T1']:.3f}" if m1 else '–'
        out.append(f"| {i} | {val('proj', i, 'T1'):.3f} | {val('levels', i, 'T1'):.3f} | {b1} | {h[0] * P_MEH:.3f} | {r1} | {val('proj', i, 'T1') / (h[0] * P_MEH):.3f} | {val('proj', i, 'T2'):.3f} | {val('terms', i, 'T2'):.3f} | {val('levels', i, 'T2'):.3f} | {b2} | {h[2] * P_MEH:.3f} |")
    out.append('')
    for rp in ('proj', 'terms', 'levels'):
        out.append(f"Mn4+ -> Fe5+ drop of computed T2, {rp}: {val(rp, 'Mn⁴⁺', 'T2') - val(rp, 'Fe⁵⁺', 'T2'):.3f} mEh; V2+ -> Fe5+: {val(rp, 'V²⁺', 'T2') - val(rp, 'Fe⁵⁺', 'T2'):.3f}")
        out.append('')
    if S1:
        out.append(f"(1b) T2 Mn4+ -> Fe5+ drop {S1['Mn⁴⁺']['T2'] - S1['Fe⁵⁺']['T2']:.3f}; V2+ -> Fe5+ {S1['V²⁺']['T2'] - S1['Fe⁵⁺']['T2']:.3f} mEh")
        out.append('')
    out.append(f"Hansen T2 Mn4+ -> Fe5+ drop {(hs['Mn⁴⁺'][2] - hs['Fe⁵⁺'][2]) * P_MEH:.3f}; V2+ -> Fe5+ {(hs['V²⁺'][2] - hs['Fe⁵⁺'][2]) * P_MEH:.3f} mEh")
    s = d['second']
    his = lambda t: -t / P_MEH                     # noqa: E731
    U, kk, R = s['uylings'], s['kk'], s['rows']
    out += ['', '## Uylings (2021) Table 14.5, his normalization (cm-1), Fe VI', '', '| class | this work (closure, Step 8) | Uylings | difference |', '|---|---|---|---|']

    def line(nm, ours, hv):
        out.append(f'| {nm} | {ours:+.3f} | {hv:+.3f} | {ours - hv:+.3f} |')
    sd3, sd2 = his(R['s->3d']['T1']), his(R['2s->3d']['T1'])
    out.append(f'| 3s->3d only | {sd3:+.3f} | | |')
    out.append(f'| 2s->3d only | {sd2:+.3f} | | |')
    line('s->3d (3s + 2s)', sd3 + sd2, U['s->3d'])
    line('3d->s', his(R['3d->s']['T1']), U['3d->s'])
    for q in ('22', '24', '44'):
        line(f"3d->d' {q}", his(kk['d']['T1'][q]), U[f'3d->d {q}'])
    for q in ('22', '24', '44'):
        line(f'3d->g {q}', his(kk['g']['T1'][q]), U[f'3d->g {q}'])
    if '3d->i' in R:
        line('3d->i 44', his(R['3d->i']['T1']), U['3d->i 44'])
    line('total second order', his(R['all configurations']['T1']), U['total second order'])
    tot = R['all configurations']['T1']
    out += ['', f"fitted (Uylings) {U['fitted']:+.3f}; closure / fit: {his(tot) / U['fitted']:.3f} (Uylings), {tot / H['T1']:.3f} (Hansen); Uylings' own total / fitted {U['total second order'] / U['fitted']:.3f}"]
    open(os.path.join(OUT, 'phase18_fe5_tables.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


def refit():
    """Fe VI (1b) NIST fit (free), and refits with T1, T2 fixed at the corrected Fe5+ values of each representation
    (phase17q11.refit logic: both, T1 alone, T2 alone; F test on the constraints); free fits of V2+, Cr3+, Mn4+ for the series."""
    import phase15u_bfit as BF
    import phase17q11 as Q11
    import phase18_nist  # noqa: F401
    from scipy.stats import f as fdist
    d0 = json.load(open(JS))
    ch = d0['chain']
    d = BF.D(3)
    lev = BF.nist('Fe_VI', 3)
    f0 = Q11.bfit_free('fe5')
    c0, n = float(np.sum(np.square(f0['resid']))), f0['n']
    nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
    p = Q11.to_bc(f0['p'])
    out = dict(free=dict(n=n, mean_error=f0['rms'], p={k: float(v) for k, v in p.items()}, err={k: float(v) for k, v in f0['err'].items()}))
    for R in ('proj', 'terms', 'levels'):
        r = ch[R]
        for k, fx in (('both', dict(T1=r['T1'], T2=r['T2'])), ('T1', dict(T1=r['T1'])), ('T2', dict(T2=r['T2']))):
            free = [q for q in nm if q not in fx]
            f = BF.fit(d, lev, free, fx, [f0['p'][q] for q in free])
            c = float(np.sum(np.square(f['resid'])))
            q = len(fx)
            F = (c - c0) / q / (c0 / (n - 8))
            out[f'{R}:{k}'] = dict(fixed=fx, mean_error=f['rms'], F=F, p=float(fdist.sf(F, q, n - 8)), order_ok=bool(f.get('order_ok', True)),
                                   **{f'free_{x}': float(f['p'][x]) for x in ('T1', 'T2') if x in f['p']})
    ser = {}
    for ion, lab in (('v', 'V²⁺'), ('cr', 'Cr³⁺'), ('mn', 'Mn⁴⁺')):
        g = Q11.bfit_free(ion)
        ser[lab] = dict(T1=float(g['p']['T1']), u1=float(g['err']['T1']), T2=float(g['p']['T2']), u2=float(g['err']['T2']), mean_error=float(g['rms']))
    ser['Fe⁵⁺'] = dict(T1=float(p['T1']), u1=float(f0['err']['T1']), T2=float(p['T2']), u2=float(f0['err']['T2']), mean_error=float(f0['rms']))
    out['series_1b'] = ser
    d0['refit'] = out
    json.dump(d0, open(JS, 'w'), indent=1, default=float)
    print('free', out['free'])
    for k, v in out.items():
        if ':' in k:
            print(f"{k:12s} mean error {v['mean_error']:.2f} F {v['F']:.2f} p {v['p']:.3g}", {x: round(y, 3) for x, y in v.items() if x.startswith('free_')})
    print(ser)


if __name__ == '__main__':
    {'final': final, 'second': second, 'tables': tables, 'refit': refit}[sys.argv[1]]()
