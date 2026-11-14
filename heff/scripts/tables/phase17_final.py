#!/usr/bin/env python3
"""Phase 17 final assembly (no new runs): "Final estimates" (largest computed space + corrections, error budget) and
the method summary table (Q1 levels against exact), from the stored results.
  -> results/phase17/phase17_final.json, results/phase17/phase17_final_tables.md"""
import json
import math
import os

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
R15, R17 = os.path.join(ROOT, 'results', 'phase15'), os.path.join(ROOT, 'results', 'phase17')
IONS = ['V²⁺', 'Cr³⁺', 'Mn⁴⁺', 'Fe²⁺']
D3 = IONS[:3]


def j(*p):
    return json.load(open(os.path.join(*p)))


# ---- largest computed space (exact H_eff) ----
# d3: Step 10 besth_sall_s4 (15u Step 10 table; phase15k_u_<ion>_besth_sall_s4.json); Fe2+: fam_d5f2g2h1 (Q2)
BASE = {}
for ion, tag in zip(D3, ('vnd', 'crnd', 'mnnd')):
    p = j(R15, f'phase15k_u_{tag}_besth_sall_s4.json')['besth_sall_s4']['exact']['proj']
    BASE[ion] = dict(T1=p['T1'], T2=p['T2'], B=p['B'], C=p['C'], beta=p['beta'], space='Step 10 `besth_sall_s4`')
fe = j(R15, 'phase15j_17_fend_fam_d5f2g2h1.json')['fam_d5f2g2h1']['exact']
BASE['Fe²⁺'] = dict(T1=fe['proj_T1'], T2=fe['proj_T2'], space='`fam_d5f2g2h1` (Q2)')

# ---- measured (1b, 15u B): T1, T2 in mEh; F2, F4 in cm-1 -> B = F2/49 - 5 F4/441, C = 35 F4/441 ----
MEAS = {'V²⁺': (8.217, 0.160, 1.802, 0.198, 58250.1, 24.0, 34664.4, 28.2, -102.6, 6.7),
        'Cr³⁺': (11.492, 0.221, 1.341, 0.277, 72041.0, 33.4, 43745.6, 39.2, 34.1, 9.1),
        'Mn⁴⁺': (13.652, 0.318, 1.123, 0.402, 84466.4, 48.3, 51833.9, 56.8, 125.8, 13.0),
        'Fe²⁺': (9.305, 0.204, 4.506, 0.316, None, None, None, None, None, None)}
UYL_CR = 11.58


def bc(F2, s2, F4, s4):
    B, C = F2 / 49 - 5 * F4 / 441, 35 * F4 / 441
    return B, math.hypot(s2 / 49, 5 * s4 / 441), C, 35 * s4 / 441


# ---- corrections ----
# per-ion DK3 (item 1(a) / Q4-R): % of T1, ΔT2, ΔB, ΔC, Δβ; NR vs DK3 in one opened basis, RAS k = 1 + h (Fe2+ rasK1r)
DK = {'V²⁺': (-3.10, -0.0037, -3.6, -14.2, -6.9), 'Cr³⁺': (-2.67, -0.0046, -3.3, -13.6, -6.9),
      'Mn⁴⁺': (-2.46, -0.0068, -3.2, -13.2, -6.8), 'Fe²⁺': (-3.30, -0.0089, None, None, None)}
# Cr3+ RAS k = 1 (+ h) items carried over to the other ions: (name, ΔT1 %, ΔT2 mEh, ΔB, ΔC, Δβ, source)
CARRY = [
    ('three virtual electrons', -0.081, +0.0130, -0.14, -0.44, -0.13, 'B1: crnd3 `k1fh_v3d3` − `k1fh`'),
    ('2s open', -0.240, +0.0005, -1.56, +1.44, -5.49, 'item 2(b): crt23 `k1fh_2s_t` − `k1fh_t` (B3 without tight: −0.24 %)'),
    ('2p open, tight d, f (+ 2nd pair, g)', -1.150, +0.0010, -1.05, +7.13, +4.62,
     'Q4-n2: crt23 `k1fh_2p_h_t` − `k1fh_t` (−1.094 %) + item 2(a) P′: crt33 − crt23 (−0.026), crt2g3 − crt23 (−0.030)'),
    ('i functions', -0.67, -0.212, -0.53, -18.44, +7.15, 'item 1(c): crti53 `k1fhi` − crnd3 `k1fh` (Step 8 + i: −0.62 %, −0.204)'),
    ('l ≥ 7', -0.09, +0.0045, -0.32, -6.60, +2.79, 'item 1(c): crtk3 `k1fhik` − crti13 `k1fhi` (−0.077 %) + geometric tail'),
]
HOLE = ('hole remainder (u3c1 − k1fh_n3)', +0.097, -0.220, -0.4, -19.4, +20.9, '15u Step 10: crnd u3c1 − k1fh_n3 (mEh)')


# ---- per-ion items (paper item 3, 1 Nov): i (two i, 2.2137 + 1.0, as crti53) and the d3 hole remainder, each ion's
# own RAS k = 1 + h pair in place of the carried Cr3+ value (only the ±10 % RAS k = 1 -> large-space transfer remains)
def _own(f, s):
    e = j(R15, f)[s]['exact']
    p = e['proj'] if 'proj' in e else dict(T1=e['proj_T1'], T2=e['proj_T2'])
    return [p.get(k) for k in ('T1', 'T2', 'B', 'C', 'beta')]


def _diff(a, b, pct=True):
    out = [100 * (a[0] / b[0] - 1) if pct else a[0] - b[0], a[1] - b[1]]
    return out + [(x - y) if x is not None else None for x, y in zip(a[2:], b[2:])]


OWN_I = {ion: _diff(_own(f'phase15{k}_17_{t}ti5_k1fhi.json', 'k1fhi'), _own(f'phase15{k}_17_{t}nd_k1fh.json', 'k1fh'))
         for ion, t, k in (('V²⁺', 'v', 'k'), ('Mn⁴⁺', 'mn', 'k'), ('Fe²⁺', 'fe', 'j'))}
OWN_HOLE = {ion: _diff(_own(f'phase15k_17_{t}nd_u3c1.json', 'u3c1'), _own(f'phase15k_17_{t}nd_k1fh_n3.json', 'k1fh_n3'),
                       pct=False) for ion, t in (('V²⁺', 'v'), ('Mn⁴⁺', 'mn'))}
OWN_HOLE['Cr³⁺'] = list(HOLE[1:6])


# paper item 3 (2 Nov): Fe2+ own 2p, P'' = P' without the virtual g (fet26 k1f_tnfg0 / k1f_2p_h_tnfg0), scaled to
# RAS k = 1 + h by the Cr3+ crt23 calibration ratio % k1fh_t / % P'', plus the Cr3+ second-pair and g increments
# (item 2(a), -0.056 points); ΔT2 raw (P'')
def _p2(f, s):
    return _diff(_own(f'{f}_k1f_2p_h_{s}.json', f'k1f_2p_h_{s}'), _own(f'{f}_k1f_{s}.json', f'k1f_{s}'))


CAL_2P = dict(cr_k1fh=_diff(_own('phase15k_17_crt2_k1fh_2p_h_t.json', 'k1fh_2p_h_t'), _own('phase15k_17_crt2_k1fh_t.json', 'k1fh_t')),
              cr_p2=_p2('phase15k_17_crt2', 'tnfg0'), cr_p1=_p2('phase15k_17_crt2', 'tnf'), fe_p2=_p2('phase15j_17_fet2', 'tnfg0'))
CAL_2P['ratio'] = CAL_2P['cr_k1fh'][0] / CAL_2P['cr_p2'][0]
CAL_2P['ratio_mEh'] = (CAL_2P['cr_k1fh'][0] / 100 * _own('phase15k_17_crt2_k1fh_t.json', 'k1fh_t')[0]) / \
    (CAL_2P['cr_p2'][0] / 100 * _own('phase15k_17_crt2_k1f_tnfg0.json', 'k1f_tnfg0')[0])
OWN_2P = {'Fe²⁺': [CAL_2P['fe_p2'][0] * CAL_2P['ratio'] - 0.026 - 0.030, CAL_2P['fe_p2'][1], None, None, None]}


# paper item 11 (3-4 Nov): Fe2+ n = 3 and i measured directly in the T2-generating space fam - 12f + 11i + n3 (`famiq`,
# feti16 orbitals = fend6 for s-h; exact sub-spaces, phase17q18.json), replacing the RAS k = 1 + h values:
#   n = 3: X-11i - X-11i,h2 (no i present); i: X-h2 - X-11i,h2 (no n = 3 present); the two are additive (cross term
#   <= 1.4 % of either item). famiq has ONE i (2.2137); the Final i item is two i (2.2137 + 1.0, `feti5`), so the
#   second i is added from RAS k = 1 + h: feti5 `k1fhi` - feti1 `k1fhi`.
def _large_fe():
    q = j(R17, 'phase17q18.json')['fe']
    d = lambda a, b: (100 * (q[a]['T1'] / q[b]['T1'] - 1), q[a]['T2'] - q[b]['T2'])     # noqa: E731
    i1_ras = _diff(_own('phase15j_17_feti1_k1fhi.json', 'k1fhi'), _own('phase15j_17_fend_k1fh.json', 'k1fh'))
    i2_ras = _diff(_own('phase15j_17_feti5_k1fhi.json', 'k1fhi'), _own('phase15j_17_feti1_k1fhi.json', 'k1fhi'))
    n3, i1 = d('X-11i', 'X-11i,h2'), d('X-h2', 'X-11i,h2')
    return dict(n3=n3, i1=i1, i1_ras=i1_ras[:2], i2_ras=i2_ras[:2], n3_with_i=d('X-', 'X-h2'), i1_with_n3=d('X-', 'X-11i'),
                i=(i1[0] + i2_ras[0], i1[1] + i2_ras[1]))


LARGE_FE = _large_fe()
U_LARGE = 0.05      # item measured in fam - 12f: cross terms with the other measured item <= 1.4 %; 12f itself not in the
#                     measuring space (its cross term with n = 3 / i not tested)
OWN_I['Fe²⁺'] = [LARGE_FE['i'][0], LARGE_FE['i'][1], None, None, None]


def carry(ion):
    """CARRY with the ion's own i and 2p items where they were run (flag: own)."""
    own = {'i functions': OWN_I, '2p open, tight d, f (+ 2nd pair, g)': OWN_2P}
    return [(c[0], *own[c[0]][ion], c[6], True) if c[0] in own and ion in own[c[0]] else (*c, False) for c in CARRY]


# ---- Fe2+ completion (paper item 2, 31 Oct): the increments the d3 Step 10 space has and fam does not, measured in
# Fe2+ RAS k = 1 + h (fend6); n = 3 and all-s in % with the Cr3+ k1fh -> large-space transfer, hole in mEh (as HOLE)
def _ex(p, s, f=None):
    e = j(R15, f'{p}_{f or s}.json')[s]['exact']
    return (e['proj']['T1'], e['proj']['T2']) if 'proj' in e else (e['proj_T1'], e['proj_T2'])


def fe_items():
    fe = {s: _ex('phase15j_17_fend', s) for s in ('k1fh', 'k1fh_n3', 'u3c1', 'k1fh_sall')}
    cr = dict(k1fh=_ex('phase15k_t_crnd', 'k1fh'), k1fh_n3=_ex('phase15k_u_crnd', 'k1fh_n3'),
              k1fh_sall=_ex('phase15k', 'k1fh_sall', '17_crnd_k1fh_sall'), besth=_ex('phase15k_t_crnd', 'besth'),
              s8=_ex('phase15k_u_crnd', 'besth_sall'), s10=_ex('phase15k_u_crnd', 'besth_sall_s4'))
    pc = lambda a, b: 100 * (a[0] / b[0] - 1)          # noqa: E731
    out = {}
    for name, (fa, fb), (ca, cb), (la, lb) in (
            ('n = 3 configurations', ('k1fh_n3', 'k1fh'), ('k1fh_n3', 'k1fh'), ('s10', 's8')),
            ('all virtual s', ('k1fh_sall', 'k1fh'), ('k1fh_sall', 'k1fh'), ('s8', 'besth'))):
        raw1, raw2 = pc(fe[fa], fe[fb]), fe[fa][1] - fe[fb][1]
        tf1 = pc(cr[la], cr[lb]) / pc(cr[ca], cr[cb])
        tf2 = (cr[la][1] - cr[lb][1]) / (cr[ca][1] - cr[cb][1])
        a1, a2 = raw1 * tf1, raw2 * tf2
        out[name] = dict(raw1=raw1, raw2=raw2, tf1=tf1, tf2=tf2, d1=a1, d2=a2,
                         # transfer ±10 %; a Cr3+ transfer far from 1 (all s: 1.7) adds half the difference raw / applied
                         u1=math.hypot(U_TRANSFER * a1, 0.5 * (a1 - raw1) if abs(tf1 - 1) > 0.2 else 0.0),
                         u2=math.hypot(U_TRANSFER * a2, 0.5 * (a2 - raw2) if abs(tf2 - 1) > 0.2 else 0.0))
    # item 11: n = 3 measured in fam - 12f (no transfer factor); the RAS-transferred value is kept as 'ras'
    a1, a2 = LARGE_FE['n3']
    out['n = 3 configurations'] = dict(raw1=a1, raw2=a2, tf1=1.0, tf2=1.0, d1=a1, d2=a2, u1=U_LARGE * abs(a1),
                                       u2=U_LARGE * abs(a2), ras=out['n = 3 configurations'], large=True)
    # fam -> besth (the remaining virtual f, h): not run for Fe2+, the Cr3+ value carried (±10 % ⊕ ion ±30 %)
    cf = _ex('phase15k', 'fam_d5f2g2h1', '17_crnd_fam_d5f2g2h1')
    a1, a2 = pc(cr['besth'], cf), cr['besth'][1] - cf[1]
    f = math.hypot(U_TRANSFER, U_ION['Fe²⁺'])
    out['fam → besth (f, h)'] = dict(raw1=a1, raw2=a2, tf1=1.0, tf2=1.0, d1=a1, d2=a2, u1=f * abs(a1), u2=f * abs(a2))
    out['hole'] = dict(d1=fe['u3c1'][0] - fe['k1fh_n3'][0], d2=fe['u3c1'][1] - fe['k1fh_n3'][1],
                       pct=pc(fe['u3c1'], fe['k1fh_n3']))
    out['raw'] = fe
    return out


# ---- uncertainty model (stated in the report) ----
U_TRANSFER = 0.10           # RAS k = 1 -> largest space, fraction of the item (i: 0.94, pure n = 3: 0.92 measured)
U_ION = {'V²⁺': 0.15, 'Cr³⁺': 0.0, 'Mn⁴⁺': 0.15, 'Fe²⁺': 0.30}   # Cr3+ -> other ion (d3 DK3 spread ±12 %)
U_FIX = {'three virtual electrons': (0.08, 0.013), 'l ≥ 7': (0.05, 0.005)}   # % of T1, mEh: not run in the full form
U_HOLE = (0.03, 0.05)       # mEh (15u Step 10: scaling ±0.02 / ±0.05, missing 4-hole 2-virtual −0.010, transfer < 0.02)


FE = fe_items()


def final():
    out = {}
    for ion in IONS:
        b = BASE[ion]
        d3 = ion in D3
        bud = []
        C = carry(ion)
        if d3:
            T1h, T2 = b['T1'] + OWN_HOLE[ion][0], b['T2'] + OWN_HOLE[ion][1]
        else:      # Fe2+: fam + n = 3 + all s (the d3 Step 10 content), + its own hole remainder (item 2)
            n3, sa, fb, ho = FE['n = 3 configurations'], FE['all virtual s'], FE['fam → besth (f, h)'], FE['hole']
            T1s = b['T1'] * (1 + (n3['d1'] + sa['d1'] + fb['d1']) / 100)
            T1h, T2 = T1s + ho['d1'], b['T2'] + n3['d2'] + sa['d2'] + fb['d2'] + ho['d2']
            for nm_, x in (('n = 3 configurations', n3), ('all virtual s', sa), ('fam → besth (f, h)', fb)):
                bud.append((nm_, x['d1'], x['d2'], x['u1'] / 100 * b['T1'], x['u2']))
            bud.append(('hole remainder', ho['d1'], ho['d2'], U_HOLE[0], U_HOLE[1]))
        pct = sum(c[1] for c in C) + DK[ion][0]
        T1 = T1h * (1 + pct / 100)
        T2 += sum(c[2] for c in C) + DK[ion][1]
        for name, p, t2, *_, own in C:
            f = U_TRANSFER if own else math.hypot(U_TRANSFER, U_ION[ion])
            u1 = math.hypot(f * abs(p), U_FIX.get(name, (0, 0))[0]) / 100 * T1h
            u2 = math.hypot(f * abs(t2), U_FIX.get(name, (0, 0))[1])
            if ion == 'Fe²⁺' and name == 'i functions':      # item 11: one i measured in fam - 12f, second i from RAS
                L1, R2 = LARGE_FE['i1'], LARGE_FE['i2_ras']
                u1 = math.hypot(U_LARGE * L1[0], U_TRANSFER * R2[0]) / 100 * T1h
                u2 = math.hypot(U_LARGE * L1[1], U_TRANSFER * R2[1])
            bud.append((name, p, t2, u1, u2))
        bud.append(('DK3 (per ion)', DK[ion][0], DK[ion][1], U_TRANSFER * abs(DK[ion][0]) / 100 * T1h,
                    U_TRANSFER * abs(DK[ion][1])))
        if d3:      # own hole remainder for each d3 ion (item 3): U_HOLE only
            bud.append(('hole remainder', OWN_HOLE[ion][0], OWN_HOLE[ion][1], U_HOLE[0], U_HOLE[1]))
        u1 = math.sqrt(sum(x[3] ** 2 for x in bud))
        u2 = math.sqrt(sum(x[4] ** 2 for x in bud))
        m = MEAS[ion]
        r = dict(base=b, T1=T1, T2=T2, T1h=T1h, pct=pct, u1=u1, u2=u2, budget=bud, meas=m[:4],
                 ratio=T1 / m[0], dev1=(T1 - m[0]) / m[1], dev1c=(T1 - m[0]) / math.hypot(m[1], u1),
                 dev2=(T2 - m[2]) / m[3], dev2c=(T2 - m[2]) / math.hypot(m[3], u2))
        if d3:
            r['B'] = b['B'] + sum(c[3] for c in C) + DK[ion][2] + OWN_HOLE[ion][2]
            r['C'] = b['C'] + sum(c[4] for c in C) + DK[ion][3] + OWN_HOLE[ion][3]
            r['beta'] = b['beta'] + sum(c[5] for c in C) + DK[ion][4] + OWN_HOLE[ion][4]
            r['measBC'] = bc(*m[4:8]) + (m[8], m[9])
        out[ion] = r
    return out


def methods():
    """T1/ex − 1, T2 − ex, β − ex and fraction per Q1 level, five runs."""
    runs = {'V²⁺ S8': ('vnd', 's8'), 'Cr³⁺ S8': ('crnd', 's8'), 'Mn⁴⁺ S8': ('mnnd', 's8'),
            'Cr³⁺ fam': ('crnd', 'fam'), 'Fe²⁺ fam': ('fend', 'fam')}
    A = j(R17, 'phase17a_tables.json')['A1']
    res = {}
    for run, (tag, sp) in runs.items():
        row = {}
        if sp == 's8':
            a = A[f'{tag}:besth_sall']
            ex = a['exact']
            ntot = j(R17, f'c_{tag}_besth_sall.json')['ncsf_total']
            for k, lab in (('C2', 'closure (C2)'), ('S2-full', 'S2-full ("Sylvester")'), ('S2-diag', 'S2-diag')):
                row[lab] = dict(a[k], frac=0.0)
            c1 = j(R17, f'c_{tag}_besth_sall.json')['variants']
            c2 = j(R17, f'c_{tag}_besth_sall_r2.json')['variants']
            for k in ('C-a', 'C-b'):
                row[k] = dict(c1[k]['balanced'], frac=c1[k]['frac'])
            row['C-e'] = dict(c2['C-e']['balanced'], frac=c2['C-e']['frac'])
            rule = j(R17, f'crule_{tag}_s8.json')['sets']['R']
        else:
            if tag == 'crnd':
                p = j(R15, 'phase15k_17_crnd_fam_d5f2g2h1.json')['fam_d5f2g2h1']
                ex = p['exact']['proj']
                row['closure (C2)'] = dict(p['closure']['proj'], frac=0.0)
                row['S2-full ("Sylvester")'] = dict(p['sylvester']['proj'], frac=0.0)
            else:
                p = j(R15, 'phase15j_17_fend_fam_d5f2g2h1.json')['fam_d5f2g2h1']
                ex = dict(T1=p['exact']['proj_T1'], T2=p['exact']['proj_T2'])
                for k, lab in (('closure', 'closure (C2)'), ('sylvester', 'S2-full ("Sylvester")')):
                    row[lab] = dict(T1=p[k]['proj_T1'], T2=p[k]['proj_T2'], frac=0.0)
            c = j(R17, f'c_{tag}_fam_d5f2g2h1_q2.json')['variants']
            for k in ('C-b', 'C-e'):
                row[k] = dict(c[k]['balanced'], frac=c[k]['frac'])
            # 30 Oct (paper item 1a): S2-diag (phase17a / phase17a_fe on the fam H) and C-a (phase17c, C_TAG=ca)
            s2 = j(R17, 'a_crnd_fam_d5f2g2h1_tot.json')['S2-diag'] if tag == 'crnd' else \
                j(R17, 'a_fend_fam_d5f2g2h1.json')['S2-diag']
            row['S2-diag'] = dict(T1=s2['T1'], T2=s2['T2'], beta=s2.get('beta'), frac=0.0)
            ca = j(R17, f'c_{tag}_fam_d5f2g2h1_ca.json')['variants']['C-a']
            row['C-a'] = dict(ca['balanced'], frac=ca['frac'])
            rule = j(R17, f'crule_{tag}_fam.json')['sets']['R']
        if True:     # 30 Oct: V2+ and Cr3+ Step 8 from cscan_vnd_g0 / cscan_crnds8_g0 (paper item 1a)
            g = j(R17, {('vnd', 's8'): 'cscan_vnd_g0.json', ('crnd', 's8'): 'cscan_crnds8_g0.json'}.get(
                (tag, sp), f'cscan_{tag}_g0.json'))['sets']['g']
            row['C-e + g'] = dict(g, frac=g['frac'])
        row['Q1_rule'] = dict(rule, frac=rule['frac'])
        res[run] = dict(exact=ex, rows={k: dict(dT1=100 * (v['T1'] / ex['T1'] - 1), dT2=v['T2'] - ex['T2'],
                                                dbeta=(v['beta'] - ex['beta']) if 'beta' in ex and v.get('beta') == v.get('beta')
                                                and v.get('beta') is not None else None,
                                                frac=v['frac']) for k, v in row.items()})
    return res


def fmt(x, f):
    return '–' if x is None else format(x, f)


def hole(ion):
    return tuple(OWN_HOLE[ion][:2]) if ion in D3 else (FE['hole']['d1'], FE['hole']['d2'])


def ns1(ion):
    return 'in space' if ion in D3 else '{:+.2f} / {:+.2f} / {:+.2f}'.format(
        *(FE[k]['d1'] for k in ('n = 3 configurations', 'all virtual s', 'fam → besth (f, h)')))


def ns2(ion):
    return 'in space' if ion in D3 else f"{sum(FE[k]['d2'] for k in ('n = 3 configurations', 'all virtual s', 'fam → besth (f, h)')):+.3f}"


def md(F, M):
    L = ['### Final T1, T2 per ion', '',
         '| ion | space (exact H_eff) | T1 computed | n = 3 / all s / fam → besth (%) | + hole rem. (mEh) | CI + basis (Cr³⁺ RAS k = 1, %) | '
         'DK3 (per ion, %) | **T1 corrected** | (1b) | T1 / (1b) | dev. (σ meas.) | dev. (σ comb.) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for ion, r in F.items():
        m = r['meas']
        L.append(f"| {ion} | {r['base']['space']} | {r['base']['T1']:.3f} | {ns1(ion)} | {hole(ion)[0]:+.3f} | "
                 f"{sum(c[1] for c in carry(ion)):+.2f} | {DK[ion][0]:+.2f} | **{r['T1']:.3f} ± {r['u1']:.3f}** | "
                 f"{m[0]:.3f} ± {m[1]:.3f} | **{r['ratio']:.3f}** | {r['dev1']:+.1f} | {r['dev1c']:+.1f} |")
    L += ['', '| ion | T2 computed | n = 3 + all s + fam → besth | + hole rem. | i | other CI + basis | DK3 | **T2 corrected** | (1b) | '
          'T2 − (1b) | dev. (σ meas.) | dev. (σ comb.) |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for ion, r in F.items():
        m = r['meas']
        oth = sum(c[2] for c in carry(ion) if c[0] != 'i functions')
        ti = next(c[2] for c in carry(ion) if c[0] == 'i functions')
        L.append(f"| {ion} | {r['base']['T2']:.3f} | {ns2(ion)} | {hole(ion)[1]:+.3f} | {ti:+.3f} | {oth:+.3f} | "
                 f"{DK[ion][1]:+.4f} | **{r['T2']:.3f} ± {r['u2']:.3f}** | {m[2]:.3f} ± {m[3]:.3f} | {r['T2'] - m[2]:+.3f} | "
                 f"{r['dev2']:+.1f} | {r['dev2c']:+.1f} |")
    L += ['', '| ion | B computed | B corrected | B (1b) | C computed | C corrected | C (1b) | β computed | β corrected | β (1b) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for ion in D3:
        r, b = F[ion], F[ion]['base']
        B, sB, C, sC, be, sbe = r['measBC']
        L.append(f"| {ion} | {b['B']:.1f} | {r['B']:.1f} | {B:.1f} ± {sB:.1f} | {b['C']:.1f} | {r['C']:.1f} | {C:.1f} ± {sC:.1f} | "
                 f"{b['beta']:+.1f} | {r['beta']:+.1f} | {be:+.1f} ± {sbe:.1f} |")
    L += ['', '### Corrections and error budget (T1 uncertainty in mEh, T2 in mEh)', '',
          '| item | ΔT1 (%) | ΔT2 (mEh) | source | carried to | ' + ' | '.join(f'u(T1) {i}' for i in IONS) + ' | '
          + ' | '.join(f'u(T2) {i}' for i in IONS) + ' |', '|' + '---|' * (5 + 2 * len(IONS))]
    names = ['n = 3 configurations', 'all virtual s', 'fam → besth (f, h)'] + [c[0] for c in CARRY] + ['DK3 (per ion)', 'hole remainder']
    src = {c[0]: c[6] for c in CARRY}
    src['DK3 (per ion)'] = 'item 1(a) / Q4-R: `<el>dk` − `<el>op` (k1fhr; Fe²⁺ rasK1r), per ion'
    src['hole remainder'] = '<el>nd `u3c1` − `k1fh_n3` per ion (Cr³⁺ 15u Step 10, V²⁺ / Mn⁴⁺ item 3, Fe²⁺ item 2)'
    for k, a, b in (('all virtual s', '`k1fh_sall` − `k1fh`', 'Step 8 − besth'),):
        x = FE[k]
        src[k] = (f"item 2: fend {a} ({x['raw1']:+.3f} %, {x['raw2']:+.4f}) × Cr³⁺ transfer {x['tf1']:.3f} / "
                  f"{x['tf2']:.3f} ({b} over crnd {a})")
    src['n = 3 configurations'] = ('item 11: feti1 `famiq` sub-spaces `X-11i` − `X-11i,h2` (fam − 12f + n3 vs fam − 12f), '
                                   'measured directly (±5 %; was fend RAS k = 1 + h × Cr³⁺ transfer, '
                                   f"{FE['n = 3 configurations']['ras']['d1']:+.3f} %)")
    src['fam → besth (f, h)'] = 'not run for Fe²⁺: Cr³⁺ `besth` − `fam_d5f2g2h1` carried (±10 % ⊕ ±30 %)'
    for n in names:
        rows = {ion: next((x for x in F[ion]['budget'] if x[0] == n), None) for ion in IONS}
        r0 = rows['Cr³⁺']
        if n == 'DK3 (per ion)':
            d1, d2 = '/'.join(f'{DK[i][0]:+.2f}' for i in IONS), '/'.join(f'{DK[i][1]:+.4f}' for i in IONS)
            to = 'per ion'
        elif n == 'hole remainder':
            d1 = '/'.join(f'{hole(i)[0]:+.3f}' for i in IONS) + ' mEh'
            d2, to = '/'.join(f'{hole(i)[1]:+.3f}' for i in IONS), 'per ion (item 3; Fe²⁺ item 2)'
        elif n == 'i functions':
            ci = {i: next(c for c in carry(i) if c[0] == n) for i in IONS}
            d1, d2 = '/'.join(f'{ci[i][1]:+.3f}' for i in IONS), '/'.join(f'{ci[i][2]:+.4f}' for i in IONS)
            to = ('per ion (item 3: `<el>ti5` `k1fhi` − `<el>nd` `k1fh`); Fe²⁺ item 11: one i from `famiq` `X-h2` − '
                  '`X-11i,h2` (±5 %) + second i from RAS feti5 − feti1 `k1fhi` (±10 %)')
        elif n in ('2p open, tight d, f (+ 2nd pair, g)',):
            ci = {i: next(c for c in carry(i) if c[0] == n) for i in IONS}
            d1, d2 = '/'.join(f'{ci[i][1]:+.3f}' for i in IONS), '/'.join(f'{ci[i][2]:+.4f}' for i in IONS)
            to = 'V²⁺, Mn⁴⁺ (Cr³⁺ value); Fe²⁺ own (item 3: `fet26` P″ × Cr³⁺ ratio k1fh_t / P″ + 2nd pair, g)'
        elif r0 is None:
            d1, d2, to = f"{FE[n]['d1']:+.3f}", f"{FE[n]['d2']:+.4f}", 'Fe²⁺ only (in the d³ Step 10 space)'
        else:
            d1, d2, to = f'{r0[1]:+.3f}', f'{r0[2]:+.4f}', 'V²⁺, Mn⁴⁺, Fe²⁺'
        L.append(f'| {n} | {d1} | {d2} | {src[n]} | {to} | ' + ' | '.join(fmt(rows[i][3] if rows[i] else None, '.3f') for i in IONS)
                 + ' | ' + ' | '.join(fmt(rows[i][4] if rows[i] else None, '.3f') for i in IONS) + ' |')
    L.append('| **total (quadrature)** | | | | | ' + ' | '.join(f"**{F[i]['u1']:.3f}**" for i in IONS) + ' | '
             + ' | '.join(f"**{F[i]['u2']:.3f}**" for i in IONS) + ' |')
    L += ['', '### Method summary: Q1 levels against exact (balanced form; T1 in %, T2 in mEh, β in cm⁻¹)', '',
          '| Q1 level | ' + ' | '.join(M) + ' |', '|---|' + '---|' * len(M)]
    order = ['closure (C2)', 'S2-full ("Sylvester")', 'S2-diag', 'C-a', 'C-b', 'C-e', 'C-e + g', 'Q1_rule']
    for k in order:
        cells = []
        for run, v in M.items():
            x = v['rows'].get(k)
            if x is None:
                cells.append('not computed')
                continue
            s = f"{x['dT1']:+.2f} % / {x['dT2']:+.3f}"
            s += f" / {x['dbeta']:+.1f}" if x['dbeta'] is not None else ''
            s += f" ({100 * x['frac']:.2f} %)" if x['frac'] else ''
            cells.append(s)
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    F, M = final(), methods()
    json.dump(dict(final=F, methods=M, fe_items=FE, own_i=OWN_I, own_hole=OWN_HOLE, cal_2p=CAL_2P, own_2p=OWN_2P), open(os.path.join(R17, 'phase17_final.json'), 'w'), indent=1, default=str)
    t = md(F, M)
    open(os.path.join(R17, 'phase17_final_tables.md'), 'w').write(t)
    print(t)
    print('Cr3+ vs Uylings', F['Cr³⁺']['T1'] / UYL_CR)
