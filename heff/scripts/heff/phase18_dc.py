#!/usr/bin/env python3
"""Phase 18, 10 Nov A and B: projections (T1..T4 mEh, f1..f12 cm-1 with Tr f^2 = 1 in d4) of exact H_eff dumps.
  ref   B, 5 % check: plain fam_d5f2g2h1 against its sub-spaces X-16s,15p,11h (the famdc_nsph rule) and
        X-16s,15p,11h,14d (without the most compact plain virtual d), Co3+ and Fe2+ (phase18_dcref.sh dumps)
  dc    B: the opened-d space famdc_nsph_dd7 (fedc6, codc6; queue dumps 18_<ion>_<sp>) against both plain references
  qd    A: Co3+ famdq in co.bas (coqd6) sub-spaces, as the Fe2+ test (phase17q16) -> phase18_dc.json
Four-body part = the vector (T3, T4 in cm-1 and f1..f12, normalized): relative difference |a - b| / |b| and cos."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
HD = os.path.join(ROOT, 'results', 'phase18', 'heff_dump')
HD17 = os.path.join(ROOT, 'results', 'phase17', 'heff_dump')
JS = os.path.join(ROOT, 'results', 'phase18', 'phase18_dc.json')
FS = [f'f{i}' for i in range(1, 13)]
FAM = {'co': (os.path.join(HD, 'q18_cond_fam.npz'), 'X-'), 'fe': (os.path.join(HD17, 'q13_fend_fam.npz'), 'X-')}
REF = {'co': os.path.join(HD, 'q18_cond_fam_dcref.npz'), 'fe': os.path.join(HD, 'q18_fend_fam_dcref.npz')}
RSETS = ['X-16s,15p,11h', 'X-16s,15p,11h,14d']


def proj(npz, s):
    import phase17q11 as Q
    import phase17q20 as Q20
    import phase18_co as C
    z = np.load(npz)
    c, r, rank = Q.ds(6).decompose(C.x_of(z, s))
    A = Q20.jlA()
    out = {k: float(c[k]) for k in ('A', 'B', 'C', 'zeta', 'alpha', 'beta', 'T1', 'T2', 'T3', 'T4') if k in c}
    out.update({k: float(c[k] * np.sqrt(A[k])) for k in FS})
    out['raw_extra'] = {k: float(c[k]) for k in ['T3', 'T4'] + FS}     # phase17q12.EXTRA, unscaled (phase18_co.tests)
    out['resid'] = float(r)
    return out


def four(p):
    import phase17q11 as Q
    return np.array([p['T3'] * Q.MEH_CM, p['T4'] * Q.MEH_CM] + [p[k] for k in FS])


def cmp(a, b):
    """a against b: T1 %, dT2, T2 %, four-body relative difference and cos, f relative difference and cos."""
    va, vb = four(a), four(b)
    fa, fb = va[2:], vb[2:]
    return dict(dT1pct=100 * (a['T1'] / b['T1'] - 1), dT2=a['T2'] - b['T2'], dT2pct=100 * (a['T2'] / b['T2'] - 1),
                four_rel=float(np.linalg.norm(va - vb) / np.linalg.norm(vb)),
                four_cos=float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)),
                f_rel=float(np.linalg.norm(fa - fb) / np.linalg.norm(fb)),
                f_cos=float(fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb)),
                dT3=a['T3'] - b['T3'], dT4=a['T4'] - b['T4'])


def load():
    return json.load(open(JS)) if os.path.exists(JS) else {}


def line(nm, p, c=None):
    s = f"{nm:28s} T1 {p['T1']:.3f} T2 {p['T2']:.3f} T3 {p['T3']:+.3f} T4 {p['T4']:+.3f} | " + ' '.join(f"{p[k]:+.0f}" for k in FS)
    if c:
        s += (f" || dT1 {c['dT1pct']:+.2f} % dT2 {c['dT2']:+.3f} ({c['dT2pct']:+.1f} %) 4-body rel {100 * c['four_rel']:.1f} % "
              f"cos {c['four_cos']:.4f}; f rel {100 * c['f_rel']:.1f} % cos {c['f_cos']:.4f}")
    print(s)


def ref():
    js = load()
    for ion in ('co', 'fe'):
        fam = proj(*FAM[ion])
        r = dict(fam=fam)
        line(f'{ion} plain fam', fam)
        for s in RSETS:
            p = proj(REF[ion], s)
            r[s] = dict(p, vs_fam=cmp(p, fam))
            line(f'{ion} {s}', p, r[s]['vs_fam'])
        c = cmp(r[RSETS[1]], r[RSETS[0]])
        r['d14_effect'] = c
        print(f'{ion} effect of the most compact plain d (14d) at fixed rest: dT2 {-c["dT2"]:+.3f}, dT1 {-c["dT1pct"]:+.2f} %')
        js.setdefault('ref', {})[ion] = r
    json.dump(js, open(JS, 'w'), indent=1)


def qd():
    """A: Co3+ famdq (every virtual d incl. the diffuse 11d, f 1, g 2, h 1, the most compact + the most diffuse 13s) in
    co.bas, coqd6 orbitals; sets as phase17q16 (Fe2+): X- (+d +s), X-13s (+d), X-11d,13s (neither), against the plain
    fam - 12f (q18_cond_fam_loo X-12f); NIST (Co IV) refit with each set's T3, T4, f fixed (phase18_co.tests)."""
    import phase18_co as C
    js = load()
    orb = json.load(open(os.path.join(ROOT, 'results', 'phase18', 'phase18_qd_orb.json')))['coqd']['r_virt']
    assert max(orb['d']) == orb['d'][0] and max(orb['s']) == orb['s'][2], orb     # 11d, 13s the most diffuse
    plain = proj(os.path.join(HD, 'q18_cond_fam_loo.npz'), 'X-12f')
    r = {'plain X-12f': plain}
    srcs = {'X-': (C.dump_file('coqd', 'famdq'), 'exact'), 'X-13s': (os.path.join(HD, 'q18_coqd_famdq.npz'), 'X-13s'),
            'X-11d,13s': (os.path.join(HD, 'q18_coqd_famdq.npz'), 'X-11d,13s')}
    line('plain fam - 12f (cond6)', plain)
    for k, (f, s) in srcs.items():
        p = proj(f, s)
        r[k] = dict(p, vs_plain=cmp(p, plain), dC=p['C'] - plain['C'], dCpct=100 * (p['C'] / plain['C'] - 1))
        line(f'coqd {k}', p, r[k]['vs_plain'])
        print(f"   C {p['C']:.1f} ({r[k]['dCpct']:+.2f} %), B {p['B']:.1f}")
    sets = {k: r[k]['raw_extra'] for k in ['plain X-12f'] + list(srcs)}
    r['fit'] = C.tests(sets, ion='co')
    js['qd'] = r
    fe = json.load(open(os.path.join(ROOT, 'results', 'phase17', 'phase17q16.json')))
    js['qd_fe_ref'] = dict(sets={k: {q: v[q] for q in ('T1', 'T2', 'T3', 'T4', 'C', 'B')} for k, v in fe['fe'].items()},
                           plain={q: fe['plain_fam_m12f'][q] for q in ('T1', 'T2', 'T3', 'T4', 'C', 'B')},
                           fit={k: {q: v.get(q) for q in ('mean_error', 'T1', 'T2', 'T2_err', 'lam', 'lam_err')}
                                for k, v in fe['fe_fit'].items()})
    json.dump(js, open(JS, 'w'), indent=1)


def dc():
    """B: famdc_nsph_dd7 in the opened-d basis (fedc6, codc6) against the plain references (same rule, 4 virtual d;
    and without the most compact plain d)."""
    import phase18_co as C
    js = load()
    orb = json.load(open(os.path.join(ROOT, 'results', 'phase18', 'phase18_qd_orb.json')))
    for ion, t in (('fe', 'fedc'), ('co', 'codc')):
        rf = js['ref'][ion]
        p = proj(C.dump_file(t, 'famdc_nsph_dd7'), 'exact')
        rd = sorted(orb[t]['r_virt']['d'])
        r = dict(p, vs_P0=cmp(p, rf[RSETS[0]]), vs_P0m14d=cmp(p, rf[RSETS[1]]), vs_fam=cmp(p, rf['fam']),
                 n_virt=dict(d=7, f=2, g=2, d_r=rd[-7:], d_r_left_out=rd[:-7]))
        js.setdefault('dc', {})[ion] = r
        line(f'{ion} opened d, dd7', p)
        for nm in ('vs_P0', 'vs_P0m14d', 'vs_fam'):
            c = r[nm]
            print(f"   {nm:10s} dT1 {c['dT1pct']:+.2f} % dT2 {c['dT2']:+.3f} ({c['dT2pct']:+.1f} %) four-body rel "
                  f"{100 * c['four_rel']:.1f} % cos {c['four_cos']:.4f}; f cos {c['f_cos']:.4f}")
    json.dump(js, open(JS, 'w'), indent=1)


if __name__ == '__main__':
    {'ref': ref, 'qd': qd, 'dc': dc}[sys.argv[1]]()
