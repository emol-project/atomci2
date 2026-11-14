#!/usr/bin/env python3
"""Phase 15m: is the 3p2 -> 3d2 reduction real and uniform, and does class-by-class addition fail?

Ion via env ION = cr (default) | v | mn (d3, 15e orbitals) | fe (d6, 15h orbitals; c2s and c2s_a only).
  dims [spaces]          CSF counts per (L,S) block, time estimates          -> phase15m_dims.json
  prep <spaces>          decks;  run_names <names>   ATOMCI (one per call)
  analyze <tag> <sp..>   d3: 15k analyze3 (one block at a time) -> phase15k_<tag>.json; fe: 15j analyze2
  eav                    configuration-average energies (frozen orbitals, Slater averages) -> phase15m_eav.json
  spectral               step 3a: Q eigenstates carrying the P-Q coupling, c2s and c2s_a (Cr3+)
  step4                  facts for the basis consultation                    -> phase15m_step4.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15l as L15  # noqa: E402  (patches phase15k: spaces3, rname_k)
import phase15k as K  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402

OUT = K.OUT
IONS = {'v': (3, 23, 'v_qzp.bas', 'o14n3'), 'cr': (3, 24, 'cr.bas', 'cr15n3'), 'mn': (3, 25, 'mn.bas', 'mn15b3'),
        'fe': (6, 26, 'fe.bas', 'fe15h6'),
        'crnd': (3, 24, 'cr.without-diffuse.bas', 'crnd3'),    # 15s addendum: Cr3+ in plain Sapporo-QZP-2012
        # 15t: plain Sapporo-QZP-2012 + one or two uncontracted primitives (f, g, h unoccupied)
        'crtfm': (3, 24, 'cr_qzp2012_add_f2.75.bas', 'crtfm3'), 'crtft': (3, 24, 'cr_qzp2012_add_f14.0.bas', 'crtft3'),
        'crtfb': (3, 24, 'cr_qzp2012_add_f2.75_f14.0.bas', 'crtfb3'),
        'crtg': (3, 24, 'cr_qzp2012_add_g4.7.bas', 'crtg3'), 'crth': (3, 24, 'cr_qzp2012_add_h4.4.bas', 'crth3'),
        # 15u series: V2+ and Mn4+ in plain Sapporo-QZP-2012
        'vnd': (3, 23, 'v.without-diffuse.bas', 'vnd3'), 'mnnd': (3, 25, 'mn.without-diffuse.bas', 'mnnd3'),
        'fend': (6, 26, 'fe.without-diffuse.bas', 'fend6'),
        # 17 Q3: plain Sapporo-QZP-2012 + one i (2.2136970) or two i (4.4, 2.2136970)
        'crti1': (3, 24, 'cr_qzp2012_add_i2.2137.bas', 'crti13'), 'crti2': (3, 24, 'cr_qzp2012_add_i4.4_i2.2137.bas', 'crti23'),
        'crti3': (3, 24, 'cr_qzp2012_add_i1.bas', 'crti33'), 'crti4': (3, 24, 'cr_qzp2012_add_i4.4.bas', 'crti43'),
        'crti5': (3, 24, 'cr_qzp2012_add_i2.2137_i1.bas', 'crti53'),
        'crtk1': (3, 24, 'cr_qzp2012_add_i2.2137_k2.2137.bas', 'crtk3'),     # 17 item 1c: + one k (l = 7)
        # 17 paper item 3: per-ion i (2.2136970; + 1.0) for V2+, Mn4+, Fe2+ (phase17q9i)
        'vti1': (3, 23, 'v_qzp2012_add_i2.2137.bas', 'vti13'), 'vti5': (3, 23, 'v_qzp2012_add_i2.2137_i1.bas', 'vti53'),
        'mnti1': (3, 25, 'mn_qzp2012_add_i2.2137.bas', 'mnti13'), 'mnti5': (3, 25, 'mn_qzp2012_add_i2.2137_i1.bas', 'mnti53'),
        'feti1': (6, 26, 'fe_qzp2012_add_i2.2137.bas', 'feti16'), 'feti5': (6, 26, 'fe_qzp2012_add_i2.2137_i1.bas', 'feti56'),
        # 17 item 1a: DK3 test for V2+, Mn4+, Fe2+ (phase17q4r recipe; NR op / DK dk in the opened basis)
        # 17 item 2a: a second tight d, f pair (crt3) or a tight g (crt2g) on top of crt2
        'crt3': (3, 24, 'cr_qzp2012_add_d45_f58_d112.5_f145.bas', 'crt33'),
        'crt2g': (3, 24, 'cr_qzp2012_add_d45_f58_g78.bas', 'crt2g3'),
        'vop': (3, 23, 'v_qzp2012_open_spd.bas', 'vop3'), 'vdk': (3, 23, 'v_qzp2012_open_spd.bas', 'vdk3'),
        'mnop': (3, 25, 'mn_qzp2012_open_spd.bas', 'mnop3'), 'mndk': (3, 25, 'mn_qzp2012_open_spd.bas', 'mndk3'),
        'feop': (6, 26, 'fe_qzp2012_open_spd.bas', 'feop6'), 'fedk': (6, 26, 'fe_qzp2012_open_spd.bas', 'fedk6'),     # 17 item 1c (i exponents)
        # 17 Q4-R: s, p, d contractions opened; NR (crop) and DK3 (crdk, ntype=4 decks) orbitals
        'crop': (3, 24, 'cr_qzp2012_open_spd.bas', 'crop3'), 'crdk': (3, 24, 'cr_qzp2012_open_spd.bas', 'crdk3'),
        # 17 Q4-n2: + tight (2p-matched) D 45, F 58
        'crt2': (3, 24, 'cr_qzp2012_add_d45_f58.bas', 'crt23'),
        # 17 paper items 3-5 (phase17q10): Fe2+ tight d, f; crt2 + one i; V2+, Mn4+ in Sapporo-QZP-2012-diffuse with
        # own-recipe orbitals (vqd3 new: o14n3 is not the recipe; mn15b3 is) under new ion names (the 15n-15u 'v' / 'mn' setup unchanged)
        'fet2': (6, 26, 'fe_qzp2012_add_d54.6_f70.3.bas', 'fet26'),
        'crt2i': (3, 24, 'cr_qzp2012_add_d45_f58_i2.2137.bas', 'crt2i3'),
        'vqd': (3, 23, 'v_qzp.bas', 'vqd3'), 'mnqd': (3, 25, 'mn.bas', 'mn15b3'),
        # 17 item 10: Fe2+ in Sapporo-QZP-2012-diffuse, own-recipe orbitals feqd6 (phase17q10 feqd)
        'feqd': (6, 26, 'fe.bas', 'feqd6'),
        # 17 test 1 (3 Nov): Cr3+ k1fh orbital dependence, 4F-HF occupied (cr4f3) and k1fh natural orbitals (crno3)
        'cr4f': (3, 24, 'cr.without-diffuse.bas', 'cr4f3'), 'crno': (3, 24, 'cr.without-diffuse.bas', 'crno3'),
        # Phase 18: Fe5+ (d3), same chain as V2+, Cr3+, Mn4+; the bases are the Fe2+ ones (same element)
        'fe5nd': (3, 26, 'fe.without-diffuse.bas', 'fe5nd3'),
        'fe5ti1': (3, 26, 'fe_qzp2012_add_i2.2137.bas', 'fe5ti13'), 'fe5ti5': (3, 26, 'fe_qzp2012_add_i2.2137_i1.bas', 'fe5ti53'),
        'fe5op': (3, 26, 'fe_qzp2012_open_spd.bas', 'fe5op3'), 'fe5dk': (3, 26, 'fe_qzp2012_open_spd.bas', 'fe5dk3'),
        'fe5t2': (3, 26, 'fe_qzp2012_add_d54.7_f70.5.bas', 'fe5t23'),     # crt2 pair at the Fe5+ 2p radius (phase18 fe5t2)
        # Phase 18 Step 2: Co3+ 3d6, as fend (co.without-diffuse.bas = BSE v0.8.13 copy, see basis/README.md)
        'cond': (6, 27, 'co.without-diffuse.bas', 'cond6'),
        'coti1': (6, 27, 'co_qzp2012_add_i2.2137.bas', 'coti16'), 'coti5': (6, 27, 'co_qzp2012_add_i2.2137_i1.bas', 'coti56'),
        'coop': (6, 27, 'co_qzp2012_open_spd.bas', 'coop6'), 'codk': (6, 27, 'co_qzp2012_open_spd.bas', 'codk6'),
        'cot2': (6, 27, 'co_qzp2012_add_d59.7_f77.bas', 'cot26'),
        # Phase 18, 10 Nov: A Co3+ in Sapporo-QZP-2012-diffuse (as feqd); B Fe2+, Co3+ with the d contractions opened
        'coqd': (6, 27, 'co.bas', 'coqd6'),
        'fedc': (6, 26, 'fe_qzp2012_open_d.bas', 'fedc6'), 'codc': (6, 27, 'co_qzp2012_open_d.bas', 'codc6')}
ION = os.environ.get('ION', 'cr')
N_, Z_, BAS_, TAG_ = IONS[ION]
K.Z3, K.BAS3, K.TAG3 = Z_, BAS_, TAG_
_rname15l = K.rname_k
_SP = {}


# ---------------------------------------------------------------- spaces

def cfg(d):
    import phase15e as E15
    return E15.full_cfg('all', d)


def classify(c):
    """Class of one RAS k = 1 configuration (step 1 partition)."""
    d = dict(c)
    core = ('1s', '2s', '2p', '3s', '3p', '3d')
    ns, np_, nd = d.get('3s', 0), d.get('3p', 0), d.get('3d', 0)
    v = {o: q for o, q in d.items() if o not in core}
    nv = sum(v.values())
    h = (2 - ns) + (6 - np_)
    if d.get('2s', 2) < 2 or (nv == 0 and h <= 1):
        return 'base'                                   # P, 3s 3d4, 2s 3d4 (c2s)
    if nv == 1 and np_ == 5 and list(v) == ['11f'] and ns in (1, 2):
        return 'F'                                      # 3p -> 11f on P and on 3s 3d4
    if nv == 0 and np_ == 4:
        return 'A'                                      # (a) 3p2 -> 3d2 (P side; Q side is 3 holes, outside RAS)
    if nv == 0 and ns == 0:
        return 'B'                                      # (b) 3s2 -> 3d2
    if nv == 2 and np_ == 6:
        return 'PR'                                     # valence pairs 3d2 -> ab on P and 3s 3d4 (incl. 3s3d -> ab)
    if nv == 2 and np_ == 5:
        return 'CV'                                     # core-valence 3p3d -> ab on P and 3s 3d4 (incl. 3s3p -> ab)
    if nv == 1 and h == 2:
        return 'CC1'                                    # core-core with one virtual: (c), (e), 3s2 -> 3d v, 3p2 -> 3d s
    if nv == 2 and h == 2:
        return 'CC2'                                    # core-core pairs into virtuals: 3p2 -> ab ((d) 11f2), 3s2 -> ab
    if nv == 1 and h <= 1:
        return 'S'                                      # remaining singles (on P and on 3s 3d4)
    raise ValueError(sorted(d.items()))


CLASSES = ['F', 'A', 'B', 'PR', 'CV', 'CC1', 'CC2', 'S']


def relax(cf_list, ks=2):
    """Single excitations 3s -> n's, 3p -> n'p, 3d -> n'd (ks most compact virtuals of each l) on the given configs."""
    out = []
    for d in cf_list:
        for sh, l in (('3s', 's'), ('3p', 'p'), ('3d', 'd')):
            if d.get(sh, 0) == 0:
                continue
            for v in K.compact3(l, ks):
                e = dict(d)
                e[sh] -= 1
                e[v] = e.get(v, 0) + 1
                out.append(e)
    return out


def spaces():
    if _SP:
        return _SP
    import phase15e as E15
    c2s = list(E15.core_spaces(N_)['c2s'])
    sp = {}
    if N_ == 3:
        XA = L15.core_core('a')
        sp['c2s_a'] = ('allcore', c2s + XA)
    else:
        sp['c2s_a'] = ('allcore', c2s + [cfg({'3p': 4, '3d': 8}), cfg({'3s': 1, '3p': 4, '3d': 9})])
    if ION != 'cr':
        _SP.update(sp)
        return _SP
    # ---- step 1: RAS k = 1 (+ 2s 3d4 so that the base is c2s), partitioned into classes
    ras = L15.ras_cfgs(1)
    ras = ras[:1] + [c2s[2]] + ras[1:] if c2s[2] not in ras else ras
    assert set(c2s) <= set(ras), 'c2s not in RAS'
    cls = {}
    for c in ras:
        cls.setdefault(classify(c), []).append(c)
    assert sorted(cls['base'], key=str) == sorted(c2s, key=str), cls['base']
    base = c2s
    sp['rasK1'] = ('allcore', base + [c for k in CLASSES for c in cls.get(k, [])])
    for k in CLASSES:
        sp[f'r1_{k}'] = ('allcore', base + cls[k])
    for a, b in [('A', x) for x in CLASSES if x != 'A'] + [('F', x) for x in CLASSES if x not in ('A', 'F')]:
        sp[f'r1_{a}x{b}'] = ('allcore', base + cls[a] + cls[b])
    # S split: valence singles 3d -> v (3d2 v), core singles 3s -> v on P (= 3d -> v on 3s 3d4), 3p -> 16p
    sub = {'Sv': [c for c in cls['S'] if dict(c).get('3s') == 2 and dict(c).get('3p') == 6],
           'Sc': [c for c in cls['S'] if dict(c).get('3s') == 1],
           'Sp': [c for c in cls['S'] if dict(c).get('3p') == 5]}
    assert sum(map(len, sub.values())) == len(cls['S'])
    for k, v in sub.items():
        sp[f'r1_{k}'] = ('allcore', base + v)
    _CLS.update(cls)
    # ---- step 3b/3c: relaxation singles (2 most compact s, p, d) on the hole states
    Q1 = {'3s': 1, '3d': 4}
    AP, AQ = {'3p': 4, '3d': 5}, {'3s': 1, '3p': 4, '3d': 6}
    R = lambda lst: [cfg(d) for d in relax(lst)]  # noqa: E731
    sp['c2s_R'] = ('allcore', c2s + R([Q1]))
    sp['c2s_a_R'] = ('allcore', c2s + [cfg(AP), cfg(AQ)] + R([Q1, AP, AQ]))
    sp['c2s_aP_R'] = ('allcore', c2s + [cfg(AP)] + R([Q1, AP]))
    sp['c2s_aQ_R'] = ('allcore', c2s + [cfg(AQ)] + R([Q1, AQ]))
    sp['c0_aP_R'] = ('allcore', c2s[:1] + [cfg(AP)] + R([AP]))
    for name, (kind, cf) in sp.items():
        assert len(set(cf)) == len(cf), name
    _SP.update(sp)
    return _SP


_CLS = {}
MSP = None


def rname(space, t):
    if space in spaces():
        return f'v15m_{space}_d{N_}_{t}e_{TAG_}'
    return _rname15l(space, t)


K.rname_k = rname


def term_list():
    import phase13d as D
    return D.term_list(N_)


def dims(names):
    import counts as CN
    import phase15e as E15
    sp = spaces()
    res = json.load(open(os.path.join(OUT, 'phase15m_dims.json'))) if os.path.exists(
        os.path.join(OUT, 'phase15m_dims.json')) else {}
    for name in names or list(sp):
        kind, cf = sp[name]
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in term_list()}
        mx = max(per.values())
        f = 1 / 0.6 if N_ == 6 else 1.0          # d6 runs 1/0.6 x slower than d3 at the same size
        r = dict(ncfg=len(cf), per=per, max=mx, sum=sum(per.values()), est_s=f * sum(K.tsec(d)[0] for d in per.values()),
                 maxjob_s=f * K.tsec(mx)[0])
        res[f'{ION}:{name}'] = r
        print(f'{ION}:{name:12s} cfg {len(cf):4d} largest {mx:6d} sum {r["sum"]:7d} est {r["est_s"] / 60:7.1f} CPU-min '
              f'largest job {r["maxjob_s"] / 60:6.1f} min (x3 {3 * r["maxjob_s"] / 60:6.1f})', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase15m_dims.json'), 'w'), indent=1)


def prep(names):
    import phase15j as J
    sp = spaces()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in term_list():
            if N_ == 3:
                K.write_deck3(rname(name, t), kind, cf, t)
            else:
                J.write_deck(rname(name, t), kind, cf, t)
    print('decks written:', ION, ' '.join(names))


def run_names(names):
    for n in names:
        try:
            K.H.F.run_list([(n, TAG_)])
        except Exception as e:  # noqa: BLE001
            print(f'FAILED {n}: {e!r}', flush=True)


def analyze(tag, names):
    if N_ == 3:
        K.analyze3(tag, names)
        return
    import phase15h as H
    import phase15j as J

    def rn(sp, t):
        return H.rname(sp, 6, t, TAG_) if sp == 'c2s' else rname(sp, t)
    J.analyze2(names, rn, tag)


# ---------------------------------------------------------------- configuration averages

def w3j0(a, b, c):
    """(a b c; 0 0 0)."""
    from sympy.physics.wigner import wigner_3j
    return float(wigner_3j(a, b, c, 0, 0, 0))


def orbitals_of(tag, bas, Z):
    import gen_v
    from atomci_heff.reader import read_vectors
    bs = K.H.F.Bas(bas, Z)
    C = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO'), bs.nbf)[0]
    P, h = {}, {}
    for o in ('1s', '2s', '2p', '3s', '3p', '3d'):
        l = 'spd'.index(o[-1])
        c = C[l][:, gen_v.vec(o) - 1]
        P[o] = bs.P(l, c)
        h[o] = float(c @ bs.h[l] @ c)
    return bs, P, h


def e_av(occ, bs, P, h):
    """Slater configuration-average energy (Cowan 6.38): one-electron + same-shell + shell-pair averages."""
    lof = {o: 'spd'.index(o[-1]) for o in occ}

    def R(a, b, c, d, k):     # R^k(ab; cd), a c on electron 1
        return float(I.slater(P[a], P[c], P[b], P[d], k, bs.x, bs.r))
    E = sum(q * h[o] for o, q in occ.items())
    os_ = [o for o in occ if occ[o]]
    for i, a in enumerate(os_):
        la, qa = lof[a], occ[a]
        Fs = R(a, a, a, a, 0) - (2 * la + 1) / (4 * la + 1) * sum(
            w3j0(la, k, la) ** 2 * R(a, a, a, a, k) for k in range(2, 2 * la + 1, 2))
        E += qa * (qa - 1) / 2 * Fs
        for b in os_[i + 1:]:
            lb, qb = lof[b], occ[b]
            G = sum(w3j0(la, k, lb) ** 2 * R(a, b, b, a, k) for k in range(abs(la - lb), la + lb + 1, 2))
            E += qa * qb * (R(a, b, a, b, 0) - 0.5 * G)
    return E


def eav():
    res = {}
    for ion, (N, Z, bas, tag) in IONS.items():
        bs, P, h = orbitals_of(tag, bas, Z)
        base = {'1s': 2, '2s': 2, '2p': 6}
        occ = {'P': {**base, '3s': 2, '3p': 6, '3d': N}, 'Q3s': {**base, '3s': 1, '3p': 6, '3d': N + 1},
               'Q3p2': {**base, '3s': 2, '3p': 4, '3d': N + 2}, 'Q3s3p2': {**base, '3s': 1, '3p': 4, '3d': N + 3}}
        E = {k: e_av(v, bs, P, h) for k, v in occ.items()}
        r = {k: E[k] - E['P'] for k in E if k != 'P'}
        r['gap_Q3p2_minus_Q3s'] = r['Q3p2'] - r['Q3s']
        r['E_P'] = E['P']
        res[ion] = r
        print(f"{ion:3s} d{N}: E_av(3s 3d^{N + 1}) - P {r['Q3s']:8.4f} Eh, E_av(3p4 3d^{N + 2}) - P {r['Q3p2']:8.4f} Eh, "
              f"gap {r['gap_Q3p2_minus_Q3s']:+8.4f} Eh ({r['gap_Q3p2_minus_Q3s'] * 27.2114:+.2f} eV); "
              f"3s 3p4 3d^{N + 3} - P {r['Q3s3p2']:8.4f}", flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase15m_eav.json'), 'w'), indent=1)


# ---------------------------------------------------------------- step 3a

def spectral(spaces_=('c2s', 'c2s_a', 'c2s_a_R'), thr=0.05):
    import phase13b as B
    res = {}
    for sp in spaces_:
        for L, twoS, m, t in term_list():
            name = K.rname_any(sp, t)
            Hm, lab = B.load_run(name)
            P = list(range(m))
            Q = list(range(m, len(lab)))
            e, U = np.linalg.eigh(Hm[np.ix_(Q, Q)])
            V = Hm[np.ix_(P, Q)] @ U                      # <P_i|H|Q_k>
            c2 = (V ** 2).sum(axis=0)
            tot = c2.sum()
            E0 = float(np.mean(np.diag(Hm)[P]))
            labs = np.array(lab)[Q]
            rows = []
            for k in np.argsort(-c2):
                if c2[k] < thr * tot:
                    break
                w = {}
                for cl, x in zip(labs, U[:, k] ** 2):
                    key = '3s 3d^(N+1)' if cl.endswith('3s1 2p6 3p6 3d4') and '2s2' in cl else \
                        ('3p4 3d5' if '3p4 3d5' in cl and '3s2' in cl else
                         ('3s 3p4 3d6' if '3s1 2p6 3p4 3d6' in cl else 'other'))
                    w[key] = w.get(key, 0.0) + float(x)
                rows.append(dict(E_minus_P=float(e[k] - E0), share=float(c2[k] / tot), c2=float(c2[k]), w=w))
            res[f'{sp} {t}'] = dict(total_c2=float(tot), n_Q=len(Q), states=rows,
                                    sum_c2_over_dE=float(np.sum(c2 / (e - E0))))
            print(f'{sp:8s} {t}: sum|V|^2 {tot:.5f}  sum |V|^2/dE {res[f"{sp} {t}"]["sum_c2_over_dE"]:.5f}  nQ {len(Q)}')
            for r in rows:
                print(f'    dE {r["E_minus_P"]:8.4f} Eh  share {r["share"]:.3f}  ' +
                      '  '.join(f'{k} {v:.3f}' for k, v in r['w'].items()))
    json.dump(res, open(os.path.join(OUT, 'phase15m_spectral.json'), 'w'), indent=1)


def doorway(spaces_=('c2s', 'c2s_a', 'c2s_a_R'), thr=0.05):
    """Step 3a, doorway view: V3s = <P|H|Q> restricted to the 3s 3p6 3d4 CSFs, V3p2 = restricted to 3p4 3d5 CSFs.
    Their strength over the Q eigenstates k, and the second-order sums sum_k |V_k|^2 / (E_k - E_P), per (L,S) block
    (E_P = mean P diagonal). The matrix elements <P|H|3s 3d4> are the same in every space (same orbitals)."""
    import phase13b as B
    res = {}
    for sp in spaces_:
        for L, twoS, m, t in term_list():
            Hm, lab = B.load_run(K.rname_any(sp, t))
            P, Q = list(range(m)), list(range(m, len(lab)))
            e, U = np.linalg.eigh(Hm[np.ix_(Q, Q)])
            E0 = float(np.mean(np.diag(Hm)[P]))
            labs = np.array(lab)[Q]
            m3s = np.array([x == '1s2 2s2 3s1 2p6 3p6 3d4' for x in labs])
            m3p = np.array(['3s2' in x and '3p4 3d5' in x for x in labs])
            V = Hm[np.ix_(P, Q)]
            out = {}
            for key, mk in (('3s', m3s), ('3p2', m3p), ('all', np.ones(len(Q), bool))):
                Vk = (V * mk) @ U
                c2 = (Vk ** 2).sum(axis=0)
                tot = float(c2.sum())
                if tot == 0:
                    continue
                order = [k for k in np.argsort(-c2) if c2[k] >= thr * tot]
                out[key] = dict(total=tot, second_order=float(np.sum(c2 / (e - E0))),
                                centroid=float(np.sum(c2 * (e - E0)) / tot),
                                states=[dict(dE=float(e[k] - E0), share=float(c2[k] / tot),
                                             w3s=float((U[m3s, k] ** 2).sum()), w3p2=float((U[m3p, k] ** 2).sum()))
                                        for k in order])
            res[f'{sp} {t}'] = out
            line = '  '.join(f'{k}: tot {v["total"]:.4f} centroid {v["centroid"]:.3f} sum2 {v["second_order"]:.5f}'
                             for k, v in out.items())
            print(f'{sp:8s} {t}: {line}')
            if t in ('2D', '4F'):
                for k, v in out.items():
                    if k == '3s':
                        for r in v['states']:
                            print(f'      3s-doorway: dE {r["dE"]:7.3f}  share {r["share"]:.3f}  w3s {r["w3s"]:.3f}  '
                                  f'w3p4 3d5 {r["w3p2"]:.3f}')
    json.dump(res, open(os.path.join(OUT, 'phase15m_doorway.json'), 'w'), indent=1)


# ---------------------------------------------------------------- step 4

def step4():
    from atomci_heff import gto
    from math import gamma
    res = {}
    for ion, Z, bas, tag in (('cr', 24, 'cr.bas', 'cr15n3'), ('fe', 26, 'fe.bas', 'fe15h6')):
        fns = gto.read_bas(os.path.join(_C.BASIS, bas))
        ff = [(e, c) for l, e, c in fns if l == 3]
        prim = []
        for e, c in ff:
            prim.append(dict(exps=list(map(float, e)), coefs=list(map(float, c)),
                             r_prim=[gamma(5) / gamma(4.5) / np.sqrt(2 * a) for a in e]))
        bs, P, h = orbitals_of(tag, bas, Z)
        r = {o: float(np.sum(P[o] ** 2 * bs.r * bs.w)) for o in ('3s', '3p', '3d')}
        import gen_v
        from atomci_heff.reader import read_vectors
        C = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO'), bs.nbf)[0]
        vf = []
        for k in range(bs.nbf[3]):
            c = C[3][:, k]
            p = bs.P(3, c)
            vf.append(dict(orb=f'{11 + k}f', r=float(np.sum(p * p * bs.r * bs.w)), h=float(c @ bs.h[3] @ c)))
        # are the virtual f the eigenvectors of h? residual of h C = S C eps
        hs = C[3].T @ bs.h[3] @ C[3]
        res[ion] = dict(nbf_f=bs.nbf[3], f_functions=prim, r_core=r, virtual_f=vf,
                        offdiag_h_in_virtual_f=float(np.abs(hs - np.diag(np.diag(hs))).max()))
        print(ion, 'f functions:', bs.nbf[3], prim, r, vf, 'max |offdiag h|', res[ion]['offdiag_h_in_virtual_f'])
    json.dump(res, open(os.path.join(OUT, 'phase15m_step4.json'), 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dims':
        dims(sys.argv[2:])
    elif cmd == 'prep':
        prep(sys.argv[2:])
    elif cmd == 'run_names':
        run_names(sys.argv[2:])
    elif cmd == 'analyze':
        analyze(sys.argv[2], sys.argv[3:])
    elif cmd == 'eav':
        eav()
    elif cmd == 'spectral':
        spectral(sys.argv[2:] or ('c2s', 'c2s_a'))
    elif cmd == 'doorway':
        doorway(sys.argv[2:] or ('c2s', 'c2s_a', 'c2s_a_R'))
    elif cmd == 'step4':
        step4()
