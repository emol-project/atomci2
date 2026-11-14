#!/usr/bin/env python3
"""Phase 15k: Cr IV 3d3 (Cr3+) as a second test of the three-body parameters.

In d3 the H_eff is exactly two-body + three-body: 9 components (7 diagonal term values incl. both 2D seniority states,
one 2D off-diagonal) = A, B, C, alpha, beta (pure two-body up to a constant) + 4 pure three-body operators. The
decomposition is a 9 x 9 solve, no fit:
  two-body: lifts of d2 operators with the d2 term values of Racah's A, B, C (1S 3P 1D 3F 1G: B 14 7 -3 -8 4,
            C 7 0 2 0 2), alpha L(L+1) (0 2 6 12 20; the lift is L(L+1) + 18, the constant goes to A) and beta
            Q = 5 P(1S) (the lift is Racah's Q(3, v) = 4, 0 for v = 1, 3; same normalisation as 15i fit_terms.py);
  three-body: O1 = O2', O2 = O2'' (15c axes, unit norm in d3), O3 = t3 orthogonalised to O1, O2, O4 = the rest of t1
            (JL; t4 lies in the O1-O2 plane = span{u, t4}), all unit norm; T1..T4 in mEh. JL t1..t4 coefficients are given as well.
Fits: 2-body + T1 O1 + T2 O2 to the 8 term energies (1 degree of freedom), the same model as 15j.

usage: phase15k.py item1
-> results/phase15/phase15k_item1.json, results/phase15/15k_terms/terms_cr3_<case>.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase13b as B  # noqa: E402
import phase13f as F13  # noqa: E402
import phase15h as H  # noqa: E402
from atomci_heff import dshell as S  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
EH_CM = H.EH_CM
MEH_CM = 219.4746
TAG3 = 'cr15n3'
PATHS3 = ['c2s', 's7', 'd5', 'g3']
TWO = ['A', 'B', 'C', 'alpha', 'beta']
D2 = {'A': [1, 1, 1, 1, 1], 'B': [14, 7, -3, -8, 4], 'C': [7, 0, 2, 0, 2], 'alpha': [0, 2, 6, 12, 20],
      'beta': [5, 0, 0, 0, 0]}


def op_basis():
    """{name: d3 component vector}: A..beta (cm-1 per unit parameter) and unit O1..O4."""
    assert [(L, twoS) for L, twoS, _ in S.terms(2)] == [(0, 0), (1, 2), (2, 0), (3, 2), (4, 0)]
    proj = []
    for i in range(5):
        v = np.zeros(5)
        v[i] = 1.0
        proj.append(S.to_vec(3, S.lift(S.from_vec(2, v), 2, 3)))
    ops = {k: sum(c * p for c, p in zip(cs, proj)) for k, cs in D2.items()}
    ops['A'] = S.to_vec(3, np.eye(len(S.dets(3))))
    O1, O2 = H.o_axes()
    T3 = S.jl_table(3)
    ip = lambda x, y: S.ip(3, x, y)  # noqa: E731
    ax = [O1, O2]
    for k in ('t3', 't1'):
        v = np.array(T3[k], float)
        for a in ax:
            v = v - ip(v, a) * a
        ax.append(v / np.sqrt(ip(v, v)))
    G = np.array([[ip(a, b) for b in ax] for a in ax])
    assert np.abs(G - np.eye(4)).max() < 1e-10, G
    for i, a in enumerate(ax):
        ops[f'O{i + 1}'] = a
        for k in TWO:
            assert abs(ip(a, ops[k])) < 1e-9, (i, k)       # three-body axes orthogonal to all two-body operators
    return ops


def decompose3(x):
    """x: d3 component vector in Eh -> A..beta (cm-1), T1..T4 (mEh), JL t1..t4 (mEh)."""
    ops = op_basis()
    names = TWO + ['O1', 'O2', 'O3', 'O4']
    M = np.array([ops[n] for n in names]).T
    assert M.shape == (9, 9)
    c = np.linalg.solve(M, x * EH_CM)
    r = {n: float(v) for n, v in zip(TWO, c[:5])}
    r.update({f'T{i + 1}': float(c[5 + i] / MEH_CM) for i in range(4)})
    r['out'] = float(np.hypot(r['T3'], r['T4']) / np.sqrt(sum(r[f'T{i}'] ** 2 for i in range(1, 5))))
    r['ang'] = float(np.degrees(np.arctan2(r['T2'], r['T1'])))
    t = sum(c[5 + i] * ops[f'O{i + 1}'] for i in range(4)) / MEH_CM
    T3 = S.jl_table(3)
    G = np.array([[S.ip(3, T3[a], T3[b]) for b in B.TK] for a in B.TK])
    r['jl'] = dict(zip(B.TK, np.linalg.solve(G, [S.ip(3, T3[a], t) for a in B.TK]).tolist()))
    return r


def term_energies(x):
    """{term: [E] (cm-1, above 4F)} from a d3 component vector (Eh)."""
    bl = B.blocks_from_vec(3, x)
    E = {}
    for L, twoS, m, t in __import__('phase13d').term_list(3):
        E[t] = np.linalg.eigvalsh(bl[(L, twoS)]).tolist()
    e0 = E['4F'][0]
    return {t: [(v - e0) * EH_CM for v in E[t]] for t in sorted(E, key=lambda t: E[t][0])}


def fit_terms(data, extra=('O1', 'O2'), iters=60):
    """Gauss-Newton fit of 2-body + T*O to term energies {term: [E]} (cm-1). Returns params (cm-1; T in mEh), errors,
    rms. A is absorbed by fitting relative to the ground term via the free constant."""
    ops = op_basis()
    names = TWO + list(extra)
    import phase13d as D
    tl = {t: (L, twoS) for L, twoS, m, t in D.term_list(3)}
    obs = [(t, k, e) for t, es in data.items() for k, e in enumerate(sorted(es))]
    E = np.array([e for *_, e in obs])
    p = np.array([0, 450, 3700, 0, 0] + [0] * len(extra), float)
    for _ in range(iters):
        x = sum(pi * ops[n] for pi, n in zip(p, names))
        bl = B.blocks_from_vec(3, x)
        calc, J = [], []
        eig = {t: np.linalg.eigh(bl[tl[t]]) for t in data}
        for t, k, e in obs:
            w, v = eig[t]
            calc.append(w[k])
            J.append([v[:, k] @ B.blocks_from_vec(3, ops[n])[tl[t]] @ v[:, k] for n in names])
        calc, J = np.array(calc), np.array(J)
        dp = np.linalg.lstsq(J, E - calc, rcond=None)[0]
        p += dp
        if np.abs(dp).max() < 1e-9:
            break
    r = E - calc
    dof = max(len(E) - len(names), 1)
    s = float(np.sqrt(r @ r / dof))
    err = np.sqrt(np.diag(np.linalg.pinv(J.T @ J))) * s
    out = {n: float(v) for n, v in zip(names, p)}
    out.update({f'{n}_err': float(e) for n, e in zip(names, err)})
    for n in extra:
        out[n] /= MEH_CM
        out[f'{n}_err'] /= MEH_CM
    out['rms'] = s
    out['dof'] = len(E) - len(names)
    return out


def path_blocks3(space):
    import phase13d as D
    names = [f'v15e_{space}_d3_{t}e_{TAG3}' for L, twoS, m, t in D.term_list(3)]
    out, info = {}, []
    for b in H.blocks(3, names):
        Bk, inf = F13.heff_block(b, b['H'])
        out[(b['L'], b['twoS'])] = (Bk, b['U'].T @ b['H'][np.ix_(b['P'], b['P'])] @ b['U'])
        info.append(inf)
    return out, dict(wP=min(i['wP'] for i in info), below=sum(i['below'] for i in info))


def item1():
    import phase13d as D
    names = [f'vcm_d3_{t}e_{TAG3}' for L, twoS, m, t in D.term_list(3)]
    hf = {(b['L'], b['twoS']): b['U'].T @ b['H'][:b['m'], :b['m']] @ b['U'] for b in H.blocks(3, names)}
    blk, info = {}, {}
    for sp in PATHS3:
        blk[sp], info[sp] = path_blocks3(sp)
        dev = [blk[sp][k][1] - hf[k] for k in hf]
        off = np.mean([np.diag(d).mean() for d in dev])
        assert max(np.abs(d - off * np.eye(len(d))).max() for d in dev) < 1e-7, sp
        info[sp]['php_offset_Eh'] = float(off)
    cases = {'hf': [], **{f'hf+{sp}': [sp] for sp in PATHS3}, 'all': PATHS3}
    d = os.path.join(OUT, '15k_terms')
    os.makedirs(d, exist_ok=True)
    res = {}
    xhf = B.vec_from_blocks(3, hf)
    for name, sps in cases.items():
        x = B.vec_from_blocks(3, {k: hf[k] + sum(blk[sp][k][0] - blk[sp][k][1] for sp in sps) for k in hf})
        dec = decompose3(x)
        corr = decompose3(x - xhf)
        te = term_energies(x)
        json.dump(te, open(os.path.join(d, f'terms_cr3_{name}.json'), 'w'), indent=1)
        fit = fit_terms(te)
        res[name] = dict(proj=dec, corr=corr, fit8=fit, terms=te, **({'info': info[sps[0]]} if len(sps) == 1 else {}))
        print(f'{name:8s} proj T1 {corr["T1"]:+8.3f} T2 {corr["T2"]:+7.3f} T3 {corr["T3"]:+7.3f} T4 {corr["T4"]:+7.3f} '
              f'out {corr["out"]:.3f} | B {dec["B"]:7.1f} C {dec["C"]:7.1f} al {dec["alpha"]:6.1f} be {dec["beta"]:7.1f}'
              f' | fit8 T1 {fit["O1"]:+7.3f}+-{fit["O1_err"]:.3f} T2 {fit["O2"]:+7.3f}+-{fit["O2_err"]:.3f} B {fit["B"]:7.1f}'
              f' C {fit["C"]:7.1f} rms {fit["rms"]:.2f}', flush=True)
    json.dump(dict(cases=res, info=info), open(os.path.join(OUT, 'phase15k_item1.json'), 'w'), indent=1, default=float)


# ---------------------------------------------------------------- item 2: new CI for Cr3+ d3, mirroring Fe (15j)

Z3, BAS3 = 24, 'cr.bas'
_RV3 = {}


def r_virt3():
    """<r> (bohr) of every virtual orbital of the Cr3+ own set cr15n3 (orbsets_15a '3|l', cr.bas)."""
    if not _RV3:
        import gen_v
        import phase15j as J
        bs = H.F.Bas(BAS3, Z3)
        z = np.load(os.path.join(OUT, 'orbsets_15a.npz'))
        for l, lst in J.VIRT.items():
            L = J.LV.index(l)
            for o in lst:
                P = bs.P(L, z[f'3|{L}'][:, gen_v.vec(o) - 1])
                _RV3[o] = float(np.sum(P * P * bs.r * bs.w))
        P = bs.P(2, z['3|2'][:, 0])
        _RV3['3d'] = float(np.sum(P * P * bs.r * bs.w))
    return _RV3


def compact3(l, n):
    import phase15j as J
    rv = r_virt3()
    return sorted(J.VIRT[l], key=rv.get)[:n]


def pair_cfgs3(nper, types=None):
    """[valence dicts] 3d1 a b (a <= b, l_a + l_b even): nper most compact virtuals per l (None = all)."""
    import itertools as it
    import phase15j as J
    allv = [(o, l) for l in J.LV for o in (J.VIRT[l] if nper is None else compact3(l, nper))]
    out = []
    for (a, la), (b, lb) in it.combinations_with_replacement(allv, 2):
        t = ''.join(sorted(la + lb, key=J.LV.index))
        if t not in J.PTYPES or (types and t not in types):
            continue
        d = {'3d': 1}
        d[a] = d.get(a, 0) + 1
        d[b] = d.get(b, 0) + 1
        out.append(d)
    return out


def spaces3():
    """{name: (kind, [frozenset configs], P first)} for Cr3+ d3."""
    import phase15e as E15
    import phase15j as J
    fs = J.CP['f'][1]
    c2s = list(E15.core_spaces(3)['c2s'])
    P0 = E15.full_cfg('vcm', {'3d': 3})
    sp = {}
    sp['cFf'] = ('allcore', c2s + [E15.full_cfg('all', {'3p': 5, '3d': 3, v: 1}) for v in fs])
    sp['cFf7'] = ('allcore', sp['cFf'][1] + [E15.full_cfg('all', {'3s': 1, '3p': 5, '3d': 4, v: 1}) for v in fs])
    Pc = pair_cfgs3(2)
    sp['pr2c_all'] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in Pc])
    sp['cpr2cP'] = ('allcore', c2s + [E15.full_cfg('all', d) for d in Pc])
    sp['cpr2cPQ'] = ('allcore', sp['cpr2cP'][1] + [E15.full_cfg('all', {**d, '3s': 1, '3d': 2}) for d in Pc])
    for ty in (['ff', 'pf', 'pp'], ['ff'], ['pf'], ['pp']):
        sp['prA_' + ''.join(ty)] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in pair_cfgs3(None, ty)])
    # extras (if cheap): pairs ff, pf, pp with all virtuals on top of the core path, P side and both sides;
    # all pair types with all virtuals, alone and with the core path on both sides
    PA = pair_cfgs3(None, ['ff', 'pf', 'pp'])
    sp['cprFP'] = ('allcore', c2s + [E15.full_cfg('all', d) for d in PA])
    sp['cprFPQ'] = ('allcore', sp['cprFP'][1] + [E15.full_cfg('all', {**d, '3s': 1, '3d': 2}) for d in PA])
    AA = pair_cfgs3(None)
    sp['prA_all'] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in AA])
    sp['cprAPQ'] = ('allcore', c2s + [E15.full_cfg('all', d) for d in AA] +
                    [E15.full_cfg('all', {**d, '3s': 1, '3d': 2}) for d in AA])
    # core-valence 3p 3d -> a b (l_a + l_b odd), 2 most compact virtuals per l, on P (3p5 3d2 a b) and on 3s 3d4
    # (3s 3p5 3d3 a b). All virtuals: largest block 13 005 (P side) / 58 257 (both sides), over budget.
    CVP = cv_cfgs3(2, 'P')
    CVQ = cv_cfgs3(2, 'Q')
    sp['cv2P'] = ('allcore', sp['cFf7'][1] + CVP)
    sp['cv2PQ'] = ('allcore', sp['cv2P'][1] + CVQ)
    # everything around the core path: cFf7 + pairs (2 compact per l, both sides) + cv (2 compact per l, both sides)
    sp['big2'] = ('allcore', sp['cv2PQ'][1] + [c for c in sp['cpr2cPQ'][1] if c not in c2s])
    # + pairs (1 compact per l) on top of the 3p -> f states with the 2 most compact f: 3p5 f 3d1 a b, 3s 3p5 f 3d2 a b
    f2 = compact3('f', 2)
    P1 = pair_cfgs3(1)
    PF = [E15.full_cfg('all', {**d, '3p': 5, '3d': 1, f: d.get(f, 0) + 1}) for d in P1 for f in f2]
    PF7 = [E15.full_cfg('all', {**d, '3s': 1, '3p': 5, '3d': 2, f: d.get(f, 0) + 1}) for d in P1 for f in f2]
    sp['big2T'] = ('allcore', sp['big2'][1] + PF + PF7)
    # 1 most compact virtual per l for the core-valence doubles
    CV1P, CV1Q = cv_cfgs3(1, 'P'), cv_cfgs3(1, 'Q')
    sp['cv1P'] = ('allcore', sp['cFf7'][1] + CV1P)
    sp['cv1PQ'] = ('allcore', sp['cv1P'][1] + CV1Q)
    sp['big1'] = ('allcore', sp['cv1PQ'][1] + [c for c in sp['cpr2cPQ'][1] if c not in c2s])
    sp['big1T'] = ('allcore', sp['big1'][1] + PF + PF7)
    # smaller: pairs (1 compact per l) on top of the 3p -> f states with the single most compact f
    f1 = compact3('f', 1)
    PF1 = [E15.full_cfg('all', {**d, '3p': 5, '3d': 1, f: d.get(f, 0) + 1}) for d in P1 for f in f1]
    PF17 = [E15.full_cfg('all', {**d, '3s': 1, '3p': 5, '3d': 2, f: d.get(f, 0) + 1}) for d in P1 for f in f1]
    sp['big1T1'] = ('allcore', sp['big1'][1] + PF1 + PF17)
    sp['cFf7T1'] = ('allcore', sp['cFf7'][1] + PF1 + PF17)
    # 3p -> f x valence pairs without the core-valence doubles (cv barely acts, cv1PQ - cFf7 = -0.01):
    # cFf7 + pairs 2 compact per l on P and on 3s 3d4, + pairs 1 compact per l on the 3p -> f states (1 f)
    PR = [c for c in sp['cpr2cPQ'][1] if c not in c2s]
    sp['cpFT1'] = ('allcore', sp['cFf7'][1] + PR + PF1 + PF17)
    # P-like part only: pairs on 3p5 f 3d3 (3p5 f 3d1 a b), not on 3s 3p5 f 3d4
    sp['cFf7T1P'] = ('allcore', sp['cFf7'][1] + PF1)
    sp['cpFT1P'] = ('allcore', sp['cFf7'][1] + PR + PF1)
    sp['cpF'] = ('allcore', sp['cFf7'][1] + PR)
    return sp


def cv_cfgs3(nper, side):
    """3p 3d -> a b (a, b virtual, l_a + l_b odd; nper most compact per l, None = all) on P or on 3s 3d4."""
    import itertools as it
    import phase15e as E15
    import phase15j as J
    allv = [(o, l) for l in J.LV for o in (J.VIRT[l] if nper is None else compact3(l, nper))]
    out = []
    for (a, la), (b, lb) in it.combinations_with_replacement(allv, 2):
        if (J.LV.index(la) + J.LV.index(lb)) % 2 == 0:
            continue
        d = {'3p': 5, '3d': 2} if side == 'P' else {'3s': 1, '3p': 5, '3d': 3}
        d[a] = d.get(a, 0) + 1
        d[b] = d.get(b, 0) + 1
        out.append(E15.full_cfg('all', d))
    return out


def rname_k(space, t):
    return f'v15k_{space}_d3_{t}e_{TAG3}'


def tsec(dim):
    """Time per job (s). The 15j fit 1.3e-4 dim^1.66 underestimates large jobs 10x (Fe cvFP 3F: 9982 CSFs, 5678 s;
    cvF 3F: 7102, 2307 s -> exponent 2.65). d3 runs 0.6 x faster: central 273 (dim / 3811)^2.65, high x 2."""
    t = 273 * (dim / 3811) ** 2.65      # recalibrated on Cr3+ d3 (cv2P 2G: 3811 CSFs, 273 s), 0.6 x the d6 value
    return t, 2 * t


def dims3():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for name, (kind, cf) in spaces3().items():
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(3)}
        mx = max(per.values())
        res[name] = dict(kind=kind, ncfg=len(cf), per=per, max=mx, sum=sum(per.values()),
                         est_s=sum(tsec(d)[0] for d in per.values()), est_hi_s=sum(tsec(d)[1] for d in per.values()),
                         maxjob_s=tsec(mx)[0], maxjob_hi_s=tsec(mx)[1])
        r = res[name]
        print(f'{name:10s} cfg {len(cf):4d} largest {mx:6d} sum {r["sum"]:7d} est {r["est_s"] / 60:7.1f} min '
              f'(hi {r["est_hi_s"] / 60:7.1f}) largest job {r["maxjob_s"] / 60:6.1f} min (hi {r["maxjob_hi_s"] / 60:6.1f})')
    json.dump(dict(r_virt=r_virt3(), compact={l: compact3(l, 2) for l in 'spdfgh'}, spaces=res),
              open(os.path.join(OUT, 'phase15k_dims.json'), 'w'), indent=1)


STOP_AFTER_H = False     # Phase 15q: True writes decks that stop ATOMCI after H (units 38/39) is written


def stop_after_h(deck):
    """CIMAIN section 1 only (&ci nrnfg=1,0,0,0,0,0) and RSMAIN off (&cntl nrnfg(8)=0); validated in 15p Addendum 2."""
    import re
    deck, n1 = re.subn(r'(&cntl\nnrnfg=1,0,1,1,0,1,1,)1,', r'\g<1>0,', deck)
    deck, n2 = re.subn(r'(&ci\nnrnfg=)1,1,1,0,1,1,', r'\g<1>1,0,0,0,0,0,', deck)
    assert n1 == n2 == 1, (n1, n2)
    return deck


def write_deck3(dst, kind, cf, t):
    """Deck + CSF list for one d3 term: 'vcm' (frozen [Ne], inputs/vcm_d3_<t>e) or 'allcore' (15e allcore_deck).
    STOP_AFTER_H: the deck stops ATOMCI after H is written (no ATOMCI diagonalization)."""
    import re
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    import phase15j as J
    m = next(mm for L, twoS, mm, tt in D.term_list(3) if tt == t)
    if kind == 'vcm':
        src = os.path.join(ROOT, 'inputs', f'vcm_d3_{t}e')
        deck = open(os.path.join(src, 'be.deck')).read()
        csf = open(os.path.join(src, 'be.csf')).read()
        assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
        deck = deck.replace('z=23,', f'z={Z3},').replace('FILE=v_qzp.bas', f'FILE={BAS3}')
    else:
        deck, csf = E15.allcore_deck(3, t, Z3, BAS3)
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGHIK]', header.split(')')[0])]
    tot, ncfg, per = CN.count(E15.LMAP, cf, t)
    assert per[cf[0]] == m, (dst, t)
    order = [c for c in cf if c in per]
    assert not [o for c in order for o in dict(c) if o not in seq], dst
    lines = [J.wrap80(['  ' + f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)]) for c in order]
    deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
    if kind == 'allcore':
        deck = gen_v.sub_all(deck, 'nstate', m)
        nob, mpob = [], []
        for let in 'spdfghik'[:1 + max('spdfghik'.index(o[-1]) for o in seq)]:     # Q3: i when the basis has it
            orbs = [o for o in seq if o.endswith(let)]
            nob.append(len(orbs))
            mpob += [gen_v.vec(o) for o in orbs]
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
        deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
        deck = gen_v.sub_all(deck, 'nind', len(nob))
        deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{k}' for l, k in enumerate(nob)))
    deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
    if STOP_AFTER_H:
        deck = stop_after_h(deck)
    out = os.path.join(ROOT, 'inputs', dst)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'be.deck'), 'w').write(deck)
    open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')


def prep3(names):
    import phase13d as D
    sp = spaces3()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in D.term_list(3):
            write_deck3(rname_k(name, t), kind, cf, t)
    print('decks written:', ' '.join(names))


def run_names(names):
    """Run each job (Cr3+ own orbitals cr15n3); a failure is logged (FAILED <name>) and skipped."""
    for n in names:
        try:
            H.F.run_list([(n, TAG3)])
        except Exception as e:  # noqa: BLE001
            print(f'FAILED {n}: {e!r}', flush=True)


def launch(spaces, nproc, log, per_job_s=14400):
    """Shell launcher balanced over nproc processes by estimated time; each list runs smallest first."""
    import phase13d as D
    dims = json.load(open(os.path.join(OUT, 'phase15k_dims.json')))['spaces']
    jobs = [(tsec(dims[sp]['per'][t])[0], rname_k(sp, t)) for sp in spaces for L, twoS, m, t in D.term_list(3)]
    jobs.sort(reverse=True)
    lists, load = [[] for _ in range(nproc)], [0.0] * nproc
    for w, n in jobs:
        i = load.index(min(load))
        lists[i].append(n)
        load[i] += w
    lines = ['#!/bin/bash', f'cd {ROOT}',
             'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1', f'L={log}']
    for i, lst in enumerate(lists):
        # one job per call under a per-job wall limit; a timeout (exit 124) is logged as TIMEOUT and the list goes on
        body = '; '.join(f'timeout {per_job_s} {sys.executable} {_C.script("phase15k.py")} run_names {n} || echo "TIMEOUT-or-FAILED {n} $?"'
                         for n in reversed(lst))
        lines.append(f'( {body} ) > $L.{i} 2>&1 &')
        lines.append(f'p{i}=$!')
    lines.append(' '.join(f'wait $p{i}; e{i}=$?;' for i in range(nproc)))
    lines.append(f'cat {" ".join(f"$L.{i}" for i in range(nproc))} > $L; rm -f {" ".join(f"$L.{i}" for i in range(nproc))}')
    lines.append('echo "exit ' + ' '.join(f'$e{i}' for i in range(nproc)) + '" >> $L')
    lines.append('echo done >> $L')
    print(f'estimated load per process (min): {[round(x / 60, 1) for x in load]}')
    return '\n'.join(lines) + '\n'


def rname_any(sp, t):
    return f'v15e_{sp}_d3_{t}e_{TAG3}' if sp in PATHS3 else rname_k(sp, t)


def analyze3(tag, spaces):
    """Exact / Sylvester / closure H_eff of each space (15j heff_levels), placed on the HF P block (vcm cr15n3):
    HF + (H_eff - PHP). Exact decomposition (A..beta, T1..T4, JL t) and the 8-term fit (2-body + T1 O1 + T2 O2).
    -> phase15k_<tag>.json, 15k_terms/terms_cr3_<tag>.json"""
    import phase13d as D
    import phase15j as J
    names = [f'vcm_d3_{t}e_{TAG3}' for L, twoS, m, t in D.term_list(3)]
    hf = {b['t']: b['U'].T @ b['H'][:b['m'], :b['m']] @ b['U'] for b in H.blocks(3, names)}
    tl = {t: (L, twoS) for L, twoS, m, t in D.term_list(3)}
    res, ens = {}, {}
    for space in spaces:
        # two passes, one dense block in memory at a time (all 7 cpFT1 blocks together exceed the RAM):
        # pass 1 keeps only the CSF diagonals, pass 2 reloads each block for heff_levels
        names_sp = [rname_any(space, t) for L, twoS, m, t in D.term_list(3)]
        bl = []
        for (L, twoS, m, t), name in zip(D.term_list(3), names_sp):
            Hm, lab, dg = J.load_block(name)
            bl.append(dict(t=t, L=L, twoS=twoS, lab=lab, m=m, diag=dg, name=name))
            del Hm
        maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
        acc = {}
        for b in bl:
            g = (2 * b['L'] + 1) * (b['twoS'] + 1)
            for i, lab in enumerate(b['lab']):
                s_, w = acc.get(lab, (0.0, 0.0))
                acc[lab] = (s_ + g * b['diag'][i], w + g)
        eav = {k: s_ / w for k, (s_, w) in acc.items()}
        p0 = bl[0]['lab'][0]
        for b in bl:
            b['P'] = [i for i, l in enumerate(b['lab']) if l == b['lab'][0]]
            assert len(b['P']) == b['m'] and b['P'] == list(range(b['m'])), (b['name'], b['P'])
            b['Q'] = [i for i in range(len(b['lab'])) if i not in b['P']]
            b['U'] = np.array(maps[f'vcm_d3_{b["t"]}e'])
            b['N'] = 3
            b['delta'] = {c: eav[c] - eav[p0] for c in set(b['lab'][i] for i in b['Q'])}
        lev, info = {}, []
        for b in bl:
            b['op' if J.SOLVER == 'iter' else 'H'] = J.load_block(b['name'])[0]
            hb, inf = J.heff_levels(b)
            b.pop('op', None), b.pop('H', None)
            info.append(inf)
            off = np.diag(hb['php'] - hf[b['t']])
            assert np.abs(hb['php'] - hf[b['t']] - off.mean() * np.eye(len(off))).max() < 1e-7, (space, b['t'])
            for k, v in hb.items():
                if k != 'php':
                    lev.setdefault(k, {})[tl[b['t']]] = hf[b['t']] + v - hb['php']
        r = dict(dims={b['t']: len(b['lab']) for b in bl}, wP=min(i['wP'] for i in info),
                 below=sum(i['below'] for i in info), gap_eV=min(i['gap'] for i in info))
        J.dump_levels(tag, space, lev)
        for k, by in lev.items():
            x = B.vec_from_blocks(3, by)
            te = term_energies(x)
            ens[f'{space} {k}'] = te
            r[k] = dict(proj=decompose3(x), fit8=fit_terms(te))
        res[space] = r
        e, f8 = r['exact']['proj'], r['exact']['fit8']
        print(f'{space:10s} max {max(r["dims"].values()):6d} wP {r["wP"]:.3f} below {r["below"]} | exact T1 {e["T1"]:+8.3f} '
              f'T2 {e["T2"]:+7.3f} T3 {e["T3"]:+7.3f} T4 {e["T4"]:+7.3f} B {e["B"]:7.1f} C {e["C"]:7.1f} al {e["alpha"]:6.1f} '
              f'be {e["beta"]:7.1f} | syl T1 {r["sylvester"]["proj"]["T1"]:+8.3f} T2 {r["sylvester"]["proj"]["T2"]:+7.3f} | '
              f'clo T1 {r["closure"]["proj"]["T1"]:+8.3f} | fit8 T1 {f8["O1"]:+8.3f} T2 {f8["O2"]:+7.3f} rms {f8["rms"]:.1f}',
              flush=True)
    json.dump(res, open(os.path.join(OUT, f'phase15k_{tag}.json'), 'w'), indent=1, default=float)
    os.makedirs(os.path.join(OUT, '15k_terms'), exist_ok=True)
    json.dump(ens, open(os.path.join(OUT, '15k_terms', f'terms_cr3_{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'item1':
        item1()
    elif cmd == 'dims3':
        dims3()
    elif cmd == 'prep3':
        prep3(sys.argv[2:])
    elif cmd == 'analyze3':
        analyze3(sys.argv[2], sys.argv[3:])
    elif cmd == 'run_names':
        run_names(sys.argv[2:])
    elif cmd == 'launch':
        sh = sys.argv[2]
        open(sh, 'w').write(launch(sys.argv[5:], int(sys.argv[3]), sys.argv[4]))
        os.chmod(sh, 0o755)
