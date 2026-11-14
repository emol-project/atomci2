#!/usr/bin/env python3
"""Phase 15j: breakdown of the remaining Fe III 3d6 discrepancy (15h gamma_eff 6.16 vs Trees' 5.0 cm-1).

Item 1 (no new CI): Fe2+ 3d6 H_eff = H_HF (P block) + sum over paths of (H_eff,path - PHP_path), from the 15h runs.
Term energies (cm-1, relative to 5D; repeated terms as both eigenvalues) are written for 15i's fit_terms.py:
HF only, HF + each path alone, all paths.
usage: phase15j.py terms
-> results/phase15/15j_terms/terms_*.json
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

OUT = os.path.join(ROOT, 'results', 'phase15')
EH_CM = H.EH_CM
PATHS = ['c2s', 's7', 'dSD', 'g2']
SOLVER = os.environ.get('HEFF_SOLVER', 'dense')     # dense | iter (Phase 15o: atomci_heff.iterative, no dense H)


def load_block(name):
    """(H, labels, CSF diagonal) of one run: dense H, or SymLower when HEFF_SOLVER=iter."""
    if SOLVER == 'iter':
        from atomci_heff.sparse import SymLower
        d = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
        op = SymLower.from_npz(d)
        return op, [str(x) for x in d['labels']], op.d.copy()
    Hm, lab = B.load_run(name)
    return Hm, lab, np.diag(Hm).copy()


def dump_levels(tag, space, lev):
    """HEFF_DUMP=<dir>: every H_eff level of every term block -> <dir>/<tag>_<space>_<solver>.npz (validation)."""
    dd = os.environ.get('HEFF_DUMP')
    if dd:
        os.makedirs(dd, exist_ok=True)
        np.savez(os.path.join(dd, f'{tag}_{space}_{SOLVER}.npz'),
                 **{f'{k}|{t if isinstance(t, str) else "%d_%d" % t}': v for k, by in lev.items() for t, v in by.items()})


def path_blocks(space):
    """{term: (H_eff, PHP)} in the canonical seniority basis (Eh) for one 15h Fe2+ d6 space."""
    import phase13d as D
    names = [H.rname(space, 6, t, 'fe15h6') for L, twoS, m, t in D.term_list(6)]
    out = {}
    for b in H.blocks(6, names):
        Bk, _ = F13.heff_block(b, b['H'])
        out[b['t']] = (Bk, b['U'].T @ b['H'][np.ix_(b['P'], b['P'])] @ b['U'])
    return out


def terms():
    blk = {sp: path_blocks(sp) for sp in PATHS}
    # H_HF: the P block of a frozen-[Ne] (vcm) space; all vcm spaces share it
    hf = {t: v[1] for t, v in blk['dSD'].items()}
    for sp in ('s7', 'g2'):
        assert max(np.abs(blk[sp][t][1] - hf[t]).max() for t in hf) < 1e-9, sp
    sets = {'hf': [], **{f'hf+{sp}': [sp] for sp in PATHS}, 'all': PATHS}
    d = os.path.join(OUT, '15j_terms')
    os.makedirs(d, exist_ok=True)
    for name, sps in sets.items():
        E = {t: np.linalg.eigvalsh(hf[t] + sum(blk[sp][t][0] - blk[sp][t][1] for sp in sps)) for t in hf}
        e0 = E['5D'][0]
        js = {t: [float((x - e0) * EH_CM) for x in E[t]] for t in sorted(E, key=lambda t: E[t][0])}
        json.dump(js, open(os.path.join(d, f'terms_{name}.json'), 'w'), indent=1)
        print(name, ' '.join(f'{t} ' + ','.join(f'{x:.1f}' for x in v) for t, v in js.items()))


def check0():
    """Step 0.2: 15i's E1_d6, E2_d6 (dn determinant basis) vs our O2', O2'' (15c axes, unit norm in d3) lifted to d6."""
    from atomci_heff import dshell as S
    T = os.path.join(OUT, '15i_trees')
    E = [np.load(os.path.join(T, f'E{i}_d6.npy')) for i in (1, 2)]
    O = [S.lift(S.from_vec(3, o), 3, 6) for o in H.o_axes()]
    # dn: spin orbitals (m, sigma) in the cfp_l order, determinants sorted by bitmask; same creation-order sign
    ix = S.index(6)
    perm = []
    for m in range(1 << 10):
        if bin(m).count('1') == 6:
            perm.append(ix[tuple(i for i in range(10) if m >> i & 1)])
    assert len(perm) == len(ix) == E[0].shape[0]
    res = {}
    for i in range(2):
        for j in range(2):
            A, Bm = E[i], O[j][np.ix_(perm, perm)]
            res[f'E{i + 1}.O{j + 1}'] = float(np.sum(A * Bm) / np.linalg.norm(A) / np.linalg.norm(Bm))
    # basis-free invariants: norms, tr(A^2), tr(A^3), sorted eigenvalues
    for i in range(2):
        ea, eo = np.linalg.eigvalsh(E[i]), np.linalg.eigvalsh(O[i])
        res[f'norm E{i + 1}, O{i + 1} (d6)'] = [float(np.linalg.norm(E[i])), float(np.linalg.norm(O[i]))]
        res[f'tr3 E{i + 1}, O{i + 1}'] = [float(np.sum(ea ** 3)), float(np.sum(eo ** 3))]
        res[f'max |eig diff| {i + 1}'] = float(np.abs(ea - eo).max())
    res['tr(E1 E2), tr(O1 O2)'] = [float(np.sum(E[0] * E[1])), float(np.sum(O[0] * O[1]))]
    for k, v in res.items():
        print(k, v)
    json.dump(res, open(os.path.join(OUT, 'phase15j_check0.json'), 'w'), indent=1)


def fit1():
    """Item 1: fit_terms.py models (a) and (b) on 16 terms and on the 14 observed terms (upper 1S, 1D dropped), for HF,
    HF + each path, all paths, and the experimental term centroids."""
    T = os.path.join(OUT, '15i_trees')
    os.chdir(T)
    sys.path.insert(0, T)
    import fit_terms as FT
    c = 403.4228550788862
    h15 = json.load(open(os.path.join(OUT, 'phase15h.json')))
    proj = {'hf': (0.0, 0.0), **{f'hf+{sp}': (1e3 * h15[f'Fe2+ {sp}']['T1'], 1e3 * h15[f'Fe2+ {sp}']['T2'])
                                 for sp in PATHS}}
    proj['all'] = tuple(sum(proj[f'hf+{sp}'][i] for sp in PATHS) for i in range(2))
    cases = {k: json.load(open(os.path.join(OUT, '15j_terms', f'terms_{k}.json'))) for k in proj}
    cases['expt'] = json.load(open(os.path.join(T, 'expt_term_centroids.json')))
    two = FT.two
    res = {}
    for name, data in cases.items():
        for sel in ('16', '14'):
            if name == 'expt' and sel == '16':
                continue
            d = data if sel == '16' else {t: (v[:1] if t in ('1S', '1D') else v) for t, v in data.items()}
            pa, ea, sa = FT.fit(d, two + ['gamma'], FT.ops, [85000, 975, 3600, 69, -156, 5])
            pb, eb, sb = FT.fit(d, two + ['T1', 'T2'], FT.ops, [85000, 975, 3600, 69, -156, 2000, 700])
            r = dict(gamma=pa[-1], gamma_err=ea[-1], rms_a=sa, T1=pb[5] / 219.4746, T1_err=eb[5] / 219.4746,
                     T2=pb[6] / 219.4746, T2_err=eb[6] / 219.4746, gamma_eff=pb[5] / c, rms_b=sb,
                     **{n: pb[i] for i, n in enumerate(two)}, two_a={n: pa[i] for i, n in enumerate(two)})
            if name in proj:
                r['proj_T1'], r['proj_T2'] = proj[name]
            res[f'{name} {sel}'] = r
            print(f'{name:8s} {sel}: (a) gamma {r["gamma"]:6.3f}+-{r["gamma_err"]:.3f} rms {sa:6.1f} | (b) T1 '
                  f'{r["T1"]:6.2f}+-{r["T1_err"]:.2f} T2 {r["T2"]:6.2f}+-{r["T2_err"]:.2f} B {r["B"]:7.1f} C '
                  f'{r["C"]:7.1f} al {r["alpha"]:6.1f} be {r["beta"]:7.1f} rms {sb:6.1f}'
                  + (f' | proj {proj[name][0]:6.2f} {proj[name][1]:6.2f}' if name in proj else ''), flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase15j_fit1.json'), 'w'), indent=1, default=float)


# ---------------------------------------------------------------- item 2: core polarization (3s, 3p singles)

CP = {'p': ('3p', ['11p', '12p', '13p', '14p', '15p', '16p']), 'f': ('3p', ['11f', '12f', '13f', '14f']),
      'h': ('3p', ['11h', '12h']), 'd': ('3s', ['11d', '12d', '13d', '14d', '15d']), 'g': ('3s', ['11g', '12g', '13g'])}
CP_SPACES = {'cpp': 'p', 'cpf': 'f', 'cph': 'h', 'csd': 'd', 'csg': 'g', 'cpall': 'pfhdg'}


def configs2(space):
    import phase15e as E15
    P0 = E15.full_cfg('vcm', {'3d': 6})
    cf = [P0]
    for cls in CP_SPACES[space]:
        h, vs = CP[cls]
        q = {'3s': 2, '3p': 6}
        q[h] -= 1
        cf += [E15.full_cfg('vcm', {**q, '3d': 6, v: 1}) for v in vs]
    return cf


def dims2():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for sp in CP_SPACES:
        cf = configs2(sp)
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(6)}
        mx = max(per.values())
        # 15h fit t = 8e-7 dim^2.40 s (15f d4) x 3 for d6; dim^2.9 as the pessimistic bound
        est = sum(8e-7 * d ** 2.4 * 3 for d in per.values())
        res[sp] = dict(per=per, max=mx, sum=sum(per.values()), est_s=est,
                       est_hi_s=sum(8e-7 * d ** 2.9 * 3 / 1107 ** 0.5 for d in per.values()))
        print(f'{sp:6s} largest {mx:6d} ({max(per, key=per.get)}) sum {res[sp]["sum"]:7d} configs {len(cf)} '
              f'est {est:8.0f} s (hi {res[sp]["est_hi_s"]:8.0f} s)')
    json.dump(res, open(os.path.join(OUT, 'phase15j_dims2.json'), 'w'), indent=1)


def rname2(space, t):
    return f'v15j_{space}_d6_{t}e_fe15h6'


def prep2():
    """vcm decks (frozen [Ne], 3s 3p active, 15h Fe2+ orbitals fe15h6) for the core-polarization spaces."""
    import re
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    for space in CP_SPACES:
        cf = configs2(space)
        for L, twoS, m, t in D.term_list(6):
            src = os.path.join(ROOT, 'inputs', f'vcm_d6_{t}e')
            deck = open(os.path.join(src, 'be.deck')).read()
            csf = open(os.path.join(src, 'be.csf')).read()
            assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
            deck = deck.replace('z=23,', 'z=26,').replace('FILE=v_qzp.bas', 'FILE=fe.bas')
            header = csf.split(' ))\n', 1)[0] + ' ))\n'
            seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
            tot, ncfg, per = CN.count(E15.LMAP, cf, t)
            assert per[cf[0]] == m, (space, t)
            order = [c for c in cf if c in per]
            lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c))
                     + '  ))' for c in order]
            deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
            dst = rname2(space, t)
            deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
            out = os.path.join(ROOT, 'inputs', dst)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, 'be.deck'), 'w').write(deck)
            open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
    print('decks written')


def run2(spaces):
    import phase13d as D
    H.F.run_list([(rname2(sp, t), 'fe15h6') for sp in spaces for L, twoS, m, t in D.term_list(6)])


def heff_levels(b):
    """{level: H_eff (m x m, canonical seniority basis)} for one term block: exact (des Cloizeaux), Sylvester (O(V^2),
    denominators dressed to all orders by H_PP and H_QQ), closure (one constant denominator per Q configuration)."""
    if 'op' in b:
        from atomci_heff import iterative as IT
        return IT.heff_levels(b)
    H_, P, Q, U = b['H'], b['P'], b['Q'], b['U']
    Hpp, Hqq, V = H_[np.ix_(P, P)], H_[np.ix_(Q, Q)], H_[np.ix_(P, Q)]
    ex, inf = F13.heff_block(b, H_)
    p, A = np.linalg.eigh(Hpp)
    q, Bq = np.linalg.eigh(Hqq)
    X = Bq @ ((Bq.T @ V.T @ A) / (p[None, :] - q[:, None])) @ A.T     # X Hpp - Hqq X = V^T
    syl = Hpp + 0.5 * (V @ X + X.T @ V.T)
    clo = Hpp.copy()
    for c, dl in b['delta'].items():
        idx = [k for k, i in enumerate(Q) if b['lab'][i] == c]
        clo -= V[:, idx] @ V[:, idx].T / dl
    return {'exact': ex, 'sylvester': U.T @ syl @ U, 'closure': U.T @ clo @ U, 'php': U.T @ Hpp @ U}, inf


def analyze2(spaces=None, rn=None, tag='item2'):
    import phase13d as D
    spaces = spaces or list(CP_SPACES)
    rn = rn or rname2
    O1, O2 = H.o_axes()
    hf = {t: v[1] for t, v in path_blocks('dSD').items()}
    res, ens = {}, {}
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    for space in spaces:
        # two passes, one block in memory at a time (15k analyze3): pass 1 keeps the CSF diagonals only
        bl = []
        for L, twoS, m, t in D.term_list(6):
            name = rn(space, t)
            Hm, lab, dg = load_block(name)
            del Hm
            P = [i for i, l in enumerate(lab) if l == lab[0]]
            assert len(P) == m and P == list(range(m)), (name, P)
            bl.append(dict(N=6, t=t, L=L, twoS=twoS, lab=lab, diag=dg, name=name, U=np.array(maps[f'vcm_d6_{t}e']),
                           m=m, P=P, Q=[i for i in range(len(lab)) if i not in P]))
        # configuration averages of the CSF diagonals over all term blocks, weight (2L+1)(2S+1)
        acc = {}
        for b in bl:
            g = (2 * b['L'] + 1) * (b['twoS'] + 1)
            for i, lab in enumerate(b['lab']):
                s, w = acc.get(lab, (0.0, 0.0))
                acc[lab] = (s + g * b['diag'][i], w + g)
        eav = {k: s / w for k, (s, w) in acc.items()}
        p0 = bl[0]['lab'][0]
        for b in bl:
            b['delta'] = {c: eav[c] - eav[p0] for c in set(b['lab'][i] for i in b['Q'])}
        lev, info = {}, []
        for b in bl:
            b['op' if SOLVER == 'iter' else 'H'] = load_block(b['name'])[0]
            hb, inf = heff_levels(b)
            b.pop('op', None), b.pop('H', None)
            info.append(inf)
            # all-electron decks: same P block up to the frozen-core constant
            off = np.diag(hb['php'] - hf[b['t']])
            assert np.abs(hb['php'] - hf[b['t']] - off.mean() * np.eye(len(off))).max() < 1e-7, (space, b['t'])
            for k, v in hb.items():
                # HF (vcm P block) + (H_eff - PHP)
                lev.setdefault(k, {})[b['t']] = hf[b['t']] + v - hb['php'] if k != 'php' else hf[b['t']]
        r = dict(dims={b['t']: len(b['lab']) for b in bl}, wP=min(i['wP'] for i in info),
                 below=sum(i['below'] for i in info), dmin=min(eav[c] - eav[p0] for c in eav if c != p0))
        dump_levels(tag, space, lev)
        for k in ('exact', 'sylvester', 'closure'):
            by = {(b['L'], b['twoS']): lev[k][b['t']] for b in bl}
            c, dinf = H.decompose(6, B.vec_from_blocks(6, by))
            T1, T2, out = H.t1t2(c, O1, O2)
            E = {t: np.linalg.eigvalsh(lev[k][t]) for t in lev[k]}
            e0 = E['5D'][0]
            ens[f'{space} {k}'] = {t: [float((x - e0) * EH_CM) for x in E[t]] for t in E}
            r[k] = dict(proj_T1=1e3 * T1, proj_T2=1e3 * T2, out=out, ang=float(np.degrees(np.arctan2(T2, T1))))
        res[space] = r
        print(space, r['dims']['3F'], {k: (round(r[k]['proj_T1'], 4), round(r[k]['proj_T2'], 4))
                                       for k in ('exact', 'sylvester', 'closure')}, flush=True)
    json.dump(ens, open(os.path.join(OUT, '15j_terms', f'terms_{tag}.json'), 'w'), indent=1)
    fits(ens, res)
    json.dump(res, open(os.path.join(OUT, f'phase15j_{tag}.json'), 'w'), indent=1, default=float)


def fits(ens, res):
    T = os.path.join(OUT, '15i_trees')
    os.chdir(T)
    sys.path.insert(0, T)
    import fit_terms as FT
    for key, data in ens.items():
        space, k = key.split()
        for sel in ('16', '14'):
            d = data if sel == '16' else {t: (v[:1] if t in ('1S', '1D') else v) for t, v in data.items()}
            pb, eb, sb = FT.fit(d, FT.two + ['T1', 'T2'], FT.ops, [85000, 975, 3600, 69, -156, 2000, 700])
            res[space][k][sel] = dict(T1=pb[5] / 219.4746, T1_err=eb[5] / 219.4746, T2=pb[6] / 219.4746,
                                      T2_err=eb[6] / 219.4746, rms=sb, **{n: pb[i] for i, n in enumerate(FT.two)})
            f = res[space][k][sel]
            print(f'{space:6s} {k:9s} {sel}: T1 {f["T1"]:7.3f}+-{f["T1_err"]:.3f} T2 {f["T2"]:7.3f}+-{f["T2_err"]:.3f}'
                  f' B {f["B"]:7.1f} C {f["C"]:7.1f} al {f["alpha"]:6.1f} be {f["beta"]:7.1f} rms {sb:5.1f}', flush=True)


# ---------------------------------------------------------------- item 3.2: core path + relaxation / polarization

VP = CP['p'][1]
VS = ['11s', '12s', '13s', '14s', '15s', '16s', '17s']
C3 = {'cAp': [('3p', 'p')], 'cAs': [('3s', 's')], 'cA2p': [('2p', 'p')], 'cApsp': [('3p', 'p'), ('3s', 's'), ('2p', 'p')],
      'cFf': 'Pf', 'cFf7': 'Pf+'}


def configs3(space, nv=None):
    """All-electron configurations: c2s (P, 3s 3d7, 2s 3d7) plus the class (on top of 3s -> 3d)."""
    import phase15e as E15
    cf = list(E15.core_spaces(6)['c2s'])
    if space in ('cFf', 'cFf7'):
        for v in CP['f'][1]:
            cf.append(E15.full_cfg('all', {'3p': 5, '3d': 6, v: 1}))
            if space == 'cFf7':
                cf.append(E15.full_cfg('all', {'3s': 1, '3p': 5, '3d': 7, v: 1}))
        return cf
    for h, kind in C3[space]:
        vs = (VP if kind == 'p' else VS)[:nv]
        for v in vs:
            if h == '3s':
                cf.append(E15.full_cfg('all', {'3s': 0, '3d': 7, v: 1}))
            elif h == '3p':
                cf.append(E15.full_cfg('all', {'3s': 1, '3p': 5, '3d': 7, v: 1}))
            else:
                cf.append(E15.full_cfg('all', {'2p': 5, '3s': 1, '3d': 7, v: 1}))
    return cf


def dims3():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for sp in C3:
        for nv in ((None, 3, 2) if sp in ('cAp', 'cA2p', 'cApsp') else (None,)):
            cf = configs3(sp, nv)
            per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(6)}
            est = sum(8e-7 * d ** 2.4 * 3 for d in per.values())
            hi = sum(8e-7 * d ** 2.9 * 3 / 1107 ** 0.5 for d in per.values())
            key = sp + ('' if nv is None else f'_{nv}')
            res[key] = dict(per=per, max=max(per.values()), sum=sum(per.values()), est_s=est, est_hi_s=hi,
                            ncfg=len(cf), max_est_s=8e-7 * max(per.values()) ** 2.4 * 3)
            print(f'{key:9s} largest {max(per.values()):6d} ({max(per, key=per.get)}) sum {sum(per.values()):7d} '
                  f'configs {len(cf):3d} est {est:8.0f} s (hi {hi:8.0f} s; largest job {res[key]["max_est_s"]:.0f} s)')
    json.dump(res, open(os.path.join(OUT, 'phase15j_dims3.json'), 'w'), indent=1)


SP3 = ['cAp', 'cAs', 'cA2p', 'cApsp', 'cFf', 'cFf7']


def rname3(space, t):
    return f'v15j_{space}_d6_{t}e_fe15h6'


def prep3():
    """All-electron decks (15e allcore_deck, Fe, fe.bas) for the item 3.2 spaces."""
    import re
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    for space in SP3:
        cf = configs3(space)
        for L, twoS, m, t in D.term_list(6):
            deck, csf = E15.allcore_deck(6, t, 26, 'fe.bas')
            header = csf.split(' ))\n', 1)[0] + ' ))\n'
            seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
            tot, ncfg, per = CN.count(E15.LMAP, cf, t)
            assert per[cf[0]] == m, (space, t)
            order = [c for c in cf if c in per]
            missing = [o for c in order for o in dict(c) if o not in seq]
            assert not missing, (space, missing)
            lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c))
                     + '  ))' for c in order]
            deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
            deck = gen_v.sub_all(deck, 'nstate', m)
            nob, mpob = [], []
            for let in 'spdfgh':
                orbs = [o for o in seq if o.endswith(let)]
                nob.append(len(orbs))
                mpob += [gen_v.vec(o) for o in orbs]
                assert sorted(gen_v.vec(o) for o in orbs) == list(range(1, len(orbs) + 1)), (let, orbs)
            deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
            deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
            deck = gen_v.sub_all(deck, 'nind', len(nob))
            deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{k}' for l, k in enumerate(nob)))
            dst = rname3(space, t)
            deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
            out = os.path.join(ROOT, 'inputs', dst)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, 'be.deck'), 'w').write(deck)
            open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
    print('decks written')


def run3(spaces):
    import phase13d as D
    H.F.run_list([(rname3(sp, t), 'fe15h6') for sp in spaces for L, twoS, m, t in D.term_list(6)])


# ---------------------------------------------------------------- valence pairs (P + 3d4 a b) and core + pairs

LV = 'spdfgh'
VIRT = {'s': VS, 'p': VP, 'd': CP['d'][1], 'f': CP['f'][1], 'g': CP['g'][1], 'h': CP['h'][1]}
PTYPES = ['ss', 'pp', 'dd', 'ff', 'gg', 'hh', 'sd', 'sg', 'dg', 'pf', 'ph', 'fh']


def pair_cfgs(nper, types=None, pick='order'):
    """[valence dicts] of 3d4 a b (a <= b, l_a + l_b even), nper virtuals per l: the first in orbital order
    (pick 'order') or the most compact (pick 'compact', smallest <r>)."""
    import itertools as it
    allv = [(o, l) for l in LV for o in (VIRT[l][:nper] if pick == 'order' or nper is None else compact(l, nper))]
    out = []
    for (a, la), (b, lb) in it.combinations_with_replacement(allv, 2):
        t = ''.join(sorted(la + lb, key=LV.index))
        if t not in PTYPES or (types and t not in types):
            continue
        d = {'3d': 4}
        d[a] = d.get(a, 0) + 1
        d[b] = d.get(b, 0) + 1
        out.append(d)
    return out


def spaces4():
    """{name: (kind, [frozenset configs], P first)}."""
    import phase15e as E15
    sp = {}
    P0 = E15.full_cfg('vcm', {'3d': 6})
    for nper in (1, 2):
        for ty in [None] + PTYPES:
            name = f'pr{nper}_{ty or "all"}'
            sp[name] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in pair_cfgs(nper, [ty] if ty else None)])
    for ty in (['ff', 'pf', 'pp'], ['ff'], ['pf'], ['pp'], ['dg'], ['sd'], ['dd'], ['ss']):
        sp['prA_' + ''.join(ty)] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in pair_cfgs(None, ty)])
    c2sF = list(E15.core_spaces(6)['c2s'])
    sp['cprFP'] = ('allcore', c2sF + [E15.full_cfg('all', d) for d in pair_cfgs(None, ['ff', 'pf', 'pp'])])
    c2s = list(E15.core_spaces(6)['c2s'])
    Pp = [E15.full_cfg('all', d) for d in pair_cfgs(1)]
    Qp = [E15.full_cfg('all', {**d, '3s': 1, '3d': 5}) for d in pair_cfgs(1)]
    sp['cpr1P'] = ('allcore', c2s + Pp)
    sp['cpr1PQ'] = ('allcore', c2s + Pp + Qp)
    return sp


def dims4():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for name, (kind, cf) in spaces4().items():
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(6)}
        mx = max(per.values())
        res[name] = dict(kind=kind, ncfg=len(cf), per=per, max=mx, sum=sum(per.values()),
                         est_s=sum(8e-7 * d ** 2.4 * 3 for d in per.values()),
                         est_hi_s=sum(8e-7 * d ** 2.9 * 3 / 1107 ** 0.5 for d in per.values()),
                         maxjob_s=8e-7 * mx ** 2.4 * 3, maxjob_hi_s=8e-7 * mx ** 2.9 * 3 / 1107 ** 0.5)
        r = res[name]
        print(f'{name:10s} cfg {len(cf):4d} largest {mx:6d} sum {r["sum"]:7d} est {r["est_s"] / 60:7.1f} min '
              f'(hi {r["est_hi_s"] / 60:7.1f}) largest job {r["maxjob_s"] / 60:6.1f} min (hi {r["maxjob_hi_s"] / 60:6.1f})')
    json.dump(res, open(os.path.join(OUT, 'phase15j_dims4.json'), 'w'), indent=1)


def wrap80(shells):
    """One configuration line '  a* n  )  b* m  ))', continued on the next line at a shell boundary when longer than 80
    columns (ATOMCI reads 80-column cards, readia %lnmax 80; the token reader continues on the next card)."""
    out, cur = [], ''
    for i, sh in enumerate(shells):
        piece = sh + ('  ))' if i == len(shells) - 1 else '  )')
        if cur and len(cur) + len(piece) > 80:
            out.append(cur)
            cur = ' ' + piece
        else:
            cur += piece
    out.append(cur)
    assert all(len(x) <= 80 for x in out)
    return '\n'.join(out)


BAS6 = 'fe.bas'      # basis of the d6 decks; phase15n.prep sets it from phase15m.IONS (fend: fe.without-diffuse.bas)


def write_deck(dst, kind, cf, t):
    """Deck + CSF list for one term: 'vcm' (frozen [Ne], from inputs/vcm_d6_<t>e) or 'allcore' (15e allcore_deck)."""
    import re
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    m = next(mm for L, twoS, mm, tt in D.term_list(6) if tt == t)
    if kind == 'vcm':
        src = os.path.join(ROOT, 'inputs', f'vcm_d6_{t}e')
        deck = open(os.path.join(src, 'be.deck')).read()
        csf = open(os.path.join(src, 'be.csf')).read()
        assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
        deck = deck.replace('z=23,', 'z=26,').replace('FILE=v_qzp.bas', f'FILE={BAS6}')
    else:
        deck, csf = E15.allcore_deck(6, t, 26, BAS6)
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGHIK]', header.split(')')[0])]     # 30 Oct: i, k as write_deck3
    tot, ncfg, per = CN.count(E15.LMAP, cf, t)
    assert per[cf[0]] == m, (dst, t)
    order = [c for c in cf if c in per]
    assert not [o for c in order for o in dict(c) if o not in seq], dst
    lines = [wrap80(['  ' + f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)]) for c in order]
    deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
    if kind == 'allcore':
        deck = gen_v.sub_all(deck, 'nstate', m)
        nob, mpob = [], []
        for let in 'spdfghik'[:1 + max(5, max('spdfghik'.index(o[-1]) for o in seq))]:     # 30 Oct: i when the basis has it
            orbs = [o for o in seq if o.endswith(let)]
            nob.append(len(orbs))
            mpob += [gen_v.vec(o) for o in orbs]
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
        deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
        deck = gen_v.sub_all(deck, 'nind', len(nob))
        deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{k}' for l, k in enumerate(nob)))
    deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
    import phase15k as K
    if K.STOP_AFTER_H:     # Phase 17 Q2: d6 decks follow the d3 switch (ATOMCI's own diagonalization segfaults at N > 40k)
        deck = K.stop_after_h(deck)
    out = os.path.join(ROOT, 'inputs', dst)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'be.deck'), 'w').write(deck)
    open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')


def prep4(names):
    import phase13d as D
    sp = spaces4()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in D.term_list(6):
            write_deck(rname3(name, t), kind, cf, t)
    print('decks written:', ' '.join(names))


_RV = {}


def r_virt():
    """<r> (bohr) of every virtual orbital name (15h orbitals; column = gen_v.vec - 1)."""
    if not _RV:
        import gen_v
        bs = H.F.Bas('fe.bas', 26)
        z = np.load(os.path.join(OUT, 'orbsets_15h.npz'))
        for l, lst in VIRT.items():
            L = LV.index(l)
            for o in lst:
                P = bs.P(L, z[f'Fe|6|{L}'][:, gen_v.vec(o) - 1])
                _RV[o] = float(np.sum(P * P * bs.r * bs.w))
    return _RV


def compact(l, n):
    """The n most compact virtuals of symmetry l (smallest <r>)."""
    rv = r_virt()
    return sorted(VIRT[l], key=rv.get)[:n]


def cv_cfgs(families, side='P'):
    """3p 3d -> a b core-valence doubles. families: [(l_a, n_a, l_b, n_b)] with the n most compact virtuals of each l.
    side 'P': on 3s2 3p6 3d6 (3p5 3d5 a b); 'Q': on 3s 3p6 3d7 (3s 3p5 3d6 a b)."""
    import phase15e as E15
    out = []
    for la, na, lb, nb in families:
        for a in compact(la, na):
            for b in compact(lb, nb):
                d = {'3p': 5, '3d': 5, a: 1, b: 1} if side == 'P' else {'3s': 1, '3p': 5, '3d': 6, a: 1, b: 1}
                out.append(E15.full_cfg('all', d))
    return out


CV = {'cvFP': [('f', 4, 'd', 3), ('p', 3, 'd', 3)], 'cvF': [('f', 4, 'd', 3)], 'cvG': [('f', 4, 'g', 2)]}


def spaces5():
    import phase15e as E15
    sp = {name: ('allcore', configs3('cFf7') + cv_cfgs(fam, 'P')) for name, fam in CV.items()}
    P0 = E15.full_cfg('vcm', {'3d': 6})
    c2s = list(E15.core_spaces(6)['c2s'])
    Pc = pair_cfgs(2, pick='compact')
    sp['pr2c_all'] = ('vcm', [P0] + [E15.full_cfg('vcm', d) for d in Pc])
    sp['cpr2cP'] = ('allcore', c2s + [E15.full_cfg('all', d) for d in Pc])
    sp['cpr2cPQ'] = ('allcore', c2s + [E15.full_cfg('all', d) for d in Pc] +
                     [E15.full_cfg('all', {**d, '3s': 1, '3d': 5}) for d in Pc])
    for name, ng, lvl in (('g3S', 3, 1), ('g3SD', 3, 2), ('g2SD', 2, 2)):
        cf = [d for d in E15.fci(6, ['3d'] + E15.GV[:ng]) if 6 - d.get('3d', 0) <= lvl]
        sp[name] = ('vcm', [P0] + [c for c in (E15.full_cfg('vcm', d) for d in cf) if c != P0])
    return sp


def dims5():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for name, (kind, cf) in spaces5().items():
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(6)}
        res[name] = dict(kind=kind, ncfg=len(cf), per=per, max=max(per.values()), sum=sum(per.values()))
        print(name, res[name]['max'], res[name]['sum'])
    fp = os.path.join(OUT, 'phase15j_dims5.json')
    old = json.load(open(fp)) if os.path.exists(fp) else {}
    old.update(res)
    json.dump(old, open(fp, 'w'), indent=1)


def prep5(names):
    import phase13d as D
    sp = spaces5()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in D.term_list(6):
            write_deck(rname3(name, t), kind, cf, t)
    print('decks written:', ' '.join(names))


def run_names(names):
    """Run each job; a failure is logged (FAILED <name>) and skipped, the rest continue."""
    for n in names:
        try:
            H.F.run_list([(n, 'fe15h6')])
        except Exception as e:  # noqa: BLE001
            print(f'FAILED {n}: {e!r}', flush=True)


def launch(spaces, nproc, log):
    """Shell launcher: the run names of the spaces, balanced over nproc processes by estimated time (largest first,
    to the least loaded process); each process runs its list sequentially with one thread."""
    import phase13d as D
    dims = {}
    for f in ('phase15j_dims4.json', 'phase15j_dims5.json'):
        fp = os.path.join(OUT, f)
        if os.path.exists(fp):
            dims.update(json.load(open(fp)))
    jobs = [(dims[sp]['per'][t] ** 2.4, rname3(sp, t)) for sp in spaces for L, twoS, m, t in D.term_list(6)]
    jobs.sort(reverse=True)
    lists, load = [[] for _ in range(nproc)], [0.0] * nproc
    for w, n in jobs:
        i = load.index(min(load))
        lists[i].append(n)
        load[i] += w
    # run each list smallest first so finished blocks accumulate early
    lines = ['#!/bin/bash', f'cd {ROOT}',
             'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1', f'L={log}']
    for i, lst in enumerate(lists):
        lines.append(f'{sys.executable} {_C.script("phase15j.py")} run_names {" ".join(reversed(lst))} > $L.{i} 2>&1 &')
        lines.append(f'p{i}=$!')
    lines.append(' '.join(f'wait $p{i}; e{i}=$?;' for i in range(nproc)))
    lines.append(f'cat {" ".join(f"$L.{i}" for i in range(nproc))} > $L; rm -f {" ".join(f"$L.{i}" for i in range(nproc))}')
    lines.append('echo "exit ' + ' '.join(f'$e{i}' for i in range(nproc)) + '" >> $L')
    lines.append('echo done >> $L')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    if sys.argv[1] == 'dims5':
        dims5()
    elif sys.argv[1] == 'prep5':
        prep5(sys.argv[2:])
    elif sys.argv[1] == 'run_names':
        run_names(sys.argv[2:])
    elif sys.argv[1] == 'launch':
        sh, nproc, log = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        open(sh, 'w').write(launch(sys.argv[5:], nproc, log))
        os.chmod(sh, 0o755)
    elif sys.argv[1] == 'dims4':
        dims4()
    elif sys.argv[1] == 'prep4':
        prep4(sys.argv[2:])
    elif sys.argv[1] == 'analyze4':
        analyze2(sys.argv[3:], lambda sp, t: H.rname(sp, 6, t, 'fe15h6') if sp == 'c2s' else rname3(sp, t),
                 sys.argv[2])
    elif sys.argv[1] == 'analyze3':
        analyze2(['c2s'] + SP3, lambda sp, t: H.rname(sp, 6, t, 'fe15h6') if sp == 'c2s' else rname3(sp, t), 'item3')
    elif sys.argv[1] == 'prep3':
        prep3()
    elif sys.argv[1] == 'run3':
        run3(sys.argv[2:])
    elif sys.argv[1] == 'dims3':
        dims3()
    elif sys.argv[1] == 'analyze2':
        analyze2()
    elif sys.argv[1] == 'dims2':
        dims2()
    elif sys.argv[1] == 'prep2':
        prep2()
    elif sys.argv[1] == 'run2':
        run2(sys.argv[2:])
    if sys.argv[1] == 'terms':
        terms()
    elif sys.argv[1] == 'fit1':
        fit1()
    elif sys.argv[1] == 'check0':
        check0()
