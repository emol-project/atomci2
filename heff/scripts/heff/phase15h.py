#!/usr/bin/env python3
"""Phase 15h: Trees' (1963) Fe III 3d6 three-body effect (3s -> 3d) vs our T1.

Trees: the 3s2 3d6 - 3s 3d7 interaction gives a three-body term with gamma = H2^2 / dE, H2 = R2(3d3d;3d3s)/35,
dE = E_av(3s 3d7) - E_av(3s2 3d6). Our T1 (projection on Uylings' O2', 15c normalization) = c * gamma.
Step 1: Fe2+ 3d6 configuration-average HF (own orbitals, fe.bas = Sapporo-QZP-2012-diffuse, contracted); E, 3d <r>,
        F2, F4; R2(3d3d;3d3s), H2, dE (same orbitals), gamma_HF.
Step 2: c = T1 of the 3s -> 3d second-order closure with H2 = 1 (R2 = 35) and dE = 1 (15c closure_t, k = 2), in d3,
        d4 and d6 (d6: Gram decomposition on e0..e4, t1..t4 lifted to d6 and the JL d4 f1..f12 lifted to d6); check
        against the 15e core-path T1 and each ion's own gamma; particle-hole check of the lifted t (d4 vs d6).
Step 3: paths in d6 as 15e (core 3s -> 3d, + 2s -> 3d; s path; d path, all virtual d; g path, 2 virtual g).
usage: phase15h.py step1 | step2 | dims | prep | run ... | analyze
-> results/phase15/phase15h_*.json, phase15h.txt
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff.reader import read_vectors  # noqa: E402
import phase13b as B  # noqa: E402
import phase14a_orb as O  # noqa: E402
import phase15f as F  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
EH_CM = 219474.6313705
RY_CM = 109737.31568160
TREES = dict(R2_Ry=0.80838, H2_cm=2535.0, dE_cm=783600.0, gamma_exp=5.0, gamma_exp_err=0.3, gamma_HF=8.2)
F.QP[6] = (6, 15)
FE = (26, 'fe.bas')
# 15e ions with their own orbital sets (runs/vq_orb_<tag>/be.NO) for the c check
CHECK = [('V2+', 3, 23, 'v_qzp.bas', 'o14n3'), ('Cr3+', 3, 24, 'cr.bas', 'cr15n3'), ('Mn4+', 3, 25, 'mn.bas', 'mn15b3'),
         ('Cr2+', 4, 24, 'cr.bas', 'cr15n4')]


# ---------------------------------------------------------------- step 1

def eav(bs, C, q3s, qd):
    """Configuration-average energy of 1s2 2s2 2p6 3s^q3s 3p6 3d^qd with fixed orbitals (O.energy generalized)."""
    sh = [(nm, l, bs.P(l, C[l][:, i]), (q3s if nm == '3s' else 4 * l + 2), C[l][:, i]) for nm, l, i in O.CLOSED]
    sh.append(('3d', 2, bs.P(2, C[2][:, 0]), qd, C[2][:, 0]))
    E = 0.0
    for i, (na, la, Pa, qa, ca) in enumerate(sh):
        E += qa * ca @ bs.h[la] @ ca
        E += qa * (qa - 1) / 2 * (bs.R(Pa, Pa, Pa, Pa, 0) - (2 * la + 1) / (4 * la + 1) *
                                  sum(O.w3(la, k, la) * bs.R(Pa, Pa, Pa, Pa, k) for k in range(2, 2 * la + 1, 2)))
        for nb, lb, Pb, qb, cb in sh[i + 1:]:
            E += qa * qb * (bs.R(Pa, Pa, Pb, Pb, 0) -
                            0.5 * sum(O.w3(la, k, lb) * bs.R(Pa, Pb, Pa, Pb, k) for k in O.exch_ks(la, lb)))
    return E


def gamma_of(bs, C, N):
    """R2(3d3d;3d3s) = int int P3d(1) P3d(2) r<^2/r>^3 P3d(1) P3s(2), H2 = R2/35, dE, gamma = H2^2/dE (Eh)."""
    d, s = bs.P(2, C[2][:, 0]), bs.P(0, C[0][:, 2])
    R2 = bs.R(d, d, d, s, 2)
    E0, E1 = eav(bs, C, 2, N), eav(bs, C, 1, N + 1)
    H2 = R2 / 35
    return dict(R2=float(R2), H2=float(H2), E_av_3s2=float(E0), E_av_3s=float(E1), dE=float(E1 - E0),
                gamma=float(H2 ** 2 / (E1 - E0)))


def step1():
    t0 = time.time()
    Z, bas = FE
    bs = F.Bas(bas, Z)
    C, inf = F.own_set(bs, 6)
    assert abs(eav(bs, C, 2, 6) - inf['E']) < 1e-8, (eav(bs, C, 2, 6), inf['E'])
    g = gamma_of(bs, C, 6)
    js = dict(Fe2p=dict(inf, **g, nbf=bs.nbf))
    np.savez(os.path.join(OUT, 'orbsets_15h.npz'), **{f'Fe|6|{l}': C[l] for l in C})
    # gamma of the 15e ions with their own orbitals (for the c check)
    for name, N, Zc, basc, tag in CHECK:
        bsc = F.Bas(basc, Zc)
        Cc = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO'), bsc.nbf)[0]
        js[name] = dict(N=N, tag=tag, **gamma_of(bsc, Cc, N))
    json.dump(js, open(os.path.join(OUT, 'phase15h_step1.json'), 'w'), indent=1)
    x = js['Fe2p']
    print(f'Fe2+ d6: E {x["E"]:.8f} <r> {x["r"]:.6f} F2 {x["F2"]:.6f} F4 {x["F4"]:.6f} Eh ({x["it"]} it, orth '
          f'{x["orth_dev"]:.1e})')
    print(f'  R2 {x["R2"]:.6f} Eh = {2 * x["R2"]:.5f} Ry = {x["R2"] * EH_CM:.0f} cm-1; H2 {x["H2"] * EH_CM:.1f} cm-1; '
          f'dE {x["dE"]:.6f} Eh = {x["dE"] * EH_CM:.0f} cm-1; gamma_HF {x["gamma"] * EH_CM:.3f} cm-1')
    for name, *_ in CHECK:
        y = js[name]
        print(f'  {name}: R2 {y["R2"]:.6f} Eh, dE {y["dE"]:.6f} Eh, gamma {y["gamma"] * EH_CM:.3f} cm-1')
    print(f'{time.time() - t0:.0f} s')


# ---------------------------------------------------------------- step 2

def o_axes():
    import phase15e as E15
    c15 = json.load(open(os.path.join(OUT, 'phase15c.json')))
    O1 = E15.d3(dict(zip(B.TK, c15['O1'])))
    O2 = E15.d3(dict(zip(B.TK, c15['O2'])))
    return O1 / E15.nrm3(O1), O2 / E15.nrm3(O2)


_B6 = None


def basis6():
    """e0..e4, t1..t4 lifted to d6 and the JL d4 f1..f12 (pure 4-body on d4) lifted to d6."""
    global _B6
    if _B6 is None:
        ops, _, _ = B.ops_fock()
        names, vecs = [], []
        for k in B.EK + B.TK:
            M, n = ops[k]
            names.append(k)
            vecs.append(S.to_vec(6, S.lift(M, n, 6)))
        for k, v in S.jl_table(4).items():
            if k.startswith('f'):
                names.append(k)
                vecs.append(S.to_vec(6, S.lift(S.from_vec(4, v), 4, 6)))
        _B6 = names, vecs
    return _B6


def decompose(N, x):
    """Coefficients of the d^N component vector x on the operator basis (Gram solve); d6 via basis6."""
    import phase14b as P14
    names, vecs = basis6() if N == 6 else P14.basis_ops(N)
    G = np.array([[S.ip(N, p, q) for q in vecs] for p in vecs])
    c = np.linalg.solve(G, [S.ip(N, p, x) for p in vecs])
    r = x - sum(ci * v for ci, v in zip(c, vecs))
    return dict(zip(names, c)), dict(rank=int(np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max())), n=len(vecs),
                                     dim=len(x), resid=float(np.sqrt(abs(S.ip(N, r, r)))),
                                     cond=float(np.linalg.cond(G)))


def t1t2(c, O1, O2):
    import phase15e as E15
    tt = E15.d3(c)
    T1, T2 = F.S_ip(tt, O1), F.S_ip(tt, O2)
    return T1, T2, E15.nrm3(tt - T1 * O1 - T2 * O2) / max(E15.nrm3(tt), 1e-300)


def step2():
    import phase15c as Q
    O1, O2 = o_axes()
    js = {}
    for N in (3, 4, 6):
        X = Q.closure_t(N, 0, True, [2])[(2, 2)] * 35 ** 2      # H2 = 1 (R2 = 35), dE = 1
        c, inf = decompose(N, S.to_vec(N, X))
        T1, T2, out = t1t2(c, O1, O2)
        js[f'd{N}'] = dict(c=T1, T2=T2, out=out, t={k: float(c[k]) for k in B.TK}, **inf)
        print(f'd{N}: c = T1/gamma = {T1:.6f} (T2 {T2:.2e}, out {out:.1e}); basis rank {inf["rank"]}/{inf["n"]}, '
              f'dim {inf["dim"]}, resid {inf["resid"]:.1e}, cond {inf["cond"]:.1e}', flush=True)
    # particle-hole: diagonal elements of the lifted t_i, d4 vs d6 (13b: diag(10 - N) = -diag(N))
    ops, _, _ = B.ops_fock()
    ph = {}
    for k in B.TK:
        x4, x6 = S.to_vec(4, S.lift(*ops[k], 4)), S.to_vec(6, S.lift(*ops[k], 6))
        d6 = {S.comp_label(6, key): y for (key, _), y in zip(S.components(6), x6) if key[2] == key[3]}
        pr = [(x, d6[S.comp_label(4, key)]) for (key, _), x in zip(S.components(4), x4) if key[2] == key[3]]
        ph[k] = dict(sum=max(abs(x + y) for x, y in pr), diff=max(abs(x - y) for x, y in pr),
                     mx=max(abs(x) for x, _ in pr))
        print(f'  {k}: max|diag(4) + diag(6)| {ph[k]["sum"]:.1e}, max|diag(4) - diag(6)| {ph[k]["diff"]:.3g} '
              f'(max|diag| {ph[k]["mx"]:.3g})')
    js['particle_hole'] = ph
    # the d6 closure read as a d4-hole operator: its d6 diagonal components decomposed in d4
    X6 = S.to_vec(6, Q.closure_t(6, 0, True, [2])[(2, 2)] * 35 ** 2)
    d6 = {S.comp_label(6, key): (key, y) for (key, _), y in zip(S.components(6), X6)}
    x4 = np.array([d6[S.comp_label(4, key)][1] for key, _ in S.components(4)])
    c4, _ = decompose(4, x4)
    T1h, _, _ = t1t2(c4, O1, O2)
    js['d6_as_d4'] = dict(T1=T1h)
    print(f'  d6 closure read as d4 components: T1 {T1h:.6f} (vs d6 decomposition {js["d6"]["c"]:.6f})')
    # 15e core-path T1 vs own gamma
    s1 = json.load(open(os.path.join(OUT, 'phase15h_step1.json')))
    e15 = json.load(open(os.path.join(OUT, 'phase15e.json')))
    ions = {'V2+': '3 V', 'Cr3+': '3 Cr', 'Mn4+': '3 Mn', 'Cr2+': '4 Cr'}
    js['check'] = {}
    for name, key in ions.items():
        g = s1[name]['gamma']
        T1c = e15[key]['c3s']['T1']
        cN = js[f'd{s1[name]["N"]}']['c']
        js['check'][name] = dict(T1_core=T1c, gamma=g, gamma_eff=T1c / cN, ratio=T1c / (cN * g))
        print(f'  {name}: 15e core 3s->3d T1 {1e3 * T1c:+.4f} mEh, own gamma {g * EH_CM:.3f} cm-1, '
              f'gamma_eff {T1c / cN * EH_CM:.3f} cm-1, ratio {T1c / (cN * g):.3f}')
    json.dump(js, open(os.path.join(OUT, 'phase15h_step2.json'), 'w'), indent=1, default=float)


# ---------------------------------------------------------------- step 3

# (ion label, N, Z, basis, orbital tag, spaces). Fe2+: the paths; Cr2+ d4 (15e own orbitals): validation of the
# excitation-limited d path against the 15e full d5 (v15e_d5_d4_*_cr15n4).
RUNS = [('Fe2+', 6, 26, 'fe.bas', 'fe15h6', ['c3s', 'c2s', 's7', 'dSD', 'g2']),
        ('Cr2+', 4, 24, 'cr.bas', 'cr15n4', ['dSD', 'dSDT'])]


def rname(space, N, t, tag):
    return f'v15h_{space}_d{N}_{t}e_{tag}'


def configs(N, space):
    """('vcm' | 'allcore', [frozenset configs]), P configuration first (15e definitions)."""
    import phase15e as E15
    P0 = E15.full_cfg('vcm', {'3d': N})
    if space in ('c3s', 'c2s'):
        return 'allcore', E15.core_spaces(N)[space]
    if space.startswith('dSD'):
        lvl = 3 if space == 'dSDT' else 2
        cf = [d for d in E15.fci(N, ['3d'] + E15.DV[:5]) if N - d.get('3d', 0) <= lvl]
    else:
        cf = E15.spaces(N)[space][1]
    cfs = [E15.full_cfg('vcm', d) for d in cf]
    return 'vcm', [P0] + [c for c in cfs if c != P0]


def prep():
    import re
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    z = np.load(os.path.join(OUT, 'orbsets_15h.npz'))
    F.write_be_no({l: z[f'Fe|6|{l}'] for l in range(6)}, 'fe15h6', F.nbf_of('fe.bas'))
    dims = {}
    for ion, N, Z, bas, tag, sps in RUNS:
        for space in sps:
            kind, cf = configs(N, space)
            for L, twoS, m, t in D.term_list(N):
                if kind == 'vcm':
                    src = os.path.join(ROOT, 'inputs', f'vcm_d{N}_{t}e')
                    deck = open(os.path.join(src, 'be.deck')).read()
                    csf = open(os.path.join(src, 'be.csf')).read()
                    assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
                    deck = deck.replace('z=23,', f'z={Z},').replace('FILE=v_qzp.bas', f'FILE={bas}')
                else:
                    deck, csf = E15.allcore_deck(N, t, Z, bas)
                header = csf.split(' ))\n', 1)[0] + ' ))\n'
                seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
                tot, ncfg, per = CN.count(E15.LMAP, cf, t)
                assert per[cf[0]] == m, (space, t)
                dims[f'{ion} {space} {t}'] = tot
                order = [c for c in cf if c in per]
                lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c))
                         + '  ))' for c in order]
                deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
                if kind == 'allcore':
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
                dst = rname(space, N, t, tag)
                deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
                out = os.path.join(ROOT, 'inputs', dst)
                os.makedirs(out, exist_ok=True)
                open(os.path.join(out, 'be.deck'), 'w').write(deck)
                open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
    json.dump(dims, open(os.path.join(OUT, 'phase15h_dims.json'), 'w'), indent=1)
    print(f'{len(dims)} decks; largest {max(dims.values())} ({max(dims, key=dims.get)})')


def run(ion, spaces):
    import phase13d as D
    ion_, N, Z, bas, tag, _ = next(r for r in RUNS if r[0] == ion)
    F.run_list([(rname(sp, N, t, tag), tag) for sp in spaces for L, twoS, m, t in D.term_list(N)])


def blocks(N, names):
    import phase13d as D
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    out = []
    for (L, twoS, m, t), name in zip(D.term_list(N), names):
        Hm, lab = B.load_run(name)
        P = [i for i, l in enumerate(lab) if l == lab[0]]
        assert len(P) == m and P == list(range(m)), (name, P)
        out.append(dict(N=N, t=t, L=L, twoS=twoS, H=Hm, lab=lab, U=np.array(maps[f'vcm_d{N}_{t}e']), m=m, P=P,
                        Q=[i for i in range(len(lab)) if i not in P]))
    return out


def analyze_space(N, names, O1, O2):
    import phase13f as F13
    bl = blocks(N, names)
    by, info = {}, []
    for b in bl:
        Bk, inf = F13.heff_block(b, b['H'])
        by[(b['L'], b['twoS'])] = Bk
        info.append(inf)
    c, dinf = decompose(N, B.vec_from_blocks(N, by))
    T1, T2, out = t1t2(c, O1, O2)
    return dict(T1=T1, T2=T2, out=out, ang=float(np.degrees(np.arctan2(T2, T1))), wP=min(i['wP'] for i in info),
                below=sum(i['below'] for i in info), dims={b['t']: len(b['lab']) for b in bl}, resid=dinf['resid'],
                t={k: float(c[k]) for k in B.TK})


def analyze():
    import phase13d as D
    t0 = time.time()
    O1, O2 = o_axes()
    c = json.load(open(os.path.join(OUT, 'phase15h_step2.json')))['d6']['c']
    res = {}
    for ion, N, Z, bas, tag, sps in RUNS:
        for space in sps:
            names = [rname(space, N, t, tag) for L, twoS, m, t in D.term_list(N)]
            res[f'{ion} {space}'] = analyze_space(N, names, O1, O2)
            print(f'{ion} {space}: T1 {1e3 * res[f"{ion} {space}"]["T1"]:+.4f} T2 {1e3 * res[f"{ion} {space}"]["T2"]:+.4f}'
                  f' mEh ({time.time() - t0:.0f} s)', flush=True)
    names = [f'v15e_d5_d4_{t}e_cr15n4' for L, twoS, m, t in D.term_list(4)]
    res['Cr2+ d5full'] = analyze_space(4, names, O1, O2)
    for k, v in res.items():
        if k.startswith('Fe2+'):
            v['gamma_eff_cm'] = v['T1'] / c * EH_CM
    json.dump(res, open(os.path.join(OUT, 'phase15h.json'), 'w'), indent=1, default=float)
    print(f'analysis {time.time() - t0:.0f} s')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'step1':
        step1()
    elif cmd == 'step2':
        step2()
    elif cmd == 'prep':
        prep()
    elif cmd == 'run':
        run(sys.argv[2], sys.argv[3:])
    elif cmd == 'analyze':
        analyze()
