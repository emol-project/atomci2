#!/usr/bin/env python3
"""Phase 15t: basis sensitivity of Cr3+ T1 to single added f, g, h primitives (plain Sapporo-QZP-2012 + one or two).

Ions (phase15m.IONS, env ION): crnd (standard plain set, tag crnd3), crtfm (+f 2.75), crtft (+f 14.0), crtfb (+f 2.75
and 14.0), crtg (+g 4.7), crth (+h 4.4). Basis files basis/cr_qzp2012_add_*.bas (cr.without-diffuse.bas plus the
uncontracted primitives; no existing file touched). Orbitals: the 15a recipe (phase15f.own_set): occupied s, p, d from
the same config-average HF (the added l are unoccupied, so s, p, d are unchanged; checked against crnd3), f, g, h =
bare-nucleus h eigenvectors in the extended basis. Virtual lists (phase15j.VIRT) follow the basis size.
Spaces (15n RAS rules, ras_cfgs_x default limits; s, p, d at k = 1, most compact):
  k1f     all f, g k = 1            (step 1 screen; the 15n rasK1 with every f)
  k1f0    no f, g k = 1
  k1fg    all f, all g
  k1fh    all f, g k = 1, all h     (RAS k = 1 has no h: h enters only here)
  best    all virtual d, all f, g k = 2, s p 1   (= 15s rasDallG2_fall; step 2)
  bestg   all virtual d, all f, all g, s p 1
  besth   all virtual d, all f, g k = 2, all h, s p 1
  besth_h3, besth_h3v1, besth_h3v0, k1fh_h3   3s3p hole limit 3 (v<k>: 3-hole configurations with <= k virtual e)
Run names v15t_<space>_d3_<t>e_<tag>.
Commands: bases / orb / vcm / dims <spaces> / prep <spaces> / run_names <names> / analyze <tag> <space>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
ROOT = _C.root()  # HEFF_WORK
OUT = os.path.join(ROOT, 'results', 'phase15')
BASE = 'cr.without-diffuse.bas'
ADD = {'crtfm': [('F', 2.75)], 'crtft': [('F', 14.0)], 'crtfb': [('F', 14.0), ('F', 2.75)],
       'crtg': [('G', 4.7)], 'crth': [('H', 4.4)]}
NEW_IONS = ['crnd', 'crtfm', 'crtft', 'crtfb', 'crtg', 'crth']


def bas_name(ion):
    return 'cr_qzp2012_add_' + '_'.join(f'{l.lower()}{e}' for l, e in sorted(ADD[ion], key=lambda x: x[1])) + '.bas'


def bases():
    """Each added primitive is inserted among the single primitives of its l in descending exponent order."""
    from atomci_heff import gto
    src = open(os.path.join(_C.BASIS, BASE)).read().splitlines()
    for ion, add in ADD.items():
        lines = list(src)
        lines[0] = (f'" Cr  Sapporo-QZP-2012 plain (cr.without-diffuse.bas) + uncontracted '
                    + ', '.join(f'{l} {e}' for l, e in add) + ' (Phase 15t)')
        for l, e in add:
            idx = [k for k, x in enumerate(lines) if x.split()[:2] == [l, '1']]
            pos = next((k for k in idx if float(lines[k].split()[3]) < e), None)
            if pos is None:        # tighter than none / l absent: after the last function of l or before the next l
                pos = idx[-1] + 1
            lines.insert(pos, f'  {l}  1  1.0  {e:14.7f}  1.0000000')
        p = os.path.join(_C.BASIS, bas_name(ion))
        assert not os.path.exists(p), p
        open(p, 'w').write('\n'.join(lines) + '\n')
        a = gto.read_bas(os.path.join(_C.BASIS, BASE))
        b = gto.read_bas(p)
        print(ion, bas_name(ion), len(a), '->', len(b), 'functions')


def orb():
    import phase15f as F
    import phase15m as M
    from atomci_heff.reader import read_vectors
    js = {}
    ref = None
    for ion in NEW_IONS:
        N, Z, bas, tg = M.IONS[ion]
        bs = F.Bas(bas, Z)
        C, inf = F.own_set(bs, N)
        rv, mineig = {}, {}
        for l, let in enumerate('spdfgh'[:len(bs.nbf)]):
            w = np.linalg.eigvalsh(bs.S[l])
            mineig[let] = float(w.min())
            first = {0: 3, 1: 2, 2: 1}.get(l, 0)
            rv[let] = [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w)) for k in range(first, bs.nbf[l])]
        inf.update(nbf=bs.nbf, r_virt=rv, S_min_eig=mineig)
        if ion == 'crnd':
            R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tg}', 'be.NO'), bs.nbf)[0]
            inf['max_dev_vs_existing'] = max(float(np.abs(C[l] * np.sign(np.sum(C[l] * R[l], 0)) - R[l]).max())
                                             for l in C)
            ref = (C, inf['E'])
        else:
            inf['spd_max_dev_vs_crnd3'] = max(float(np.abs(C[l] - ref[0][l]).max()) for l in range(3))
            inf['dE_vs_crnd3'] = inf['E'] - ref[1]
            F.write_be_no(C, tg, bs.nbf)
        js[ion] = inf
        print(ion, tg, bs.nbf, 'E', f'{inf["E"]:.8f}', {k: v for k, v in inf.items() if 'dev' in k or k == 'dE_vs_crnd3'},
              'S_min', {k: f'{v:.2e}' for k, v in mineig.items()},
              {k: [round(x, 3) for x in v] for k, v in rv.items() if k in 'fgh'}, flush=True)
    json.dump(js, open(os.path.join(OUT, 'phase15t_orb.json'), 'w'), indent=1)


def vcm():
    import phase13d as D
    import phase15m as M
    import phase15s_nd as ND
    for ion in NEW_IONS[1:]:
        N, Z, bas, tg = M.IONS[ion]
        for L, twoS, m, t in D.term_list(3):
            dst, deck, csf = ND.vcm_deck(t, bas, tg)
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
        print(ion, 'vcm decks for', tg)


# ---------------------------------------------------------------- ion-dependent setup (spaces, virtual lists)

if os.environ.get('ION', 'cr') in NEW_IONS + ['vnd', 'mnnd', 'fend', 'crti1', 'crti2', 'crti3', 'crti4', 'crti5', 'crtk1', 'crop', 'crdk', 'crt2', 'crt3', 'crt2g',
                                                                 'vop', 'vdk', 'mnop', 'mndk', 'feop', 'fedk',
                                                                 'vti1', 'vti5', 'mnti1', 'mnti5', 'feti1', 'feti5',
                                                                 'fet2', 'crt2i', 'vqd', 'mnqd', 'feqd', 'cr4f', 'crno',
                                                                 'fe5nd', 'fe5ti1', 'fe5ti5', 'fe5op', 'fe5dk', 'fe5t2',
                                                                 'cond', 'coti1', 'coti5', 'coop', 'codk', 'cot2', 'coqd', 'fedc', 'codc',
    'fe5dc', 'fe5td', 'fe5qd', 'crdc', 'crtd', 'crqd']:   # 15u; 17 Q3 (i), Q4; paper 3-5, 10; test 1
    import gen_v
    import phase15s as S
    import phase15j as J
    Q = S.Q
    N15 = S.N15
    _nbf = Q.K.H.F.Bas(Q.M.BAS_, Q.M.Z_).nbf
    if len(_nbf) > 6:                       # 17 Q3: a basis with i functions
        J.VIRT.setdefault('i', [])
        J.LV = 'spdfghi'
    if len(_nbf) > 7:                       # 17 item 1c: a basis with k functions (l = 7)
        J.VIRT.setdefault('k', [])
        J.LV = 'spdfghik'
    for _l, _lst in J.VIRT.items():         # the virtuals this basis has (the 15s crnd trim generalised to extension)
        _L = 'spdfghik'.index(_l)
        _nv = _nbf[_L] - gen_v.CORE[_L] - (1 if _L == 2 else 0) if _L < len(_nbf) else 0
        _lst[:] = [f'{11 + k}{_l}' for k in range(_nv)]
        assert [gen_v.vec(o) for o in _lst] == list(range(_nbf[_L] - _nv + 1, _nbf[_L] + 1)) if _nv else True
    import phase17q4r as _Q4R
    if Q.ION in _Q4R.ELOF:                  # 17 Q4-R: the k = 1 picks are the crnd3 ones (phase17q4r docstring);
        import functools                     # item 1a: vnd3 / mnnd3 / fend6 for V, Mn, Fe
        Q.K.r_virt3 = functools.partial(_Q4R.r_virt_matched, _Q4R.ELOF[Q.ION])
    import phase15e as _E15
    for _l, _lst in J.VIRT.items():
        for _o in _lst:
            _E15.LMAP.setdefault(_o, 'spdfghik'.index(_l))     # 15f (crtfb) is new
            _E15.CAP.setdefault(_o, 2 * (2 * _E15.LMAP[_o] + 1))     # 17 Q4-R: 15d.. 20d etc. are new

    def _virt(f='all', g=1, h=0, d=1):
        def pick(l, k):
            return Q.K.compact3(l, len(J.VIRT[l]) if k == 'all' else k)     # compact first (15s order)
        return pick('s', 1) + pick('p', 1) + pick('d', d) + pick('f', f) + pick('g', g) + pick('h', h)

    SPACES = {'k1f': dict(), 'k1f0': dict(f=0), 'k1fg': dict(g='all'), 'k1fh': dict(h='all'),
              'best': dict(d='all', g=2), 'bestg': dict(d='all', g='all'), 'besth': dict(d='all', g=2, h='all')}
    # 15t addendum 2: 3s3p hole limit 2 -> 3 (phase15s rasK1_h3 rule); _v<k>: configurations with 3 holes keep at
    # most k virtual electrons (all others as in the parent space)
    H3 = {'besth_h3': ('besth', 2), 'besth_h3v1': ('besth', 1), 'besth_h3v0': ('besth', 0), 'k1fh_h3': ('k1fh', 2)}
    _sp0 = S.spaces

    def h3_cfgs(parent, v3):
        virt = _virt(**SPACES[parent])
        base = S.ras_cfgs_x(virt)
        out = []
        for c in S.ras_cfgs_x(virt, maxh=3):
            d = dict(c)
            holes = (2 - d.get('3s', 0)) + (6 - d.get('3p', 0))
            nv = sum(q for o, q in d.items() if o in virt)
            if holes <= 2 or nv <= v3:
                out.append(c)
        assert all(c in out for c in base)
        return out

    def spaces():
        sp = _sp0()
        for k, kw in SPACES.items():
            if k not in sp:
                sp[k] = ('allcore', S.ras_cfgs_x(_virt(**kw)))
        for k, (par, v3) in H3.items():
            if k not in sp:
                sp[k] = ('allcore', h3_cfgs(par, v3))
        if 'besth_h3k1' not in sp:     # best + h with only the 3-hole configurations of k1fh_h3 (RAS k = 1 virtuals)
            new3 = [c for c in sp['k1fh_h3'][1] if c not in sp['k1fh'][1]]
            sp['besth_h3k1'] = ('allcore', sp['besth'][1] + [c for c in new3 if c not in sp['besth'][1]])
        return sp

    N15.spaces = spaces
    Q.spaces = spaces
    _rn = S.rname

    def rname(space, t):
        if space in SPACES or space in H3 or space == 'besth_h3k1':
            return f'v15t_{space}_d{S.N_}_{t}e_{Q.TAG_}'
        return _rn(space, t)

    N15.rname = rname
    Q.K.rname_k = rname

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd in ('bases', 'orb', 'vcm'):
        {'bases': bases, 'orb': orb, 'vcm': vcm}[cmd]()
    elif cmd == 'check':     # 'best' must equal the 15s rasDallG2_fall configuration list for crnd
        sp = spaces()
        print(Q.ION, 'best == rasDallG2_fall:', sp['best'][1] == sp['rasDallG2_fall'][1],
              'k1f == 15n rasK1 with every f:', sp['k1f'][1] == N15.ras_cfgs(N15.virt_k({'s': 1, 'p': 1, 'd': 1, 'f': len(J.VIRT['f']), 'g': 1})),
              {k: len(v[1]) for k, v in sp.items() if k in SPACES},
              {k: _virt(**kw) for k, kw in SPACES.items()})
    elif cmd == 'dims':
        Q.dims(sys.argv[2:], 'phase15t_dims.json')
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        Q.M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
