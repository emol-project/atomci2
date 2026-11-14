#!/usr/bin/env python3
"""Phase 17 B: three electrons in the virtual space, Cr3+ RAS k = 1 + all h (15t k1fh; plain Sapporo-QZP-2012, own
orbitals crnd3; n_hole <= 2 as in k1fh).

The RAS rule is phase15s.ras_cfgs_x(maxv=3) (the 15s rasK1_v3 rule): up to 3 electrons in the virtuals, and a
configuration with 3 virtual electrons may be a triple from the references 3s2 3p6 3d3 / 3s 3p6 3d4 (it cannot be
less); everything else (excitation level <= 2, n_hole <= 2) is unchanged.
  k1fh_v3     every such configuration
  k1fh_v3l1   only the 3-virtual-electron configurations with at most one of the three in l >= 3
  k1fh_v3d3   only 3s2 3p6 v3 (3d3 -> v3 from 3s2 3p6 3d3)
Run names v17_<space>_d3_<t>e_crnd3. Commands: classes / dims <spaces> / prep <spaces> / run_names <names> /
analyze <tag> <spaces>"""
import collections
import json
import os
import sys

os.environ.setdefault('ION', 'crnd')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase15u as U  # noqa: E402

T, S, Q, N15, E15 = U.T, U.S, U.Q, U.N15, U.E15
VIRT = U.VIRT
LOF = {o: 'spdfghik'.index(o[-1]) for o in VIRT}
OUT17 = os.path.join(_C.root(), 'results', 'phase17')


def core3(c):
    d = dict(c)
    return d.get('3s', 0), d.get('3p', 0), d.get('3d', 0)


def nhigh(c):
    return sum(q for o, q in dict(c).items() if o in LOF and LOF[o] >= 3)


_sp_u = U.spaces
B17 = ('k1fh_v3', 'k1fh_v3l1', 'k1fh_v3d3', 'k1fh_2p', 'k1fh_2p_h', 'k1fh_2s', 'k1fh_2s2p',
       'fam_d5f2g2h1_sall', 'k1fhi', 'k1fhik', 'k1fhr', 'rasK1r', 'besth_salli', 'k1fh_t',
       'k1fh_2p_h_t', 'k1fh_2s_t', 'k1fh_t2', 'k1fh_2p_h_t2', 'k1fh_tg', 'k1fh_2p_h_tg',
       'k1fh_t2d', 'k1fh_2p_h_t2d', 'k1fh_t2f', 'k1fh_2p_h_t2f', 'k1f_tn', 'k1f_2p_h_tn', 'k1fh_tnf', 'k1fh_2p_h_tnf', 'k1f_tnf', 'k1f_2p_h_tnf',
       'k1f_tnfi', 'k1f_2p_h_tnfi', 'k1fh_sdall', 'k1f_tnfg0', 'k1f_2p_h_tnfg0', 'famdq', 'famiq', 'famdc', 'famdc_d6', 'famdc_d7', 'famdc_d8', 'famdc_nsph', 'famdc_nsp', 'famdc_nh', 'famdc_nsphg', 'famdc_nsph_dd7', 'famdc_nsph_dd8')


def ras_inner(virt, inner, maxh=2, maxv=2, inner_in_h=False):
    """phase15s.ras_cfgs_x (k1fh rule) with 2s / 2p opened: at most inner[o] holes in o (2s, 2p), at most one
    inner hole in total; excitation level from the references 3s2 3p6 3d3 / 3s 3p6 3d4 (2s2 2p6) <= 2 (<= 3 only for
    3 virtual electrons, unchanged), <= maxv virtual electrons, <= maxh holes in 3s3p (inner_in_h: the inner hole
    counts towards maxh). Parity as the references. The c2s configurations are kept."""
    import itertools as it
    import phase15e as E15
    N = S.N_
    lof = {o: 'spdfghik'.index(o[-1]) for o in virt + ['2s', '2p', '3s', '3p', '3d']}
    refs = [{'2s': 2, '2p': 6, '3s': 2, '3p': 6, '3d': N}, {'2s': 2, '2p': 6, '3s': 1, '3p': 6, '3d': N + 1}]
    nel = 16 + N
    rv = []
    for n in range(maxv + 1):
        for comb in it.combinations_with_replacement(virt, n):
            v = dict(collections.Counter(comb))
            if all(q <= 2 * (2 * lof[o] + 1) for o, q in v.items()):
                rv.append(v)
    out = []
    for h2s in range(inner.get('2s', 0) + 1):
        for h2p in range(inner.get('2p', 0) + 1):
            if h2s + h2p > 1:
                continue
            for ns in range(0, 3):
                for np_ in range(0, 7):
                    if (2 - ns) + (6 - np_) + (h2s + h2p if inner_in_h else 0) > maxh:
                        continue
                    for v in rv:
                        nd = nel - (2 - h2s) - (6 - h2p) - ns - np_ - sum(v.values())
                        if nd < 0 or nd > 10:
                            continue
                        c = {'2s': 2 - h2s, '2p': 6 - h2p, '3s': ns, '3p': np_, '3d': nd, **v}
                        if sum(lof[o] * q for o, q in c.items()) % 2:
                            continue
                        lim = max(2, sum(v.values()))
                        if min(sum(max(0, r.get(o, 0) - c.get(o, 0)) for o in set(r) | set(c)) for r in refs) <= lim:
                            out.append(c)
    P0 = refs[0]
    out.sort(key=lambda c: c != P0)
    cf = [E15.full_cfg('all', c) for c in out]
    c2s = list(E15.core_spaces(N)['c2s'])
    return cf + [c for c in c2s if c not in cf]


def spaces():
    sp = _sp_u()
    # Q2 (long queue): the 15u D candidate fam_d5f2g2h1 (besth without the most diffuse virtual f) with every virtual
    # s, as Step 8 did for besth (sall_virt)
    import phase15j as J
    v = U.T._virt(**U.FAM['fam_d5f2g2h1'])
    v = [o for o in v if o[-1] != 's'] + Q.K.compact3('s', len(J.VIRT['s']))
    sp['fam_d5f2g2h1_sall'] = ('allcore', S.ras_cfgs_x(v))
    if J.VIRT.get('i'):     # Q3 (ION crti1 / crti2): k1fh + every i of the basis, same RAS rule
        sp['k1fhi'] = ('allcore', S.ras_cfgs_x(U.T._virt(h='all') + J.VIRT['i']))
        assert all(c in sp['k1fhi'][1] for c in sp['k1fh'][1])
        # Q4-i: the Step 8 space (besth_sall) + every i of the basis, same RAS rule
        sp['besth_salli'] = ('allcore', S.ras_cfgs_x(U.sall_virt() + J.VIRT['i']))
        assert all(c in sp['besth_salli'][1] for c in sp['besth_sall'][1])
    if J.VIRT.get('k'):     # item 1c (ION crtk1): k1fhi + every k of the basis
        sp['k1fhik'] = ('allcore', S.ras_cfgs_x(U.T._virt(h='all') + J.VIRT['i'] + J.VIRT['k']))
        assert all(c in sp['k1fhik'][1] for c in sp['k1fhi'][1])
    if S.N_ == 6:           # item 1a (ION feop / fedk; fend as the check): the 15n rasK1 rule (one virtual per l,
        # s..g, by <r>; for feop / fedk ranked by the matched fend6 <r>, phase17q4r) under a v17 run name
        sp['rasK1r'] = ('allcore', N15.ras_cfgs(N15.virt_k({l: 1 for l in 'spdfg'})))
    if Q.ION in ('fet2', 'fe5t2', 'cot2'):     # paper item 3 (phase17q10): Fe2+ 2p with the tight d, f; the crt2 rules of Q4-n2 (k1fh_t,
        # two most compact d, every f) and of item 2a P' (k1f_tnf: no h, without the most diffuse f), 2p closed / opened
        # P'' = P' without the virtual g (the Fe2+ P' 2p block 83,684 CSFs sits at the 14 GB limit; P'' 60,342)
        for ref, op, kw in (('k1fh_t', 'k1fh_2p_h_t', dict(h='all', d=2)), ('k1f_tnf', 'k1f_2p_h_tnf', dict(h=0, d=2, f=3)),
                            ('k1f_tnfg0', 'k1f_2p_h_tnfg0', dict(h=0, d=2, f=3, g=0))):
            vt = U.T._virt(**kw)
            bt = S.ras_cfgs_x(vt)
            sp[ref] = ('allcore', bt)
            cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
            sp[op] = ('allcore', bt + [c for c in cf if c not in bt])
    if Q.ION in ('feqd', 'coqd'):     # item 10: fam_d5f2g2h1 - 12f in Sapporo-QZP-2012-diffuse: every virtual d (the diffuse
        # one included), the most compact f, g 2, h 1, and the most compact plus the most diffuse virtual s
        s_all = Q.K.compact3('s', len(J.VIRT['s']))
        vq = [o for o in U.T._virt(d='all', f=1, g=2, h=1) if o[-1] != 's'] + [s_all[0], s_all[-1]]
        sp['famdq'] = ('allcore', S.ras_cfgs_x(vq))
    if Q.ION in ('fedc', 'codc'):   # Phase 18, 10 Nov B: fam_d5f2g2h1 (every virtual d, f 2, g 2, h 1, s 1, p 1) with the
        # d contractions opened (10 virtual d instead of 4); famdc_d<k>: only the k most compact virtual d
        sp['famdc'] = ('allcore', S.ras_cfgs_x(U.T._virt(d='all', f=2, g=2, h=1)))
        for k in (6, 7, 8):
            sp[f'famdc_d{k}'] = ('allcore', S.ras_cfgs_x(U.T._virt(d=k, f=2, g=2, h=1)))
        # reduced: every virtual d, without the virtual l in the suffix (n<l...>: s, p, h; the f, g counts of fam)
        for nm, drop in (('famdc_nsph', 'sph'), ('famdc_nsp', 'sp'), ('famdc_nh', 'h'), ('famdc_nsphg', 'sphg')):
            v = [o for o in U.T._virt(d='all', f=2, g=2, h=1) if o[-1] not in drop.replace('g', '')]
            if 'g' in drop:
                v = [o for o in v if o[-1] != 'g' or o == Q.K.compact3('g', 1)[0]]
            sp[nm] = ('allcore', S.ras_cfgs_x(v))
        # famdc_nsph_dd<k>: the k most diffuse virtual d (the tightest opened d, <r> below the plain set's most compact
        # virtual d, left out), f 2, g 2, no virtual s, p, h
        dall = Q.K.compact3('d', len(J.VIRT['d']))
        for k in (7, 8):
            v = [o for o in U.T._virt(d='all', f=2, g=2, h=1) if o[-1] not in 'sphd'] + dall[-k:]
            sp[f'famdc_nsph_dd{k}'] = ('allcore', S.ras_cfgs_x(v))
    if Q.ION in ('feti1', 'coti1'):    # item 11: fam_d5f2g2h1 - 12f + the i (2.2136970) + the three pure n = 3 configurations
        # of Step 10 (feti16 s..h orbitals = fend6); its exact sub-spaces (phase17q18) separate i and n = 3
        vi = U.T._virt(d='all', f=1, g=2, h=1) + J.VIRT['i']
        bi = S.ras_cfgs_x(vi)
        n3 = [E15.full_cfg('all', d) for d in U.S4['s4']]
        sp['famiq'] = ('allcore', bi + [c for c in n3 if c not in bi])
    if S.N_ == 3:
        base = sp['k1fh'][1]
        sp['k1fhr'] = sp['k1fh']     # Q4-R (ION crop / crdk): k1fh under a v17 run name
        v3 = S.ras_cfgs_x(VIRT, maxv=3)
        assert all(c in v3 for c in base)
        new = [c for c in v3 if c not in base]
        sp['k1fh_v3'] = ('allcore', v3)
        sp['k1fh_v3l1'] = ('allcore', base + [c for c in new if nhigh(c) <= 1])
        sp['k1fh_v3d3'] = ('allcore', base + [c for c in new if core3(c)[:2] == (2, 6)])
        for nm, inn, ih in (('k1fh_2p', {'2p': 1}, False), ('k1fh_2p_h', {'2p': 1}, True), ('k1fh_2s', {'2s': 1}, False),
                            ('k1fh_2s2p', {'2s': 1, '2p': 1}, False)):
            cf = ras_inner(VIRT, inn, inner_in_h=ih)
            sp[nm] = ('allcore', base + [c for c in cf if c not in base])
        # paper item 5: k1fh with every virtual s and every virtual d (the 3d2 ns / 3d2 nd configurations in full)
        vs = [o for o in VIRT if o[-1] not in 'sd'] + Q.K.compact3('s', len(J.VIRT['s'])) + \
            Q.K.compact3('d', len(J.VIRT['d']))
        sp['k1fh_sdall'] = ('allcore', S.ras_cfgs_x(vs))
        assert all(c in sp['k1fh_sdall'][1] for c in base)
        if Q.ION == 'crt2':     # paper item 3: the Fe2+ P'' (no virtual g) for Cr3+, to calibrate P'' against P'
            vt = U.T._virt(h=0, d=2, f=3, g=0)
            bt = S.ras_cfgs_x(vt)
            sp['k1f_tnfg0'] = ('allcore', bt)
            cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
            sp['k1f_2p_h_tnfg0'] = ('allcore', bt + [c for c in cf if c not in bt])
        if Q.ION == 'crt2i':    # paper item 4 (phase17q10): P' (crt2 k1f_tnf) + every i, 2p closed and opened
            vt = U.T._virt(h=0, d=2, f=3)
            vi = vt + J.VIRT['i']
            bt = S.ras_cfgs_x(vi)
            sp['k1f_tnfi'] = ('allcore', bt)
            cf = ras_inner(vi, {'2p': 1}, inner_in_h=True)
            sp['k1f_2p_h_tnfi'] = ('allcore', bt + [c for c in cf if c not in bt])
            noi = lambda cs: [c for c in cs if not any(o[-1] == 'i' for o in dict(c))]     # noqa: E731
            assert noi(sp['k1f_tnfi'][1]) == S.ras_cfgs_x(vt)
            assert noi(sp['k1f_2p_h_tnfi'][1]) == S.ras_cfgs_x(vt) + [c for c in ras_inner(vt, {'2p': 1}, inner_in_h=True)
                                                                       if c not in S.ras_cfgs_x(vt)]
        if Q.ION == 'crt2':     # Q4-n2 (phase17q4n): the two most compact virtual d (tight + k1fh), every f
            vt = U.T._virt(h='all', d=2)
            bt = S.ras_cfgs_x(vt)
            sp['k1fh_t'] = ('allcore', bt)
            cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
            sp['k1fh_2p_h_t'] = ('allcore', bt + [c for c in cf if c not in bt])
            cf = ras_inner(vt, {'2s': 1}, inner_in_h=False)      # item 2b: k1fh_2s rule over the same virtuals
            sp['k1fh_2s_t'] = ('allcore', bt + [c for c in cf if c not in bt])
        # item 2a: crt3 = two tight d and two tight f (the three most compact d), crt2g = crt2 + a tight g (the two
        # most compact g: tight + the k1fh one); 2p-closed reference and 2p_h as Q4-n2
        # item 2a fallback: the same 2p comparison in the k1f parent (no h) for all three bases
        # fallback 2: k1fh without the most diffuse virtual f (<r> 2.27 bohr), crt2 (3 f) vs crt3 (4 f)
        for ion, kw in (('crt2', dict(d=2, f=3)), ('crt3', dict(d=3, f=4)), ('crt2g', dict(d=2, f=3, g=2))):
            if Q.ION == ion:
                vt = U.T._virt(h=0, **kw)               # fallback 3: k1f parent and no diffuse f
                bt = S.ras_cfgs_x(vt)
                sp['k1f_tnf'] = ('allcore', bt)
                cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
                sp['k1f_2p_h_tnf'] = ('allcore', bt + [c for c in cf if c not in bt])
                vt = U.T._virt(h='all', **kw)
                bt = S.ras_cfgs_x(vt)
                sp['k1fh_tnf'] = ('allcore', bt)
                cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
                sp['k1fh_2p_h_tnf'] = ('allcore', bt + [c for c in cf if c not in bt])
        for ion, kw in (('crt2', dict(d=2)), ('crt3', dict(d=3)), ('crt2g', dict(d=2, g=2))):
            if Q.ION == ion:
                vt = U.T._virt(h=0, **kw)
                bt = S.ras_cfgs_x(vt)
                sp['k1f_tn'] = ('allcore', bt)
                cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
                sp['k1f_2p_h_tn'] = ('allcore', bt + [c for c in cf if c not in bt])
        for ion, nm, kw in (('crt3', 't2', dict(d=3)), ('crt2g', 'tg', dict(d=2, g=2))):
            if Q.ION == ion:
                vt = U.T._virt(h='all', **kw)
                bt = S.ras_cfgs_x(vt)
                sp[f'k1fh_{nm}'] = ('allcore', bt)
                cf = ras_inner(vt, {'2p': 1}, inner_in_h=True)
                sp[f'k1fh_2p_h_{nm}'] = ('allcore', bt + [c for c in cf if c not in bt])
                if ion == 'crt3':       # fallbacks (the full t2 analysis does not fit): only one of the two new tight
                    import phase15j as J        # functions, same crt33 orbitals; 'd' drops the tightest f, 'f' the tightest d
                    rd = lambda l: min(J.VIRT[l], key=Q.K.r_virt3().get)    # noqa: E731
                    for sub, drop in (('d', rd('f')), ('f', rd('d'))):
                        vs = [o for o in vt if o != drop]
                        bs_ = S.ras_cfgs_x(vs)
                        sp[f'k1fh_t2{sub}'] = ('allcore', bs_)
                        cf = ras_inner(vs, {'2p': 1}, inner_in_h=True)
                        sp[f'k1fh_2p_h_t2{sub}'] = ('allcore', bs_ + [c for c in cf if c not in bs_])
    return sp


N15.spaces = spaces
Q.spaces = spaces
_rn = U.rname


def rname(space, t):
    if space in B17:
        return f'v17_{space}_d{S.N_}_{t}e_{Q.TAG_}'
    return _rn(space, t)


N15.rname = rname
Q.K.rname_k = rname


def own_hf(names):
    """d6 (Fe2+ fend6): phase15j.analyze2 places the levels on the P block of the 15h fe15h6 'dSD' space, which needs
    the fe15h6 orbitals. For own orbitals the HF P block is the space's own PHP (all-electron decks: the vcm P block
    up to a constant, which drops out of every parameter), so path_blocks('dSD') is replaced by the PHP of the
    first space's runs, in the canonical seniority basis."""
    import numpy as np
    import phase13d as D
    import phase15j as J
    from atomci_heff.sparse import SymLower
    root = _C.root()                     # HEFF_WORK (was the research repo root, from __file__)
    maps = json.load(open(os.path.join(root, 'results', 'phase13', 'phase13b_b.json')))['maps']
    out = {}
    for L, twoS, m, t in D.term_list(6):
        d = np.load(os.path.join(root, 'results', N15.rname(names[0], t) + '.npz'), allow_pickle=True)
        op = SymLower.from_npz(d)
        U = np.array(maps[f'vcm_d6_{t}e'])
        out[t] = (None, U.T @ op.sub(np.arange(m)).dense() @ U)
        del d, op
    J.path_blocks = lambda space: out


def classes():
    """The added 3-virtual-electron configurations by (3s, 3p, 3d) occupation and by number of l >= 3 virtuals."""
    sp = spaces()
    base = sp['k1fh'][1]
    by = collections.defaultdict(collections.Counter)
    for c in sp['k1fh_v3'][1]:
        if c not in base:
            by[core3(c)][nhigh(c)] += 1
    print('virtuals', VIRT, '| k1fh', len(base), '| k1fh_v3', len(sp['k1fh_v3'][1]))
    for k, n in sorted(by.items(), reverse=True):
        print(f'  3s{k[0]} 3p{k[1]} 3d{k[2]} v3: {sum(n.values()):3d}  by number of l >= 3 virtual e {dict(sorted(n.items()))}')
    json.dump({f'3s{k[0]} 3p{k[1]} 3d{k[2]}': dict(n) for k, n in by.items()},
              open(os.path.join(OUT17, 'phase17_b0_classes.json'), 'w'), indent=1)


if __name__ == '__main__':
    os.makedirs(OUT17, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'classes':
        classes()
    elif cmd == 'dims':
        Q.dims(sys.argv[2:], os.path.join('..', 'phase17', 'phase17_dims.json'))
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        Q.M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        if S.N_ == 6:
            own_hf(sys.argv[3:])
        N15.analyze(sys.argv[2], sys.argv[3:])
