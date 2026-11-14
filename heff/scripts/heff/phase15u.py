#!/usr/bin/env python3
"""Phase 15u: what the 3-hole increase of Cr3+ T1 is made of, and how the 3s3p hole count converges (crnd only).

Spaces on top of phase15t (ION=crnd: standard plain Sapporo-QZP-2012, own orbitals crnd3). n_hole = maximum number
of holes in 3s3p relative to 3s2 3p6.
  Step 1 (parent k1fh = RAS k = 1 + all h; k1fh_h3 its n_hole = 3 version, 15t addendum 2): the 3-hole
  configurations added by k1fh_h3, grouped by the number of virtual electrons nv
    u1a     k1fh + (a) nv = 0
    u1ab    k1fh + (a) + (b) nv <= 1
  Step 2: the n = 3 complex alone (3s, 3p, 3d; no virtuals, no 2s hole), every even-parity distribution of the
  11 electrons with at most n holes in 3s3p (no excitation-level limit)
    n3h<n>  n = 0 .. 7
  Step 3: n_hole = 4 on k1fh. ras_cfgs_x(maxh=4) keeps the excitation level <= 2 from the two references, which
  allows at most 3 holes (3s 3p4 from 3s 3p6 3d4), so it equals k1fh_h3. The variants add the 4-hole configurations
  with at most v virtual electrons and no excitation-level limit:
    u3v<v>  k1fh_h3 + 4-hole configurations with nv <= v
    u3c1    k1fh_h3 + every 3- and 4-hole configuration with nv <= 1 (also the 3-hole ones with excitation level 3)
  Step 4: best space + all h (15t besth) plus pure n = 3 configurations
    s4a     besth + 3s 3p4 3d6
    s4      besth + 3s 3p4 3d6, 3s2 3p2 3d7, 3s0 3p4 3d7
  Step 5: u5 = k1fh_h3 + the five 3s2 3p3 3d5 v (v = 15p, 11f, 12f, 13f, 11h), the 3-hole part of u3c1
  Step 8: besth_sall = besth with every virtual s (all ions)
  Step 10: besth_sall_s4 = besth_sall + the three pure n = 3 configurations of s4 (all ions); k1fh_n3 = k1fh + the
    same three (d3; the pure n = 3 part in RAS k = 1 + all h, for the carry-over)
  Step 6: reduced virtuals RVIRT (16s 15p 14d 11f): r2 (n_hole 2), r3 (n_hole 3), r3c1 / r3c2 = r3 + every 3- and
    4-hole configuration with <= 1 / <= 2 virtual electrons, no excitation limit
Run names v15u_<space>_d3_<t>e_crnd3. Commands: groups / dims <spaces> / prep <spaces> / run_names <names> /
analyze <tag> <spaces>"""
import itertools as it
import collections
import os
import sys

os.environ.setdefault('ION', 'crnd')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase15t as T  # noqa: E402

S, Q, N15, E15 = T.S, T.Q, T.N15, T._E15
NEL, P0 = 8 + S.N_, {'3s': 2, '3p': 6, '3d': S.N_}     # 30 Oct: from the ion's N (was d3 only: 11, 3d3)
LOF = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}


def holes(d):
    return (2 - d.get('3s', 0)) + (6 - d.get('3p', 0))


def nvirt(d, virt):
    return sum(q for o, q in d.items() if o in virt)


def val(c):
    """frozenset full configuration -> {orbital: occupation} outside the [Ne] core (2s/2p/1s holes kept)."""
    return {o: q for o, q in dict(c).items() if o not in ('1s', '2s', '2p') or q != {'1s': 2, '2s': 2, '2p': 6}[o]}


def n3_cfgs(nh):
    out = []
    for ns in range(3):
        for np_ in range(7):
            nd = NEL - ns - np_
            if 0 <= nd <= 10 and (2 - ns) + (6 - np_) <= nh and np_ % 2 == 0:
                out.append({'3s': ns, '3p': np_, '3d': nd})
    out.sort(key=lambda c: c != P0)
    return out


def four_hole(virt, v, nh=(4,)):
    """Even-parity configurations with exactly nh holes in 3s3p and <= v electrons in virt, no excitation limit."""
    out = []
    for n in range(v + 1):
        for comb in it.combinations_with_replacement(virt, n):
            w = dict(collections.Counter(comb))
            if any(q > 2 * (2 * LOF[o[-1]] + 1) for o, q in w.items()):
                continue
            for ns, k in it.product(range(3), nh):
                np_ = 6 - (k - (2 - ns))
                nd = NEL - ns - np_ - n
                if 0 <= np_ <= 6 and 0 <= nd <= 10:
                    c = {'3s': ns, '3p': np_, '3d': nd, **w}
                    if sum(LOF[o[-1]] * q for o, q in c.items()) % 2 == 0:
                        out.append(c)
    return out


_sp_t = T.spaces
RVIRT = T._virt(f=1, g=0, h=0)          # Step 6: RAS k = 1 with one virtual per l for s, p, d, f (no g, no h)
R6 = {'r2': None, 'r3': None, 'r3c1': 1, 'r3c2': 2}
VIRT = T._virt(**T.SPACES['k1fh'])
U1 = {'u1a': 0, 'u1ab': 1}
U3 = {'u3v0': 0, 'u3v1': 1, 'u3v2': 2, 'u3c1': (1, (3, 4))}
N3 = {f'n3h{n}': n for n in range(8)}
# Step 4: best space + all h (15t besth) plus pure n = 3 configurations (no virtual electron)
# (3d count from the ion's N: 3s 3p4 3d(N+3), 3s2 3p2 3d(N+4), 3s0 3p4 3d(N+4); d3: 3d6, 3d7, 3d7; Fe2+ d6: 3d9, 3d10, 3d10)
S4 = {'s4a': [{'3s': 1, '3p': 4, '3d': S.N_ + 3}],
      's4': [{'3s': 1, '3p': 4, '3d': S.N_ + 3}, {'3s': 2, '3p': 2, '3d': S.N_ + 4}, {'3s': 0, '3p': 4, '3d': S.N_ + 4}]}


# D: reduced members of the best-space family (15t _virt knobs: virtual d, f, g, h counts, compact first; s 1, p 1),
# named fam_d<d>f<f>g<g>h<h>; best = fam_d5f4g2h0, besth = fam_d5f4g2h2
FAM = {f'fam_d{d}f{f}g{g}h{h}': dict(d=d, f=f, g=g, h=h) for d in (3, 5) for f in (1, 2, 3, 4) for g in (1, 2)
       for h in (0, 1, 2)}


# Step 8: best space + all h with every virtual s (besth keeps the most compact one)
def sall_virt():
    import phase15j as J
    v = T._virt(**T.SPACES['besth'])
    return [o for o in v if o[-1] != 's'] + Q.K.compact3('s', len(J.VIRT['s']))


def spaces():
    sp = _sp_t()
    for k, kw in FAM.items():
        sp[k] = ('allcore', S.ras_cfgs_x(T._virt(**kw)))
    sp['besth_sall'] = ('allcore', S.ras_cfgs_x(sall_virt()))
    n3add = [E15.full_cfg('all', d) for d in S4['s4']]
    sp['besth_sall_s4'] = ('allcore', sp['besth_sall'][1] + [c for c in n3add if c not in sp['besth_sall'][1]])
    base, h3 = sp['k1fh'][1], sp['k1fh_h3'][1]
    # 30 Oct (Phase 17 paper items): RAS k = 1 + all h with every virtual s (the all-s increment in a cheap space)
    vs = [o for o in VIRT if o[-1] != 's'] + Q.K.compact3('s', len(__import__('phase15j').VIRT['s']))
    sp['k1fh_sall'] = ('allcore', S.ras_cfgs_x(vs))
    sp['k1fh_n3'] = ('allcore', base + [c for c in n3add if c not in base])
    if S.N_ != 3:          # d6 (Fe2+): besth / s4, and the Step 10 carry-over spaces k1fh_n3, u3c1 (NEL, P0 from N)
        for k, add in S4.items():
            add = [E15.full_cfg('all', d) for d in add]
            sp[k] = ('allcore', sp['besth'][1] + [c for c in add if c not in sp['besth'][1]])
        add = [E15.full_cfg('all', d) for d in four_hole(VIRT, *U3['u3c1'])]
        sp['u3c1'] = ('allcore', h3 + [c for c in add if c not in h3])
        return sp
    new3 = [c for c in h3 if c not in base]
    for k, v in U1.items():
        sp[k] = ('allcore', base + [c for c in new3 if nvirt(dict(c), VIRT) <= v])
    for k, n in N3.items():
        sp[k] = ('allcore', [E15.full_cfg('all', d) for d in n3_cfgs(n)])
    for k, v in U3.items():
        add = [E15.full_cfg('all', d) for d in (four_hole(VIRT, *v) if isinstance(v, tuple) else four_hole(VIRT, v))]
        sp[k] = ('allcore', h3 + [c for c in add if c not in h3])
    sp['u5'] = ('allcore', h3 + [E15.full_cfg('all', {'3s': 2, '3p': 3, '3d': 5, v: 1})
                                 for v in VIRT if v[-1] in 'pfh'])
    r3 = S.ras_cfgs_x(RVIRT, maxh=3)
    sp['r2'], sp['r3'] = ('allcore', S.ras_cfgs_x(RVIRT)), ('allcore', r3)
    for k in ('r3c1', 'r3c2'):
        add = [E15.full_cfg('all', d) for d in four_hole(RVIRT, R6[k], (3, 4))]
        sp[k] = ('allcore', r3 + [c for c in add if c not in r3])
    for k, add in S4.items():
        add = [E15.full_cfg('all', d) for d in add]
        sp[k] = ('allcore', sp['besth'][1] + [c for c in add if c not in sp['besth'][1]])
    return sp


N15.spaces = spaces
Q.spaces = spaces
_rn = T.rname


def rname(space, t):
    if space in U1 or space in U3 or space in N3 or space in S4 or space in R6 or space == 'u5' or space in FAM or space in ('besth_sall', 'besth_sall_s4', 'k1fh_n3', 'k1fh_sall') or \
            (space == 'besth' and Q.ION in ('vnd', 'mnnd', 'fend')):  # series: V2+, Mn4+, Fe2+ best space + all h
        return f'v15u_{space}_d{S.N_}_{t}e_{Q.TAG_}'
    return _rn(space, t)


N15.rname = rname
Q.K.rname_k = rname


def groups():
    """Step 1/2/3 configuration checks."""
    sp = spaces()
    base, h3 = sp['k1fh'][1], sp['k1fh_h3'][1]
    new3 = [dict(c) for c in h3 if c not in base]
    by = collections.defaultdict(list)
    for d in new3:
        assert holes(d) == 3, d
        by[nvirt(d, VIRT)].append(d)
    print('virtuals', VIRT)
    print('k1fh', len(base), 'k1fh_h3', len(h3), 'new 3-hole', len(new3), {k: len(v) for k, v in sorted(by.items())})

    def s(d):
        return ' '.join(f'{o}{q}' for o, q in d.items() if o not in ('1s', '2s', '2p'))
    for k in sorted(by):
        print(f'  nv={k}:', '; '.join(s(d) for d in by[k]))
    lv = collections.Counter(''.join(sorted(o[-1] for o, q in d.items() if o in VIRT for _ in range(q))) for d in by[2])
    print('  nv=2 by l of the virtual pair', dict(lv))
    lv = collections.Counter(next(o[-1] for o in d if o in VIRT) for d in by[1])
    print('  nv=1 by l', dict(lv))
    # parity check of the 0-virtual 3-hole distributions
    for ns in range(3):
        np_ = 6 - (3 - (2 - ns))
        if 0 <= np_ <= 6:
            print(f'  3 holes, no virtual: 3s{ns} 3p{np_} 3d{NEL - ns - np_}', 'even' if np_ % 2 == 0 else 'odd')
    for k, n in N3.items():
        print(k, len(sp[k][1]), '; '.join(f'3s{d["3s"]} 3p{d["3p"]} 3d{d["3d"]}' for d in n3_cfgs(n) if holes(d) == n))
    for k, v in U3.items():
        a = [d for d in (four_hole(VIRT, *v) if isinstance(v, tuple) else four_hole(VIRT, v))
             if E15.full_cfg('all', d) not in sp['k1fh_h3'][1]]
        print(k, len(sp[k][1]), 'added', len(a), '; '.join(s(d) for d in a if nvirt(d, VIRT) == 0))
    print('ras_cfgs_x(maxh=4) == k1fh_h3:', S.ras_cfgs_x(VIRT, maxh=4) == h3)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'groups':
        groups()
    elif cmd == 'dims':
        Q.dims(sys.argv[2:], 'phase15u_dims.json')
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        Q.M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
