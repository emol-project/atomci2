#!/usr/bin/env python3
"""Phase 15e: completeness of the paths and virtual-space convergence of T1, T2 (own orbitals).

Ions: d3 V2+, Cr3+, Mn4+; d4 Cr2+. Separate spaces (3s2 3p6 doubly occupied unless stated):
  (a) s path: 3d^N + single and double excitations 3d -> virtual s (11s..), k = 1, 2, 3, 7 (all) virtual s
      (s virtuals = Fock eigenvectors of the own set, as in 14b/15a/15b);
  (b) d path: full CI of N electrons in {3d} + virtual d {11d,12d} / {11d,12d,13d} / {11d..15d};
  (c) g path: full CI in {3d} + {11g} / {11g,12g} / {11g,12g,13g} (g^3 in one shell excluded, as ATOMCI requires);
  (d) core path: 3s2 3p6 3d^N + 3s -> 3d, then + 2s -> 3d, then + 1s -> 3d (single holes, the hole shell's other
      shells full; orbitals unrelaxed).
(a)-(c) use the 13b vcm deck of the term ([Ne] frozen, 3s 3p 3d active, all outer orbitals) with a new CSF list
(the 15d construction); (d) uses a deck from atinputs with every shell active (no frozen core), patched as 13b.
Check: 15d combined space + 11s for Cr3+.
usage: phase15e.py dims | prep | run | analyze
-> results/phase15/phase15e.txt/.json
"""
import itertools
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import counts as CN  # noqa: E402
import gen_v  # noqa: E402
import trial  # noqa: E402
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase14b as P14  # noqa: E402
import phase15b as Q15  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
LMAP = dict(CN.orbitals(trial.V), **{'1s': 0, '2s': 0, '2p': 1})
CAP = {o: 2 * (2 * l + 1) for o, l in LMAP.items()}
IONS = [(3, 'V'), (3, 'Cr'), (3, 'Mn'), (4, 'Cr')]
SV = [f'{k}s' for k in range(11, 18)]
DV = [f'{k}d' for k in range(11, 16)]
GV = ['11g', '12g', '13g']
CORE0 = {'1s': 2, '2s': 2, '2p': 6}
VAL0 = {'3s': 2, '3p': 6}


def fci(N, orbs):
    out = []
    for occ in itertools.product(*[range(min(CAP[o], N) + 1) for o in orbs]):
        if sum(occ) != N:
            continue
        d = {o: q for o, q in zip(orbs, occ) if q}
        if any(o.endswith('g') and q >= 3 for o, q in d.items()):
            continue
        out.append(d)
    return out


def spaces(N):
    """{space: (deck kind, [config dicts])}; config dicts hold the valence part (3s 3p added below)."""
    P = {'3d': N}
    sp = {}
    for k in (1, 2, 3, 7):
        cf = [P]
        s = SV[:k]
        cf += [{'3d': N - 1, o: 1} for o in s]
        cf += [{'3d': N - 2, o: 2} for o in s]
        cf += [{'3d': N - 2, a: 1, b: 1} for a, b in itertools.combinations(s, 2)]
        sp[f's{k}'] = ('vcm', cf)
    for k in (2, 3, 5):
        sp[f'd{k}'] = ('vcm', fci(N, ['3d'] + DV[:k]))
    for k in (1, 2, 3):
        sp[f'g{k}'] = ('vcm', fci(N, ['3d'] + GV[:k]))
    return sp


def full_cfg(kind, d, hole=None):
    base = dict(CORE0)
    base.update(VAL0)
    if hole:
        base[hole] -= 1
    base.update(d)
    return frozenset({o: q for o, q in base.items() if q}.items()) if kind == 'all' else \
        frozenset({o: q for o, q in {**VAL0, **d}.items() if q}.items())


def core_spaces(N):
    P = {'3d': N}
    out = {}
    holes = []
    for name, h in (('c3s', '3s'), ('c2s', '2s'), ('c1s', '1s')):
        holes.append(h)
        out[name] = [full_cfg('all', P)] + [full_cfg('all', {'3d': N + 1}, hole=x) for x in holes]
    return out


def all_configs(N, el):
    """{space: [frozenset configs]} incl. kind; P config first."""
    res = {}
    P0 = full_cfg('vcm', {'3d': N})
    for name, (kind, cf) in spaces(N).items():
        cfs = [full_cfg('vcm', d) for d in cf]
        res[name] = ('vcm', [P0] + [c for c in cfs if c != P0])
    for name, cf in core_spaces(N).items():
        res[name] = ('allcore', cf)
    if (N, el) == (3, 'Cr'):
        cf = [full_cfg('vcm', d) for d in fci(N, ['3d', '11d', '12d', '11g', '11s'])]
        cf.append(frozenset({'3s': 1, '3p': 6, '3d': N + 1}.items()))
        P = full_cfg('vcm', {'3d': N})
        res['comb+s'] = ('vcm', [P] + [c for c in cf if c != P])
    return res


def dims():
    w = ['| ion | sector | space | largest block CSFs | sum over terms | configurations |', '|---|---|---|---|---|---|']
    for N, el in IONS:
        for name, (kind, cf) in all_configs(N, el).items():
            if (N, el) != (3, 'V') and N == 3 and name != 'comb+s':
                continue  # identical for every d3 ion
            mx, tot = 0, 0
            for L, twoS, m, t in D.term_list(N):
                a = CN.count(LMAP, cf, t)
                mx, tot = max(mx, a[0]), tot + a[0]
            w.append(f'| {el if name == "comb+s" or N == 4 else "any d3"} | d{N} | {name} | {mx} | {tot} | {len(cf)} |')
    print('\n'.join(w))
    return w


def rname(space, N, t, tag):
    return f'v15e_{space}_d{N}_{t}e_{tag}'


def allcore_deck(N, term, Z, bas):
    import gen_v
    from atomci_heff import gto
    L = 'SPDFGHI'.index(term[1])
    ls = [f[0] for f in gto.read_bas(os.path.join(_C.BASIS, bas))]
    nbf = [ls.count(l) for l in range(max(ls) + 1)]     # outer counts from the basis (15s: 9s7p5d3f2g1h set)
    outer = ', '.join(f'{n - gen_v.CORE[l] - (1 if l == 2 else 0)}{"SPDFGHIK"[l]}' for l, n in enumerate(nbf))
    txt = f"""title:  " allcore "
deck:   be.deck
csf:    be.csf
atom:
  z:     {Z}
  ang:   {term[1]}
  spin:  {term[0]}
files:
  cfptable:  ./CFPTBL
  basis:     {bas}
  format:    atomci
  mcp:       mcp.dat
  open:
    - unit: 27, file: be.NO, format: binary, status: old
orbital:
  active:       1S, 2S, 2P, 3S, 3P, 3D
  outer:        {outer}
reference:
  - scf: true, occ: 1S*2 2S*2 2P*6 3S*2 3P*6 3D*{N}
ctrl:
  ai:  type: 2, idprnt: 0
  wf:  rule: 1, npflg: 0
  tr:
    record: 1
    ind:
      - record: 2, l: 0, first: 1, last: {nbf[0]}
  ci:  ips: 10, maxit: 400, nsdim: 100, nstate: 1
  rs:  print: -0.001
"""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, 'x.inp'), 'w').write(txt)
    shutil.copy(os.path.join(_C.BASIS, bas), tmp)
    env = dict(os.environ, PYTHONPATH=_C.ATOMCI_SCRIPTS, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', '-m', 'atinputs', 'x.inp'], cwd=tmp, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deck, csf = open(os.path.join(tmp, 'be.deck')).read(), open(os.path.join(tmp, 'be.csf')).read()
    shutil.rmtree(tmp)
    assert re.search(r'^itnel=%d,$' % (18 + N), deck, flags=re.M)
    deck = gen_v.sub_first(deck, 'nrnfg', '1,0,1,1,0,1,1,1')
    wf = deck.index(' &wf\n')
    head, tail = deck[:wf], deck[wf:]
    tail = gen_v.sub_first(tail, 'ispin', int(term[0]))
    tail = gen_v.sub_first(tail, 'iorb', 2 * L + 1)
    tail = gen_v.sub_first(tail, 'ipar', 0)
    deck = head + tail
    deck = re.sub(r'^rule=1,$', 'rule=0,', deck, count=1, flags=re.M)
    deck = gen_v.sub_first(deck, 'npflg', 0)
    deck = deck.replace(' &wf\n', ' &wf\nlrps=4000000,\n', 1)
    deck = deck.replace(' &ci\n', ' &ex\nretry=0,\n /\n &ci\n', 1)
    return deck, csf


def prep():
    n = 0
    for N, el in IONS:
        Z, bas = Q15.EL[el]
        tag = Q15.tag_of(el, N)
        for space, (kind, cf) in all_configs(N, el).items():
            for L, twoS, m, t in D.term_list(N):
                if kind == 'vcm':
                    src = os.path.join(ROOT, 'inputs', f'vcm_d{N}_{t}e')
                    deck = open(os.path.join(src, 'be.deck')).read()
                    csf = open(os.path.join(src, 'be.csf')).read()
                    assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
                    deck = deck.replace('z=23,', f'z={Z},').replace('FILE=v_qzp.bas', f'FILE={bas}')
                else:
                    deck, csf = allcore_deck(N, t, Z, bas)
                header = csf.split(' ))\n', 1)[0] + ' ))\n'
                seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
                tot, ncfg, per = CN.count(LMAP, cf, t)
                assert per[cf[0]] == m, (space, t)
                order = [c for c in cf if c in per]
                lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)) + '  ))'
                         for c in order]
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
                n += 1
    print(f'{n} decks')


def run():
    t0 = time.time()
    for N, el in IONS:
        tag = Q15.tag_of(el, N)
        for space in all_configs(N, el):
            for L, twoS, m, t in D.term_list(N):
                dst = rname(space, N, t, tag)
                if os.path.isdir(os.path.join(ROOT, 'runs', dst)):
                    continue
                subprocess.run([sys.executable, _C.script('run_atomci.py'), dst,
                                '--vectors-from', f'vq_orb_{tag}'], check=True, stdout=subprocess.DEVNULL)
            print(f'{el} d{N} {space} {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- analysis

T3 = S.jl_table(3)


def nrm3(v):
    return float(np.sqrt(max(S.ip(3, v, v), 0.0)))


def d3(c):
    return sum(c[k] * T3[k] for k in B.TK)


def blocks(space, N, tag):
    js = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))
    out = []
    for L, twoS, m, t in D.term_list(N):
        Hm, lab = B.load_run(rname(space, N, t, tag))
        P = [i for i, l in enumerate(lab) if l == lab[0]]
        assert len(P) == m and P == list(range(m)), (space, t, P)
        out.append(dict(N=N, t=t, L=L, twoS=twoS, H=Hm, lab=lab, U=np.array(js['maps'][f'vcm_d{N}_{t}e']), m=m, P=P,
                        Q=[i for i in range(len(lab)) if i not in P]))
    return out


def analyze():
    import phase13f as F
    t0 = time.time()
    c15 = json.load(open(os.path.join(OUT, 'phase15c.json')))
    O1 = d3(dict(zip(B.TK, c15['O1'])))
    O2 = d3(dict(zip(B.TK, c15['O2'])))
    O1, O2 = O1 / nrm3(O1), O2 / nrm3(O2)
    j15d = json.load(open(os.path.join(OUT, 'phase15d.json')))
    res = {}
    for N, el in IONS:
        tag = Q15.tag_of(el, N)
        # reference H_PP (seniority) from the own vcm run
        ref = {}
        for L, twoS, m, t in D.term_list(N):
            Hm, lab = B.load_run(f'vcm_d{N}_{t}e_{tag}')
            U = np.array(json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps'][f'vcm_d{N}_{t}e'])
            ref[t] = U.T @ Hm[:m, :m] @ U
        r = {}
        for space in all_configs(N, el):
            bl = blocks(space, N, tag)
            dev = max(float(np.abs(b['U'].T @ b['H'][np.ix_(b['P'], b['P'])] @ b['U'] - ref[b['t']]).max()) for b in bl)
            by, info = {}, []
            for b in bl:
                Bk, inf = F.heff_block(b, b['H'])
                by[(b['L'], b['twoS'])] = Bk
                info.append(inf)
            x = B.vec_from_blocks(N, by)
            dec = P14.decompose(N, x)
            t = d3(dec['c'])
            T1, T2 = S.ip(3, t, O1), S.ip(3, t, O2)
            r[space] = dict(T1=T1, T2=T2, t=nrm3(t), out=nrm3(t - T1 * O1 - T2 * O2) / max(nrm3(t), 1e-300),
                            ang=float(np.degrees(np.arctan2(T2, T1))), c=dec['c'], f=dec['n']['f'],
                            wP=min(i['wP'] for i in info), below=sum(i['below'] for i in info), hpp_dev=dev)
        res[f'{N} {el}'] = r
        print(f'{el} d{N} {time.time() - t0:.0f} s', flush=True)
    w = ['# Phase 15e: completeness of the paths; virtual-space convergence of T1, T2',
         open(__file__).read().split('"""')[1].split('-> results')[0].strip(), '']
    w += dims()
    w.append('\n## Per path and step: T1, T2 (mEh, d3-normalized O2\', O2\'\'), angle, out of plane; P weight, '
             'states below; max |H_PP - H_PP(vcm run)| (Eh)')
    w.append('| ion | space | T1 | T2 | angle | out | min P | below | H_PP check |')
    w.append('|---|---|---|---|---|---|---|---|---|')
    for key, r in res.items():
        N, el = key.split()
        for sp, x in r.items():
            w.append(f'| {el} d{N} | {sp} | {1e3 * x["T1"]:+.4f} | {1e3 * x["T2"]:+.4f} | {x["ang"]:+.2f} | {x["out"]:.4f} | '
                     f'{x["wP"]:.3f} | {x["below"]} | {x["hpp_dev"]:.1e} |')
    w.append('\n## Totals: t(a)+t(b)+t(c)+t(d) at the largest size vs at the 15d size, and vs the 15d combined space')
    w.append('| ion | 15d-size sum (s0 + d2 + g1 + c3s): T1, T2 | largest sum (s7 + d5 + g3 + c1s): T1, T2 | 15d combined T1, T2 | '
             'T1 largest / 15d combined | change of T2 (largest - 15d size) |')
    w.append('|---|---|---|---|---|---|')
    tot = {}
    for key, r in res.items():
        N, el = key.split()
        small = [r['d2'], r['g1'], r['c3s']]
        large = [r['s7'], r['d5'], r['g3'], r['c1s']]
        S1 = (sum(x['T1'] for x in small), sum(x['T2'] for x in small))
        S2 = (sum(x['T1'] for x in large), sum(x['T2'] for x in large))
        cmb = j15d[f'{N} {el}']['all']
        tot[key] = dict(small=S1, large=S2, comb=(cmb['T1'], cmb['T2']))
        w.append(f'| {el} d{N} | {1e3 * S1[0]:+.3f}, {1e3 * S1[1]:+.3f} | {1e3 * S2[0]:+.3f}, {1e3 * S2[1]:+.3f} | '
                 f'{1e3 * cmb["T1"]:+.3f}, {1e3 * cmb["T2"]:+.3f} | {S2[0] / cmb["T1"]:.3f} | {1e3 * (S2[1] - S1[1]):+.3f} |')
    w.append('\nStep increments (mEh): s path T1 by k = 1, 2, 3, 7; d path T2 at 11-12d, +13d, all; g path T2 at 11g, +12g, '
             'all; core T1 3s, +2s, +1s')
    for key, r in res.items():
        N, el = key.split()
        w.append(f'- {el} d{N}: s T1 ' + ' / '.join(f'{1e3 * r[f"s{k}"]["T1"]:+.3f}' for k in (1, 2, 3, 7)) +
                 '; d T2 ' + ' / '.join(f'{1e3 * r[f"d{k}"]["T2"]:+.3f}' for k in (2, 3, 5)) +
                 '; g T2 ' + ' / '.join(f'{1e3 * r[f"g{k}"]["T2"]:+.3f}' for k in (1, 2, 3)) +
                 '; core T1 ' + ' / '.join(f'{1e3 * r[c]["T1"]:+.3f}' for c in ('c3s', 'c2s', 'c1s')))
    # additivity check Cr3+
    r = res['3 Cr']
    cmb = j15d['3 Cr']['all']
    pred = (cmb['T1'] + r['s1']['T1'], cmb['T2'] + r['s1']['T2'])
    a = r['comb+s']
    dd = np.hypot(a['T1'] - pred[0], a['T2'] - pred[1]) / np.hypot(a['T1'], a['T2'])
    w.append(f'\n## Additivity check (Cr3+ d3): 15d combined + 11s in one space: T1 {1e3 * a["T1"]:+.3f}, T2 {1e3 * a["T2"]:+.3f}; '
             f'15d combined + s1 path: T1 {1e3 * pred[0]:+.3f}, T2 {1e3 * pred[1]:+.3f}; in-plane difference / |t| = {dd:.4f}')
    res['_tot'] = tot
    res['_add'] = dict(comb_s=(a['T1'], a['T2']), pred=pred, rel=dd)
    json.dump(res, open(os.path.join(OUT, 'phase15e.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    w.append(f'\nWall time {time.time() - t0:.0f} s.')
    open(os.path.join(OUT, 'phase15e.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dims':
        dims()
    elif cmd == 'prep':
        prep()
    elif cmd == 'run':
        run()
    elif cmd == 'analyze':
        analyze()
