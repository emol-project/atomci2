#!/usr/bin/env python3
"""Phase 10 recon trial: [Ne] frozen, 3s 3p opened, V QZP, orbitals = set 2 of runs/vq_orb (Phase 9b).

trial.py gen NAME     NAME = vc_<space>_<ion>_<term>, space in S, C1, C2, C3a, C3, C3u (= C3a with uniform parents), T2a, T2r, C3r (Phase 10c), C3k, C3kT (Phase 10d; T2r/C3r/C3k/C3kT need --vectors-from vq_cno) (scripts/phase10_recon/counts.py),
                      ion v5p (d0), v4p (d1), v3p (d2) or v2p (d3), term e.g. 3Fe; writes inputs/NAME/
Run with scripts/run_atomci.py NAME --vectors-from vq_orb.

Deck patches (traps as in scripts/gen_v.py): ITNEL (atinputs counts 2 electrons per frozen shell),
MPOB written from the orbital names (core n -> n - l), outer counts chosen not to clash with labels,
shells with l >= 4 and >= 3 electrons excluded.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
_C.setup_path()
_C.setup_path()
import counts as CN  # noqa: E402
import gen_v  # noqa: E402

NBF = gen_v.nbf('vq')                 # (10, 8, 6, 4, 3, 2)
V = (7, 6, 6, 4, 3, 2)                # complete valence/virtual space per l (3d in d)
ION = {'v5p': 0, 'v4p': 1, 'v3p': 2, 'v2p': 3}
LET = 'spdfgh'
_KF = os.path.join(ROOT, 'results', 'phase10', 'phase10c_keep.json')
KEEP = set(json.load(open(_KF))['keep']) if os.path.exists(_KF) else None   # Phase 10c kept virtual NOs (T2r, C3r)


def inp_text(ang, mult, nd, record):
    ind = '\n'.join(f'      - record: {record}, l: {l}, first: 1, last: {n}' for l, n in enumerate(NBF))
    return f"""title:  " V core-open d{nd} "
deck:   be.deck
csf:    be.csf
atom:
  z:     23
  ang:   {ang}
  spin:  {mult}
files:
  cfptable:  ./CFPTBL
  basis:     v_qzp.bas
  format:    atomci
  mcp:       mcp.dat
  open:
    - unit: 27, file: be.NO, format: binary, status: old
orbital:
  frozen_core:  1S, 2S, 2P
  active:       3S, 3P, 3D
  outer:        7S, 6P, 5D, 4F, 3G, 2H
reference:
  - scf: true, occ: 3S*2 3P*6{f' 3D*{nd}' if nd else ''}
ctrl:
  ai:  type: 2, idprnt: 0
  wf:  rule: 1, npflg: 0
  tr:
    record: 1
    ind:
{ind}
  ci:  ips: 10, maxit: 400, nsdim: 100, nstate: 1
  rs:  print: -0.001
"""


def gen(name):
    m = re.fullmatch(r'vc_(S|C1|C2|C3a|C3u|C3r|C3kT|C3k|C3|T2a|T2r)_(v5p|v4p|v3p|v2p)_(\d)([SPDFGH])e', name)
    assert m, name
    space, nd = m.group(1), ION[m.group(2)]
    mult, L = int(m.group(3)), 'SPDFGH'.index(m.group(4))
    term = f'{mult}{m.group(4)}'
    deck, csf = gen_v.run_atinputs('vq', inp_text(m.group(4), mult, nd, 2), os.path.dirname(_C.ATOMCI_SCRIPTS))
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
    nob, mpob = [], []
    for let in LET:
        orbs = [o for o in seq if o.endswith(let)]
        nob.append(len(orbs))
        mpob += [gen_v.vec(o) for o in orbs]
        assert sorted(gen_v.vec(o) for o in orbs) == list(range(1, len(orbs) + 1)), (let, orbs)
    assert nob == list(NBF), nob
    if space in ('C3k', 'C3kT'):
        Lmap, sp = CN.kspaces(nd, V, term, KEEP)
    elif space.startswith('T'):
        Lmap, sp = CN.tspaces(nd, V, term, keep=KEEP if space == 'T2r' else None)
    else:
        Lmap, sp = CN.spaces(nd, V, term, uniform=space in ('C3u', 'C3r'))
        if space == 'C3r':      # uniform C3a with every virtual electron in the kept NOs (Phase 10c check)
            sp['C3r'] = {c for c in sp['C3a'] if all(o in ('3s', '3p', '3d', '11s') or o in KEEP for o in dict(c))}
    if space == 'C3u':
        space = 'C3a'
    assert set(Lmap) == {o for o in seq if o not in ('1s', '2s', '2p')}, (sorted(Lmap), seq)
    tot, ncfg, per = CN.count(Lmap, sp[space], term)
    ref = frozenset({'3s': 2, '3p': 6, **({'3d': nd} if nd else {})}.items())
    order = ([ref] if ref in per else []) + sorted((c for c in per if c != ref), key=lambda c: tuple(sorted(dict(c).items())))
    lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)) + '  ))'
             for c in order]
    deck = gen_v.sub_all(deck, 'itnel', 18 + nd, 14 + nd)      # atinputs: 2 x 3 frozen shells + 8 + nd
    deck = gen_v.sub_first(deck, 'nrnfg', '1,0,1,1,0,1,1,1')
    wf = deck.index(' &wf\n')
    head, tail = deck[:wf], deck[wf:]
    tail = gen_v.sub_first(tail, 'ispin', mult)
    tail = gen_v.sub_first(tail, 'iorb', 2 * L + 1)
    tail = gen_v.sub_first(tail, 'ipar', 0)
    deck = head + tail
    deck = re.sub(r'^rule=1,$', 'rule=0,', deck, count=1, flags=re.M)
    deck = gen_v.sub_first(deck, 'npflg', 0)
    deck = deck.replace(' &wf\n', ' &wf\nlrps=4000000,\n', 1)
    deck = deck.replace(' &ci\n', ' &ex\nretry=0,\n /\n &ci\n', 1)
    deck = gen_v.sub_all(deck, 'nsdim', tot + 10 if tot <= 3000 else 100)
    deck = gen_v.sub_all(deck, 'nstate', 2 if (nd == 3 and term == '2D') else 1)
    deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
    deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
    deck = gen_v.sub_all(deck, 'nind', len(nob))
    deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{n}' for l, n in enumerate(nob)))
    deck = re.sub(r"^title='.*',$", f"title=' {name} ',", deck, count=1, flags=re.M)
    out = os.path.join(ROOT, 'inputs', name)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'be.deck'), 'w').write(deck)
    open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
    print(f'{name}: {tot} CSFs in {ncfg} configurations; coupling sequence {" ".join(seq)}')


if __name__ == '__main__':
    assert sys.argv[1] == 'gen'
    for nm in sys.argv[2:]:
        gen(nm)
