#!/usr/bin/env python3
"""Phase 18: Z-generic copy of phase15j.write_deck (d6 decks) for Co3+. phase15j.py stays unchanged. The nuclear charge
comes from phase15m.Z_ (env ION) and the basis from phase15j.BAS6, as phase15n.prep sets it. phase18.py installs this
function as phase15j.write_deck before phase15n.prep. Regression: phase18.py deckcheck (fend decks reproduced bytewise)."""
import os
import re

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)


def write_deck(dst, kind, cf, t, out_root=None):
    """Deck + CSF list for one term: 'vcm' (frozen [Ne], from inputs/vcm_d6_<t>e) or 'allcore' (15e allcore_deck)."""
    import counts as CN
    import gen_v
    import phase13d as D
    import phase15e as E15
    import phase15j as J
    import phase15k as K
    import phase15m as M
    Z = M.Z_
    m = next(mm for L, twoS, mm, tt in D.term_list(6) if tt == t)
    if kind == 'vcm':
        src = os.path.join(ROOT, 'inputs', f'vcm_d6_{t}e')
        deck = open(os.path.join(src, 'be.deck')).read()
        csf = open(os.path.join(src, 'be.csf')).read()
        assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
        deck = deck.replace('z=23,', f'z={Z},').replace('FILE=v_qzp.bas', f'FILE={J.BAS6}')
    else:
        deck, csf = E15.allcore_deck(6, t, Z, J.BAS6)
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
        for let in 'spdfghik'[:1 + max(5, max('spdfghik'.index(o[-1]) for o in seq))]:
            orbs = [o for o in seq if o.endswith(let)]
            nob.append(len(orbs))
            mpob += [gen_v.vec(o) for o in orbs]
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
        deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
        deck = gen_v.sub_all(deck, 'nind', len(nob))
        deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{k}' for l, k in enumerate(nob)))
    deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
    if K.STOP_AFTER_H:
        deck = K.stop_after_h(deck)
    out = os.path.join(out_root or os.path.join(ROOT, 'inputs'), dst)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'be.deck'), 'w').write(deck)
    open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
