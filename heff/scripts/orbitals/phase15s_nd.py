#!/usr/bin/env python3
"""Phase 15s addendum: the Cr3+ best space in Sapporo-QZP-2012 without the diffuse functions (9s7p5d3f2g1h).

Basis basis/cr.without-diffuse.bas = cr.bas minus the most diffuse single-primitive function of each l (derived; the
same rule reproduces the downloaded V plain set exactly, basis/src/cr_sapporo_qzp2012_derived.gamess). Ion 'crnd'
(phase15m.IONS: Z 24, tag crnd3). Orbitals: the 15a recipe (phase15f.own_set; reproduces cr15n3 exactly from cr.bas).
Decks: allcore decks from phase15e.allcore_deck (outer counts from the basis); the frozen-[Ne] HF P-block decks
vcm_d3_<t>e_crnd3 from the 13b vcm decks with every basis-size field adapted (reproduces vcm_d3_<t>e_cr15n3 for cr.bas).
  orb                     own orbitals -> runs/vq_orb_crnd3/be.NO, results/phase15/phase15s_nd_orb.json
  vcm                     HF P-block decks for crnd3 (+ regression against cr15n3)
  (dims / prep / run_names / analyze via phase15s.py with ION=crnd)
d-path T2 by virtual d (both bases): dpath <tag>   15p dpathSD construction (3d -> virtual d, 3d >= N - 2) with each
                        virtual d left out in turn and each one alone, -> phase15s_nd_dpath.json"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase15')
BAS_ND, Z, TAG_ND = 'cr.without-diffuse.bas', 24, 'crnd3'


def orb():
    import phase15f as F
    from atomci_heff.reader import read_vectors
    js = {}
    for bas, tg in (('cr.bas', 'cr15n3'), (BAS_ND, TAG_ND)):
        bs = F.Bas(bas, Z)
        C, inf = F.own_set(bs, 3)
        rv = {}
        for l, let in ((0, 's'), (1, 'p'), (2, 'd'), (3, 'f'), (4, 'g')):
            first = {0: 3, 1: 2, 2: 1}.get(l, 0)
            for k in range(first, bs.nbf[l]):
                P = bs.P(l, C[l][:, k])
                rv[f'{11 + k - first}{let}'] = float(np.sum(P * P * bs.r * bs.w))
        inf.update(nbf=bs.nbf, r_virt=rv)
        if tg == 'cr15n3':     # the recipe reproduces the existing set
            R = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb_cr15n3', 'be.NO'), bs.nbf)[0]
            inf['max_dev_vs_existing'] = max(float(np.abs(C[l] * np.sign(np.sum(C[l] * R[l], 0)) - R[l]).max())
                                             for l in C)
        else:
            F.write_be_no(C, tg, bs.nbf)
            hb = open(os.path.join(ROOT, 'runs', 'hfb_cr3p', 'be.NO'), 'rb').read()
            mine = open(os.path.join(ROOT, 'runs', f'vq_orb_{tg}', 'be.NO'), 'rb').read()
            n = int.from_bytes(hb[:4], 'little')
            inf['header_equals_atomci_hfb_cr3p'] = hb[:8 + n] == mine[:8 + n]
        js[tg] = inf
        print(tg, {k: v for k, v in inf.items() if k not in ('r_virt', 'eps_virt_d')},
              {o: round(r, 3) for o, r in rv.items() if o[-1] in 'df'}, flush=True)
    json.dump(js, open(os.path.join(OUT, 'phase15s_nd_orb.json'), 'w'), indent=1)


def vcm_deck(t, bas, tg, z=Z):
    """13b frozen-[Ne] deck of term t with z, basis and every basis-size field (nob, nbf, mpob, ind, outer header)
    adapted to bas; CSF list unchanged (3s2 3p6 3d3 and 3s 3p6 3d4, no outer orbitals), as vcm_d3_<t>e_<tag>."""
    import gen_v
    import phase14a_orb as O
    import phase15f as F
    src = os.path.join(ROOT, 'inputs', f'vcm_d3_{t}e')
    deck = open(os.path.join(src, 'be.deck')).read()
    csf = open(os.path.join(src, 'be.csf')).read()
    assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
    deck = deck.replace('z=23,', f'z={z},').replace('FILE=v_qzp.bas', f'FILE={bas}')
    nbf = F.nbf_of(bas)
    if nbf != list(O.NBF):
        head, rest = csf.split('  3S   3P   3D\n', 1)
        csf = F.outer_header(bas, z) + '  3S   3P   3D\n' + rest
        old = ','.join(map(str, O.NBF))
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nbf)), expect=old)
        deck = gen_v.sub_all(deck, 'nbf', ','.join(map(str, nbf)), expect=old)
        mp = [int(x) for x in re.search(r'^mpob=([\d,]+),$', deck, flags=re.M).group(1).split(',')]
        cut = np.cumsum([0] + list(O.NBF))
        parts = [[x for x in mp[cut[l]:cut[l + 1]] if x <= nbf[l]] for l in range(len(O.NBF))]
        for l in range(len(O.NBF)):    # 15t: an l larger than in v_qzp.bas (descending order there) is extended
            if nbf[l] > O.NBF[l]:
                # descending run from O.NBF[l], then a fixed tail (s: ..., 3, 1, 2 since 17 Q4-R; 15t: no tail)
                k = next((i for i, x in enumerate(parts[l]) if x != O.NBF[l] - i), len(parts[l]))
                tail = parts[l][k:]
                assert sorted(tail) == list(range(1, len(tail) + 1)) and k + len(tail) == O.NBF[l], (l, parts[l])
                parts[l] = list(range(nbf[l], len(tail), -1)) + tail
        parts += [list(range(nbf[l], 0, -1)) for l in range(len(O.NBF), len(nbf))]     # 17 Q3: i (l = 6)
        if len(nbf) != len(O.NBF):
            deck = gen_v.sub_all(deck, 'nsym', len(nbf), expect=len(O.NBF))
            deck = gen_v.sub_all(deck, 'nind', len(nbf), expect=len(O.NBF))
        assert [len(p) for p in parts] == nbf
        deck = gen_v.sub_first(deck, 'mpob', ','.join(str(x) for p in parts for x in p))
        old_ind = ',\n'.join(f'2,{l},1,{n}' for l, n in enumerate(O.NBF))
        assert f'ind={old_ind},' in deck, t
        deck = deck.replace(f'ind={old_ind},', 'ind=' + ',\n'.join(f'2,{l},1,{n}' for l, n in enumerate(nbf)) + ',')
    lines = csf.split(' ))\n', 1)
    body = lines[1].splitlines()
    assert body[0].strip().startswith('3S* 2') and '3D* 3' in body[0], body[0]
    dst = f'vcm_d3_{t}e_{tg}'
    deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
    return dst, deck, csf


def vcm():
    import phase13d as D
    n = 0
    for L, twoS, m, t in D.term_list(3):
        dst, deck, csf = vcm_deck(t, 'cr.bas', 'cr15n3')
        d0 = os.path.join(ROOT, 'inputs', dst)
        same = open(os.path.join(d0, 'be.deck')).read() == deck and open(os.path.join(d0, 'be.csf')).read() == csf
        print(t, 'cr.bas regression (identical to existing vcm_d3 cr15n3 deck):', same)
        dst, deck, csf = vcm_deck(t, BAS_ND, TAG_ND)
        os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
        open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
        open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
        n += 1
    print(n, 'crnd3 vcm decks')


if __name__ == '__main__':
    {'orb': orb, 'vcm': vcm}[sys.argv[1]]()
