#!/usr/bin/env python3
"""Phase 9 vanadium decks ([Ar] frozen), generated with atinputs and patched.

Runs (inputs/<name>/be.deck + be.csf), <b> = basis prefix: v (basis/v.bas, Sapporo-TZP-2012-diffuse)
or vq (basis/v_qzp.bas, Sapporo-QZP-2012-diffuse):
  <b>_hf_<core>               HF (&MC) of the core-defining state + CI of its single CSF
                              core = v2 (V2+ 3d3 4F) or v3 (V3+ 3d2 3F)
  <b>_<ion>_<term>_<orb>[_r<spec>|_s<spec>]  valence full CI (18 core electrons frozen) over ALL
                              configurations of the valence electrons in the complete
                              valence space (TZP: 6s5p5d3f2g, 3d included); shells with l >= 4
                              and 3 electrons (g^3, h^3) excluded
  ion = v4p (d1), v3p (d2), v2p (d3); term e.g. 4Fe; orb = hfv2 (set 1 of runs/<b>_hf_v2),
  ivo (set 2 of runs/<b>_orb, scripts/v_orb.py: virtual s = IVOs of 3d2(3F)ns 4F), hfv3
  (set 1 of runs/<b>_hf_v3)

Traps (Phase 9 recon), all patched here and asserted:
  * atinputs ITNEL counts 2 electrons per frozen shell (13 instead of 21)
  * the coupling sequence lists the core as 1s 2s 2p 3s 3p, so the reversed MPOB
    would swap 1s/3s and 2p/3p: MPOB is written explicitly from the orbital names
  * outer counts are parsed as labels: asserted not to clash with the core and 3D labels
  * ATOMCI has no CFPs for g^3 and silently drops such configurations: excluded here
    (h^3 likewise, by assumption; the CSF count check of every run catches any other drop)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import counting as K  # noqa: E402
from atomci_heff import gto  # noqa: E402

BASES = {'v': 'v.bas', 'vq': 'v_qzp.bas'}
LET = 'spdfghik'
CORE = {0: 3, 1: 2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
LCH = 'SPDFGHIK'
ION = {'v4p': 1, 'v3p': 2, 'v2p': 3, 'v1p': 4, 'v0p': 5}
HF = {'v2': ('F', 4, 3), 'v3': ('F', 3, 2)}          # core-defining state: L letter, 2S+1, d electrons
ORB = {'hfv2': 1, 'ivo': 2, 'hfv3': 1}     # record of the orbital set


def nbf(b):
    """Contracted functions per l of basis prefix b (trailing zeros dropped)."""
    ls = [l for l, _, _ in gto.read_bas(os.path.join(_C.BASIS, BASES[b]))]
    return tuple(ls.count(l) for l in range(max(ls) + 1))


def outer_counts(NBF):
    out = []
    for l, n in enumerate(NBF):
        k = n - CORE[l] - (1 if l == 2 else 0)
        lab = f'{k}{LET[l].upper()}'
        assert lab not in ('1S', '2S', '3S', '2P', '3P', '3D'), f'outer count {lab} clashes with an orbital label'
        out.append(lab)
    return ', '.join(out)


def inp_text(b, ang, mult, nd, mc, record):
    NBF = nbf(b)
    ind = '\n'.join(f'      - record: {record}, l: {l}, first: 1, last: {n}' for l, n in enumerate(NBF))
    return f"""title:  " V d{nd} "
deck:   be.deck
csf:    be.csf
atom:
  z:     23
  ang:   {ang}
  spin:  {mult}
files:
  cfptable:  ./CFPTBL
  basis:     {BASES[b]}
  format:    atomci
  mcp:       mcp.dat
  open:
    - unit: 27, file: be.NO, format: binary, status: {'unknown' if mc else 'old'}
orbital:
  frozen_core:  1S, 2S, 2P, 3S, 3P
  active:       3D
  outer:        {outer_counts(NBF)}
reference:
  - scf: true, occ: 3D*{nd}
ctrl:
  ai:  type: 2, idprnt: 0
{'  mc:  itmax: 50' if mc else ''}
  wf:  rule: 1, npflg: 0
  tr:
    record: 1
    ind:
{ind}
  ci:  ips: 10, maxit: 400, nsdim: 100, nstate: 1
  rs:  print: -0.001
"""


def run_atinputs(b, text, atomci_repo):
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, 'v.inp'), 'w').write(text)
    shutil.copy(os.path.join(_C.BASIS, BASES[b]), tmp)
    env = dict(os.environ, PYTHONPATH=os.path.join(atomci_repo, 'scripts'), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', '-m', 'atinputs', 'v.inp'], cwd=tmp, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deck, csf = open(os.path.join(tmp, 'be.deck')).read(), open(os.path.join(tmp, 'be.csf')).read()
    shutil.rmtree(tmp)
    return deck, csf


def sub_all(deck, key, val, expect=None):
    hits = re.findall(rf'^{key}=(.*),$', deck, flags=re.M)
    assert hits, key
    if expect is not None:
        assert all(h == str(expect) for h in hits), (key, hits, expect)
    return re.sub(rf'^{key}=.*,$', f'{key}={val},', deck, flags=re.M)


def sub_first(deck, key, val):
    assert re.search(rf'^{key}=.*,$', deck, flags=re.M), key
    return re.sub(rf'^{key}=.*,$', f'{key}={val},', deck, count=1, flags=re.M)


def sub_block(deck, key, val):
    m = re.search(rf'^{key}=([\d,\n]+?),\n /', deck, flags=re.M)
    assert m, key
    return deck[:m.start()] + f'{key}={val},\n /' + deck[m.end():]


def vec(o):
    """Vector index of an orbital name: core n -> n - l (3d -> 1), outer 1k<l> -> CORE + [d: 1] + (k - 10)."""
    n, l = int(o[:-1]), LET.index(o[-1])
    return n - l if n < 11 else CORE[l] + (1 if l == 2 else 0) + (n - 10)


def parse(name):
    m = re.fullmatch(r'(v|vq)_hf_(v2|v3)', name)
    if m:
        return dict(b=m.group(1), kind='hf', core=m.group(2))
    m = re.fullmatch(r'(v|vq)_(v4p|v3p|v2p|v1p|v0p)_(\d)([SPDFGHI])e_(hfv2|ivo|hfv3)(_b|_[rs](?:\d[spdfgh])+)?', name)
    if not m:
        raise SystemExit(f'bad name {name}')
    suf = m.group(6) or ''
    return dict(b=m.group(1), kind='ci', nd=ION[m.group(2)], mult=int(m.group(3)), L=LCH.index(m.group(4)),
                orb=m.group(5), level_b=suf == '_b', reduced=reduced_orbitals(suf[2:]) if suf[:2] in ('_r', '_s') else None,
                singles=suf.startswith('_s'))


def reduced_orbitals(spec):
    """Orbital names of a reduced valence space, e.g. '3d1g' -> 3d 11d 12d 11g (counts per l, 3d counted in d;
    the first orbitals of each l in vq_orb order, as Phase 13a/13b)."""
    names = {'3d'}
    for n, l in re.findall(r'(\d)([spdfgh])', spec):
        names |= {f'{10 + k}{l}' for k in range(1, int(n) - (1 if l == 'd' else 0) + 1)}
    return names


def make(name, atomci_repo):
    p = parse(name)
    mc = p['kind'] == 'hf'
    if mc:
        ang, mult, nd = HF[p['core']]
        L, record = LCH.index(ang), 1
    else:
        nd, mult, L = p['nd'], p['mult'], p['L']
        ang, record = LCH[L], ORB[p['orb']]
    b, NBF = p['b'], nbf(p['b'])
    deck, csf = run_atinputs(b, inp_text(b, ang, mult, nd, mc, record), atomci_repo)
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
    nob, mpob = [], []
    for let in LET[:len(NBF)]:
        orbs = [o for o in seq if o.endswith(let)]
        nob.append(len(orbs))
        mpob += [vec(o) for o in orbs]
        assert sorted(vec(o) for o in orbs) == list(range(1, len(orbs) + 1)), (let, orbs)
    core = [o for o in ('1s', '2s', '3s', '2p', '3p')]
    assert all(o in seq for o in core + ['3d']), seq
    assert nob == list(NBF), nob
    val = [(o, LET.index(o[-1])) for o in seq if o not in core]
    twoS = mult - 1
    if mc:
        space = {frozenset({'3d': nd}.items()): 1}
    else:
        # level_b (Phase 13a): 3d/4s complex only, i.e. configurations within {3d, 11s}
        # reduced (Phase 13b): all configurations within the named orbitals (suffix _r<spec>, e.g. _r3d1g)
        # singles (Phase 13d, suffix _s<spec>): as _r<spec>, but at most one electron outside 3d (3d^N, 3d^(N-1) x)
        sel = (('3d', '11s') if p.get('level_b') else p['reduced']) if (p.get('level_b') or p.get('reduced')) else None
        space = K.count_space([x for x in val if x[0] in sel] if sel else val, nd, L, twoS, 0,
                              keep=lambda occ: all(q < 3 for o, q in occ.items() if o[-1] in 'gh') and
                              (not p['singles'] or occ.get('3d', 0) >= nd - 1))
    ref = frozenset({'3d': nd}.items())
    order = ([ref] if ref in space else []) + sorted((c for c in space if c != ref),
                                                     key=lambda c: tuple(sorted(dict(c).items())))
    lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)) + '  ))'
             for c in order]
    ncsf = sum(space.values())
    nel = 18 + nd
    deck = sub_all(deck, 'itnel', nel, 10 + nd)          # atinputs: 2 per frozen shell + nd
    if mc:
        deck = sub_first(deck, 'nrnfg', '1,1,1,1,0,1,1,1')
    else:
        deck = sub_first(deck, 'nrnfg', '1,0,1,1,0,1,1,1')
        wf = deck.index(' &wf\n')
        head, tail = deck[:wf], deck[wf:]
        tail = sub_first(tail, 'ispin', mult)
        tail = sub_first(tail, 'iorb', 2 * L + 1)
        tail = sub_first(tail, 'ipar', 0)
        deck = head + tail
    deck = re.sub(r'^rule=1,$', 'rule=0,', deck, count=1, flags=re.M)
    deck = sub_first(deck, 'npflg', 0)
    deck = deck.replace(' &wf\n', ' &wf\nlrps=4000000,\n', 1)
    deck = deck.replace(' &ci\n', ' &ex\nretry=0,\n /\n &ci\n', 1)
    deck = sub_all(deck, 'nsdim', ncsf + 10 if ncsf <= 3000 else 100)
    deck = sub_all(deck, 'nstate', 1 if mc else max(1, K.n_csf([(2, nd)], L, twoS)))
    deck = sub_all(deck, 'nob', ','.join(map(str, nob)))
    deck = sub_block(deck, 'mpob', ','.join(map(str, mpob)))
    deck = sub_all(deck, 'nind', sum(1 for n in nob if n))
    deck = sub_block(deck, 'ind', ',\n'.join(f'{record},{l},1,{n}' for l, n in enumerate(nob) if n))
    deck = re.sub(r"^title='.*',$", f"title=' {name} ',", deck, count=1, flags=re.M)
    out = os.path.join(ROOT, 'inputs', name)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'be.deck'), 'w').write(deck)
    open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
    return ncsf, len(space), seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--atomci-repo', default=os.path.dirname(_C.ATOMCI_SCRIPTS))
    a = ap.parse_args()
    for name in a.names:
        n, c, seq = make(name, a.atomci_repo)
        print(f'{name}: {n} CSFs in {c} configurations; coupling sequence {" ".join(seq)}')


if __name__ == '__main__':
    main()
