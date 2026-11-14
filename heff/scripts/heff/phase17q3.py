#!/usr/bin/env python3
"""Phase 17 Q3: i functions (l = 6) for Cr3+ in the plain Sapporo-QZP-2012 set.

Exponents, by the 15t rule (one uncontracted primitive per test; 15t added g 4.7 = 2 x 2.374 and h 4.4 = 2 x 2.214, i.e.
about twice the tightest existing exponent of that l). The set has no i, so the single i takes the tightest h
exponent (2.2136970; in Sapporo-QZP the tightest exponent falls only slowly with l: g 2.374, h 2.214), and the
two-i variant adds 4.4 (twice that, as the 15t h test).
  crti1  cr.without-diffuse.bas + I 2.2136970          tag crti13
  crti2  cr.without-diffuse.bas + I 4.4, I 2.2136970   tag crti23
Orbitals: the 15a recipe (phase15f.own_set): s, p, d from the configuration-average HF (unchanged by an unoccupied
l), f..i = bare-nucleus h eigenvectors. be.NO with 7 symmetries (WMEGU reads NAME, NSET, ISYM, IBF(20), IOB(20)).
Test (l = 6 through ATOMCI): 1s2 2s2 2p6 3s2 3p6 3d 11x (x = h control, i), terms 1G and 3G, one CSF each; the CI
energy against the Slater formula from our own radial integrals.
Follow-up test2: two electrons in one l (11x2, x = g control, i), L + S even, against analytic2.
Commands: bases / orb / test / test2 / vcm"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase17')
BASE = 'cr.without-diffuse.bas'
EXP_I1 = 2.2136970
ADD = {'crti1': [EXP_I1], 'crti2': [4.4, EXP_I1],
       'crti3': [1.0], 'crti4': [4.4], 'crti5': [EXP_I1, 1.0]}       # 29 Oct: i-exponent check (item 1c)
TAGS = {'crti1': 'crti13', 'crti2': 'crti23', 'crti3': 'crti33', 'crti4': 'crti43', 'crti5': 'crti53'}
Z = 24
LET = 'SPDFGHIK'     # 29 Oct: K (l = 7) for the item 1c k test


def bas_name(ion):
    return 'cr_qzp2012_add_' + '_'.join(f'i{e:g}' for e in ADD[ion]) + '.bas'


def bases(ions=None):
    from atomci_heff import gto
    src = open(os.path.join(_C.BASIS, BASE)).read().splitlines()
    end = next(k for k, x in enumerate(src) if x.strip().startswith(')'))
    for ion, ex in ADD.items():
        if ions and ion not in ions:
            continue
        lines = list(src)
        lines[0] = (f'" Cr  Sapporo-QZP-2012 plain (cr.without-diffuse.bas) + uncontracted '
                    + ', '.join(f'I {e:g}' for e in ex) + ' (Phase 17 Q3)')
        lines[end:end] = [f'  I  1  1.0  {e:14.7f}  1.0000000' for e in ex]
        p = os.path.join(_C.BASIS, bas_name(ion))
        assert not os.path.exists(p), p
        open(p, 'w').write('\n'.join(lines) + '\n')
        a, b = gto.read_bas(os.path.join(_C.BASIS, BASE)), gto.read_bas(p)
        assert [f[0] for f in b].count(6) == len(ex) and len(b) == len(a) + len(ex)
        print(ion, bas_name(ion), len(a), '->', len(b), 'functions')


def write_be_no(C, tg, nbf):
    """be.NO (two identical sets) for nsym = len(nbf) <= 20: header of runs/vq_orb/be.NO with ISYM, IBF, IOB set."""
    import phase15f as F
    from atomci_heff.reader import read_vectors
    rec = F.records(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'))
    nl = len(nbf)
    hdrs = []
    for s in (0, 1):
        hdr = rec[s * 7]
        iv = list(struct.unpack(f'<{(len(hdr) - 72) // 4}i', hdr[72:]))
        assert iv[3] == 6 and iv[4:10] == list(F.O.NBF) and iv[24:30] == list(F.O.NBF)
        iv[3] = nl
        for l in range(nl):
            iv[4 + l] = iv[24 + l] = nbf[l]
        hdrs.append(hdr[:72] + struct.pack(f'<{len(iv)}i', *iv))
    work = os.path.join(ROOT, 'runs', f'vq_orb_{tg}')
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, 'be.NO'), 'wb') as fo:
        for s in (0, 1):
            fo.write(struct.pack('<i', len(hdrs[s])) + hdrs[s] + struct.pack('<i', len(hdrs[s])))
            for l in range(nl):
                r = struct.pack('<i', l) + C[l].T.astype('<f8').tobytes()
                fo.write(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)))
    chk = read_vectors(os.path.join(work, 'be.NO'), nbf)
    assert all(np.array_equal(chk[k][l], C[l]) for k in (0, 1) for l in range(nl))


def orb(ions=None):
    import phase15f as F
    from atomci_heff.reader import read_vectors
    bs0 = F.Bas(BASE, Z)
    R = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb_crnd3', 'be.NO'), bs0.nbf)[0]
    jp = os.path.join(OUT, 'phase17q3_orb.json')
    js = json.load(open(jp)) if ions else {}
    for ion in ADD:
        if ions and ion not in ions:
            continue
        bs = F.Bas(bas_name(ion), Z)
        C, inf = F.own_set(bs, 3)
        inf['nbf'] = bs.nbf
        inf['spd_max_dev_vs_crnd3'] = max(float(np.abs(C[l] * np.sign(np.sum(C[l] * R[l], 0)) - R[l]).max())
                                          for l in range(3))
        inf['fgh_max_dev_vs_crnd3'] = max(float(np.abs(C[l] * np.sign(np.sum(C[l] * R[l], 0)) - R[l]).max())
                                          for l in range(3, 6))
        inf['S_min_eig_i'] = float(np.linalg.eigvalsh(bs.S[6]).min())
        inf['r_virt_i'] = [float(np.sum(bs.P(6, C[6][:, k]) ** 2 * bs.r * bs.w)) for k in range(bs.nbf[6])]
        # fix signs of s..h to crnd3 so that every shared orbital is bitwise the crnd3 one up to round-off
        for l in range(6):
            C[l] = C[l] * np.sign(np.sum(C[l] * R[l], 0))
        write_be_no(C, TAGS[ion], bs.nbf)
        js[ion] = inf
        print(ion, TAGS[ion], bs.nbf, f'E {inf["E"]:.8f}', 'dev spd', f'{inf["spd_max_dev_vs_crnd3"]:.1e}',
              'fgh', f'{inf["fgh_max_dev_vs_crnd3"]:.1e}', '<r> i', [round(x, 3) for x in inf['r_virt_i']])
    json.dump(js, open(jp, 'w'), indent=1)


def allcore_deck7(term, bas, occ_open, nstate=1, ipar=0):
    """phase15e.allcore_deck for any number of symmetries (one reference configuration occ_open on top of
    1s2 2s2 2p6 3s2 3p6)."""
    import gen_v
    from atomci_heff import gto
    ls = [f[0] for f in gto.read_bas(os.path.join(_C.BASIS, bas))]
    nbf = [ls.count(l) for l in range(max(ls) + 1)]
    core = {**gen_v.CORE, 6: 0, 7: 0}
    outer = ', '.join(f'{n - core[l] - (1 if l == 2 else 0)}{LET[l]}' for l, n in enumerate(nbf))
    txt = f"""title:  " q3test "
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
  - scf: true, occ: 1S*2 2S*2 2P*6 3S*2 3P*6 {occ_open}
ctrl:
  ai:  type: 2, idprnt: 0
  wf:  rule: 1, npflg: 0
  tr:
    record: 1
    ind:
      - record: 2, l: 0, first: 1, last: {nbf[0]}
  ci:  ips: 10, maxit: 400, nsdim: 20, nstate: {nstate}
  rs:  print: -0.001
"""
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, 'x.inp'), 'w').write(txt)
    shutil.copy(os.path.join(_C.BASIS, bas), tmp)
    env = dict(os.environ, PYTHONPATH=_C.ATOMCI_SCRIPTS, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-B', '-m', 'atinputs', 'x.inp'], cwd=tmp, env=env, capture_output=True,
                       text=True)
    if r.returncode:
        raise SystemExit(f'atinputs failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    deck, csf = open(os.path.join(tmp, 'be.deck')).read(), open(os.path.join(tmp, 'be.csf')).read()
    shutil.rmtree(tmp)
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGHIK]', header.split(')')[0])]
    nob, mpob = [], []
    for let in LET[:len(nbf)].lower():
        orbs = [o for o in seq if o.endswith(let)]
        nob.append(len(orbs))
        mpob += [gen_v.vec(o) if let in 'spdfgh' else int(o[:-1]) - 10 for o in orbs]
    deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
    deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
    deck = gen_v.sub_all(deck, 'nind', len(nob))
    deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{k}' for l, k in enumerate(nob)))
    deck = gen_v.sub_first(deck, 'nrnfg', '1,0,1,1,0,1,1,1')
    wf = deck.index(' &wf\n')
    head, tail = deck[:wf], deck[wf:]
    tail = gen_v.sub_first(tail, 'ispin', int(term[0]))
    tail = gen_v.sub_first(tail, 'iorb', 2 * 'SPDFGHIKLMN'.index(term[1]) + 1)
    tail = gen_v.sub_first(tail, 'ipar', ipar)     # 29 Oct: 1 for odd configurations (3d 11h, 3d 11k)
    deck = head + tail
    deck = re.sub(r'^rule=1,$', 'rule=0,', deck, count=1, flags=re.M)
    deck = gen_v.sub_first(deck, 'npflg', 0)
    deck = deck.replace(' &wf\n', ' &wf\nlrps=4000000,\n', 1)
    deck = deck.replace(' &ci\n', ' &ex\nretry=0,\n /\n &ci\n', 1)
    return deck, csf


def w3j0(a, k, b):
    from sympy.physics.wigner import wigner_3j
    return float(wigner_3j(a, k, b, 0, 0, 0))


def sixj(a, b, c, d, e, f):
    from sympy.physics.wigner import wigner_6j
    return float(wigner_6j(a, b, c, d, e, f))


def analytic(bas, x, twoS1, L):
    """E(1s2 2s2 2p6 3s2 3p6 3d 11x, (2S+1)L) from our integrals: E_av minus the average d-x pair energy plus the
    term's pair energy (two non-equivalent electrons)."""
    import phase15f as F
    import phase15m as M
    from atomci_heff import integrals as I
    from atomci_heff.reader import read_vectors
    bs = F.Bas(bas, Z)
    tg = TAGS[next(k for k in ADD if bas_name(k) == bas)] if bas != BASE else 'crnd3'
    C = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tg}', 'be.NO'), bs.nbf)[0]
    lx = 'spdfghik'.index(x)
    col = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '2p': (1, 0), '3p': (1, 1), '3d': (2, 0), '11' + x: (lx, 0)}
    P, h = {}, {}
    for o, (l, k) in col.items():
        c = C[l][:, k]
        P[o], h[o] = bs.P(l, c), float(c @ bs.h[l] @ c)
    occ = {'1s': 2, '2s': 2, '2p': 6, '3s': 2, '3p': 6, '3d': 1, '11' + x: 1}
    lof = {o: 'spdfghik'.index(o[-1]) for o in occ}

    def R(a, b, c, d, k):
        return float(I.slater(P[a], P[c], P[b], P[d], k, bs.x, bs.r))
    E = sum(q * h[o] for o, q in occ.items())
    os_ = list(occ)
    for i, a in enumerate(os_):
        la, qa = lof[a], occ[a]
        Fs = R(a, a, a, a, 0) - (2 * la + 1) / (4 * la + 1) * sum(
            w3j0(la, k, la) ** 2 * R(a, a, a, a, k) for k in range(2, 2 * la + 1, 2))
        E += qa * (qa - 1) / 2 * Fs
        for b in os_[i + 1:]:
            lb, qb = lof[b], occ[b]
            G = sum(w3j0(la, k, lb) ** 2 * R(a, b, b, a, k) for k in range(abs(la - lb), la + lb + 1, 2))
            E += qa * qb * (R(a, b, a, b, 0) - 0.5 * G)
    d, v, l1, l2 = '3d', '11' + x, 2, lx
    S = (twoS1 - 1) / 2
    avg = R(d, v, d, v, 0) - 0.5 * sum(w3j0(l1, k, l2) ** 2 * R(d, v, v, d, k) for k in range(abs(l1 - l2), l1 + l2 + 1, 2))
    red = lambda a, k, b: (-1) ** a * np.sqrt((2 * a + 1) * (2 * b + 1)) * w3j0(a, k, b)    # noqa: E731
    term = R(d, v, d, v, 0)
    for k in range(2, min(2 * l1, 2 * l2) + 1, 2):
        # 29 Oct: (-1)^(l1 + l2) added to the direct term (the old (-1)^L was right only for even l1 + l2)
        term += (-1) ** (L + l1 + l2) * red(l1, k, l1) * red(l2, k, l2) * sixj(l1, l2, L, l2, l1, k) * R(d, v, d, v, k)
    for k in range(abs(l1 - l2), l1 + l2 + 1, 2):
        term += (-1) ** S * red(l1, k, l2) ** 2 * sixj(l1, l2, L, l1, l2, k) * R(d, v, v, d, k)
    return E - avg + term


def analytic2(bas, x, L):
    """E(1s2 2s2 2p6 3s2 3p6 11x2, (2S+1)L), L + S even, from our integrals: E_av minus the average x2 pair energy plus
    the term's pair energy F0 + sum_k (-1)^L <x||C^k||x>^2 {x x L; x x k} F^k (two equivalent electrons)."""
    import phase15f as F
    from atomci_heff import integrals as I
    from atomci_heff.reader import read_vectors
    bs = F.Bas(bas, Z)
    tg = TAGS[next(k for k in ADD if bas_name(k) == bas)]
    C = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tg}', 'be.NO'), bs.nbf)[0]
    lx = 'spdfghik'.index(x)
    col = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '2p': (1, 0), '3p': (1, 1), '11' + x: (lx, 0)}
    P, h = {}, {}
    for o, (l, k) in col.items():
        c = C[l][:, k]
        P[o], h[o] = bs.P(l, c), float(c @ bs.h[l] @ c)
    occ = {'1s': 2, '2s': 2, '2p': 6, '3s': 2, '3p': 6, '11' + x: 2}
    lof = {o: 'spdfghik'.index(o[-1]) for o in occ}

    def R(a, b, c, d, k):
        return float(I.slater(P[a], P[c], P[b], P[d], k, bs.x, bs.r))
    E = sum(q * h[o] for o, q in occ.items())
    os_ = list(occ)
    for i, a in enumerate(os_):
        la, qa = lof[a], occ[a]
        Fs = R(a, a, a, a, 0) - (2 * la + 1) / (4 * la + 1) * sum(
            w3j0(la, k, la) ** 2 * R(a, a, a, a, k) for k in range(2, 2 * la + 1, 2))
        E += qa * (qa - 1) / 2 * Fs
        for b in os_[i + 1:]:
            lb, qb = lof[b], occ[b]
            G = sum(w3j0(la, k, lb) ** 2 * R(a, b, b, a, k) for k in range(abs(la - lb), la + lb + 1, 2))
            E += qa * qb * (R(a, b, a, b, 0) - 0.5 * G)
    v = '11' + x
    avg = R(v, v, v, v, 0) - (2 * lx + 1) / (4 * lx + 1) * sum(
        w3j0(lx, k, lx) ** 2 * R(v, v, v, v, k) for k in range(2, 2 * lx + 1, 2))
    term = R(v, v, v, v, 0) + sum((-1) ** L * (2 * lx + 1) ** 2 * w3j0(lx, k, lx) ** 2 * sixj(lx, lx, L, lx, lx, k)
                                  * R(v, v, v, v, k) for k in range(2, 2 * lx + 1, 2))
    return E - avg + term


def test2():
    """Q3 follow-up: two electrons in one l (the i2 coupling, not covered by test()): single-CSF runs of
    1s2 2s2 2p6 3s2 3p6 11x2 in the crti1 basis, x = g (control) and x = i, terms with L + S even (one CSF each);
    E = H[0, 0] of the exported H against analytic2. Output phase17q3_test2.json."""
    import phase15j as J
    bas, tg = bas_name('crti1'), TAGS['crti1']
    TL = 'SPDFGHIKLMN'
    res = {}
    for x, terms in (('g', ('1S', '3P', '1G', '3H')), ('i', ('1S', '3P', '1D', '1G', '3H', '1I', '3K', '1L', '3M', '1N'))):
        for t in terms:
            dst = f'v17q3_test_{x}2_{t}e_{tg}'
            deck, csf = allcore_deck7(t, bas, '3D*2')
            header = csf.split(' ))\n', 1)[0] + ' ))\n'
            seq = [y.lower() for y in re.findall(r'\d+[SPDFGHIK]', header.split(')')[0])]
            occ = {'1s': 2, '2s': 2, '2p': 6, '3s': 2, '3p': 6, '11' + x: 2}
            assert set(occ) <= set(seq)
            csf = header + J.wrap80(['  ' + f'{o[:-1]}{o[-1].upper()}* {occ[o]}' for o in seq if o in occ]) + '\n )\n'
            deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
            out = os.path.join(ROOT, 'inputs', dst)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, 'be.deck'), 'w').write(deck)
            open(os.path.join(out, 'be.csf'), 'w').write(csf)
            shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
            r = subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                                f'vq_orb_{tg}'], capture_output=True, text=True,
                               env=dict(os.environ, OMP_NUM_THREADS='1'))
            npz = os.path.join(ROOT, 'results', dst + '.npz')
            e = float(np.load(npz)['H'][0, 0]) if r.returncode == 0 and os.path.exists(npz) else None
            ea = analytic2(bas, x, TL.index(t[1]))
            res[f'{x}2:{t}'] = dict(rc=r.returncode, atomci=e, analytic=ea, diff=(e - ea) if e is not None else None)
            print(x, t, 'rc', r.returncode, 'ATOMCI', e, 'analytic', f'{ea:.10f}',
                  'diff', f'{e - ea:.2e}' if e is not None else '-', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q3_test2.json'), 'w'), indent=1)


def test():
    """Single-CSF runs of 3d 11x in the crti1 basis (7 symmetries): x = g (control, l = 4 known to work), x = i;
    terms with one CSF each. E = H[0, 0] of the exported H against the Slater formula. (3d 11h single-CSF decks
    segfault in ATOMCI after WFMAIN in the standard 6-symmetry basis too, so h is not the control here.)"""
    import numpy as np
    import phase15j as J
    bas, tg = bas_name('crti1'), TAGS['crti1']
    TL = 'SPDFGHIKLMN'
    res = {}
    for x, terms in (('g', ('1G', '3G', '3I')), ('i', ('1G', '3G', '1I', '3I', '3L'))):
        for t in terms:
            dst = f'v17q3_test_d1{x}1_{t}e_{tg}'
            deck, csf = allcore_deck7(t, bas, '3D*2')
            header = csf.split(' ))\n', 1)[0] + ' ))\n'
            seq = [y.lower() for y in re.findall(r'\d+[SPDFGHIK]', header.split(')')[0])]
            occ = {'1s': 2, '2s': 2, '2p': 6, '3s': 2, '3p': 6, '3d': 1, '11' + x: 1}
            assert set(occ) <= set(seq)
            csf = header + J.wrap80(['  ' + f'{o[:-1]}{o[-1].upper()}* {occ[o]}' for o in seq if o in occ]) + '\n )\n'
            deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
            out = os.path.join(ROOT, 'inputs', dst)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, 'be.deck'), 'w').write(deck)
            open(os.path.join(out, 'be.csf'), 'w').write(csf)
            shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
            r = subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                                f'vq_orb_{tg}'], capture_output=True, text=True,
                               env=dict(os.environ, OMP_NUM_THREADS='1'))
            npz = os.path.join(ROOT, 'results', dst + '.npz')
            e = float(np.load(npz)['H'][0, 0]) if r.returncode == 0 and os.path.exists(npz) else None
            ea = analytic(bas, x, int(t[0]), TL.index(t[1]))
            res[f'{x}:{t}'] = dict(rc=r.returncode, atomci=e, analytic=ea, diff=(e - ea) if e is not None else None)
            print(x, t, 'rc', r.returncode, 'ATOMCI', e, 'analytic', f'{ea:.10f}',
                  'diff', f'{e - ea:.2e}' if e is not None else '-', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q3_test.json'), 'w'), indent=1)


def vcm(ions=None):
    """Frozen-[Ne] P-block decks (phase15s_nd.vcm_deck, now with 7 symmetries) for crti13 / crti23, run, and compared
    with vcm_d3_<t>e_crnd3: the occupied orbitals are the same, so the P-block H must agree."""
    import numpy as np
    import phase13d as D
    import phase15s_nd as ND
    jp = os.path.join(OUT, 'phase17q3_vcm.json')
    res = json.load(open(jp)) if ions else {}
    for ion, tg in TAGS.items():
        if ions and ion not in ions:
            continue
        for L, twoS, m, t in D.term_list(3):
            dst, deck, csf = ND.vcm_deck(t, bas_name(ion), tg)
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
            shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
            subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                            f'vq_orb_{tg}'], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
            a = np.load(os.path.join(ROOT, 'results', dst + '.npz'))['H']
            b = np.load(os.path.join(ROOT, 'results', f'vcm_d3_{t}e_crnd3.npz'))['H']
            res[dst] = float(np.abs(a - b).max())
            print(dst, a.shape, 'max |H - H(crnd3)|', f'{res[dst]:.1e}', flush=True)
    json.dump(res, open(jp, 'w'), indent=1)


def split(ion='crti1', percfg=True):
    """Which i configurations carry the k1fhi dT2 (crti13, one i): exact H_eff (des Cloizeaux, no dressing) on the
    sub-space k1fh + one class of i configurations, and k1fh + one configuration, from the stored crti13 H. Classes by
    the partner of the i electron (i alone, i2, s i, p i, d i, f i, g i, h i). Checks: no i -> k1fh (crnd3), all i ->
    the crti13 exact. -> results/phase17/phase17q3_split.json"""
    import time
    tg = TAGS[ion]
    os.environ.update(A_RUNFMT='v17_k1fhi_d3_{t}e_' + tg, A_TAG=tg, A_ION=ion)
    import phase17a as A
    import phase17c_scan as S
    import atomci_heff.approx as AP
    from atomci_heff.sparse import SymLower
    t0 = time.time()
    bl, eav, hf, _ = A.blocks('k1fhi')
    p0 = bl[0]['lab'][0]
    icf = sorted(c for c in eav if c != p0 and any(k.endswith('i') for k in AP.parse_label(c)))

    def partner(c):
        o = AP.parse_label(c)
        v = [(k, q) for k, q in o.items() if int(k[:-1]) >= 11 and not k.endswith('i')]
        ni = sum(q for k, q in o.items() if k.endswith('i'))      # 29 Oct: any i orbitals (two-i bases)
        return 'i2' if ni >= 2 else (v[0][0][-1] + ' i' if v else 'i')

    def holes(c):
        o = AP.parse_label(c)
        h = [(2 - o.get('3s', 0), '3s'), (6 - o.get('3p', 0), '3p'), (3 - o.get('3d', 0), '3d')]
        return ' '.join(n + ('²' if k == 2 else '') for k, n in h if k > 0) or '–'

    cls = {c: partner(c) for c in icf}
    names = ['i', 'i2', 's i', 'p i', 'd i', 'f i', 'g i', 'h i']
    def d1i(c):        # 29 Oct: the 3d -> i singles 3s2 3p6 3d2 ni alone
        o = AP.parse_label(c)
        return (o.get('3s', 0), o.get('3p', 0), o.get('3d', 0)) == (2, 6, 2) and cls[c] == 'i'

    sets = {'none': set(), 'all': set(icf), 'd1i': {c for c in icf if d1i(c)},
            'd1i+i2': {c for c in icf if d1i(c) or cls[c] == 'i2'}} | {n: {c for c in icf if cls[c] == n} for n in names} | \
           ({'cfg ' + c: {c} for c in icf} if percfg else {})
    by = {s: {} for s in sets}
    ncsf = {s: 0 for s in sets}
    for b in bl:
        m, t, U = b['m'], b['t'], b['U']
        op = SymLower.from_npz(np.load(os.path.join(ROOT, 'results', b['name'] + '.npz'), allow_pickle=True))
        lab = b['lab']
        php = U.T @ op.sub(np.arange(m)).dense() @ U
        noi = [i for i in range(op.n) if lab[i] not in sets['all']]
        for s, add in sets.items():
            keep = np.array(sorted(noi + [i for i in range(op.n) if lab[i] in add]))
            sub = op.sub(keep)
            h, inf = S.heff(sub, m, np.arange(m, sub.n), sub.sub(np.arange(m)).dense())
            by[s][t] = hf[t] + U.T @ h @ U - php
            ncsf[s] += sum(1 for i in range(op.n) if lab[i] in add)
        print(t, f'{time.time() - t0:.0f} s', flush=True)
        del op
    res = dict(classes={n: sorted(c for c in icf if cls[c] == n) for n in names}, holes={c: holes(c) for c in icf},
               sets={})
    for s in sets:
        pr = A.params(by[s])
        res['sets'][s] = dict(ncsf=ncsf[s], **{q: float(pr[q]) for q in ('T1', 'T2', 'B', 'C', 'beta')})
    res['wall_s'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, 'phase17q3_split.json' if ion == 'crti1' else f'phase17q5_split_{ion}.json'), 'w'),
              indent=1)
    r = res['sets']
    for s in ['none', 'all', 'd1i', 'd1i+i2'] + names:
        print(f'{s:5s} {r[s]["ncsf"]:6d} T1 {r[s]["T1"]:.4f} T2 {r[s]["T2"]:.4f} dT2 {r[s]["T2"] - r["none"]["T2"]:+.4f}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'split':
        split(*(args[:1] or ['crti1']), percfg=(args[1:2] or ['1'])[0] == '1')
    elif cmd in ('bases', 'orb', 'vcm'):
        {'bases': bases, 'orb': orb, 'vcm': vcm}[cmd](args or None)
    else:
        {'test': test, 'test2': test2}[cmd]()
