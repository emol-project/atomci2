#!/usr/bin/env python3
"""Phase 18: extend the d3 series (Fe5+, Fe VI) and test the four-body result on a second d6 ion (Co3+, Co IV).
Setup commands (orbitals and P-block decks; the CI runs go through phase18_queue.py and phase17.py with ION=<ion>):
  r2p <tag> <Z> <basis>   <r>_2p of an orbital set, and the crt2 tight pair at that radius (phase17q10 rule)
  fe5vcm                  frozen-[Ne] HF P-block runs of fe5nd3 (phase15s_nd.vcm_deck with Z = 26, Fe plain basis)
  fe5i                    fe5ti1 / fe5ti5 orbitals (phase17q9i recipe, s..h checked against fe5nd3) + P-block runs
  fe5dk                   fe5op / fe5dk: AI-only runs, orbitals, P-block runs (phase17q4r recipe, element 'fe5')
  fe5t2                   the Cr3+ crt2 tight d, f pair at the Fe5+ 2p radius: basis, orbitals, P-block runs
  coorb / coi / codk / cot2  Co3+ orbitals as fend6 / feti / feop+fedk / fet2 (d6: no vcm runs, own_hf uses the runs' PHP)
  dbases                  10 Nov B: Fe, Co plain sets with the d contractions opened (<el>_qzp2012_open_d.bas)
  qdorb [ion ...]         10 Nov A, B: own orbitals coqd6 (co.bas), fedc6, codc6 (opened d) -> phase18_qd_orb.json
  deckcheck               phase18_write_deck with Z = 26 reproduces the stored fend fam decks bytewise
  prep <ion> <spaces>     decks through phase15n.prep with the Z-generic writer
  dims <ion> <spaces>     CSF counts -> results/phase18/phase18_dims.json (run with ION=<ion>)
Output json goes to results/phase18/ (the phase 17 result files stay untouched)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase18')
R2P_CR = 0.2600291745443122                # <r>_2p of crnd3: the crt2 pair (d 45, f 58) sits at this radius
FE5T2 = ('fe5t2', 26, 3, 'fe.without-diffuse.bas', 'fe5t23', 'fe5nd3')


def vcm_runs(bas, tag, z, dk=False):
    import phase13d as D
    import phase15s_nd as ND
    names = []
    for L, twoS, m, t in D.term_list(3):
        dst, deck, csf = ND.vcm_deck(t, bas, tag, z=z)
        os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
        open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
        open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
        names.append(dst)
    if dk:
        import phase17q4r as Q4R
        Q4R.dkpatch(names)
    for dst in names:
        shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
        subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                        f'vq_orb_{tag}'], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
        assert np.load(os.path.join(ROOT, 'results', dst + '.npz'))['H'].size
    print('P-block runs', tag, 'done', flush=True)


def r2p(tag, Z, bas):
    import phase15f as F
    from atomci_heff.reader import read_vectors
    bs = F.Bas(bas, int(Z))
    R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO'), bs.nbf)[0]
    p = bs.P(1, R[1][:, 0])
    r = float(np.sum(p * p * bs.r * bs.w))
    sc = (R2P_CR / r) ** 2
    print(tag, '<r>_2p', r, 'scale', sc, 'D', round(45.0 * sc, 1), 'F', round(58.0 * sc, 1))
    return r, sc


def fe5i():
    import phase17q9i as Q9
    Q9.OUT = OUT
    Q9.EL['fe5'] = (26, 3, 'fe.without-diffuse.bas', 'fe5nd3', 'Fe')
    base = {'fe5ti1': 'fe_qzp2012_add_i2.2137.bas', 'fe5ti5': 'fe_qzp2012_add_i2.2137_i1.bas'}
    Q9.IONS.update(fe5ti1=('fe5', Q9.ADD['ti1'], 'fe5ti13'), fe5ti5=('fe5', Q9.ADD['ti5'], 'fe5ti53'))
    Q9.bas_name = lambda ion: base[ion]
    Q9.orb(list(base))
    Q9.vcm(list(base))


def fe5dk():
    import phase17q4r as Q4R
    Q4R.OUT = OUT
    Q4R.ai('fe5')
    Q4R.orb('fe5')
    Q4R.vcm('fe5')


def fe5t2():
    import phase17q10 as Q10
    Q10.OUT = OUT
    ion, Z, N, parent, tag, own = FE5T2
    _, sc = r2p(own, Z, parent)
    add = [('D', round(45.0 * sc, 1)), ('F', round(58.0 * sc, 1))]
    bas = 'fe_qzp2012_add_' + '_'.join(f'{l.lower()}{e:g}' for l, e in add) + '.bas'
    Q10.DEF = {ion: (Z, N, parent, add, bas, tag, own)}
    if not os.path.exists(os.path.join(_C.BASIS, bas)):
        Q10.bases()
    Q10.orb([ion])
    vcm_runs(bas, tag, Z)
    print('basis for phase15m.IONS:', bas)


def coorb():
    """Co3+ own orbitals: the 15a recipe (phase15f.own_set), as fend6."""
    import phase15f as F
    import phase15h  # noqa: F401  (F.QP[6] for d6)
    bs = F.Bas('co.without-diffuse.bas', 27)
    C, inf = F.own_set(bs, 6)
    rv = {let: [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w)) for k in range({0: 3, 1: 2, 2: 1}.get(l, 0), bs.nbf[l])]
          for l, let in enumerate('spdfgh'[:len(bs.nbf)])}
    inf.update(nbf=bs.nbf, r_virt=rv)
    assert not os.path.exists(os.path.join(ROOT, 'runs', 'vq_orb_cond6', 'be.NO'))
    F.write_be_no(C, 'cond6', bs.nbf)
    json.dump(inf, open(os.path.join(OUT, 'phase18_cond_orb.json'), 'w'), indent=1, default=float)
    print('cond6', bs.nbf, 'E', f'{inf["E"]:.8f}', '<r>3d', round(inf['r'], 4), {k: [round(x, 3) for x in v] for k, v in rv.items()})


def coi():
    import phase17q9i as Q9
    Q9.OUT = OUT
    Q9.EL['co'] = (27, 6, 'co.without-diffuse.bas', 'cond6', 'Co')
    Q9.IONS.update(coti1=('co', Q9.ADD['ti1'], 'coti16'), coti5=('co', Q9.ADD['ti5'], 'coti56'))
    Q9.bases(['coti1', 'coti5'])
    Q9.orb(['coti1', 'coti5'])


def codk():
    import phase17q4r as Q4R
    Q4R.OUT = OUT
    Q4R.bases('co')
    Q4R.ai('co')
    Q4R.orb('co')


def cot2():
    import phase17q10 as Q10
    Q10.OUT = OUT
    _, sc = r2p('cond6', 27, 'co.without-diffuse.bas')
    add = [('D', round(45.0 * sc, 1)), ('F', round(58.0 * sc, 1))]
    bas = 'co_qzp2012_add_' + '_'.join(f'{l.lower()}{e:g}' for l, e in add) + '.bas'
    Q10.DEF = {'cot2': (27, 6, 'co.without-diffuse.bas', add, bas, 'cot26', 'cond6')}
    Q10.bases()
    Q10.orb(['cot2'])
    print('basis for phase15m.IONS:', bas)


def dbases():
    """10 Nov B: plain Sapporo-QZP-2012 with only the d contractions opened into their distinct primitives (the
    phase17q4r recipe restricted to d; f is all single primitives already, so opening f changes nothing)."""
    from atomci_heff import gto
    for el, base in (('Fe', 'fe.without-diffuse.bas'), ('Co', 'co.without-diffuse.bas')):
        p = os.path.join(_C.BASIS, f'{el.lower()}_qzp2012_open_d.bas')
        assert not os.path.exists(p), p
        fns = gto.read_bas(os.path.join(_C.BASIS, base))
        src = open(os.path.join(_C.BASIS, base)).read().splitlines()
        assert all(len(e) == 1 for l, e, c in fns if l == 3), 'f not all single primitives'
        dex = sorted({e for l, ee, c in fns if l == 2 for e in ee}, reverse=True)
        i0 = next(k for k, x in enumerate(src) if x.split()[:1] == ['D'])
        i1 = next(k for k, x in enumerate(src) if x.split()[:1] == ['F'])
        lines = [f'" {el}  Sapporo-QZP-2012 plain ({base}) with the d contractions opened into their distinct primitives '
                 '(Phase 18, 10 Nov B)'] + src[1:i0] + [f'  D  1  1.0  {e:16.7f}  1.0000000' for e in dex] + src[i1:]
        open(p, 'w').write('\n'.join(lines) + '\n')
        b = gto.read_bas(p)
        key = lambda L: [(f[0], list(map(float, f[1])), list(map(float, f[2]))) for f in L if f[0] != 2]  # noqa: E731
        assert key(b) == key(fns)
        print(os.path.basename(p), [[f[0] for f in b].count(l) for l in range(6)], 'd', dex)


QD = {  # ion: (Z, basis, new tag, plain reference tag): own_set orbitals, occupied signs as the reference (feqd recipe)
    'coqd': (27, 'co.bas', 'coqd6', 'cond6', 'co.without-diffuse.bas'),
    'fedc': (26, 'fe_qzp2012_open_d.bas', 'fedc6', 'fend6', 'fe.without-diffuse.bas'),
    'codc': (27, 'co_qzp2012_open_d.bas', 'codc6', 'cond6', 'co.without-diffuse.bas')}


def qdorb(ions):
    """10 Nov A, B: Co3+ in Sapporo-QZP-2012-diffuse (coqd6) and Fe2+, Co3+ with opened d (fedc6, codc6): the 15a
    recipe (own_set, d6) as feqd6 (phase17q10.feqd); <r> of every virtual -> phase18_qd_orb.json."""
    import phase15f as F
    import phase15h  # noqa: F401  (F.QP[6] for d6)
    from atomci_heff.reader import read_vectors
    jp = os.path.join(OUT, 'phase18_qd_orb.json')
    js = json.load(open(jp)) if os.path.exists(jp) else {}
    for ion in ions:
        Z, bas, tag, ref, base = QD[ion]
        bs0 = F.Bas(base, Z)
        R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{ref}', 'be.NO'), bs0.nbf)[0]
        bs = F.Bas(bas, Z)
        C, inf = F.own_set(bs, 6)
        first = {0: 3, 1: 2, 2: 1}
        for l in range(len(bs.nbf)):
            for k in range(min(first.get(l, 0), bs.nbf[l])):
                C[l][:, k] *= np.sign(np.sum(bs.P(l, C[l][:, k]) * bs0.P(l, R[l][:, k]) * bs.w))
        inf['nbf'] = bs.nbf
        inf['S_min_eig'] = {'spdfgh'[l]: float(np.linalg.eigvalsh(bs.S[l]).min()) for l in range(len(bs.nbf))}
        inf['r_virt'] = {'spdfgh'[l]: [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w))
                                       for k in range(first.get(l, 0), bs.nbf[l])] for l in range(len(bs.nbf))}
        inf['dE_vs_ref'] = inf['E'] - F.own_set(bs0, 6)[1]['E']
        assert not os.path.exists(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO')), tag
        F.write_be_no(C, tag, bs.nbf)
        js[ion] = dict(tag=tag, basis=bas, ref=ref, **inf)
        print(tag, bs.nbf, f'E {inf["E"]:.8f} (vs {ref} {inf["dE_vs_ref"]:+.2e})', 'S_min', inf['S_min_eig'],
              {l: [round(x, 2) for x in v] for l, v in inf['r_virt'].items()}, flush=True)
    json.dump(js, open(jp, 'w'), indent=1, default=float)


def prep(ion, spaces, out_root=None):
    """Decks through phase15n.prep with the Z-generic writer (env ION set before the phase17 import)."""
    os.environ['ION'] = ion
    import functools
    import phase17  # noqa: F401
    import phase15j as J
    import phase15n as N15
    import phase18_write_deck as W
    J.write_deck = functools.partial(W.write_deck, out_root=out_root) if out_root else W.write_deck
    N15.prep(spaces)


def deckcheck():
    """The copy with Z = 26 reproduces the stored fend fam decks bytewise (run in its own process: ION fend)."""
    import filecmp
    import tempfile
    d = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
    prep('fend', ['fam_d5f2g2h1'], out_root=d)
    bad, n = [], 0
    for r in sorted(os.listdir(d)):
        for f in ('be.deck', 'be.csf'):
            n += 1
            if not filecmp.cmp(os.path.join(d, r, f), os.path.join(ROOT, 'inputs', r, f), shallow=False):
                bad.append(f'{r}/{f}')
    print('deckcheck:', n, 'files compared,', len(bad), 'differ', bad[:6])
    shutil.rmtree(d)
    assert not bad


def dims(args):
    os.environ.setdefault('ION', args[0])
    import phase17 as P
    P.Q.dims(args[1:], os.path.join('..', 'phase18', 'phase18_dims.json'))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'r2p':
        r2p(*sys.argv[2:5])
    elif c == 'fe5vcm':
        vcm_runs('fe.without-diffuse.bas', 'fe5nd3', 26)
    elif c == 'dims':
        dims(sys.argv[2:])
    elif c == 'qdorb':
        qdorb(sys.argv[2:] or list(QD))
    elif c == 'prep':
        prep(sys.argv[2], sys.argv[3:])
    else:
        {'fe5i': fe5i, 'fe5dk': fe5dk, 'fe5t2': fe5t2, 'coorb': coorb, 'coi': coi, 'codk': codk, 'cot2': cot2,
         'deckcheck': deckcheck, 'dbases': dbases}[c]()
