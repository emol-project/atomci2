#!/usr/bin/env python3
"""Phase 17 paper items 3 (Fe2+ 2p), 4 (additivity) and 5 (diffuse set, V2+ / Mn4+): bases and orbitals.
  fet2   fe.without-diffuse.bas + D 54.6, F 70.3 (uncontracted)        tag fet26
         the Cr3+ crt2 pair (d 45, f 58, 2p-matched) scaled by (<r>_2p Cr / <r>_2p Fe)^2 = (0.26003 / 0.23612)^2 =
         1.2128 (own HF, plain basis), so the pair sits at the Fe2+ 2p radius as crt2 does at the Cr3+ one
  crt2i  cr_qzp2012_add_d45_f58.bas (crt2) + I 2.2136970               tag crt2i3
         the item 4 test: P' with 2p open and one i, against P' + 2p (crt23) and P' + i (crt2i3, 2p closed)
Orbitals: phase15f.own_set (the 15a recipe), signs of the occupied orbitals as the parent set (fend6 / crt23). For
crt2i the i is unoccupied, so s..h must equal crt23 (checked); for fet2 the tight d enters the 3d HF orbital.
diffuse (item 5): checks that own_set in v_qzp.bas / mn.bas (Sapporo-QZP-2012-diffuse) reproduces the existing
o14n3 / mn15b3 sets, so that these serve as the diffuse-set orbitals with the same recipe as vnd3 / mnnd3.
Commands: bases / orb [ion] / vcm (crt2i) / diffuse"""
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
OUT = os.path.join(ROOT, 'results', 'phase17')
SC = (0.2600291745443122 / 0.23611635676860354) ** 2
# ion: (Z, N, parent basis, added primitives, basis, tag, parent orbital tag)
DEF = {'fet2': (26, 6, 'fe.without-diffuse.bas', [('D', round(45.0 * SC, 1)), ('F', round(58.0 * SC, 1))],
                'fe_qzp2012_add_d54.6_f70.3.bas', 'fet26', 'fend6'),
       'crt2i': (24, 3, 'cr_qzp2012_add_d45_f58.bas', [('I', 2.2136970)], 'cr_qzp2012_add_d45_f58_i2.2137.bas',
                 'crt2i3', 'crt23')}
DIFF = {'v': (23, 'v_qzp.bas', 'o14n3'), 'mn': (25, 'mn.bas', 'mn15b3')}


def bases():
    """Each primitive among the single primitives of its l in descending exponent order (15t rule); a new l (i) goes
    before the closing ')'."""
    from atomci_heff import gto
    for ion, (Z, N, parent, add, bas, tag, _) in DEF.items():
        assert bas.replace('.bas', '').endswith('_'.join(f'{l.lower()}{e:g}' for l, e in add)), (bas, add)
        lines = open(os.path.join(_C.BASIS, parent)).read().splitlines()
        lines[0] += ' + uncontracted ' + ', '.join(f'{l} {e:g}' for l, e in add) + ' (Phase 17 paper items 3-4)'
        for l, e in add:
            idx = [k for k, x in enumerate(lines) if x.split()[:2] == [l, '1']]
            end = next(k for k, x in enumerate(lines) if x.strip().startswith(')'))
            pos = next((k for k in idx if float(lines[k].split()[3]) < e), idx[-1] + 1) if idx else end
            lines.insert(pos, f'  {l}  1  1.0  {e:14.7f}  1.0000000')
        p = os.path.join(_C.BASIS, bas)
        assert not os.path.exists(p), p
        open(p, 'w').write('\n'.join(lines) + '\n')
        a, b = gto.read_bas(os.path.join(_C.BASIS, parent)), gto.read_bas(p)
        assert len(b) == len(a) + len(add)
        print(ion, bas, [[f[0] for f in b].count(l) for l in range(7)])


def orb(ions=None):
    import phase15f as F
    import phase15h  # noqa: F401  (F.QP[6] for d6)
    from atomci_heff.reader import read_vectors
    jp = os.path.join(OUT, 'phase17q10_orb.json')
    js = json.load(open(jp)) if os.path.exists(jp) else {}
    for ion, (Z, N, parent, add, bas, tag, own) in DEF.items():
        if ions and ion not in ions:
            continue
        bs0 = F.Bas(parent, Z)
        R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{own}', 'be.NO'), bs0.nbf)[0]
        bs = F.Bas(bas, Z)
        C, inf = F.own_set(bs, N)
        inf['nbf'] = bs.nbf
        inf['S_min_eig'] = {'spdfghi'[l]: float(np.linalg.eigvalsh(bs.S[l]).min()) for l in range(len(bs.nbf))}
        first = {0: 3, 1: 2, 2: 1}
        inf['r_virt'] = {'spdfghi'[l]: [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w))
                                        for k in range(first.get(l, 0), bs.nbf[l])] for l in range(len(bs.nbf))}
        inf['dE_vs_parent'] = inf['E'] - F.own_set(bs0, N)[1]['E']
        for l in range(len(bs.nbf)):          # signs: the occupied orbitals as the parent set (overlap of P(r))
            for k in range(min(first.get(l, 0), bs.nbf[l])):
                p, p0 = bs.P(l, C[l][:, k]), bs0.P(l, R[l][:, k])
                C[l][:, k] *= np.sign(np.sum(p * p0 * bs.w))
        if ion == 'crt2i':                    # i unoccupied: s..h as crt23, column signs fixed to it
            for l in range(6):
                C[l] = C[l] * np.sign(np.sum(C[l] * R[l], 0))
            inf['sh_max_dev_vs_parent'] = max(float(np.abs(C[l] - R[l]).max()) for l in range(6))
            assert inf['sh_max_dev_vs_parent'] < 1e-6, inf['sh_max_dev_vs_parent']
        if len(bs.nbf) > 6:                   # a basis with i: the phase17q3 writer (LMAX 6)
            import phase17q3 as Q3
            Q3.write_be_no(C, tag, bs.nbf)
        else:
            F.write_be_no(C, tag, bs.nbf)
        js[ion] = inf
        print(ion, tag, bs.nbf, f'E {inf["E"]:.8f} (vs parent {inf["dE_vs_parent"]:+.2e})', 'S_min', inf['S_min_eig'],
              {l: [round(x, 3) for x in v] for l, v in inf['r_virt'].items() if l in 'dfi'}, flush=True)
    json.dump(js, open(jp, 'w'), indent=1)


def vcm():
    """crt2i: the frozen-[Ne] P-block H against vcm_d3_<t>e_crt23 (the i does not enter the P block)."""
    import phase13d as D
    import phase15s_nd as ND
    Z, N, parent, add, bas, tag, own = DEF['crt2i']
    res = {}
    for L, twoS, m, t in D.term_list(3):
        dst, deck, csf = ND.vcm_deck(t, bas, tag, z=Z)
        os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
        open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
        open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
        shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
        subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                        f'vq_orb_{tag}'], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
        a = np.load(os.path.join(ROOT, 'results', dst + '.npz'))['H']
        b = np.load(os.path.join(ROOT, 'results', f'vcm_d3_{t}e_{own}.npz'))['H']
        res[dst] = float(np.abs(a - b).max())
        print(dst, a.shape, 'max |H - H(crt23)|', f'{res[dst]:.1e}', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q10_vcm.json'), 'w'), indent=1)


def diffuse():
    import phase15f as F
    from atomci_heff.reader import read_vectors
    res = {}
    for el, (Z, bas, tag) in DIFF.items():
        bs = F.Bas(bas, Z)
        C, inf = F.own_set(bs, 3)
        R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{tag}', 'be.NO'), bs.nbf)[0]
        dev = [float(np.abs(np.abs(C[l]) - np.abs(R[l])).max()) for l in range(len(bs.nbf))]
        res[el] = dict(tag=tag, nbf=bs.nbf, E=inf['E'], max_abs_dev_per_l=dev,
                       S_min_eig={'spdfgh'[l]: float(np.linalg.eigvalsh(bs.S[l]).min()) for l in range(len(bs.nbf))})
        print(el, tag, bs.nbf, f'E {inf["E"]:.8f}', 'max | |C| - |C(tag)| | per l', [f'{x:.1e}' for x in dev], flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q10_diffuse.json'), 'w'), indent=1)


def vqd():
    """V2+ in Sapporo-QZP-2012-diffuse: o14n3 is not the own_set recipe (diffuse check: p differs), so a new set vqd3
    (own_set in v_qzp.bas; occupied signs as vnd3 by the overlap of P(r))."""
    import phase15f as F
    from atomci_heff.reader import read_vectors
    bs0 = F.Bas('v.without-diffuse.bas', 23)
    R = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb_vnd3', 'be.NO'), bs0.nbf)[0]
    bs = F.Bas('v_qzp.bas', 23)
    C, inf = F.own_set(bs, 3)
    first = {0: 3, 1: 2, 2: 1}
    for l in range(len(bs.nbf)):
        for k in range(min(first.get(l, 0), bs.nbf[l])):
            C[l][:, k] *= np.sign(np.sum(bs.P(l, C[l][:, k]) * bs0.P(l, R[l][:, k]) * bs.w))
    inf['r_virt'] = {'spdfgh'[l]: [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w))
                                   for k in range(first.get(l, 0), bs.nbf[l])] for l in range(len(bs.nbf))}
    inf['dE_vs_vnd3'] = inf['E'] - F.own_set(bs0, 3)[1]['E']
    F.write_be_no(C, 'vqd3', bs.nbf)
    js = json.load(open(os.path.join(OUT, 'phase17q10_diffuse.json')))
    js['v_own'] = dict(tag='vqd3', **{k: v for k, v in inf.items() if k in ('E', 'dE_vs_vnd3', 'r_virt')})
    json.dump(js, open(os.path.join(OUT, 'phase17q10_diffuse.json'), 'w'), indent=1)
    print('vqd3', bs.nbf, f'E {inf["E"]:.8f} (vs vnd3 {inf["dE_vs_vnd3"]:+.2e})',
          {l: [round(x, 2) for x in v] for l, v in inf['r_virt'].items() if l in 'sd'})


def feqd():
    """Fe2+ in Sapporo-QZP-2012-diffuse (item 10): own_set in fe.bas (the 15a recipe, as fend6 in
    fe.without-diffuse.bas); occupied signs as fend6 by the overlap of P(r)."""
    import phase15f as F
    import phase15h  # noqa: F401  (F.QP[6] for d6)
    from atomci_heff.reader import read_vectors
    bs0 = F.Bas('fe.without-diffuse.bas', 26)
    R = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb_fend6', 'be.NO'), bs0.nbf)[0]
    bs = F.Bas('fe.bas', 26)
    C, inf = F.own_set(bs, 6)
    first = {0: 3, 1: 2, 2: 1}
    for l in range(len(bs.nbf)):
        for k in range(min(first.get(l, 0), bs.nbf[l])):
            C[l][:, k] *= np.sign(np.sum(bs.P(l, C[l][:, k]) * bs0.P(l, R[l][:, k]) * bs.w))
    inf['nbf'] = bs.nbf
    inf['S_min_eig'] = {'spdfgh'[l]: float(np.linalg.eigvalsh(bs.S[l]).min()) for l in range(len(bs.nbf))}
    inf['r_virt'] = {'spdfgh'[l]: [float(np.sum(bs.P(l, C[l][:, k]) ** 2 * bs.r * bs.w))
                                   for k in range(first.get(l, 0), bs.nbf[l])] for l in range(len(bs.nbf))}
    inf['dE_vs_fend6'] = inf['E'] - F.own_set(bs0, 6)[1]['E']
    F.write_be_no(C, 'feqd6', bs.nbf)
    jp = os.path.join(OUT, 'phase17q10_orb.json')
    js = json.load(open(jp)) if os.path.exists(jp) else {}
    js['feqd'] = dict(tag='feqd6', **inf)
    json.dump(js, open(jp, 'w'), indent=1)
    print('feqd6', bs.nbf, f'E {inf["E"]:.8f} (vs fend6 {inf["dE_vs_fend6"]:+.2e})', 'S_min', inf['S_min_eig'],
          {l: [round(x, 2) for x in v] for l, v in inf['r_virt'].items()})


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'orb':
        orb(sys.argv[2:] or None)
    else:
        {'bases': bases, 'vcm': vcm, 'diffuse': diffuse, 'vqd': vqd, 'feqd': feqd}[c]()
