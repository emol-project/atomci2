#!/usr/bin/env python3
"""Phase 17 paper item 3: per-ion i functions for V2+, Mn4+ (d3) and Fe2+ (d6), the Cr3+ crti1 / crti5 recipe.
  <el>ti1  <el>.without-diffuse.bas + I 2.2136970            tag <el>ti1<N>
  <el>ti5  <el>.without-diffuse.bas + I 2.2136970, I 1.0     tag <el>ti5<N>
(the Cr3+ exponents, as the brief asks; plain Sapporo-QZP-2012 of each element, -> basis/<el>_qzp2012_add_i....bas).
Orbitals: phase15f.own_set (configuration-average HF of 3d^N; s..h unchanged by the unoccupied i), signs of s..h fixed
to the element's own set (vnd3 / mnnd3 / fend6), checked bitwise up to round-off. vcm (d3 only): the frozen-[Ne]
P-block H against vcm_d3_<t>e_<own>. Space: phase17 k1fhi (k1fh + every i, same RAS rule), reference k1fh of the ion.
Commands: bases / orb / vcm [ions]"""
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
import phase17q3 as Q3  # noqa: E402

OUT = Q3.OUT
EL = {'v': (23, 3, 'v.without-diffuse.bas', 'vnd3', 'V'), 'mn': (25, 3, 'mn.without-diffuse.bas', 'mnnd3', 'Mn'),
      'fe': (26, 6, 'fe.without-diffuse.bas', 'fend6', 'Fe')}
ADD = {'ti1': [Q3.EXP_I1], 'ti5': [Q3.EXP_I1, 1.0]}
IONS = {f'{el}{k}': (el, ex, f'{el}{k}{EL[el][1]}') for el in EL for k, ex in ADD.items()}


def bas_name(ion):
    el, ex, _ = IONS[ion]
    return f'{el}_qzp2012_add_' + '_'.join(f'i{e:g}' for e in ex) + '.bas'


def bases(ions=None):
    from atomci_heff import gto
    for ion, (el, ex, tg) in IONS.items():
        if ions and ion not in ions:
            continue
        base = EL[el][2]
        src = open(os.path.join(_C.BASIS, base)).read().splitlines()
        end = next(k for k, x in enumerate(src) if x.strip().startswith(')'))
        lines = list(src)
        lines[0] = (f'" {EL[el][4]}  Sapporo-QZP-2012 plain ({base}) + uncontracted '
                    + ', '.join(f'I {e:g}' for e in ex) + ' (Phase 17 paper item 3)')
        lines[end:end] = [f'  I  1  1.0  {e:14.7f}  1.0000000' for e in ex]
        p = os.path.join(_C.BASIS, bas_name(ion))
        assert not os.path.exists(p), p
        open(p, 'w').write('\n'.join(lines) + '\n')
        a, b = gto.read_bas(os.path.join(_C.BASIS, base)), gto.read_bas(p)
        assert [f[0] for f in b].count(6) == len(ex) and len(b) == len(a) + len(ex)
        print(ion, bas_name(ion), len(a), '->', len(b), 'functions')


def orb(ions=None):
    import phase15f as F
    import phase15h  # noqa: F401  (F.QP[6] for d6)
    from atomci_heff.reader import read_vectors
    jp = os.path.join(OUT, 'phase17q9i_orb.json')
    js = json.load(open(jp)) if os.path.exists(jp) else {}
    for ion, (el, ex, tg) in IONS.items():
        if ions and ion not in ions:
            continue
        Z, N, base, own, _ = EL[el]
        bs0 = F.Bas(base, Z)
        R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{own}', 'be.NO'), bs0.nbf)[0]
        bs = F.Bas(bas_name(ion), Z)
        C, inf = F.own_set(bs, N)
        inf['nbf'] = bs.nbf
        for l in range(6):
            C[l] = C[l] * np.sign(np.sum(C[l] * R[l], 0))
        inf['sh_max_dev_vs_own'] = max(float(np.abs(C[l] - R[l]).max()) for l in range(6))
        inf['S_min_eig_i'] = float(np.linalg.eigvalsh(bs.S[6]).min())
        inf['r_virt_i'] = [float(np.sum(bs.P(6, C[6][:, k]) ** 2 * bs.r * bs.w)) for k in range(bs.nbf[6])]
        assert inf['sh_max_dev_vs_own'] < 1e-6, inf['sh_max_dev_vs_own']
        Q3.write_be_no(C, tg, bs.nbf)
        js[ion] = inf
        print(ion, tg, bs.nbf, f'E {inf["E"]:.8f}', 'dev s..h', f'{inf["sh_max_dev_vs_own"]:.1e}',
              '<r> i', [round(x, 3) for x in inf['r_virt_i']])
    json.dump(js, open(jp, 'w'), indent=1)


def vcm(ions=None):
    import phase13d as D
    import phase15s_nd as ND
    jp = os.path.join(OUT, 'phase17q9i_vcm.json')
    res = json.load(open(jp)) if os.path.exists(jp) else {}
    for ion, (el, ex, tg) in IONS.items():
        if (ions and ion not in ions) or EL[el][1] != 3:
            continue
        for L, twoS, m, t in D.term_list(3):
            dst, deck, csf = ND.vcm_deck(t, bas_name(ion), tg, z=EL[el][0])
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
            shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
            subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                            f'vq_orb_{tg}'], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
            a = np.load(os.path.join(ROOT, 'results', dst + '.npz'))['H']
            b = np.load(os.path.join(ROOT, 'results', f'vcm_d3_{t}e_{EL[el][3]}.npz'))['H']
            res[dst] = float(np.abs(a - b).max())
            print(dst, a.shape, 'max |H - H(own)|', f'{res[dst]:.1e}', flush=True)
    json.dump(res, open(jp, 'w'), indent=1)


if __name__ == '__main__':
    {'bases': bases, 'orb': orb, 'vcm': vcm}[sys.argv[1]](sys.argv[2:] or None)
