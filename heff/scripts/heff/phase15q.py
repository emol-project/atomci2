#!/usr/bin/env python3
"""Phase 15q: the stop-after-H route (ATOMCI CIMAIN section 1 only, our iterative solver) for the large d3 spaces.

Ion via env ION (cr | v | mn), as phase15n/15p. Every deck is written with phase15k.STOP_AFTER_H = True
(&ci nrnfg=1,0,0,0,0,0, &cntl nrnfg(8)=0; validated in 15p Addendum 2). Spaces (15p definitions, same RAS rules):
  rasDall       virtuals s 1, p 1, d all (5), f 1, g 1            (step 0 check block; step 2 for V2+, Mn4+)
  rasDall_nof   the same without f                                 (step 1a)
  rasDallG2     all virtual d + g 2, with f                        (step 1b)
  rasK2dall     s 2, p 2, d all (5), f 2, g 2, with f              (step 1c, full RAS k = 2 with all virtual d)
  dims [spaces]       CSF counts per term, H-build time / memory estimates  -> phase15q_dims.json
  prep <spaces>; run_names <names>; analyze <tag> <spaces>   (as phase15p)
Time model (H build only): fit over the 15k-15p d3 runs (start -> fort.39 written), see phase15q_hfit.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15p as P15  # noqa: E402  (spaces of 15n/15p, ion setup)
import phase15n as N15  # noqa: E402
import phase15m as M  # noqa: E402
import phase15k as K  # noqa: E402

K.STOP_AFTER_H = True
ION, N_, TAG_ = M.ION, M.N_, M.TAG_
OUT = K.OUT
NEW = {'rasK2dall': {'s': 2, 'p': 2, 'd': 5, 'f': 2, 'g': 2}}
Q = ['rasDall', 'rasDall_nof', 'rasDallG2', 'rasK2dall']
_sp0 = N15.spaces


def spaces():
    sp = _sp0()
    for k, v in NEW.items():
        if k not in sp:
            sp[k] = ('allcore', N15.ras_cfgs(N15.virt_k(v)))
    return sp


N15.spaces = spaces
_rn0 = N15.rname


def rname(space, t):
    if space in Q:
        return f'v15q_{space}_d{N_}_{t}e_{TAG_}'
    return _rn0(space, t)


N15.rname = rname
K.rname_k = rname


def hfit():
    """H-build time (s) and nnz per CSF^2 from earlier runs: time = start (run.log date) -> fort.39 mtime."""
    import datetime
    import glob
    p = os.path.join(OUT, 'phase15q_hfit.json')
    if os.path.exists(p):
        return json.load(open(p))
    rows = []
    for d in glob.glob(os.path.join(ROOT, 'runs', 'v15[k-p]_*_d3_*e_*')):
        try:
            lg = json.load(open(os.path.join(d, 'run.log')))
            z = np.load(os.path.join(ROOT, 'results', os.path.basename(d) + '.npz'), allow_pickle=True)
            n = int(z['N']) if 'N' in z.files else z['H'].shape[0]
            t = os.path.getmtime(os.path.join(d, 'fort.39')) - datetime.datetime.fromisoformat(lg['date']).timestamp()
            rows.append((n, t, (os.path.getsize(os.path.join(d, 'fort.39')) - 7212) / 16))   # 16 B per stored entry
        except Exception:  # noqa: BLE001
            continue
    a = np.array([r for r in rows if r[0] >= 1000 and r[1] > 5])
    x, y = np.log(a[:, 0]), np.log(a[:, 1])
    c = np.polyfit(x, y, 1)
    res = y - np.polyval(c, x)
    dens = a[:, 2] / a[:, 0] ** 2
    f = dict(n_runs=len(a), exp=float(c[0]), c=float(c[1]), lo=float(np.exp(res.min())), hi=float(np.exp(res.max())),
             density_max=float(dens.max()), density_median=float(np.median(dens)))
    json.dump(f, open(p, 'w'), indent=1)
    return f


def est(n, f):
    t = float(np.exp(f['c']) * n ** f['exp'])
    nnz = f['density_max'] * n * n / 2 + n
    return dict(t_s=t, t_hi_s=t * f['hi'], nnz_hi=nnz, ana_GB=40 * nnz / 1e9, export_GB=24 * nnz / 1e9)


def dims(names, fname='phase15q_dims.json'):
    import counts as CN
    import phase15e as E15
    f = hfit()
    p = os.path.join(OUT, fname)
    res = json.load(open(p)) if os.path.exists(p) else {}
    sp = spaces()
    for name in names:
        kind, cf = sp[name]
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in M.term_list()}
        e = {t: est(n, f) for t, n in per.items()}
        mx = max(per, key=per.get)
        r = dict(ncfg=len(cf), per=per, est=e, max=per[mx], maxjob_s=e[mx]['t_s'], maxjob_hi_s=e[mx]['t_hi_s'],
                 cpu_s=sum(v['t_s'] for v in e.values()), ana_GB=e[mx]['ana_GB'])
        res[f'{ION}:{name}'] = r
        print(f'{ION}:{name:11s} cfg {len(cf):5d} ' + ' '.join(f'{t}:{n}' for t, n in per.items()) +
              f' | largest job {r["maxjob_s"] / 3600:.2f} h (hi {r["maxjob_hi_s"] / 3600:.2f}) CPU {r["cpu_s"] / 3600:.2f} h'
              f' | analysis <= {r["ana_GB"]:.1f} GB', flush=True)
    json.dump(res, open(p, 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dims':
        dims(sys.argv[2:] or Q)
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
    elif cmd == 'virt':
        print(ION, {k: N15.virt_k(v) for k, v in {**P15.VARIANTS, **NEW}.items() if k in Q})
