#!/usr/bin/env python3
"""Phase 15r: Cr3+ f-virtual convergence on the 15q step 1b base (rasDallG2: s 1, p 1, d all, f 1, g 2).

Same route as 15q (stop-after-H deck, iterative solver); this module only adds spaces and the v15r run names.
  rasDallG2_f0 .. _f4   k_f = 0..4 (11f, 12f, 13f, 14f by <r>, compact first); k_f = 1 is 15q's rasDallG2
  rasDallG2_s2, _p2     s or p raised to k = 2 (16s+17s, 16p+15p), f 1
Commands as phase15q: dims / prep / run_names / analyze."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15q as Q  # noqa: E402  (sets STOP_AFTER_H, installs spaces() / rname())
import phase15n as N15  # noqa: E402

BASE = {'s': 1, 'p': 1, 'd': 5, 'f': 1, 'g': 2}
NEW = {f'rasDallG2_f{k}': {**BASE, 'f': k} for k in (0, 2, 3, 4)}
NEW.update(rasDallG2_s2={**BASE, 's': 2}, rasDallG2_p2={**BASE, 'p': 2})
Q.NEW.update(NEW)
R = list(NEW)
_rn = N15.rname


def rname(space, t):
    if space in R:
        return f'v15r_{space}_d{Q.N_}_{t}e_{Q.TAG_}'
    return _rn(space, t)


N15.rname = rname
Q.K.rname_k = rname

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dims':
        Q.dims(sys.argv[2:] or R, 'phase15r_dims.json')
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        Q.M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
    elif cmd == 'virt':
        print(Q.ION, {k: N15.virt_k(v) for k, v in NEW.items()})
