"""Phase 18: NIST ASD level tables with the Level column located from the header (the Fe VI and Co IV exports have no
g / Prefix column, so phase15u_bfit.nist, which reads column 4, does not apply). Importing registers fe5 and co in
phase15u_bfit.IONS and routes those two files through this parser; the other ions are untouched."""
import os

import phase15u_bfit as BF

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OWN = {'Fe_VI': ('fe5', 'Fe VI', 3), 'Co_IV': ('co', 'Co IV', 6)}
_nist0 = BF.nist


def parse(fn, N):
    """3d^N rows -> [(term, leading %, 2J, E)], columns found by header name."""
    cfg = term = None
    col = None
    out = []
    for line in open(os.path.join(_C.REFDATA, 'levels', f'NIST_ASD_levels_{fn}.txt')):
        c = [x.strip() for x in line.split('|')]
        if c[0].startswith('Config'):
            col = {k: next(i for i, h in enumerate(c) if h.startswith(k)) for k in ('Term', 'J', 'Level', 'Leading')}
            continue
        if col is None or len(c) <= col['Leading']:
            continue
        cfg, term = c[0] or cfg, c[col['Term']] or term
        if cfg == f'3d{N}' and c[col['Level']]:
            j = c[col['J']].split('/')
            out.append((term, c[col['Leading']], int(j[0]) * (1 if len(j) == 2 else 2), float(c[col['Level']].strip('[]?'))))
    return out


def nist(fn, N):
    return parse(fn, N) if fn in OWN else _nist0(fn, N)


BF.nist = nist
for fn, (ion, lab, N) in OWN.items():
    if not any(x[0] == ion for x in BF.IONS):
        BF.IONS.append((ion, lab, fn, N))
