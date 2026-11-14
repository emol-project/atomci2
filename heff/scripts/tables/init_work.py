#!/usr/bin/env python3
"""Copy the inputs of one reproduction level from HEFF_DATA (read only) into HEFF_WORK.

  init_work.py level1 [--force]   H_eff dumps + per-space result JSON + two orbital files (checks/manifest_level1.txt)
  init_work.py fixture [--force]  the bundled Cr3+ fixture (tests/fixtures/cr3_fam) instead of HEFF_DATA

Files already in HEFF_WORK are kept unless --force. A missing source file is an error (the list is printed, nothing
is guessed). HEFF_DATA is never written."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as C  # noqa: E402

MANIFESTS = {'level1': os.path.join(C.HEFF, 'checks', 'manifest_level1.txt'),
             'fixture': os.path.join(C.HEFF, 'tests', 'fixtures', 'cr3_fam', 'MANIFEST.txt')}


def files(level):
    return [x.strip() for x in open(MANIFESTS[level]) if x.strip() and not x.startswith('#')]


def init(level, force=False, src=None):
    src = src or (os.path.join(C.HEFF, 'tests', 'fixtures', 'cr3_fam') if level == 'fixture' else C.DATA)
    work = C.root()
    if os.path.realpath(work) == os.path.realpath(src) or os.path.realpath(work).startswith(os.path.realpath(src) + os.sep):
        raise SystemExit(f'heff: HEFF_WORK ({work}) must not be HEFF_DATA or inside it ({src})')
    fs = files(level)
    miss = [f for f in fs if not os.path.isfile(os.path.join(src, f))]
    if miss:
        raise SystemExit(f'heff: {len(miss)} of {len(fs)} {level} inputs missing under {src}:\n  ' + '\n  '.join(miss))
    n = 0
    for f in fs:
        d = os.path.join(work, f)
        if os.path.exists(d) and not force:
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(os.path.join(src, f), d)
        n += 1
    print(f'{level}: {n} of {len(fs)} files copied from {src} into {work}')
    return fs


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if len(a) != 1 or a[0] not in MANIFESTS:
        raise SystemExit(__doc__)
    init(a[0], force='--force' in sys.argv)
