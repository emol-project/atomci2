#!/usr/bin/env python3
"""Run ATOMCI on inputs/<RUN>/ in runs/<RUN>/ and export H + CSFs to results/.

Usage: scripts/run_atomci.py RUN [--vectors-from OTHER_RUN] [--binary PATH]

The work directory runs/<RUN>/ is wiped and recreated, so no stale fort.*
files carry over. run.log records the binary path and its sha256, the
CFPTBL sha256, the input hashes, the exit code and stderr.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff.reader import read_csfs, read_h, read_h_sparse  # noqa: E402
import resource  # noqa: E402

import numpy as np  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--vectors-from', help='copy runs/<X>/be.NO into the work dir first')
    ap.add_argument('--binary', default=_C.ATOMCI_BIN)
    ap.add_argument('--cfptbl', default=_C.ATOMCI_CFPTBL)
    ap.add_argument('--sparse-above', type=int, default=int(os.environ.get('HEFF_SPARSE_ABOVE', 20000)),
                    help='export without densifying H above this many CSFs (default 20000, or env HEFF_SPARSE_ABOVE)')
    ap.add_argument('--export-only', action='store_true',
                    help='do not run ATOMCI; re-export results from the existing runs/<RUN>/')
    a = ap.parse_args()
    if a.export_only:
        work = os.path.join(ROOT, 'runs', a.run)
        log = json.load(open(os.path.join(work, 'run.log')))
        return export(a.run, work, log, a.sparse_above)

    src = os.path.join(ROOT, 'inputs', a.run)
    work = os.path.join(ROOT, 'runs', a.run)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.symlink(a.cfptbl, os.path.join(work, 'CFPTBL'))
    for f in os.listdir(os.path.join(_C.BASIS)):
        if f.endswith('.bas'):
            shutil.copy(os.path.join(_C.BASIS, f), work)
    for f in ('be.deck', 'be.csf'):
        shutil.copy(os.path.join(src, f), work)
    if a.vectors_from:
        shutil.copy(os.path.join(ROOT, 'runs', a.vectors_from, 'be.NO'), work)

    binary = os.path.realpath(a.binary)
    log = {
        'run': a.run,
        'date': datetime.datetime.now().isoformat(timespec='seconds'),
        'binary': binary,
        'binary_sha256': sha256(binary),
        'cfptbl': os.path.realpath(a.cfptbl),
        'cfptbl_sha256': sha256(a.cfptbl),
        'deck_sha256': sha256(os.path.join(work, 'be.deck')),
        'csf_sha256': sha256(os.path.join(work, 'be.csf')),
        'vectors_from': a.vectors_from,
    }
    if a.vectors_from:
        log['vectors_in_sha256'] = sha256(os.path.join(work, 'be.NO'))
    with open(os.path.join(work, 'be.deck')) as fin, \
            open(os.path.join(work, 'be.out'), 'w') as fout, \
            open(os.path.join(work, 'be.err'), 'w') as ferr:
        t0 = time.time()
        rc = subprocess.run([binary], stdin=fin, stdout=fout, stderr=ferr, cwd=work).returncode
    log['wall_seconds'] = round(time.time() - t0, 2)
    log['peak_rss_mb_atomci'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    log['run_dir_bytes'] = sum(os.path.getsize(os.path.join(work, f)) for f in os.listdir(work)
                               if os.path.isfile(os.path.join(work, f)) and not os.path.islink(os.path.join(work, f)))
    log['returncode'] = rc
    log['stderr'] = open(os.path.join(work, 'be.err')).read()
    with open(os.path.join(work, 'run.log'), 'w') as f:
        json.dump(log, f, indent=2)
    if rc != 0:
        raise SystemExit(f'atomci exited {rc}; see {work}/be.out')

    export(a.run, work, log, a.sparse_above)


def export(run, work, log, sparse_above=20000):
    """results/<run>.npz: dense H for N <= 500, otherwise sparse lower triangle (H_rows, H_cols, H_vals, N)."""
    hd, csfs = read_csfs(os.path.join(work, 'fort.15'))
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    n = len(csfs)
    save = np.savez_compressed
    if n <= 500:
        H, nent = read_h(os.path.join(work, 'fort.38'), os.path.join(work, 'fort.39'))
        hdata = dict(H=H)
    elif n <= sparse_above:
        H, nent = read_h(os.path.join(work, 'fort.38'), os.path.join(work, 'fort.39'))
        r, c = np.nonzero(np.tril(H))
        hdata = dict(H_rows=r.astype(np.int32), H_cols=c.astype(np.int32), H_vals=H[r, c], N=n)
    else:
        # never densify: lower triangle incl. the diagonal, as the n <= 20000 branch;
        # stored uncompressed (compression of ~1e8 doubles is slow and gains little)
        d, r, c, v, nent = read_h_sparse(os.path.join(work, 'fort.38'), os.path.join(work, 'fort.39'))
        ii = np.arange(n, dtype=np.int32)
        hdata = dict(H_rows=np.concatenate([r, ii]), H_cols=np.concatenate([c, ii]),
                     H_vals=np.concatenate([v, d]), N=n)
        save = np.savez
    assert n == hd['ITSTV']
    save(os.path.join(ROOT, 'results', f'{run}.npz'),
                        **hdata, labels=np.array([c['label'] for c in csfs]),
                        couplings=np.array([' '.join(c['couplings']) for c in csfs]),
                        meta=json.dumps({**log, 'n_offdiag': int(nent),
                                         'ee_header': {k: v for k, v in hd.items()}}))
    with open(os.path.join(ROOT, 'results', f'{run}_csfs.txt'), 'w') as f:
        for i, c in enumerate(csfs, 1):
            f.write(f"{i:4d}  {c['label']:<18s} {' '.join(c['couplings']):<16s} "
                    f"{' '.join(c['shell_states'])}\n")
    peak_py = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(f'{run}: N={len(csfs)} offdiag={nent} rc={log["returncode"]} stderr={log["stderr"].strip()!r} '
          f'peak RSS export {peak_py} MB, atomci {log.get("peak_rss_mb_atomci")} MB')


if __name__ == '__main__':
    main()
