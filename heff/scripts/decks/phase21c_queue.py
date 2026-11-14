#!/usr/bin/env python3
"""Phase 21 job C queue: ATOMCI stop-after-H runs v21_famnsphf_d3_<t>e_<tag>, NSLOT parallel, one block per job,
timeout 3 h per job (skipped = reported), Fe5+ before Cr3+, longest block first. Before each job: /data free >= 150 GB
(else stop). Disk: after a job's results/<run>.npz exists (the full H), its fort.* >= 100 MB are deleted (logged in
runs/<run>/fort_deleted.json and phase21c_queue.log); the npz is all the analysis reads."""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
os.chdir(ROOT)
LOG = 'results/phase21/phase21c_queue.log'
NSLOT = int(os.environ.get('NSLOT', '3'))
TAGS = {'fe5nd': 'fe5nd3', 'fe5td': 'fe5td3', 'fe5qd': 'fe5qd3', 'fe5dc': 'fe5dc3', 'crtd': 'crtd3', 'crqd': 'cr15n3', 'crdc': 'crdc3'}
ORDER = sys.argv[1:] or ['fe5nd', 'fe5td', 'fe5qd', 'fe5dc', 'crtd', 'crqd', 'crdc']   # entries 'ion' or 'ion:space'
SPD = 'famnsphf'


def log(m):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%m-%d %H:%M:%S")} {m}\n')


def free_gb():
    return shutil.disk_usage('/data').free / 2 ** 30


def clean(run):
    d = os.path.join('runs', run)
    out = []
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if f.startswith('fort.') and os.path.getsize(p) >= 100 * 2 ** 20:
            out.append(dict(file=f, bytes=os.path.getsize(p), mtime=os.path.getmtime(p)))
            os.remove(p)
    json.dump(out, open(os.path.join(d, 'fort_deleted.json'), 'w'), indent=1)
    log(f'cleaned {run}: {sum(x["bytes"] for x in out) / 2 ** 30:.1f} GB')


dims = json.load(open('results/phase21/phase21c.json'))['dims']
jobs = []
for g, ent in enumerate(ORDER):
    ion, SP = (ent.split(':') + [SPD])[:2]
    per = dims.get(f'{ion}:{SP}', {}).get('per')
    if per is None:
        per = dims[f"{'crnd' if ion.startswith('cr') else 'fe5nd'}:{SP}"]['per']
    for t, n in sorted(per.items(), key=lambda x: -x[1]):
        run = f'v21_{SP}_d3_{t}e_{TAGS[ion]}'
        if not os.path.exists(f'results/{run}.npz'):
            jobs.append((g, -n, ion, t, run))
jobs.sort()
log(f'start: {len(jobs)} jobs, {NSLOT} slots, free {free_gb():.0f} GB')
running = {}
stop = False
while jobs or running:
    for run, (p, t0, ion, n) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        ok = os.path.exists(f'results/{run}.npz')
        log(f'done {run} n {n} rc {rc} npz {ok} {time.time() - t0:.0f} s' + (' TIMEOUT (> 3 h, skipped)' if rc == 124 else ''))
        if ok:
            clean(run)
        del running[run]
    while not stop and jobs and len(running) < NSLOT:
        if free_gb() < 150:
            log(f'DISK-STOP free {free_gb():.0f} GB')
            stop = True
            break
        g, nn, ion, t, run = jobs.pop(0)
        shutil.rmtree(os.path.join('runs', run), ignore_errors=True)
        p = subprocess.Popen(['timeout', '10800', sys.executable, _C.script('run_atomci.py'), run, '--vectors-from', f'vq_orb_{TAGS[ion]}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        running[run] = (p, time.time(), ion, -nn)
        log(f'start {run} n {-nn} free {free_gb():.0f} GB')
    if stop and not running:
        break
    time.sleep(20)
log(f'end; not started: {[j[4] for j in jobs]}')
