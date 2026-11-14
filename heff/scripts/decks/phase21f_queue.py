#!/usr/bin/env python3
"""Phase 21 A2 driver: ATOMCI runs v21_famA_d3_<t>e_crnd3 (3 slots, longest first, /data >= 150 GB before each start,
10 h guard per job), then per block `phase21f.py block <t>` (serial, RLIMIT_AS 14.3 GB) and a git commit of the block
result (results/phase21 only); `phase21f.py final` + commit at the end.
Memory (15 GB machine): ATOMCI itself ~15 MB; the export (scripts/run_atomci.py, after run.log is written) holds the
sparse H (~8-9 GB for 81k CSFs), the analysis up to ~14 GB. At most one of them runs at a time: an exporting job is
SIGSTOPped (right after its run.log appears) while another export or an analysis is active, then SIGCONTed.
fort.* >= 100 MB deleted after the npz exists (as phase21c_queue)."""
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
HERE = os.path.join(ROOT, 'results', 'phase21')  # outputs (was this script's directory)
os.makedirs(HERE, exist_ok=True)
os.chdir(ROOT)
LOG = os.path.join(HERE, 'phase21f_queue.log')
PY = sys.executable
NSLOT = 3
GUARD = 10 * 3600
DIMS = {'2P': 44823, '4P': 42730, '2D': 67001, '2F': 79169, '4F': 74757, '2G': 81305, '2H': 74564}
run_name = lambda t: f'v21_famA_d3_{t}e_crnd3'  # noqa: E731


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


def git_commit(paths, msg):
    """heff: the research queue committed its outputs to git here; the packaged queue only logs."""
    log(f'done (not committed): {msg}')


def limit():
    lim = int(14.3e9)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


todo = [t for t in sorted(DIMS, key=lambda t: -DIMS[t]) if not os.path.exists(f'results/{run_name(t)}.npz')]
ready = [t for t in sorted(DIMS, key=lambda t: -DIMS[t]) if t not in todo and not os.path.exists(os.path.join(HERE, 'a2', f'a2_{t}.json'))]
log(f'start: ATOMCI {todo}, analysis pending {ready}, free {free_gb():.0f} GB')
running = {}      # t -> dict(p, t0, stopped)
ana = None        # (t, p, t0)
done_ana = {t for t in DIMS if os.path.exists(os.path.join(HERE, 'a2', f'a2_{t}.json'))}
failed = set()
stop = False
while todo or running or ready or ana:
    # ATOMCI jobs
    for t, j in list(running.items()):
        rc = j['p'].poll()
        run = run_name(t)
        if rc is None:
            if time.time() - j['t0'] > GUARD and not os.path.exists(os.path.join('runs', run, 'run.log')):
                j['p'].kill()
                log(f'GUARD kill {run} (> 10 h)')
            continue
        ok = os.path.exists(f'results/{run}.npz')
        log(f'done {run} n {DIMS[t]} rc {rc} npz {ok} {time.time() - j["t0"]:.0f} s')
        if ok and rc == 0:
            clean(run)
            ready.append(t)
        else:
            failed.add(t)
        del running[t]
    # memory gate: exporting = alive with run.log
    exporting = [t for t, j in running.items() if os.path.exists(os.path.join('runs', run_name(t), 'run.log'))]
    active = [t for t in exporting if not running[t]['stopped']]
    allow = 0 if ana else 1
    for t in active[allow:]:
        os.kill(running[t]['p'].pid, signal.SIGSTOP)
        running[t]['stopped'] = True
        log(f'SIGSTOP export {run_name(t)} (memory gate)')
    if not ana and not active:
        wait = [t for t in exporting if running[t]['stopped']]
        if wait:
            t = wait[0]
            os.kill(running[t]['p'].pid, signal.SIGCONT)
            running[t]['stopped'] = False
            log(f'SIGCONT export {run_name(t)}')
    # analysis
    if ana:
        t, p, t0 = ana
        rc = p.poll()
        if rc is not None:
            ok = rc == 0 and os.path.exists(os.path.join(HERE, 'a2', f'a2_{t}.json'))
            log(f'analysis {t} rc {rc} {time.time() - t0:.0f} s')
            ana = None
            if ok:
                done_ana.add(t)
                info = json.load(open(os.path.join(HERE, 'a2', f'a2_{t}.json')))
                git_commit([os.path.join(HERE, 'a2', f'a2_{t}.npz'), os.path.join(HERE, 'a2', f'a2_{t}.json'),
                            os.path.join(HERE, f'phase21f_block_{t}.log'), LOG],
                           f"Phase 21 A2: block {t} (famA {info['n_famA']} / fam {info['n_fam']} CSFs; "
                           f"|H(P,class)| {info['HPclass_max']:.1e})")
            else:
                failed.add('ana:' + t)
    exporting_now = [t for t, j in running.items()
                     if os.path.exists(os.path.join('runs', run_name(t), 'run.log')) and not j['stopped']]
    if not ana and ready and not exporting_now:
        t = ready.pop(0)
        nthr = str(max(1, 4 - len(running)))
        p = subprocess.Popen([PY, _C.script('phase21f.py'), 'block', t],
                             stdout=open(os.path.join(HERE, f'phase21f_block_{t}.log'), 'w'), stderr=subprocess.STDOUT,
                             preexec_fn=limit, env=dict(os.environ, OMP_NUM_THREADS=nthr, OPENBLAS_NUM_THREADS=nthr))
        ana = (t, p, time.time())
        log(f'analysis {t} start ({nthr} threads)')
    # start ATOMCI
    while not stop and todo and len(running) < NSLOT:
        if free_gb() < 150:
            log(f'DISK-STOP free {free_gb():.0f} GB')
            stop = True
            break
        t = todo.pop(0)
        run = run_name(t)
        shutil.rmtree(os.path.join('runs', run), ignore_errors=True)
        p = subprocess.Popen([PY, _C.script('run_atomci.py'), run, '--vectors-from', 'vq_orb_crnd3'],
                             stdout=open(os.path.join(HERE, f'phase21f_atomci_{t}.log'), 'w'), stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        running[t] = dict(p=p, t0=time.time(), stopped=False)
        log(f'start {run} n {DIMS[t]} free {free_gb():.0f} GB')
    if stop and not running and not ana and not ready:
        break
    time.sleep(20)
log(f'end; ATOMCI not started {todo}; failed {sorted(failed)}; analysed {sorted(done_ana)}')
if len(done_ana) == len(DIMS):
    r = subprocess.run([PY, _C.script('phase21f.py'), 'final'], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS='2'))
    open(os.path.join(HERE, 'phase21f_final.log'), 'w').write(r.stdout + r.stderr)
    log(f'final rc {r.returncode}')
    if r.returncode == 0:
        git_commit([os.path.join(HERE, 'phase21f.json'), os.path.join(HERE, 'phase21f_final.log'), LOG],
                   'Phase 21 A2: final (T decomposition of the 7 blocks)')
