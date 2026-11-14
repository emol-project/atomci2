#!/usr/bin/env python3
"""Phase 18 queue (copy of phase17_queue.py; dims results/phase18/phase18_dims.json, log results/phase18/phase18_runs.log, analysis tags 18_<ion>_<space>).
Phase 17 B1 queue (copy of phase15u_queue.py; run names v17_*, dims phase17_dims.json, script phase17.py, analysis
tags 17_<ion>_<space>; added: memory stop, the summed RSS of all running work is sampled every loop and above MEMSTOP_MB
(default 14336) everything is killed and the queue stops, MEM-STOP).
Phase 15u queue (copy of phase15t_queue.py): stop-after-H ATOMCI jobs + iterative analyses on 4 cores.

Steps 1 and 2 (3-hole groups in k1fh, n = 3 complex by hole count), one tier, longest first. A waiting analysis that does
not fit under the memory cap holds back new jobs until it fits. A job starts only if the summed memory
estimate of everything running stays <= CAP_MB. When the 7 blocks of a space are done its analysis (iterative path)
is queued at the front. Per-job timeout JOB_H (default 2 h), total wall until DEADLINE (epoch s, argv[1]); unfinished work is
killed / not started and logged. Every finished job and analysis is committed.
After the disk-full night of 22-23 Oct:
  - before each job: free space on the runs disk >= 2 x its expected run-directory size (phase15u_diskfit.json, x1.15
    margin) plus the expected size of every running job, otherwise the queue stops starting work (DISK-STOP);
  - true return codes: a job is ok only with results/<run>.npz and run.log rc 0; rc 124 = timeout, a job that exits 0
    without its export is logged FAILED rc 1. The first real failure (not a timeout) stops the queue: nothing new
    starts, running work finishes, then the queue exits;
  - after an analysis is committed (rc 0, commit rc 0), the large fort.* (>= 100 MB) of its 7 runs are deleted, unless
    listed in results/phase15/run_files_keep.txt; size and mtime go to runs/<run>/fort_deleted.json.
After the 25 Oct all-f overrun: every loop the queue projects its finish (greedy over NCORE slots: running jobs'
remaining time + pending jobs, each fit estimate x the actual/fit ratio of the jobs finished so far in this queue,
plus ANA_S per pending analysis). When the projection passes the deadline it logs a WARNING (again at most every 30
min while it stays over, and once when it comes back under) and prints all warnings at the end.
Log: results/phase17/phase17_runs.log, analyses: phase17_ana.log. argv[2:]: ion:space pairs (default STEP1);
'--' separates tier 1 from tier 2"""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
os.chdir(ROOT)
LOG, ANA = 'results/phase18/phase18_runs.log', 'results/phase18/phase18_ana.log'
ANA_ERR = 'results/phase18/phase18_ana.err'
CAP_MB, NCORE = float(os.environ.get("CAP_MB", "11000")), int(os.environ.get("MAXJ", "3"))   # 15u: <= 3 large jobs
JOB_S = float(os.environ.get("JOB_H", "2")) * 3600      # per-job timeout
# per-space override, e.g. JOB_H_SPACE="u3c1:2.5" (Step 3: 2.5 h for u3c1 only)
SKIP = set(x for x in os.environ.get('SKIP', '').split(',') if x)       # space:term held back, e.g. u5:2H
JOB_S_SP = {k: float(v) * 3600 for k, v in (x.split(':') for x in os.environ.get('JOB_H_SPACE', '').split(',') if x)}
TAGS = {'cond': 'cond6', 'coti1': 'coti16', 'coti5': 'coti56', 'coop': 'coop6', 'codk': 'codk6', 'cot2': 'cot26', 'coqd': 'coqd6', 'fedc': 'fedc6', 'codc': 'codc6', 'fe5nd': 'fe5nd3', 'fe5ti5': 'fe5ti53', 'fe5op': 'fe5op3', 'fe5dk': 'fe5dk3', 'fe5t2': 'fe5t23', 'fend': 'fend6', 'cr': 'cr15n3', 'v': 'o14n3', 'mn': 'mn15b3', 'crnd': 'crnd3', 'vnd': 'vnd3', 'mnnd': 'mnnd3', 'crtfm': 'crtfm3', 'crtft': 'crtft3', 'crtfb': 'crtfb3', 'crtg': 'crtg3', 'crth': 'crth3',
        'crti1': 'crti13', 'crti2': 'crti23', 'crti3': 'crti33', 'crti4': 'crti43', 'crti5': 'crti53', 'crtk1': 'crtk3',
        'vop': 'vop3', 'vdk': 'vdk3', 'mnop': 'mnop3', 'mndk': 'mndk3', 'feop': 'feop6', 'fedk': 'fedk6', 'crop': 'crop3', 'crdk': 'crdk3', 'crt2': 'crt23', 'crt3': 'crt33', 'crt2g': 'crt2g3',
        'vti1': 'vti13', 'vti5': 'vti53', 'mnti1': 'mnti13', 'mnti5': 'mnti53', 'feti1': 'feti16', 'feti5': 'feti56',
        'fet2': 'fet26', 'crt2i': 'crt2i3', 'vqd': 'vqd3', 'mnqd': 'mn15b3', 'feqd': 'feqd6',
        'cr4f': 'cr4f3', 'crno': 'crno3'}
STEP1 = [('crnd', s) for s in ('u1ab', 'u1a', 'n3h0', 'n3h1', 'n3h2', 'n3h3', 'n3h4', 'n3h5', 'n3h6', 'n3h7')]
_A = sys.argv[2:]
# each '--' starts a new tier: a tier's jobs start only when every earlier tier is all started (1 Nov: > 2 tiers)
_T = [[]]
for _x in _A:
    if _x == '--':
        _T.append([])
    else:
        _T[-1].append(tuple(_x.split(':')))
TIERS = [_T[0] or STEP1] + _T[1:]
KEEP = 'results/phase15/run_files_keep.txt'
# 9 Nov (DISK-STOP at 05:36): DISKFIT=<json> replaces the 15u fit (re-derived from the Phase 18 terms already run);
# DEPTH_FIRST=1 with one space per tier: a space's terms all finish before a term of the next space starts
DFIT = json.load(open(os.path.join(ROOT, os.environ.get('DISKFIT', 'results/phase15/phase15u_diskfit.json'))))
DEPTH = os.environ.get('DEPTH_FIRST') == '1'
QSCRIPT = os.environ.get('QSCRIPT', _C.script('phase17.py'))   # Step 7: scripts/phase15u7.py (15q basis, ions cr v mn)


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')


def commit(msg, paths):
    """heff: the research queue committed each finished job to git here; the packaged queue only logs."""
    log(f'done (not committed): {msg}')
    return 0


NEL = {'fend': 6, 'feop': 6, 'fedk': 6, 'feti1': 6, 'feti5': 6, 'fet2': 6, 'feqd': 6,
       'cond': 6, 'coti1': 6, 'coti5': 6, 'coop': 6, 'codk': 6, 'cot2': 6, 'coqd': 6, 'fedc': 6, 'codc': 6}       # d count per ion (default 3)
V15U = ('fam_d5f2g2h1', 'k1fh_n3', 'u3c1', 'k1fh_sall', 'besth_sall', 'besth_sall_s4')   # spaces named by phase15u (v15u_*), the rest v17_*
V15T = ('k1fh',)                        # 30 Oct: phase15t names (v15t_*)


def name(ion, sp, t):
    pre = 'v15u' if sp in V15U else 'v15t' if sp in V15T else 'v17'
    return f'{pre}_{sp}_d{NEL.get(ion, 3)}_{t}e_{TAGS[ion]}'


def ana_json(ion, sp):
    """analysis output: phase15k_* (d3, K.analyze3) or phase15j_* (d6, J.analyze2)"""
    return f'results/phase15/phase15{"j" if NEL.get(ion, 3) == 6 else "k"}_18_{ion}_{sp}.json'


def npz_ok(n):
    """npz exists and its zip directory reads (a truncated export, as on the disk-full night, fails here)."""
    try:
        with zipfile.ZipFile(f'results/{n}.npz') as z:
            return len(z.namelist()) > 0
    except Exception:  # noqa: BLE001
        return False


def ok(n):
    try:
        return npz_ok(n) and json.load(open(f'runs/{n}/run.log'))['returncode'] == 0
    except Exception:  # noqa: BLE001
        return False


def disk_job(n):      # expected run-directory bytes, with a x1.15 margin over the fit
    return 1.15 * math.exp(DFIT['c'] + DFIT['exp'] * math.log(n))


def clean(n):
    """Delete the large fort.* of run n (npz exported, analysis committed) unless kept; returns bytes freed."""
    keep = {l.split('\t')[0] for l in open(KEEP) if not l.startswith('#')} if os.path.exists(KEEP) else set()
    d = f'runs/{n}'
    if not npz_ok(n) or not os.path.isdir(d):
        return 0
    big = [f for f in sorted(os.listdir(d)) if f.startswith('fort.') and f'{d}/{f}' not in keep and
           os.path.getsize(f'{d}/{f}') >= 100 * 2 ** 20]
    side = f'{d}/fort_deleted.json'
    rec = json.load(open(side)) if os.path.exists(side) else {}
    for f in big:
        st = os.stat(f'{d}/{f}')
        rec[f] = dict(size=st.st_size, mtime=st.st_mtime)
    json.dump(rec, open(side, 'w'), indent=1)
    freed = 0
    for f in big:
        freed += os.path.getsize(f'{d}/{f}')
        os.remove(f'{d}/{f}')
    return freed


def mem_job(n):        # fit to the 35 15q jobs: peak RSS 241 MB + 1.36e-6 N^2 MB (export dominates)
    return 300 + 1.5e-6 * n * n


def mem_ana(n):        # 15q: 3.37 GB for rasK2dall (largest block 40595)
    return 300 + 2.0e-6 * n * n


MEMSTOP_MB = float(os.environ.get('MEMSTOP_MB', '14336'))
# memory reservation for work outside the queue (1 Nov, item 9 sub-space analyses): a number (MB) in RESERVE counts
# against CAP_MB before any start; the queue writes its own running estimate (MB) to STATUS every loop
RESERVE, STATUS = 'results/phase18/queue_reserve_mb', 'results/phase18/queue_running_mb'


def reserve_mb():
    try:
        return float(open(RESERVE).read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def rss_mb(pgids):
    """Summed RSS (MB) of every process whose process group is in pgids."""
    tot = 0
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            st = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
            if int(st[2]) in pgids:
                tot += int(st[21]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, IndexError, ValueError):
            pass
    return tot / 2 ** 20


ANA_S = 600.0                  # projected analysis wall per space (s)
WARN = []
PEAK = [0.0]


def projected_end(run, pending, est, ratio, now):
    """Greedy finish estimate over NCORE slots."""
    slots = sorted(max(est(k) * ratio - (now - ts), 60.0) if k[0] == 'job' else ANA_S
                   for k, (p, m, ts) in run.items())
    slots = (slots + [0.0] * NCORE)[:max(NCORE, len(slots))]
    for k in pending:
        slots.sort()
        slots[0] += est(k) * ratio if k[0] == 'job' else ANA_S
    return now + max(slots, default=0.0)


class Adopted:
    """A job started by an earlier queue and still running (1 Nov reorder): its timeout wrapper is the session leader
    (start_new_session), so pid = pgid; poll() is None while /proc/<pid> exists, returncode stays None (ok() decides)."""
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return None if os.path.exists(f'/proc/{self.pid}') else 0


def running_jobs():
    """run name -> (pid, elapsed s) of the queue jobs already running (their 'timeout ... run_names <n>' wrapper)."""
    out = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            a = open(f'/proc/{d}/cmdline').read().split('\0')
            if a[0] == 'timeout' and 'run_names' in a:
                st = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
                el = time.time() - (_boot() + int(st[19]) / os.sysconf('SC_CLK_TCK'))
                out[a[a.index('run_names') + 1]] = (int(d), el)
        except (OSError, IndexError, ValueError):
            pass
    return out


def _boot():
    return float(next(l.split()[1] for l in open('/proc/stat') if l.startswith('btime')))


def main():
    deadline = float(sys.argv[1])
    dims = json.load(open('results/phase18/phase18_dims.json'))
    fit = json.load(open('results/phase15/phase15t_jobfit.json'))
    tiers = []
    for tier in TIERS:
        jobs = []
        for ion, sp in tier:
            for t, n in dims[f'{ion}:{sp}']['per'].items():
                if not ok(name(ion, sp, t)) and f'{sp}:{t}' not in SKIP:
                    jobs.append((float(fit['c'] + fit['exp'] * __import__('math').log(n)), n, ion, sp, t))
        tiers.append(sorted(jobs, reverse=True))
    spaces = [s for tier in TIERS for s in tier]
    ana_done, ana_q = set(), []
    run = {}
    stop = None
    jfit = {}                                   # (ion, sp, t) -> fit estimate (s)
    for tier in tiers:
        for j in tier:
            jfit[j[2:]] = math.exp(j[0])
    est = lambda k: jfit.get(k[1:], 3600.0)    # noqa: E731
    ext = running_jobs()                         # adopt jobs of an earlier queue (not restarted, tracked to the end)
    for tier in tiers:
        for j in list(tier):
            n = name(*j[2:])
            if n in ext:
                tier.remove(j)
                run[('job',) + j[2:]] = (Adopted(ext[n][0]), mem_job(j[1]), time.time() - ext[n][1])
                log(f'adopted running {n} (pid {ext[n][0]}, {ext[n][1] / 3600:.2f} h elapsed)')
    act, last_warn, over = [], 0.0, False
    n_ana = len([1 for ion, sp in spaces if not os.path.exists(ana_json(ion, sp))])
    log(f'start; deadline {time.strftime("%H:%M:%S", time.localtime(deadline))}, cap {CAP_MB} MB; tier sizes '
        f'{[len(t) for t in tiers]}')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HEFF_SOLVER='iter', HEFF_SPARSE_ABOVE='500')
    while True:
        for ion, sp in spaces:
            if (ion, sp) not in ana_done and (ion, sp) not in ana_q and \
                    not os.path.exists(ana_json(ion, sp)) and \
                    all(ok(name(ion, sp, t)) for t in dims[f'{ion}:{sp}']['per']) and \
                    not any(k[1:] == (ion, sp) for k in run):
                ana_q.append((ion, sp))
        cur = next((t for t in tiers if t), [])
        if DEPTH and any(k[0] == 'job' and k[1:3] not in {j[2:4] for j in cur} for k in run):
            cur = []            # depth-first: a term of the next space starts only when the previous space's jobs are done
        pending = [('ana', ion, sp) for ion, sp in ana_q] + [('job',) + j[2:] for j in cur]
        if stop:
            pending = []
        if not pending and not run:
            break
        now = time.time()
        ratio = sum(a for a, f in act) / sum(f for a, f in act) if act else 1.0
        n_ana_left = n_ana - len(ana_done) - len([k for k in run if k[0] == 'ana'])
        pend = [k for k in pending if k[0] == 'job'] + [('ana',)] * max(n_ana_left, 0)
        pe = projected_end(run, pend, est, ratio, now)
        if pe > deadline and (not over or now - last_warn > 1800):
            w = (f'WARNING projected finish {time.strftime("%H:%M", time.localtime(pe))} passes the total limit '
                 f'{time.strftime("%H:%M", time.localtime(deadline))} (actual/fit {ratio:.2f}, {len(pend)} pending, '
                 f'{len(run)} running)')
            log(w)
            WARN.append(time.strftime('%H:%M:%S ') + w)
            over, last_warn = True, now
        elif pe <= deadline and over:
            log(f'projected finish {time.strftime("%H:%M", time.localtime(pe))} back within the total limit')
            over = False
        r_mb = rss_mb({p.pid for p, m, ts in run.values()})
        PEAK[0] = max(PEAK[0], r_mb)
        rsv = reserve_mb()
        with open(STATUS, 'w') as f:
            f.write(f'{sum(v[1] for v in run.values()):.0f} {time.time():.0f}\n')
        if r_mb > MEMSTOP_MB:
            for k, (p, m, ts) in run.items():
                os.killpg(p.pid, signal.SIGTERM)
                log(f'KILLED (memory) {k}')
            stop = f'MEM-STOP: summed RSS {r_mb / 1024:.1f} GB > {MEMSTOP_MB / 1024:.1f} GB'
            log(stop)
            break
        if time.time() > deadline:
            for k, (p, m, ts) in run.items():
                os.killpg(p.pid, signal.SIGTERM)
                log(f'KILLED (total limit) {k}')
            for k in pending:
                log(f'NOT-STARTED (total limit) {k}')
            break
        for k in pending:
            if len(run) >= NCORE:
                break
            if k[0] == 'ana':
                ion, sp = k[1:]
                m = mem_ana(dims[f'{ion}:{sp}']['max'])
            else:
                ion, sp, t = k[1:]
                m = mem_job(dims[f'{ion}:{sp}']['per'][t])
                need = 2 * disk_job(dims[f'{ion}:{sp}']['per'][t]) + \
                    sum(disk_job(dims[f'{q[1]}:{q[2]}']['per'][q[3]]) for q in run if q[0] == 'job')
                free = shutil.disk_usage('runs').free
                if free < need:
                    stop = f'DISK-STOP before {name(ion, sp, t)}: free {free / 1e9:.1f} GB < needed {need / 1e9:.1f} GB'
                    log(stop)
                    break
            if sum(v[1] for v in run.values()) + m + rsv > CAP_MB and (run or rsv):
                if k[0] == 'ana':
                    break           # hold new jobs until the analysis fits
                continue
            e = dict(env, ION=ion)
            if k[0] == 'ana':
                tag = f'18_{ion}_{sp}'
                cmd = [sys.executable, _C.script('peakmem.py'), f'ana {tag}', sys.executable, QSCRIPT,
                       'analyze', tag, sp]
                p = subprocess.Popen(cmd, env=e, stdout=open(ANA, 'a'), stderr=open(ANA_ERR, 'a'), start_new_session=True)
                ana_q.remove((ion, sp))
                log(f'begin analysis {tag} (mem est {m / 1024:.1f} GB)')
            else:
                n = name(ion, sp, t)
                cmd = ['timeout', str(JOB_S_SP.get(sp, JOB_S)), sys.executable, _C.script('peakmem.py'), f'job {n}', sys.executable,
                       QSCRIPT, 'run_names', n]
                p = subprocess.Popen(cmd, env=e, stdout=open(LOG, 'a'), stderr=subprocess.DEVNULL, start_new_session=True)
                for tier in tiers:
                    tier[:] = [j for j in tier if j[2:] != (ion, sp, t)]
                log(f'begin {n} (N {dims[f"{ion}:{sp}"]["per"][t]}, mem est {m / 1024:.1f} GB, '
                    f'timeout {JOB_S_SP.get(sp, JOB_S) / 3600:.1f} h)')
            run[k] = (p, m, time.time())
        for k, (p, m, ts) in list(run.items()):
            if p.poll() is None:
                continue
            del run[k]
            if k[0] == 'ana':
                tag = f'18_{k[1]}_{k[2]}'
                ana_done.add(k[1:])
                js = ana_json(k[1], k[2])
                rc = p.returncode if p.returncode else (0 if os.path.exists(js) else 1)
                log(f'end analysis {tag} rc {rc} wall {time.time() - ts:.0f} s')
                crc = commit(f'Phase 18: analysis {tag}, iterative path',
                             [ANA, LOG, js, f'results/phase15/15k_terms/terms_cr3_{tag}.json',
                              f'results/phase15/15j_terms/terms_{tag}.json'])
                if rc:
                    stop = stop or f'STOP: analysis {tag} failed rc {rc}'
                    log(stop)
                elif crc == 0:
                    ion, sp = k[1:]
                    fr = sum(clean(name(ion, sp, t)) for t in dims[f'{ion}:{sp}']['per'])
                    log(f'cleaned {tag}: large fort.* deleted, {fr / 1e9:.1f} GB freed')
            else:
                n = name(*k[1:])
                rc = p.returncode if p.returncode else (0 if ok(n) else 1)
                st = 'end' if rc == 0 else 'TIMEOUT' if rc == 124 else 'FAILED'
                log(f'{st} {n} rc {rc} wall {time.time() - ts:.0f} s')
                if rc == 0:
                    act.append((time.time() - ts, est(k)))
                if rc:
                    ana_done.add(k[1:3])          # space incomplete: no analysis
                if rc not in (0, 124):
                    stop = stop or f'STOP: first real failure {n} rc {rc}'
                    log(stop)
                commit(f'Phase 18: job {n} finished', [LOG])
        time.sleep(20)
    log(f'peak summed RSS {PEAK[0] / 1024:.2f} GB')
    log(f'done{" (" + stop + ")" if stop else ""}')
    print(f'{len(WARN)} projected-finish warning(s)' + ''.join('\n  ' + w for w in WARN), flush=True)
    commit('Phase 18: queue finished', [LOG])
    sys.exit(1 if stop else 0)


if __name__ == '__main__':
    main()
