#!/usr/bin/env python3
"""Phase 17 item 6 (31 Oct brief, Fe2+ T2 definition): recompute the exact H_eff blocks of the Fe2+ chain spaces
(phase15j.analyze2 with HEFF_DUMP; tag q11_<ion>_<space>, so the stored phase15j_17_* results are untouched) ->
results/phase17/heff_dump/q11_<ion>_<space>_<space>_iter.npz; Fe2+ chain spaces, the DK3 pair and the i runs ({exact|sylvester|closure|php}|<term>: block, canonical seniority
basis, Eh). Runs two analyses at a time in the measured memory pairs (fam 9.4 GB + k1fh 3.5 GB, ...), ≤ 13 GB summed;
meant to run between the two queues (nothing else large running). Each analysis is wrapped by peakmem.py.
usage: phase17q11_dump.py"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
DUMP = os.path.join(ROOT, 'results', 'phase17', 'heff_dump')
LOG = os.path.join(ROOT, 'results', 'phase17', 'phase17q11_dump.log')
# two lanes of (ion, space); analysis peaks (GB, measured or by the queue rule): fend fam 9.4, feti5 k1fhi 8.9,
# feti1 k1fhi 5.5, u3c1 6.9 | k1fh_sall 4.4, k1fh 3.5, k1fh_n3 3.5, feop / fedk rasK1r < 2; the large ones in lane 0
LANES = [[('fend', 'fam_d5f2g2h1'), ('feti5', 'k1fhi'), ('feti1', 'k1fhi'), ('fend', 'u3c1')],
         [('fend', 'k1fh'), ('fend', 'k1fh_n3'), ('fend', 'k1fh_sall')]]      # feop / fedk rasK1r: run at once (1 Nov)


def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def cmd(ion, space):
    env = dict(os.environ, ION=ion, HEFF_DUMP=DUMP, HEFF_SOLVER='iter', HEFF_SPARSE_ABOVE='500',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return [sys.executable, _C.script('peakmem.py'), f'ana q11_{ion}_{space}', sys.executable, _C.script('phase17.py'),
            'analyze', f'q11_{ion}_{space}', space], env


def main():
    os.makedirs(DUMP, exist_ok=True)
    os.chdir(ROOT)
    log('start')
    running, todo, rc_all = {}, [list(x) for x in LANES], 0
    while running or any(todo):
        for i, lane in enumerate(todo):
            if i not in running and lane:
                ion, sp = lane.pop(0)
                c, env = cmd(ion, sp)
                sp = f'{ion}_{sp}'          # dump file: <tag>_<space>_iter.npz with tag q11_<ion>_<space>
                running[i] = (sp, subprocess.Popen(c, env=env, stdout=open(os.path.join(DUMP, f'{sp}.log'), 'w'),
                                                   stderr=subprocess.STDOUT), time.time())
                log(f'begin {sp}')
        time.sleep(20)
        for i in list(running):
            sp, p, ts = running[i]
            if p.poll() is not None:
                ok = p.returncode == 0 and os.path.exists(os.path.join(DUMP, f'q11_{sp}_{sp.split("_", 1)[1]}_iter.npz'))
                rc_all |= 0 if ok else 1
                log(f'end {sp} rc {p.returncode} dump {"ok" if ok else "MISSING"} wall {time.time() - ts:.0f} s')
                del running[i]
    log(f'done rc {rc_all}')
    sys.exit(rc_all)


if __name__ == '__main__':
    main()
