#!/usr/bin/env python3
"""Level 1: the paper and SM tables from the H_eff dumps and the per-space result JSON (no ATOMCI, no stored H).

  rebuild_level1.py [--no-init] [--check]

1. init_work.py level1 (copy the inputs from HEFF_DATA into HEFF_WORK; skipped with --no-init)
2. every assembly step of the research repo, in the order that rebuilds all intermediate JSON from the dumps (none of
   the outputs is copied from HEFF_DATA; a failing step stops the rebuild)
3. --check: compare the tables with the reference copies in checks/reference/ byte for byte (exit 1 on a difference)

Outputs in HEFF_WORK/results/: phase18/paper_tables.md, phase19/phase19_tables.md, phase17/phase17a_tables.md,
phase17/phase17c_rule_tables.md, phase17/phase17_final_tables.md, phase18/phase18_fe5_tables.md (without the Q1_rule
appendix that the research repo added with phase18_q1diag.py from the stored H), phase21/phase21_tables.md and the JSON
listed in checks/outputs_level1.txt. About 7 min (OMP_NUM_THREADS defaults to 4 here)."""
import filecmp
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as C  # noqa: E402

STEPS = [('make_operators.py',), ('phase17a_tables.py',), ('phase17q11.py', 'chain')] + \
        [('phase18_co.py', c) for c in ('free', 'ana', 'refit', 'final', 'refit2', 'fefinal', 'ferefit2')] + \
        [('phase18_fe5.py', c) for c in ('final', 'second', 'tables', 'refit')] + \
        [('phase17q20.py', 'all'), ('phase17_final.py',)] + \
        [('phase19.py', c) for c in ('nist', 'fits', 'single')] + \
        [('phase18_tables.py',), ('phase19_report.py',), ('phase17c_rule_tables.py',), ('phase19b_side.py',),
         ('phase21_report.py',)]
# reference copy (checks/reference/<name>) -> output under HEFF_WORK/results
TABLES = {'paper_tables.md': 'phase18/paper_tables.md', 'phase19_tables.md': 'phase19/phase19_tables.md',
          'phase17a_tables.md': 'phase17/phase17a_tables.md', 'phase17c_rule_tables.md': 'phase17/phase17c_rule_tables.md',
          'phase21_tables.md': 'phase21/phase21_tables.md', 'phase17_final_tables.md': 'phase17/phase17_final_tables.md'}


def run(log=print):
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '4')
    for st in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, C.script(st[0])] + list(st[1:]), cwd=C.root(), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            raise SystemExit(f'heff: step {" ".join(st)} failed (rc {r.returncode}):\n{r.stderr[-3000:]}')
        log(f'ok {time.time() - t:5.0f} s  {" ".join(st)}')
    log(f'level 1 done in {time.time() - t0:.0f} s; HEFF_WORK={C.WORK}')


def check():
    bad = []
    for ref, out in TABLES.items():
        a, b = os.path.join(C.HEFF, 'checks', 'reference', ref), os.path.join(C.WORK, 'results', out)
        same = os.path.exists(b) and filecmp.cmp(a, b, shallow=False)
        print(('identical ' if same else 'DIFFERENT ') + out)
        if not same:
            bad.append(out)
    return bad


if __name__ == '__main__':
    if '--no-init' not in sys.argv:
        import init_work
        init_work.init('level1')
    run()
    if '--check' in sys.argv and check():
        sys.exit(1)
