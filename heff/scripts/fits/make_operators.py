#!/usr/bin/env python3
"""The Trees / Uylings operator matrices of d3 and d6 (Ogamma*, e1/e2 of d3, E1/E2 of d6) -> HEFF_WORK/results/phase15/
15i_trees/*.npy, by the research steps scripts/fits/trees/step1.py, step2.py, step4.py (run with that directory as the
working directory, as in the research repo). Needed by phase15u_crfit (every NIST fit). Seconds."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as C  # noqa: E402

OUT = ('Ogamma_d3.npy', 'Ogamma3_d3.npy', 'Ogamma_d6.npy', 'Ogamma3_d6.npy', 'e1_d3.npy', 'e2_d3.npy', 'E1_d6.npy',
       'E2_d6.npy')


def main():
    os.makedirs(C.TREES, exist_ok=True)
    code = os.path.join(C.SCRIPTS, 'fits', 'trees')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([code, C.HEFF] + ([os.environ['PYTHONPATH']]
                                                                       if os.environ.get('PYTHONPATH') else [])))
    for s in ('step1.py', 'step2.py', 'step4.py'):
        subprocess.run([sys.executable, os.path.join(code, s)], cwd=C.TREES, env=env, check=True,
                       stdout=subprocess.DEVNULL)
    miss = [f for f in OUT if not os.path.exists(os.path.join(C.TREES, f))]
    if miss:
        raise SystemExit(f'heff: operator files not written: {miss}')
    print('operators written to', C.TREES)


if __name__ == '__main__':
    main()
