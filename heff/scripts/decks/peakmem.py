#!/usr/bin/env python3
"""Run a command; print its wall time and peak RSS (ru_maxrss of the child, MB). usage: peakmem.py <label> <cmd...>"""
import resource
import subprocess
import sys
import time

t0 = time.time()
r = subprocess.run(sys.argv[2:])
print(f'PEAKMEM {sys.argv[1]} maxrss_MB {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.0f} '
      f'wall_s {time.time() - t0:.1f} rc {r.returncode}', flush=True)
sys.exit(r.returncode)
