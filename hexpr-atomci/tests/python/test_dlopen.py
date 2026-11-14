"""The shared library loads at run time, and its thread-local caches work in new threads.

usage: python test_dlopen.py path/to/libhexpr_atomci.so
"""
import ctypes
import os
import sys
import threading

lib = ctypes.CDLL(os.path.abspath(sys.argv[1]))
lib.ac_sixj.restype = ctypes.c_double
lib.ac_sixj.argtypes = [ctypes.c_int] * 6
out = []


def work():
    # {1 1 1; 1 1 1} = 1/6, with the arguments as 2j
    out.append(lib.ac_sixj(2, 2, 2, 2, 2, 2))


threads = [threading.Thread(target=work) for _ in range(4)]
work()
for t in threads:
    t.start()
for t in threads:
    t.join()
if len(out) != 5 or any(abs(v - 1.0 / 6.0) > 1e-14 for v in out):
    sys.exit(f"FAILED: {out}")
print("dlopen passed")
