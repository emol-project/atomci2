import numpy as np
from ops import *
np.set_printoptions(precision=6, suppress=True)
for N in (3, 6):
    o = dN_ops(N)
    B2 = o['ops2']
    print('N', N, 'dim', len(o['dets']), 'rank 2-body span', np.linalg.matrix_rank(np.array([b.ravel() for b in B2]).T))
    # pairing eigenvalues by seniority check later
    Wc = closure(N, 'core', {2: 1.0}) * 35**2   # Trees gamma = 1
    R3, c = project_out(Wc, B2)
    print('  ||pure3(O_gamma)||_F =', np.linalg.norm(R3), '  trace', np.trace(R3))
    np.save(f'Ogamma_d{N}.npy', Wc); np.save(f'Ogamma3_d{N}.npy', R3)
