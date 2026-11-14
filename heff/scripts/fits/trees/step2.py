import numpy as np
from ops import *
N = 3
o = dN_ops(N); B2 = o['ops2']
e1 = np.load('Ogamma3_d3.npy'); e1 /= np.linalg.norm(e1)
vecs = {}
for kind, rad in [('g', {2: 1.0}), ('g', {4: 1.0}), ('g', {2: 1.0, 4: 0.6}), ('x', {2: 1.0}), ('x', {4: 1.0}), ('x', {0: 1.0}), ('x', {0: 1.0, 2: 1.0, 4: 0.6})]:
    W = closure(N, kind, rad)
    R3, _ = project_out(W, B2)
    vecs[(kind, tuple(sorted(rad.items())))] = R3
# plane
M = np.array([v.ravel() for v in vecs.values()] + [e1.ravel()]).T
s = np.linalg.svd(M, compute_uv=False); print('singular values of closures (3-body parts):', s[:5]/s[0])
# e2: orthogonal to e1 in plane, from x(k=2)
v = vecs[('x', ((2, 1.0),))].copy(); v -= np.sum(v*e1)*e1; e2 = v/np.linalg.norm(v)
# sign: d-path (x, all k) positive T2
vx = vecs[('x', ((0, 1.0), (2, 1.0), (4, 0.6)))]
if np.sum(vx*e2) < 0: e2 = -e2
np.save('e1_d3.npy', e1); np.save('e2_d3.npy', e2)
for key, v in vecs.items():
    t1, t2 = np.sum(v*e1), np.sum(v*e2)
    out = np.linalg.norm(v - t1*e1 - t2*e2)/np.linalg.norm(v)
    print(key, ' angle %.2f deg' % np.degrees(np.arctan2(t2, t1)), ' out-of-plane %.1e' % out)
# g path: angle vs R4/R2
print('g-path angle vs R4/R2:')
for r in [0, 0.3, 0.5, 0.7, 1.0, 1.5]:
    v = vecs[('g', ((2, 1.0),))] + r**2*vecs[('g', ((4, 1.0),))]
    # cross term needed: compute directly
    W = closure(N, 'g', {2: 1.0, 4: r}); v, _ = project_out(W, B2)
    print('  r=%.1f  %.1f deg' % (r, np.degrees(np.arctan2(np.sum(v*e2), np.sum(v*e1)))))
