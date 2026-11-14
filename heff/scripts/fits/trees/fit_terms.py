"""Fit d6 TERM energies (no spin-orbit) with the Trees-type model, as the experiment is analysed.
usage: python3 fit_terms.py terms.json      (json: {"5D":[E], "3P":[E1,E2], ...} in cm-1, all 16 terms or a subset)
Models: (a) 2-body + gamma*O_gamma (Trees)   (b) 2-body + T1*E1 + T2*E2 (pure 3-body, our normalisation)
Prints gamma, T1, T2 (cm-1 and mEh).  Needs Ogamma_d6.npy, E1_d6.npy, E2_d6.npy from step1/step4."""
import sys, json, numpy as np
from fitlib import D6
H2E = 219474.6
d = D6()
Og, E1, E2 = (np.load(f) for f in ('Ogamma_d6.npy', 'E1_d6.npy', 'E2_d6.npy'))
lz, sz = d.o['Lz'], d.o['Sz']
LET = 'SPDFGHI'
def terms(H):
    out = {}
    for L in range(7):
        for S2 in (0, 2, 4):  # 2S = 0,2,4
            idx = np.where((abs(np.diag(lz)-L) < 1e-8) & (abs(np.diag(sz)-S2/2) < 1e-8))[0]
            if len(idx) == 0: continue
            w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
            l2 = np.einsum('ia,ij,ja->a', v, d.L2[np.ix_(idx, idx)], v); s2 = np.einsum('ia,ij,ja->a', v, d.S2[np.ix_(idx, idx)], v)
            sel = (abs(l2-L*(L+1)) < 1e-4) & (abs(s2-S2/2*(S2/2+1)) < 1e-4)
            if sel.any(): out['%d%s' % (S2+1, LET[L])] = (w[sel], v[:, sel], idx)
    return out
def fit(data, names, ops, p0, iters=50):
    p = np.array(p0, float)
    obs = [(t, k, e) for t, es in data.items() for k, e in enumerate(sorted(es))]
    E = np.array([e for *_, e in obs])
    for _ in range(iters):
        H = sum(a*ops[n] for a, n in zip(p, names)); T = terms(H)
        calc = np.array([T[t][0][k] for t, k, e in obs])
        J = np.array([[T[t][1][:, k]@ops[n][np.ix_(T[t][2], T[t][2])]@T[t][1][:, k] for n in names] for t, k, e in obs])
        dp = np.linalg.lstsq(J, E-calc, rcond=None)[0]; p += dp
        if abs(dp).max() < 1e-9: break
    r = E-calc; dof = max(len(E)-len(names), 1); s = np.sqrt(r@r/dof)
    err = np.sqrt(np.diag(np.linalg.pinv(J.T@J)))*s
    return p, err, s
ops = {**d.ops, 'gamma': Og, 'T1': E1, 'T2': E2}
two = ['A', 'B', 'C', 'alpha', 'beta']
def analyse(data):
    pa, ea, sa = fit(data, two + ['gamma'], ops, [85000, 975, 3600, 69, -156, 5])
    pb, eb, sb = fit(data, two + ['T1', 'T2'], ops, [85000, 975, 3600, 69, -156, 2000, 700])
    print('(a) Trees model : gamma = %.3f +- %.3f  rms %.1f' % (pa[-1], ea[-1], sa))
    print('(b) pure 3-body : T1 = %.2f +- %.2f mEh, T2 = %.2f +- %.2f mEh, gamma_eff = T1/c = %.3f, rms %.2f' % (
        pb[5]/219.4746, eb[5]/219.4746, pb[6]/219.4746, eb[6]/219.4746, pb[5]/403.4228550788862, sb))
    print('    2-body: ' + ', '.join('%s %.1f' % (n, v) for n, v in zip(two, pb[:5])))
if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyse(json.load(open(sys.argv[1])))
    else:  # self-test: synthetic spectrum = expt-fit 2-body + our T1,T2 (15h) + nothing else
        p = dict(A=84763.6, B=975.0, C=3605.2, alpha=69.3, beta=-155.6)
        H = sum(v*ops[k] for k, v in p.items()) + 11.324*219.4746*E1 + 5.209*219.4746*E2
        data = {t: list(v[0]) for t, v in terms(H).items()}
        print('self-test, 16 terms:', {t: len(v) for t, v in data.items()})
        analyse(data)
