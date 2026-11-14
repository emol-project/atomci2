import numpy as np
from fitlib import *
from ops import closure, project_out, dN_ops
d = D6()
# --- O2'' in d6 via closure combination fixed in d3
o3 = dN_ops(3); B3 = o3['ops2']; B6 = d.o['ops2']
e1_3 = np.load('e1_d3.npy'); e2_3 = np.load('e2_d3.npy')
Wc3, _ = project_out(closure(3, 'core', {2: 1.0}), B3); Wx3, _ = project_out(closure(3, 'x', {2: 1.0}), B3)
M = np.array([Wc3.ravel(), Wx3.ravel()]).T
a1 = np.linalg.lstsq(M, e1_3.ravel(), rcond=None)[0]; a2 = np.linalg.lstsq(M, e2_3.ravel(), rcond=None)[0]
Wc6, _ = project_out(closure(6, 'core', {2: 1.0}), B6); Wx6, _ = project_out(closure(6, 'x', {2: 1.0}), B6)
E1 = a1[0]*Wc6 + a1[1]*Wx6; E2 = a2[0]*Wc6 + a2[1]*Wx6
print('d6 norms (expect 2,2):', np.linalg.norm(E1), np.linalg.norm(E2), ' <E1,E2> =', np.sum(E1*E2))
np.save('E1_d6.npy', E1); np.save('E2_d6.npy', E2)
Og = np.load('Ogamma_d6.npy')
c = 403.4228550788862
print('check: pure3(O_gamma)·E1/|E1|^2*... T1 per unit gamma =', np.sum(np.load('Ogamma3_d6.npy')*E1)/4)
names = ['A', 'B', 'C', 'zeta', 'alpha', 'beta', 'gamma']
p0 = [88241, 965, 3833, 437, 34, -575, 4.83]
base = d.fit(names, {'gamma': Og}, p0)
ref = base['refv']
print('ENL base: gamma %.3f +- %.3f, mean %.1f' % (base['p'][-1], base['err'][-1], base['mean']))
# --- (1) leakage: synthetic data = base model + tau*E2 ; refit Trees model
ops = {**d.ops, 'gamma': Og}
def synth(p, tau, g_extra=0.0):
    H = sum(pi*ops[n] for pi, n in zip(p, names)) + tau*E2 + g_extra*Og
    lev = d.levels(H); ks = d.assign(lev, OBS, ref)
    return np.array([lev[J][0][ks[i]] for i, (t, J, e) in enumerate(OBS)])
for tau in [100, 500, 1143.2]:
    Es = synth(base['p'], tau)
    f = d.fit(names, {'gamma': Og}, base['p'], ref=ref, Es=Es)
    print('tau(T2)=%7.1f cm-1  -> gamma_fit - gamma_true = %+.4f   (kappa=%.5f per cm-1, = %.4f in T1/T2 units)  rms %.1f' % (
        tau, f['p'][-1]-base['p'][-1], (f['p'][-1]-base['p'][-1])/tau, (f['p'][-1]-base['p'][-1])*c/tau, f['mean']))
# --- (2) same but O2'' leakage onto each param
Es = synth(base['p'], 1143.2); f = d.fit(names, {'gamma': Og}, base['p'], ref=ref, Es=Es)
print('param shifts for T2=1143 cm-1:', dict(zip(names, np.round(f['p']-base['p'], 2))))
# --- (3) fit experiment with O2'' free
names3 = names + ['tau']
f3 = d.fit(names3, {'gamma': Og, 'tau': E2}, list(base['p']) + [0.0], ref=ref)
print('\nexpt fit with O2\'\' free: gamma %.3f +- %.3f, T2 = %.0f +- %.0f cm-1 (= %.2f +- %.2f mEh), mean %.1f, corr(g,tau)=%.2f' % (
    f3['p'][6], f3['err'][6], f3['p'][7], f3['err'][7], f3['p'][7]/219.4746, f3['err'][7]/219.4746, f3['mean'],
    f3['cov'][6, 7]/np.sqrt(f3['cov'][6, 6]*f3['cov'][7, 7])))
print('   others', dict(zip(names3, np.round(f3['p'], 1))))
# --- (4) fit experiment with 3-body fixed to our direction (T2/T1 = ours), one scale
T1o, T2o = 11.324, 5.209
dirop = Og + (T2o/T1o)*c*E2      # gamma*(O_gamma + r c E2): pure-3 part = gamma*c*(E1 + r E2)
f4 = d.fit(names, {'gamma': dirop}, base['p'], ref=ref)
print('expt fit with 3-body along OUR direction (T2/T1=%.3f): gamma %.3f +- %.3f, mean %.1f' % (T2o/T1o, f4['p'][-1], f4['err'][-1], f4['mean']))
# --- (5) our prediction as Trees would extract it:  gamma_fit = gamma_true + kappa*T2
