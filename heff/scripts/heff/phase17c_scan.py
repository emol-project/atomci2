#!/usr/bin/env python3
"""Phase 17 C class scan: start from C-e (balanced, E_ref = p_j) and move one Q2 class at a time into Q1; then the
greedy sequence (add the class with the largest summed |dT2| over the ions, repeat until |T2 - exact| < 0.3 mEh and
|T1/exact - 1| < 3 % for every ion).

Q2 classes (Q minus C-e; first matching rule wins):
  inner     1s/2s/2p hole
  pf2       3p hole + f (3p -> f with a second excitation; the 3p5 3d^N nf singles are in C-b)
  3s_v      3s hole + virtual
  3p2_v     two 3p holes + virtual (3p2 -> 3d2 + virtual)
  3p_v      one 3p hole + virtual, no f
  s, p, g, h, f   virtual electrons of one l only (s: the 3d^(N-1) ns buffer is in C-d; g includes the 3d^(N-1) ng
                  singles; f: no 3p hole)
  vd        virtual d only (the 3d^(N-1) nd singles are in C-e, so doubles and more)
  rest      virtual electrons of mixed l
Same construction as phase17c 'balanced'. P+Q1 up to DENSE_MAX CSFs per block is diagonalized densely (as phase17c);
larger P+Q1 by Davidson on the dressed operator H11 + X D_j^-1 X^T, applied through two products with the full stored H
(no copy of X), lowest m+1 roots, the root with the largest P overlap taken.

Ion setup (env, as phase17c): crnd: ION=crnd <space>; mnnd: ION=mnnd besth_sall; fend: C_N=6 A_RUNFMT=... A_ION=fend.
Rule-based Q1 (sets named 'R' or 'R-<sub>', the same command): Q1_rule = n = 3 complex (n3) + 3p -> f singles (pf) +
all 3d -> virtual singles 3d^(N-1) nl (1s, 1d, 1g, 1i; 1p, 1f, 1h are odd and absent) + the 3d^2 -> v^2 doubles with
both electrons in one l, every virtual orbital of that l (2s, 2p, 2d, 2f, 2g, 2h, 2i); 'R-<sub>' leaves one out.
Exact sub-space (sets 'X-<orb>,<orb>,...', 1 Nov item 9): P+Q1 = every configuration without any of the listed virtual
orbitals, the rest dropped (no dressing); exact des Cloizeaux H_eff of that sub-space by Davidson (m + 2 lowest roots,
the m with the largest P weight). 'X-' (nothing dropped) is the exact H_eff of the full space (check).
SCAN_DUMP=<file.npz>: the H_eff blocks of every set ('<set>|<term>', canonical seniority basis, Eh).
usage: phase17c_scan.py eval <space> <out.json> <set> ...   set = '+'-joined class names after C-e ('' = C-e), or R[-sub]
       phase17c_scan.py greedy                               (driver; runs the three ions as subprocesses)"""
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase17c as C  # noqa: E402
import atomci_heff.approx as AP  # noqa: E402
from atomci_heff.sparse import SymLower, davidson  # noqa: E402

A = C.A
DENSE_MAX = int(os.environ.get('DENSE_MAX', '3000'))
CLASSES = [  # no '+' in a name: '+' joins the classes of a set
    'inner', 'pf2', '3s_v', '3p2_v', '3p_v', 's', 'p', 'vd', 'f', 'g', 'h', 'rest']


def qclass(label, N):
    o = AP.parse_label(label)
    if (o.get('1s', 0), o.get('2s', 0), o.get('2p', 0)) != (2, 2, 6):
        return 'inner'
    hs, hp = 2 - o.get('3s', 0), 6 - o.get('3p', 0)
    vir = {k: q for k, q in o.items() if int(k[:-1]) >= 11}
    ls = {k[-1] for k in vir}
    assert vir, label                       # every no-virtual configuration is in C-a
    if hp and 'f' in ls:
        return 'pf2'
    if hs:
        return '3s_v'
    if hp >= 2:
        return '3p2_v'
    if hp == 1:
        return '3p_v'
    if len(ls) == 1:
        return 'vd' if ls == {'d'} else ls.pop()
    return 'rest'


LS = 'spdfghi'
RSUB = ['n3', 'pf'] + ['1' + l for l in LS] + ['2' + l for l in LS]


def rclass(label, N):
    """sub-class of Q1_rule, or None."""
    if C.n3(label):
        return 'n3'
    if C.pf1(label):
        return 'pf'
    c, vir, inner = C.occ(label)
    ls = {k[-1] for k in vir}
    if not inner or len(ls) != 1:
        return None
    k = sum(vir.values())
    if (k == 1 and c == (2, 6, N - 1)) or (k == 2 and c == (2, 6, N - 2)):
        return f'{k}{ls.pop()}'
    return None


def rset(name, cfgs, N):
    drop = name[2:] if name.startswith('R-') else None
    return {c for c in cfgs if rclass(c, N) not in (None, drop)}


class DOp:
    """H11 + X D^-1 X^T on P+Q1 (index i1) through the full H: H [x;0] gives H11 x (rows i1) and X^T x (rows q2)."""
    def __init__(self, op, i1, q2, dinv):
        self.op, self.i1, self.q2, self.dinv = op, i1, q2, dinv
        self.n = len(i1)
        self.d = op.d[i1]

    def matvec(self, X):
        X = X.reshape(self.n, -1)
        Z = np.zeros((self.op.n, X.shape[1]))
        Z[self.i1] = X
        W = self.op.matvec(Z)
        Z[:] = 0.0
        Z[self.q2] = W[self.q2] * self.dinv[:, None]
        return W[self.i1] + self.op.matvec(Z)[self.i1]


class SubOp:
    """H restricted to the index set i1 (through the full stored H)."""
    def __init__(self, op, i1):
        self.op, self.i1, self.n = op, i1, len(i1)
        self.d = op.d[i1]

    def matvec(self, X):
        X = X.reshape(self.n, -1)
        Z = np.zeros((self.op.n, X.shape[1]))
        Z[self.i1] = X
        return self.op.matvec(Z)[self.i1]


def heff_exact(op, m, q1, Hpp):
    """des Cloizeaux H_eff (m x m) of the sub-space P+Q1 (exact; Q2 dropped)."""
    i1 = np.r_[np.arange(m), np.sort(q1)].astype(int)
    p, Am = np.linalg.eigh(Hpp)
    k = m + 2
    X0 = np.zeros((len(i1), k))
    X0[:m, :m] = Am
    lo = np.argsort(op.d[i1[m:]])[:k - m]
    for c, q in enumerate(lo):
        X0[m + q, m + c] = 1.0
    e, Cm, rn, it, nmv = davidson(SubOp(op, i1), k, X0=X0, tol=1e-6, maxit=600, max_sub=60)
    assert rn.max() < 1e-5, rn
    wp = (Cm[:m] ** 2).sum(0)
    sel = np.sort(np.argsort(-wp)[:m])
    Xp, E = Cm[:m, sel], e[sel]
    w, Us = np.linalg.eigh(Xp @ Xp.T)
    Sm = Us @ np.diag(w ** -0.5) @ Us.T
    below = int(sum(1 for j in range(k) if j not in sel and e[j] < E.max()))
    return Sm @ Xp @ np.diag(E) @ Xp.T @ Sm, dict(n1=len(i1), wP=[float(x) for x in wp[sel]], it=[int(it)], dense=False,
                                                  roots_below=below)


def heff(op, m, q1, Hpp):
    """des Cloizeaux H_eff (m x m) of the balanced form; also P weights and solver info."""
    q1 = np.sort(q1)
    i1 = np.r_[np.arange(m), q1].astype(int)
    q2 = np.setdiff1d(np.arange(m, op.n), i1)
    d2 = op.d[q2]
    p, Am = np.linalg.eigh(Hpp)
    dense = len(i1) <= DENSE_MAX
    if dense:
        H11 = op.sub(i1).dense()
        X = op.block(i1, q2)
    E, Xp, wsel, its = [], [], [], []
    for j in range(m):
        dinv = 1.0 / (p[j] - d2)
        if dense:
            import scipy.sparse as sp
            e, Cm = np.linalg.eigh(H11 + (X @ sp.diags(dinv) @ X.T).toarray())
            it = 0
        else:
            k = m + 1
            X0 = np.zeros((len(i1), k))
            X0[:m, :m] = Am
            X0[m + int(np.argmin(op.d[q1])), m] = 1.0
            e, Cm, rn, it, nmv = davidson(DOp(op, i1, q2, dinv), k, X0=X0, tol=1e-6, maxit=400, max_sub=40)
            assert rn.max() < 1e-5, rn
        ov = (Am[:, j] @ Cm[:m]) ** 2
        kk = int(np.argmax(ov))
        E.append(e[kk])
        Xp.append(Cm[:m, kk])
        wsel.append(float((Cm[:m, kk] ** 2).sum()))
        its.append(int(it))
    Xp = np.array(Xp).T
    w, Us = np.linalg.eigh(Xp @ Xp.T)
    Sm = Us @ np.diag(w ** -0.5) @ Us.T
    return Sm @ Xp @ np.diag(E) @ Xp.T @ Sm, dict(n1=len(i1), wP=wsel, it=its, dense=dense)


def evaluate(space, out, sets):
    t0 = time.time()
    N = C.N
    bl, eav, hf, _ = C.blocks6(os.environ['A_RUNFMT']) if N == 6 else A.blocks(space)
    p0 = bl[0]['lab'][0]
    cfgs = sorted(set(eav) - {p0})
    base = set(C.q1_sets(cfgs)['C-e'])
    cls = {c: qclass(c, N) for c in cfgs if c not in base}
    def xset(s):
        drop = {x for x in s[2:].split(',') if x}
        # item 11: the token 'h2' also drops every configuration with more than 2 holes in 3s3p (the pure n = 3 ones)
        hmax = 2 if 'h2' in drop else None
        drop.discard('h2')

        def holes(c):
            o = AP.parse_label(c)
            return (2 - o.get('3s', 0)) + (6 - o.get('3p', 0))
        return {c for c in cfgs if not drop & set(AP.parse_label(c)) and (hmax is None or holes(c) <= hmax)}
    q1s = {s: xset(s) if s.startswith('X-') else rset(s, cfgs, N) if s == 'R' or s.startswith('R-') else
           base | {c for c, k in cls.items() if k in s.split('+')} for s in sets}
    by = {s: {} for s in sets}
    info = {s: {} for s in sets}
    ntot = 0
    for b in bl:
        m, t, U = b['m'], b['t'], b['U']
        op = SymLower.from_npz(np.load(os.path.join(A.ROOT, 'results', b['name'] + '.npz'), allow_pickle=True))
        lab = b['lab']
        ntot += op.n
        Hpp = op.sub(np.arange(m)).dense()
        php = U.T @ Hpp @ U
        for s in sets:
            q1 = np.array([i for i in range(m, op.n) if lab[i] in q1s[s]], dtype=int)
            h, inf = (heff_exact if s.startswith('X-') else heff)(op, m, q1, Hpp)
            by[s][t] = hf[t] + U.T @ h @ U - php
            info[s][t] = inf
            print(f'{t} [{s}] n1 {inf["n1"]} / {op.n} it {inf["it"]} wP {min(inf["wP"]):.3f} {time.time() - t0:.0f} s',
                  flush=True)
        del op
    res = dict(space=space, ntot=ntot, classes={k: sorted(c for c, v in cls.items() if v == k) for k in CLASSES},
               rule={k: sorted(c for c in cfgs if rclass(c, N) == k) for k in RSUB}, sets={})
    for s in sets:
        pr = (C.params6 if N == 6 else A.params)(by[s])
        n1 = sum(x['n1'] for x in info[s].values())
        res['sets'][s] = dict(n1=n1, frac=n1 / ntot, info=info[s], **{q: float(pr[q]) for q in C.Q5})
        res['sets'][s]['ncfg'] = len(q1s[s])
    if os.environ.get('SCAN_DUMP'):
        np.savez(os.environ['SCAN_DUMP'], **{f'{s}|{t}': v for s in sets for t, v in by[s].items()})
    res['wall_s'] = time.time() - t0
    res['maxrss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(res, open(out, 'w'), indent=1)
    print('done', f'{res["wall_s"]:.0f} s, maxrss {res["maxrss_MB"]:.0f} MB', flush=True)


# ---------------------------------------------------------------- driver
class _CrFamExact(dict):
    """Cr3+ fam exact T1, T2 (mEh), B, C, beta (cm-1) from results/phase15/phase15k_17_crnd_fam_d5f2g2h1.json
    (exact.proj), read on first use; rounded as the values formerly written here."""
    def __missing__(self, k):
        if not len(self):
            from atomci_heff import config as C_
            f = os.path.join(A.ROOT, 'results', 'phase15', 'phase15k_17_crnd_fam_d5f2g2h1.json')
            p = json.load(open(C_.need(f)))['fam_d5f2g2h1']['exact']['proj']
            self.update(T1=round(p['T1'], 3), T2=round(p['T2'], 3), B=round(p['B'], 1), C=round(p['C'], 1),
                        beta=round(p['beta'], 1))
        if k not in self.keys():
            raise KeyError(k)
        return dict.__getitem__(self, k)


ROOT = A.ROOT
OUTD = os.path.join(ROOT, 'results', 'phase17')
IONS = {  # env, space, exact (T1, T2 mEh; B, C, beta cm-1)
    'fend': (dict(C_N='6', A_RUNFMT='v15u_fam_d5f2g2h1_d6_{t}e_fend6', A_ION='fend'), 'fam_d5f2g2h1',
             dict(T1=9.853, T2=3.786)),
    'crnd': (dict(ION='crnd'), 'fam_d5f2g2h1', _CrFamExact()),
    'mnnd': (dict(ION='mnnd'), 'besth_sall', dict(T1=14.317, T2=1.656, B=1159.9, C=4327.9, beta=153.7)),
}


def run_ion(ion, sets, tag):
    env, space, _ = IONS[ion]
    out = os.path.join(OUTD, f'cscan_{ion}_{tag}.json')
    if not os.path.exists(out):
        e = dict(os.environ, **env)
        with open(os.path.join(OUTD, f'run_cscan.log.{ion}'), 'a') as lg:
            subprocess.run([sys.executable, __file__, 'eval', space, out] + sets, env=e, check=True, stdout=lg,
                           stderr=subprocess.STDOUT)
    return json.load(open(out))['sets']


def ok(r, ex):
    return abs(r['T2'] - ex['T2']) < 0.3 and abs(r['T1'] / ex['T1'] - 1) < 0.03


def commit(step, cur):
    """heff: the research scan committed every greedy step to git here; the packaged scan does not."""
    print(f'greedy step {step} [{cur or "C-e"}] written (not committed)', flush=True)
    subprocess.run(['git', 'commit', '-q', '-m', f'Phase 17 C class scan: greedy step {step} [{cur or "C-e"}]\n\n'
                    'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], cwd=ROOT, check=True)


def greedy():
    chosen, step, log = [], 0, []
    while True:
        cur = '+'.join(chosen)
        rest = [c for c in CLASSES if c not in chosen]
        if step:                     # the current Q1 alone first: no candidate scan once it meets the criteria
            now = {ion: run_ion(ion, [cur], f'g{step}c') for ion in IONS}
            if all(ok(now[ion][cur], IONS[ion][2]) for ion in IONS):
                log.append(dict(step=step, q1=cur, ok=True, score={}))
                json.dump(log, open(os.path.join(OUTD, 'cscan_greedy.json'), 'w'), indent=1)
                print(f'step {step} [{cur}] ok True', flush=True)
                commit(step, cur)
                break
        sets = [cur] + ['+'.join(chosen + [c]) for c in rest]
        res = {ion: run_ion(ion, sets, f'g{step}') for ion in IONS}
        done = all(ok(res[ion][cur], IONS[ion][2]) for ion in IONS)
        sc = {c: sum(abs(res[ion]['+'.join(chosen + [c])]['T2'] - res[ion][cur]['T2']) for ion in IONS) for c in rest}
        log.append(dict(step=step, q1=cur, ok=done, score=sc))
        json.dump(log, open(os.path.join(OUTD, 'cscan_greedy.json'), 'w'), indent=1)
        print(f'step {step} [{cur}] ok {done} ' + ' '.join(f'{c} {v:.3f}' for c, v in sorted(sc.items(), key=lambda x: -x[1])),
              flush=True)
        commit(step, cur)
        if done or not rest:
            break
        chosen.append(max(sc, key=sc.get))
        step += 1


if __name__ == '__main__':
    if sys.argv[1] == 'eval':
        evaluate(sys.argv[2], sys.argv[3], [s if s != '-' else '' for s in sys.argv[4:]])
    elif sys.argv[1] == 'greedy':
        greedy()
