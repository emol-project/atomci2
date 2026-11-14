#!/usr/bin/env python3
"""Phase 13e: N dependence of the configuration-averaged denominator as the convergence criterion.

step1  Delta_i(N) = E_av(Q_i, N) - E_av(P, N) (CSF diagonals, (2L+1)(2S+1) weights over all terms; no runs):
         core  P = 3s2 3d^N, Q = 3s 3d^(N+1), N = 0..9 (vcm_* runs);
         g     Q_i = 3d^(N-1) i, i = 11g, 12g, N = 1..5 (QZP d1..d3, 6d2g d4, 3d1g d5 exports; 12g has no d5 export);
         d     Q_i = 3d^(N-1) i, i = 11d..15d, N = 1..5 (13d s6d runs).
       Fit Delta0 + delta (N - 3), rho = |delta / Delta0|. In fixed orbitals Delta_i(N) is exactly linear in N
       (E_av is linear in the averaged integrals), so the fit residual is a consistency check.
       Next to rho: r4 / |V3 t lifted| and |h4 / h3| per channel (13b-13d).
step2  frozen diagonal: H_QQ block of configuration 3d^(N-1) nd shifted by c_i(N) = Delta_i(3) - Delta_i(N);
       (a') A + M coupling, (b') M only; s3d and s6d. Control: the same construction in the core channel.
-> results/phase13/phase13e.txt/.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import phase13b as B  # noqa: E402
import phase13c as C13  # noqa: E402
import phase13d as D  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
HART_EV = D.HART_EV
EK, TK = B.EK, B.TK


def occ(label):
    return {o: int(q) for o, q in re.findall(r'(\d+[spdfgh])(\d+)', label)}


def diag_labels(run):
    """CSF diagonal and labels of results/<run>.npz without densifying."""
    z = np.load(os.path.join(ROOT, 'results', run + '.npz'), allow_pickle=True)
    lab = [str(x) for x in z['labels']]
    if 'H' in z:
        return np.diag(z['H']).copy(), lab
    r, c, v = z['H_rows'], z['H_cols'], z['H_vals']
    dg = np.zeros(len(lab))
    m = r == c
    dg[r[m]] = v[m]
    return dg, lab


def eav_gaps(rows):
    """rows: iterable of (N, weight, diag, labels, P-occupation, {name: Q-occupation}) -> {name: {N: gap}}."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for N, g, dg, lab, pocc, qocc in rows:
        for x, l in zip(dg, lab):
            o = occ(l)
            if o == pocc:
                acc[(N, 'P')][0] += g * x
                acc[(N, 'P')][1] += g
            for nm, q in qocc.items():
                if o == q:
                    acc[(N, nm)][0] += g * x
                    acc[(N, nm)][1] += g
    av = {k: v[0] / v[1] for k, v in acc.items() if v[1]}
    out = defaultdict(dict)
    for (N, nm), x in av.items():
        if nm != 'P' and (N, 'P') in av:
            out[nm][N] = x - av[(N, 'P')]
    return out


def fit(gap):
    Ns = np.array(sorted(gap))
    y = np.array([gap[N] for N in Ns])
    A = np.vstack([np.ones_like(Ns), Ns - 3]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    q = np.polyfit(Ns - 3, y, 2)[0] if len(Ns) > 2 else np.nan
    return dict(D0=float(c[0]), delta=float(c[1]), rho=float(abs(c[1] / c[0])), maxres=float(np.abs(res).max()),
                quad=float(q), Ns=Ns.tolist(), zero=float(3 - c[0] / c[1]))


def core_gaps():
    data, _ = C13.core_data()
    rows = []
    for name, d in data.items():
        N = d['N']
        lab = [f'P{i}' for i in d['P']]
        g = (2 * d['L'] + 1) * (d['twoS'] + 1)
        dg = np.diag(d['H'])
        P = set(d['P'])
        rows.append((N, g, dg, ['3s2' if i in P else '3s1' for i in range(len(dg))], {'3s': 2},
                     {'3s->3d': {'3s': 1}}))
    return eav_gaps(rows)['3s->3d'], data


def g_gaps():
    runs = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r6d2g',
            5: 'vq_v0p_{}e_ivo_r3d1g'}
    rows = []
    for N in range(1, 6):
        for L, twoS, m, t in D.term_list(N):
            dg, lab = diag_labels(runs[N].format(t))
            pocc = {'3d': N}
            qocc = {o: ({'3d': N - 1, o: 1} if N > 1 else {o: 1}) for o in ('11g', '12g')}
            rows.append((N, (2 * L + 1) * (twoS + 1), dg, lab, pocc, qocc))
    return eav_gaps(rows)


def d_gaps(data):
    rows = []
    for (N, t), d in data.items():
        L, twoS = B.term_key(N, t)
        qocc = {o: ({'3d': N - 1, o: 1} if N > 1 else {o: 1}) for o in d['Q']}
        rows.append((N, (2 * L + 1) * (twoS + 1), np.diag(d['H']), d['lab'], {'3d': N}, qocc))
    return eav_gaps(rows)


def label_fit(data):
    """Global least-squares label values over all runs of all spaces (as 13d) and the null space of the design."""
    labs = sorted({nm for dd in data.values() for d in dd.values() for nm in d['W']})
    col = {nm: k for k, nm in enumerate(labs)}
    rows, rhs = [], []
    for dd in data.values():
        for d in dd.values():
            iu = np.tril_indices(len(d['lab']))
            R = np.zeros((len(iu[0]), len(labs)))
            for nm, Wm in d['W'].items():
                R[:, col[nm]] = Wm[iu]
            rows.append(R)
            rhs.append(d['H'][iu])
    R, y = np.vstack(rows), np.concatenate(rhs)
    g, *_ = np.linalg.lstsq(R, y, rcond=None)
    _, sv, vt = np.linalg.svd(R, full_matrices=False)
    null = vt[(sv < sv[0] * 1e-10)]
    return dict(col=col, g=g, null=null)


def fav_dd(lf):
    """F_av(3d,3d) = F0 - (2/63)(F2 + F4) and F_av(3d,i) = F0 - G0/10 - (G2 + G4)/35 as label combinations
    {label: weight}; returns (value, max |null-space component| of the combination)."""
    def val(comb_):
        w = np.zeros(len(lf['col']))
        for nm, x in comb_.items():
            w[lf['col'][nm]] += x
        return float(w @ lf['g']), float(np.abs(lf['null'] @ w).max()) if len(lf['null']) else 0.0
    return val


def exact_d_gaps(data, lf):
    """Configuration-averaged gap Delta_i(N) = Delta_i(1) + (N - 1) s_i, s_i = F_av(3d,i) - F_av(3d,3d)."""
    val = fav_dd(lf)
    dd = {'R0(3d3d;3d3d)': 1.0, 'R2(3d3d;3d3d)': -2 / 63, 'R4(3d3d;3d3d)': -2 / 63}
    out, chk = {}, {}
    d1 = data[(1, '2D')]
    i3 = d1['P'][0]
    for o, Qi in d1['Q'].items():
        di = {f'R0({o}{o};3d3d)': 1.0, f'R0({o}3d;{o}3d)': -1 / 10, f'R2({o}3d;{o}3d)': -1 / 35,
              f'R4({o}3d;{o}3d)': -1 / 35}
        comb_ = dict(di)
        for k, x in dd.items():
            comb_[k] = comb_.get(k, 0.0) - x
        s_i, nul = val(comb_)
        g1 = d1['H'][Qi[0], Qi[0]] - d1['H'][i3, i3]
        out[o] = {N: g1 + (N - 1) * s_i for N in range(1, 6)}
        chk[o] = nul
    return out, val(dd), chk


def exact_core_gaps(p):
    """Delta(N) = E_av(3s 3d^(N+1)) - E_av(3s2 3d^N) = n_d - n_s - F0ss + (1 - N) F_av(sd) + N F_av(dd)."""
    fsd = p['F0sd'] - p['G2sd'] / 10
    fdd = p['F0dd'] - 2 / 63 * (p['F2dd'] + p['F4dd'])
    return {N: p['n_d'] - p['n_s'] - p['F0ss'] + (1 - N) * fsd + N * fdd for N in range(0, 10)}, fdd


def eav_scalars(Eav):
    """h_k from E_av(N) = sum_k C(N, k) h_k, N = 0..3, and h4 = E_av(4) - prediction."""
    h = []
    for N in range(4):
        h.append(Eav[N] - sum(comb(N, k) * h[k] for k in range(N)))
    h4 = Eav[4] - sum(comb(4, k) * h[k] for k in range(4))
    return h + [h4]


def eav_from_info(info, N):
    """E_av(N) from the stored target energies (13c info: trace of the H_eff block = sum of target energies)."""
    num = den = 0.0
    for L, twoS, m, t in D.term_list(N):
        g = (2 * L + 1) * (twoS + 1)
        num += g * sum(info[f'd{N} {t}']['E'])
        den += g * m
    return num / den


# ---------------------------------------------------------------- step 2

def frozen_H(d, f, kind, shift):
    H = D.variant_H(d, f, kind)
    N = int(d['lab'][d['P'][0]][2:])
    for o, Qi in d['Q'].items():
        H[Qi, Qi] += shift[o][N]
    return H


def frozen_vectors(dd, f, kind, shift):
    vecs, info = {0: np.array([B.ECORE])}, {}
    for N in range(1, 6):
        blocks = {}
        for L, twoS, m, t in D.term_list(N):
            d = dd[(N, t)]
            Bk, inf = D.heff_block(frozen_H(d, f, kind, shift), d['lab'], N, t, m)
            blocks[(L, twoS)] = Bk
            info[(N, t)] = inf
        vecs[N] = B.vec_from_blocks(N, blocks)
    return vecs, info


def target_row(info, N):
    it = {t: x for (NN, t), x in info.items() if NN == N}
    tmin = min(it, key=lambda t: it[t]['gap_eV'])
    return dict(wP=min(x['wP'] for x in it.values()), amb=max(x['amb'] for x in it.values()),
                cond=max(x['cond'] for x in it.values()), below=sum(x['below'] for x in it.values()),
                gap=it[tmin]['gap_eV'], blk=tmin, dom=it[tmin]['dom'])


def flagged(r):
    return r['wP'] < 0.9 or r['amb'] > 0.3 or r['cond'] > 1.2 or r['below'] > 0


def core_frozen(data, c):
    """Exact core H_eff vectors d^0..d^9 with the Q block of each run shifted by c[N]."""
    from atomci_heff import heff
    blocks = {}
    for d in data.values():
        H = d['H'].copy()
        Qi = d['Q']
        if Qi:
            H[Qi, Qi] += c[d['N']]
            e, Cv, wP, _ = heff.spectrum(H, d['P'])
            st, _ = heff.select_states(wP, len(d['P']))
            A = heff.exact_heff(H, d['P'], st)['hermitian']
        else:
            A = H[np.ix_(d['P'], d['P'])]
        blocks.setdefault(d['N'], {})[(d['L'], d['twoS'])] = d['U'].T @ A @ d['U']
    return {N: B.vec_from_blocks(N, blocks[N]) for N in range(10)}


def main():
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13e: N dependence of the configuration-averaged denominator; frozen-diagonal variants')

    # ---------------- step 1
    cg, cdata = core_gaps()
    gg = g_gaps()
    data = {sp: {(N, t): D.load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)} for sp in D.SPACES}
    dgr = d_gaps(data['s6d'])                                  # P-term-restricted averages (13d style)
    lf = label_fit(data)
    dgp, (fdd_lab, fdd_nul), nulchk = exact_d_gaps(data['s6d'], lf)
    pcore = json.load(open(os.path.join(OUT, 'phase13b_b.json')))['params']
    cgx, fdd_core = exact_core_gaps(pcore)
    # checks: F_av(3d,3d) from the second difference of E_av(P) (P is complete: all d^N terms were run)
    eavP = {}
    for N in range(1, 6):
        num = den = 0.0
        for L, twoS, m, t in D.term_list(N):
            d = data['s6d'][(N, t)]
            g_ = (2 * L + 1) * (twoS + 1)
            num += g_ * sum(d['H'][i, i] for i in d['P'])
            den += g_ * len(d['P'])
        eavP[N] = num / den
    fdd_2nd = [eavP[N + 1] - 2 * eavP[N] + eavP[N - 1] for N in range(2, 5)]
    a('\n## Step 1. Configuration-averaged gap Delta_i(N) = E_av(Q_i) - E_av(P) (Eh) and rho_i = |delta / Delta0|')
    a('The runs contain only the (L,S) of d^N, so a CSF-diagonal average over Q is incomplete (13d used that '
      '"P-coupled" average). Exact configuration averages from integrals:')
    a('- d: Delta_i(N) = Delta_i(1) + (N-1)[F_av(3d,i) - F_av(3d,3d)]; Delta_i(1) from the d^1 run (one CSF each); '
      'F_av(3d,i) = F0 - G0/10 - (G2+G4)/35, F_av(3d,3d) = F0 - (2/63)(F2+F4) from the tape label values (global fit, '
      f'max null-space component of the combinations {max(list(nulchk.values()) + [fdd_nul]):.1e}).')
    a(f'  Check: F_av(3d,3d) = {fdd_lab:.6f} Eh (labels) vs second differences of the complete E_av(3d^N) '
      + ', '.join(f'{x:.6f}' for x in fdd_2nd) + f' vs 13b core model {fdd_core:.6f}.')
    a('- core: Delta(N) = n_d - n_s - F0ss + (1-N) F_av(3s,3d) + N F_av(3d,3d), 13b determinant-model parameters.')
    a('- g: no tape with g labels at several N is left, so the g rows use the P-coupled CSF-diagonal average '
      '(marked *).')
    a('| channel | i | N | Delta_i(N) (Eh) | Delta0 | delta | rho | max resid | quad coef | N_0 |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    fits = {}
    for ch, gaps in (('core', {'3s->3d': cgx}), ('core*', {'3s->3d': cg}), ('g*', gg), ('d', dgp), ('d*', dgr)):
        for i, gp in gaps.items():
            r = fit(gp)
            fits[(ch, i)] = dict(**r, gaps=gp)
            a(f'| {ch} | {i} | {min(gp)}..{max(gp)} | ' + ', '.join(f'{gp[N]:.4f}' for N in sorted(gp)) +
              f' | {r["D0"]:.5f} | {r["delta"]:+.5f} | {r["rho"]:.4f} | {r["maxres"]:.1e} | {r["quad"]:+.1e} | '
              f'{r["zero"]:.2f} |')
    a('(* = P-coupled CSF-diagonal average, incomplete over the Q terms.)')
    js['fits'] = {f'{c} {i}': v for (c, i), v in fits.items()}

    # r4 / t-part and |h4/h3| per channel
    c13 = json.load(open(os.path.join(OUT, 'phase13c_valence.json')))['spaces']
    cc = json.load(open(os.path.join(OUT, 'phase13c_core.json')))
    hc = cc['h']['h0_3'] + [cc['h']['h4']]
    rows = []
    core_m = c13['core']['m']
    rows.append(('core 3s->3d (13b)', core_m['ratio_t'], abs(hc[4] / hc[3]), fits[('core', '3s->3d')]['rho']))
    for sp, lab in (('1d1g', 'g: 3d + 11g (13c)'), ('3d', 'd: 3d + 11d, 12d, all excitations (13c)'),
                    ('5d1g', '5d1g (13c)'), ('6d2g', '6d2g (13c)')):
        info = c13[sp]['info']
        Eav = [B.ECORE] + [eav_from_info(info, N) for N in range(1, 5)]
        h = eav_scalars(Eav)
        assert abs(h[4] - c13[sp]['m']['h4']) < 1e-7, (sp, h[4], c13[sp]['m']['h4'])
        rho = fits[('g*', '11g')]['rho'] if sp == '1d1g' else fits[('d', '11d')]['rho']
        rows.append((lab, c13[sp]['m']['ratio_t'], abs(h[4] / h[3]), rho))
    sres = {}
    for sp in D.SPACES:
        f, _ = D.split(data[sp])
        vecs, _ = D.vectors(data[sp], f, 'a')
        m = C13.metrics(vecs)
        h3 = np.trace(m['V'][3]) / comb(10, 3)
        sres[sp] = dict(f=f, m=m)
        rows.append((f'd: {sp} singles (13d)', m['ratio_t'], abs(m['h4'] / h3), fits[('d', '11d')]['rho']))
    a('\n| channel | r4 / |V3 t lifted| (d^4) | |h4 / h3| | rho (lowest Q_i) |')
    a('|---|---|---|---|')
    for lab, r, hh, rho in rows:
        a(f'| {lab} | {r:.4f} | {hh:.4f} | {rho:.4f} |')
    a('(rho of the lowest orbital: core 3s->3d; g 11g; d 11d. h3 = scalar 3-body part = tr V3 / C(10,3).)')
    js['criterion'] = rows

    # ---------------- step 2
    a('\n## Step 2. Frozen diagonal: H_QQ of 3d^(N-1) nd shifted by c_i(N) = Delta_i(3) - Delta_i(N)')
    shift = {o: {N: dgp[o][3] - dgp[o][N] for N in dgp[o]} for o in dgp}
    a('c_i(N) (Eh): ' + '; '.join(f'{o}: ' + ', '.join(f'{shift[o][N]:+.4f}' for N in range(1, 6)) for o in shift))
    res = {}
    for sp in D.SPACES:
        f = sres[sp]['f']
        for kind in ('a', 'b'):
            vecs, info = frozen_vectors(data[sp], f, kind, shift)
            # check: the averaged gap of the shifted H equals Delta_i(3)
            m = C13.metrics(vecs)
            m5 = D.d5_metrics(vecs, m['V'])
            res[(sp, kind + "'")] = dict(vecs=vecs, info=info, m=m, m5=m5,
                                        tg={N: target_row(info, N) for N in range(2, 6)})
        # the unfrozen 13d variants for comparison
        for kind in ('a', 'b'):
            vecs, info = D.vectors(data[sp], f, kind)
            m = C13.metrics(vecs)
            res[(sp, kind)] = dict(vecs=vecs, info=info, m=m, m5=D.d5_metrics(vecs, m['V']),
                                   tg={N: target_row(info, N) for N in range(2, 6)})
    a('\nTargets (min P weight / max ambiguity / max cond / roots below a target / min gap eV (block, lowest '
      'non-target configuration)):')
    a('| space | variant | N | min P | max amb | max cond | below | min gap (eV) | flagged |')
    a('|---|---|---|---|---|---|---|---|---|')
    for sp in D.SPACES:
        for kind in ("a", "a'", "b", "b'"):
            for N in (3, 4, 5):
                r = res[(sp, kind)]['tg'][N]
                a(f'| {sp} | {kind} | {N} | {r["wP"]:.4f} | {r["amb"]:.2e} | {r["cond"]:.3f} | {r["below"]} | '
                  f'{r["gap"]:+.1f} ({r["blk"]}, {r["dom"]}) | {"yes" if flagged(r) else "no"} |')
    js['targets'] = {f'{sp} {k} {N}': res[(sp, k)]['tg'][N] for sp in D.SPACES for k in ("a", "a'", "b", "b'")
                     for N in (3, 4, 5)}

    a('\nd^4 four-body residual (mEh; JL metric, sum over states) and d^5:')
    a('| space | variant | V3 t: u / t4 / w / t3 | |V3 t lifted| | |r4| | r4/|V3 t| | h4 | |e| | |t| | |f| | '
      'd5 rms/state V0..V3 | d5 rms/state V0..V4 | d5 E_av resid |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rowsJ = {}
    for sp in D.SPACES:
        for kind in ("a", "a'", "b", "b'"):
            r = res[(sp, kind)]
            m, m5 = r['m'], r['m5']
            _, co, _ = B.tpart(3, S.to_vec(3, m['V'][3]))
            a(f'| {sp} | {kind} | ' + ' / '.join(f'{1e3 * co[q]:+.3f}' for q in ('u', 't4', 'w', 't3')) +
              f' | {1e3 * m["V3t"]:.3f} | {1e3 * m["r4"]:.3f} | {m["ratio_t"]:.3f} | {1e3 * m["h4"]:+.3f} | '
              f'{1e3 * m["e"]:.3f} | {1e3 * m["t"]:.3f} | {1e3 * m["f"]:.3f} | {1e3 * m5["rms5"]:.3f} | '
              f'{1e3 * m5["rms5_V4"]:.3f} | {1e3 * m5["h5"]:+.3f} |')
            rowsJ[f'{sp} {kind}'] = dict(m={k: v for k, v in m.items() if k not in ('r', 'V')}, m5=m5, V3t=co)
    js['rows'] = rowsJ

    # d^5 details where well-posed
    for sp in D.SPACES:
        for kind in ("a'", "b'"):
            r = res[(sp, kind)]
            if flagged(r['tg'][5]):
                a(f'\n{sp} {kind}: d^5 flagged (min P {r["tg"][5]["wP"]:.3f}, amb {r["tg"][5]["amb"]:.2f}, below '
                  f'{r["tg"][5]["below"]}); d^5 numbers not meaningful.')
                continue
            vecs, V = r['vecs'], r['m']['V']
            V4 = V + [S.from_vec(4, vecs[4]) - S.apply_parts(V, 4)]
            e3 = B.eig_resid(5, vecs[5], B.predict(V, 5))
            e4 = B.eig_resid(5, vecs[5], B.predict(V4, 5))
            single = [t for t in e3 if len(e3[t]) == 1]
            a(f'\n{sp} {kind}: d^5 well-posed (min P {r["tg"][5]["wP"]:.4f}, amb {r["tg"][5]["amb"]:.2e}, '
              f'gap {r["tg"][5]["gap"]:+.1f} eV). Residual actual - predicted (mEh), sorted eigenvalues per block:')
            a('| d^5 term | from d^0..d^3 | from d^0..d^4 |')
            a('|---|---|---|')
            for t in e3:
                a(f'| {t}{" (single)" if t in single else ""} | ' + '/'.join(f'{1e3 * x:+.3f}' for x in e3[t]) + ' | ' +
                  '/'.join(f'{1e3 * x:+.3f}' for x in e4[t]) + ' |')
            a(f'9 singly occurring terms {single}: rms {1e3 * np.sqrt(np.mean([e3[t][0] ** 2 for t in single])):.3f} '
              f'(V0..V3) -> {1e3 * np.sqrt(np.mean([e4[t][0] ** 2 for t in single])):.3f} mEh (V0..V4)')
            rowsJ[f'{sp} {kind}']['d5'] = dict(e3=e3, e4=e4, single=single)

    # core control
    a('\n## Core control: the same frozen-diagonal construction in the core minimal space (3s -> 3d), N = 0..9')
    c = {N: cgx[3] - cgx[N] for N in cgx}
    a('c(N) (Eh): ' + ', '.join(f'{c[N]:+.4f}' for N in sorted(c)))
    ex = C13.core_vecs(cdata, 'exact')
    fr = core_frozen(cdata, c)
    a('| | |V3 t lifted| d^4 | |r4| | r4/|V3 t| | h4 | |e| | |t| | |f| | rms/state d^5..d^9 (V0..V3) |')
    a('|---|---|---|---|---|---|---|---|---|')
    cj = {}
    for lab, v in (('exact (13b)', ex), ('frozen diagonal', fr)):
        m = C13.metrics(v)
        V = m['V']
        rr = [np.sqrt(S.ip(N, v[N] - B.predict(V, N), v[N] - B.predict(V, N)) / comb(10, N)) for N in range(5, 10)]
        cj[lab] = dict(m={k: x for k, x in m.items() if k not in ('r', 'V')}, rms=rr)
        a(f'| {lab} | {1e3 * m["V3t"]:.4f} | {1e3 * m["r4"]:.4f} | {m["ratio_t"]:.4f} | {1e3 * m["h4"]:+.5f} | '
          f'{1e3 * m["e"]:.4f} | {1e3 * m["t"]:.4f} | {1e3 * m["f"]:.4f} | ' + ', '.join(f'{1e3 * x:.3f}' for x in rr) + ' |')
    dt = B.tpart(3, S.to_vec(3, B.extract(fr)[3]))[0] - B.tpart(3, S.to_vec(3, B.extract(ex)[3]))[0]
    a(f'Change of the t-part of V3 (d^3): {1e3 * np.sqrt(S.ip(3, dt, dt)):.4f} mEh')
    js['core_control'] = cj

    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13e.json'), 'w'), indent=1,
              default=lambda x: float(x) if np.isscalar(x) else (x.tolist() if hasattr(x, 'tolist') else str(x)))
    open(os.path.join(OUT, 'phase13e.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
