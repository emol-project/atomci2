#!/usr/bin/env python3
"""Phase 17 Q4-R: scalar relativity (ATOMCI's one-electron DK3 path, NTYPE = 4) for Cr3+ in RAS k = 1 + h (k1fh).

What is relativistic: only the one-electron H. ATOMCI's AIIGR4 (&AI NTYPE=4; IFDK3, IFNUCL default .TRUE.) builds
the DK3 h with a finite Gaussian nucleus in a B-spline momentum representation and projects it onto the basis; S and
the two-electron R^k stay non-relativistic (no picture change, no spin-orbit, no Breit).
Basis (both sides): cr.without-diffuse.bas with every s, p and d contraction opened into its distinct primitives
(18 s, 13 p, 11 d; f, g, h unchanged) -> basis/cr_qzp2012_open_spd.bas. Sapporo-DKH3 is not available locally.
Orbitals (both sides, the 15a recipe with that side's h): configuration-average HF of 3d3 (phase15f.own_set) with
bs.h[l] = the h of an AI-only ATOMCI run (NTYPE=2: must equal our analytic T + V; NTYPE=4: DK3), read from unit 18
(records ITYP, L, LCL, packed lower triangle; aiw1op). f, g, h = eigenvectors of that h.
Virtual s, p, d: the opened basis has 15 / 11 / 10 of them, and compact3 (smallest <r>) would pick tight core
functions instead of the k1fh ones. So both sides take the crnd3 virtuals (11s.. 16s, 11p.. 15p, 11d.. 14d; exact in
the opened basis, which contains every crnd3 primitive), Loewdin-orthonormalized after projecting out that side's
occupied orbitals, under their crnd3 names; the rest of the opened virtual space follows (Fock-canonical, never in the
CI). The k = 1 picks then use the crnd3 <r> ranking, so the 141 configurations and CSF lists are the crnd3 k1fh ones.
Ions (phase15m): crop (NR, tag crop3), crdk (DK3, tag crdk3). The crdk decks get ntype=4 (dkpatch).
29 Oct (item 1a): the same recipe for V2+ (vop / vdk, reference vnd3), Mn4+ (mnop / mndk, mnnd3) and Fe2+ d6
(feop / fedk, fend6; Fe space rasK1r = the 15n rasK1 rule); every command takes the element (cr default).
Commands: bases [el] / ai [el] / orb [el] / dkpatch <run> ... / vcm [el]"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase17')
EL = {  # element: Z, N, plain basis, opened basis, reference orbital tag, (NR ion, DK ion)
    'cr': (24, 3, 'cr.without-diffuse.bas', 'cr_qzp2012_open_spd.bas', 'crnd3', ('crop', 'crdk')),
    'v': (23, 3, 'v.without-diffuse.bas', 'v_qzp2012_open_spd.bas', 'vnd3', ('vop', 'vdk')),
    'mn': (25, 3, 'mn.without-diffuse.bas', 'mn_qzp2012_open_spd.bas', 'mnnd3', ('mnop', 'mndk')),
    'fe': (26, 6, 'fe.without-diffuse.bas', 'fe_qzp2012_open_spd.bas', 'fend6', ('feop', 'fedk')),
    'fe5': (26, 3, 'fe.without-diffuse.bas', 'fe_qzp2012_open_spd.bas', 'fe5nd3', ('fe5op', 'fe5dk')),   # Phase 18
    'co': (27, 6, 'co.without-diffuse.bas', 'co_qzp2012_open_spd.bas', 'cond6', ('coop', 'codk')),       # Phase 18
}
TAGS = {ion: ion + str(e[1]) for e in EL.values() for ion in e[5]}       # crop3, crdk3, ..., feop6, fedk6
NTYPE = {ion: 2 + 2 * k for e in EL.values() for k, ion in enumerate(e[5])}
ELOF = {ion: el for el, e in EL.items() for ion in e[5]}
ELOF.update(cr4f='cr', crno='cr')     # 17 test 1: other Cr3+ orbital sets in the plain basis, k1fh picks = crnd3's
NOCC = {0: 3, 1: 2, 2: 1}
BASE, OPEN, Z = EL['cr'][2], EL['cr'][3], EL['cr'][0]      # Cr names kept for the docstring and old callers


def _qp(N):
    import phase15f as F
    F.QP.setdefault(6, (6, 15))         # as phase15h (Fe2+ d6)
    return F.QP[N]


def bases(el='cr'):
    from atomci_heff import gto
    BASE, OPEN = EL[el][2], EL[el][3]
    fns = gto.read_bas(os.path.join(_C.BASIS, BASE))
    src = open(os.path.join(_C.BASIS, BASE)).read().splitlines()
    p = os.path.join(_C.BASIS, OPEN)
    assert not os.path.exists(p), p
    lines = [f'" {el.capitalize():2s}  Sapporo-QZP-2012 plain ({BASE}) with every s, p, d contraction opened into its '
             'distinct primitives (Phase 17 Q4-R)']
    for l, let in enumerate('SPD'):
        ex = sorted({e for ll, ee, c in fns if ll == l for e in ee}, reverse=True)
        lines += [f'  {let}  1  1.0  {e:16.7f}  1.0000000' for e in ex]
    first_f = next(k for k, x in enumerate(src) if x.split()[:1] == ['F'])
    lines += src[first_f:]
    open(p, 'w').write('\n'.join(lines) + '\n')
    b = gto.read_bas(p)
    print(OPEN, [[f[0] for f in b].count(l) for l in range(6)])


# ---------------------------------------------------------------- AI-only ATOMCI runs, unit 18 reader

def ai_dir(ion):
    return os.path.join(ROOT, 'runs', f'q4r_ai_{TAGS[ion]}')


def ai_run(ion):
    work = ai_dir(ion)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    Z, OPEN = EL[ELOF[ion]][0], EL[ELOF[ion]][3]
    shutil.copy(os.path.join(_C.BASIS, OPEN), work)
    os.symlink(_C.ATOMCI_CFPTBL, os.path.join(work, 'CFPTBL'))
    deck = (f"OPEN UNIT=10 FILE=./CFPTBL BINARY STATUS=OLD\nOPEN UNIT=3 FILE={OPEN} STATUS=OLD\nEND\n &cntl\nnrnfg=1,0,0,0,0,0,0,0,\ntitle=' q4r_ai_{TAGS[ion]} ',\n"
            f" /\n &ai\nz={Z},\niunit=3,\nntype={NTYPE[ion]},\nidprnt=0,\n /\n")
    open(os.path.join(work, 'be.deck'), 'w').write(deck)
    with open(os.path.join(work, 'be.deck')) as fi, open(os.path.join(work, 'be.out'), 'w') as fo:
        rc = subprocess.run([_C.ATOMCI_BIN], stdin=fi, stdout=fo,
                            stderr=subprocess.STDOUT, cwd=work).returncode
    assert rc == 0, (ion, rc)


def read_h18(path, nbf):
    """{(ityp, l): full symmetric matrix} from the AI integral file (unit 18): records ITYP, L, LCL, LCL reals
    (aiw1op splits a packed lower triangle into chunks of LOEI). L = l + 1."""
    from atomci_heff.reader import records
    acc = {}
    for r in records(path):
        if len(r) < 12:
            continue
        ityp, L, lcl = struct.unpack('<3i', r[:12])
        if ityp not in (1, 2) or len(r) != 12 + 8 * lcl or not 1 <= L <= len(nbf):
            continue
        acc.setdefault((ityp, L - 1), []).append(np.frombuffer(r[12:], '<f8'))
    out = {}
    for (ityp, l), parts in acc.items():
        v = np.concatenate(parts)
        n = nbf[l]
        assert len(v) == n * (n + 1) // 2, (ityp, l, len(v), n)
        M = np.zeros((n, n))
        M[np.tril_indices(n)] = v          # row-wise packed lower triangle: (1,1), (2,1), (2,2), ...
        out[(ityp, l)] = M + np.tril(M, -1).T
    return out


def ai(el='cr'):
    import phase15f as F
    Z, OPEN = EL[el][0], EL[el][3]
    bs = F.Bas(OPEN, Z)
    res = {}
    for ion in EL[el][5]:
        ai_run(ion)
        m = read_h18(os.path.join(ai_dir(ion), 'fort.18'), bs.nbf)
        res[ion] = {}
        for l in range(len(bs.nbf)):
            S, h = m[(1, l)], m[(2, l)]
            res[ion][l] = dict(dS=float(np.abs(S - bs.S[l]).max()), dh=float(np.abs(h - bs.h[l]).max()))
            np.save(os.path.join(ai_dir(ion), f'h{l}.npy'), h)
        print(ion, 'max |S - ours|', max(v['dS'] for v in res[ion].values()), '| h - ours| per l',
              [f'{v["dh"]:.2e}' for v in res[ion].values()], flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase17q4r_ai.json' if el == 'cr' else f'phase17q6r_ai_{el}.json'), 'w'),
              indent=1)


# ---------------------------------------------------------------- orbitals

def ref_virtuals(bs, el='cr'):
    """crnd3 s, p, d virtual columns expressed in the opened basis (exact: span(contracted) is in span(opened))."""
    import phase15f as F
    from atomci_heff import gto
    from atomci_heff.reader import read_vectors
    Z, BASE, ref = EL[el][0], EL[el][2], EL[el][4]
    b0 = F.Bas(BASE, Z)
    R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{ref}', 'be.NO'), b0.nbf)[0]
    Y, err = {}, {}
    for l in range(3):
        fa = [f for f in b0.fns if f[0] == l]
        fb = [f for f in bs.fns if f[0] == l]
        Sab = gto.matrices(fb + fa, l, Z)['S'][:len(fb), len(fb):]
        T = np.linalg.solve(bs.S[l], Sab)                           # contracted function -> opened coefficients
        err[l] = float(np.abs(T.T @ bs.S[l] @ T - b0.S[l]).max())  # 0 if the span is contained
        Y[l] = T @ R[l]
    for l in range(3, len(bs.nbf)):                                # f, g, h: the same functions as crnd3
        Y[l] = R[l]
    return Y, err


def own_set_dk(bs, h, el='cr'):
    """phase15f.own_set with bs.h replaced by h ({l: matrix}); s, p, d virtuals replaced as in the docstring."""
    import phase15f as F
    import phase14a_orb as O
    N = EL[el][1]
    bs.h = {l: h[l] for l in bs.h}
    _qp(N)
    C, inf = F.own_set(bs, N)
    Y, err = ref_virtuals(bs, el)
    match = {}
    for l in range(3):
        S, no = bs.S[l], NOCC[l]
        Co = C[l][:, :no]
        Pv = np.eye(len(S)) - Co @ Co.T @ S
        X = Pv @ Y[l][:, no:]                                      # crnd3 virtuals minus this side's occupied
        s, U = np.linalg.eigh(X.T @ S @ X)
        Xm = X @ U @ np.diag(s ** -0.5) @ U.T                       # Loewdin: closest orthonormal set to X
        Pm = np.eye(len(S)) - (Co @ Co.T + Xm @ Xm.T) @ S
        Xc = Pm @ C[l][:, no:]
        sc, Uc = np.linalg.eigh(Xc.T @ S @ Xc)
        Xc = Xc @ Uc[:, sc > 1e-8] / np.sqrt(sc[sc > 1e-8])
        nrest = len(S) - no - Xm.shape[1]
        assert Xc.shape[1] == nrest, (l, Xc.shape, nrest)
        Fl = O.fock(bs, C, *_qp(N))[0][l]
        e, v = np.linalg.eigh(Xc.T @ Fl @ Xc)
        Co = Co * np.sign(np.sum(Co * (S @ Y[l][:, :no]), 0))      # occupied signs as crnd3
        C[l] = np.column_stack([Co, Xm, Xc @ v])
        ovl = np.abs(Y[l][:, no:].T @ S @ Xm)
        match[l] = dict(span_err=err[l], min_diag_overlap=float(np.diag(ovl).min()),
                        lowdin_s_min=float(s.min()))
    for l in range(3, len(bs.nbf)):
        C[l] = C[l] * np.sign(np.sum(C[l] * (bs.S[l] @ Y[l]), 0))
    inf['orth_dev'] = max(float(np.abs(C[l].T @ bs.S[l] @ C[l] - np.eye(bs.nbf[l])).max()) for l in C)
    inf['match'] = match
    return C, inf


def diagnostics(bs, C):
    """HF quantities of the 15k/Q4 kind from the orbitals alone."""
    P = {'1s': bs.P(0, C[0][:, 0]), '2s': bs.P(0, C[0][:, 1]), '3s': bs.P(0, C[0][:, 2]),
         '2p': bs.P(1, C[1][:, 0]), '3p': bs.P(1, C[1][:, 1]), '3d': bs.P(2, C[2][:, 0])}
    rr = {k: float(np.sum(p * p * bs.r * bs.w)) for k, p in P.items()}
    R = bs.R
    return dict(r=rr, F2=R(P['3d'], P['3d'], P['3d'], P['3d'], 2), F4=R(P['3d'], P['3d'], P['3d'], P['3d'], 4),
                G2_3s3d=R(P['3s'], P['3d'], P['3d'], P['3s'], 2),
                R1_3s3d_3p3p=R(P['3s'], P['3d'], P['3p'], P['3p'], 1))


def orb(el='cr'):
    import phase15f as F
    import phase14a_orb as O
    Z, N, OPEN = EL[el][0], EL[el][1], EL[el][3]
    nr, dk = EL[el][5]
    js = {}
    for ion in (nr, dk):
        tg = TAGS[ion]
        bs = F.Bas(OPEN, Z)
        h = {l: np.load(os.path.join(ai_dir(ion), f'h{l}.npy')) for l in range(len(bs.nbf))}
        C, inf = own_set_dk(bs, h, el)
        r = O.scf(bs, *_qp(N))
        inf['eps_occ'] = {f'{n}{"spd"[l]}': float(r['eps'][l][k]) for l in range(3)
                          for k, n in enumerate(range(l + 1, 4))}
        inf['diag'] = diagnostics(bs, C)
        F.write_be_no(C, tg, bs.nbf)
        inf['nbf'] = bs.nbf
        js[ion] = inf
        print(ion, tg, bs.nbf, f'E {inf["E"]:.8f}', 'orth', f'{inf["orth_dev"]:.1e}', 'match', inf['match'],
              flush=True)
    d = {k: js[dk]['diag'][k] / js[nr]['diag'][k] - 1 for k in ('F2', 'F4', 'G2_3s3d', 'R1_3s3d_3p3p')}
    js['dk_vs_nr'] = dict(dE=js[dk]['E'] - js[nr]['E'], rel=d,
                          deps={k: js[dk]['eps_occ'][k] - js[nr]['eps_occ'][k] for k in js[nr]['eps_occ']},
                          dr={k: js[dk]['diag']['r'][k] - js[nr]['diag']['r'][k] for k in js[nr]['diag']['r']})
    print('DK - NR', json.dumps(js['dk_vs_nr'], indent=1))
    json.dump(js, open(os.path.join(OUT, 'phase17q4r_orb.json' if el == 'cr' else f'phase17q6r_orb_{el}.json'), 'w'),
              indent=1)


_RVM = {}


def r_virt_matched(el='cr'):
    """compact3 ranking for crop / crdk: the crnd3 <r> of 11s.. 16s, 11p.. 15p, 11d.. 14d (and of every f, g, h);
    the rest of the opened virtual space is never picked (inf)."""
    if el not in _RVM:
        import gen_v
        import phase15f as F
        import phase15j as J
        from atomci_heff.reader import read_vectors
        Z, BASE, ref = EL[el][0], EL[el][2], EL[el][4]
        b0 = F.Bas(BASE, Z)
        R = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{ref}', 'be.NO'), b0.nbf)[0]
        rv = _RVM[el] = {}
        for l, lst in J.VIRT.items():
            L = J.LV.index(l)
            for o in lst:
                k = gen_v.vec(o) - 1
                if L < len(b0.nbf) and k < b0.nbf[L]:
                    p = b0.P(L, R[L][:, k])
                    rv[o] = float(np.sum(p * p * b0.r * b0.w))
                else:
                    rv[o] = float('inf')
    return _RVM[el]


def dkpatch(names):
    for n in names:
        p = os.path.join(ROOT, 'inputs', n, 'be.deck')
        d = open(p).read()
        if 'ntype=4,' in d:
            continue
        assert d.count('ntype=2,') == 1, n
        open(p, 'w').write(d.replace('ntype=2,', 'ntype=4,'))
    print('ntype=4:', len(names), 'decks')


def vcm(el='cr'):
    import phase13d as D
    import phase15s_nd as ND
    Z, N, OPEN = EL[el][0], EL[el][1], EL[el][3]
    assert N == 3, 'd6 (Fe) needs no vcm decks: its HF P block is the space PHP (phase17.own_hf)'
    for ion in EL[el][5]:
        tg = TAGS[ion]
        names = []
        for L, twoS, m, t in D.term_list(3):
            dst, deck, csf = ND.vcm_deck(t, OPEN, tg, Z)
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            open(os.path.join(ROOT, 'inputs', dst, 'be.csf'), 'w').write(csf)
            names.append(dst)
        if NTYPE[ion] == 4:
            dkpatch(names)
        for dst in names:
            shutil.rmtree(os.path.join(ROOT, 'runs', dst), ignore_errors=True)
            subprocess.run([sys.executable, _C.script('run_atomci.py'), dst, '--vectors-from',
                            f'vq_orb_{tg}'], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
            print(dst, 'done', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dkpatch':
        dkpatch(sys.argv[2:])
    else:
        {'bases': bases, 'ai': ai, 'orb': orb, 'vcm': vcm}[cmd](*sys.argv[2:3])
