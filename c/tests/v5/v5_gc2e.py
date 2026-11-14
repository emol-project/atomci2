"""S2-M7, Z3: Delta g of the two-electron gauge correction from acfs ("gc2e")
against a determinant CI with the m-resolved operator of python/gc2eref.py.

usage: v5_gc2e.py ACFS CFPTBL prep_dir configs [parts] [--gc1 | --rec]   (parts: 1 SSO, 2 SOO, 3 both)

--rec checks the two-body recoil term (acfs's rec2 column, ac_ci_rec2),
<O_z>/M with O_z = sum_{i != j} (x_i p_y,j - y_i p_x,j) from PySCF's int1e_r
and int1e_ipovlp.

--gc1 checks the one-electron gauge correction (acfs's dg_GC) instead, with
O_z = (alpha^2/2) Z sum_i r^-1 [s_z - (s.r^) z^] on a DFT grid; its rank-2
part couples l to l +- 2 (s-d, p-f), which configurations such as 2s 2p^5 d
mixed with 2s^2 2p^5 exercise.

The determinant side: H = h + V_ee + H_SO1 on PySCF integrals (as v5_so1.py);
for each level the state of largest M within its degenerate eigenspace
(J_z = L_z + S_z, L = -i r x grad from int1e_cg_irxp), and
  Delta g = <O_z> / M,
  O_z = (alpha^2/2) sum_{i != j} sum_m W_zm(i; j) (s_m(i) + 2 s_m(j)),
W from gc2eref.gc2e_tensors (electron 1 on (a, b), electron 2 on (c, d)).
Neither the tensor algebra nor the radial integrals of the C code are used.

Debugging aids (environment): GC2E_RANK=0/1/2 keeps one spatial rank of
W_km in (field, spin) (trace, antisymmetric, symmetric traceless), to be
compared with acfs restricted to the same K'; GC2E_PIECE=dens/rd keeps only
the density term or the R grad terms of gc2eref's integrated-by-parts form.
With --gc1, GC1_SAMEL=1 drops the l -> l +- 2 elements (the size of what the
check covers).
"""
import os
import re
import subprocess
import sys

import numpy as np
from pyscf import ao2mo

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", "..", "python"))
import atomci_pyscf  # noqa: E402
import gc2eref  # noqa: E402
from v5_so1 import determinants, orbital_list, read_configs, sign_apply  # noqa: E402

ALPHA = 1 / 137.035999084
HARTREE_CM = 219474.6313632


def slater(dets, h1, G):
    """<I|sum h1 + 1/2 sum G a+ a+ a a|J> over the determinants."""
    def gas(P, Q, R, S):
        return G[P, Q, R, S] - G[P, Q, S, R]

    nd = len(dets)
    H = np.zeros((nd, nd), complex)
    for J, dj in enumerate(dets):
        sj = set(dj)
        for I, di in enumerate(dets):
            si = set(di)
            pi, qj = sorted(si - sj), sorted(sj - si)
            if len(pi) > 2:
                continue
            if not pi:
                v = sum(h1[P, P] for P in di) if h1 is not None else 0.0
                v += 0.5 * sum(gas(P, Q, P, Q) for P in di for Q in di)
                H[I, J] = v
            elif len(pi) == 1:
                P, Q = pi[0], qj[0]
                sg, _ = sign_apply(dj, [(True, P), (False, Q)])
                v = (h1[P, Q] if h1 is not None else 0.0) + sum(gas(P, R, Q, R) for R in si & sj)
                H[I, J] = sg * v
            else:
                (P1, P2), (Q1, Q2) = pi, qj
                sg, _ = sign_apply(dj, [(True, P1), (True, P2), (False, Q2), (False, Q1)])
                H[I, J] = sg * gas(P1, P2, Q1, Q2)
    return H


def main():
    gc1 = "--gc1" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    acfs, cfptbl, prep, cfgfile = args[:4]
    parts = int(args[4]) if len(args) > 4 else 3
    dump = subprocess.run([os.path.join(os.path.dirname(acfs), "nfte_dump"), f"{prep}/fort.15"],
                          capture_output=True, text=True, check=True).stdout
    orbs, closed = orbital_list(dump)
    cfgs = read_configs(cfgfile, orbs, closed)
    mol, C, mos = atomci_pyscf.build(f"{prep}/fort.18", f"{prep}/fort.21")
    seqs = sorted({s for occ in cfgs for s in occ})
    cols, shell_so = [], {}
    for s in seqs:
        l, i = orbs[s]
        ks = [k for k, (ll, ii, m) in enumerate(mos) if ll == l and ii == i]
        shell_so[s] = [2 * (len(cols) + j) + sg for j in range(len(ks)) for sg in (0, 1)]
        cols += ks
    Ca = C[:, cols]
    n = Ca.shape[1]
    nso = 2 * n
    sp, sgl = np.arange(nso) // 2, np.arange(nso) % 2
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2
    smat = np.array([sx, sy, sz])
    d2 = np.eye(2)

    # H = h + V_ee + H_SO1
    h = Ca.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ Ca
    hso = 0.5 * ALPHA ** 2 * 1j * np.einsum("ip,cij,jq->cpq", Ca, mol.intor("int1e_pnucxp", comp=3), Ca)
    h1 = np.zeros((nso, nso), complex)
    for P in range(nso):
        for Q in range(nso):
            p, a, q, b = P // 2, P % 2, Q // 2, Q % 2
            h1[P, Q] = (h[p, q] if a == b else 0.0) + sum(hso[c, p, q] * smat[c][a, b] for c in range(3))
    eri = ao2mo.restore(1, ao2mo.full(mol, Ca), n)
    G = eri[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3).astype(complex)
    G *= (sgl[:, None, None, None] == sgl[None, None, :, None]) & (sgl[None, :, None, None] == sgl[None, None, None, :])
    dets = determinants(cfgs, shell_so)
    H = slater(dets, h1, G)
    E, U = np.linalg.eigh(0.5 * (H + H.conj().T))

    # J_z
    mol.set_common_orig((0, 0, 0))
    lz = -1j * (Ca.T @ mol.intor("int1e_cg_irxp", comp=3)[2] @ Ca)  # cg_irxp = i r x p = r x grad; L = -i r x grad
    jz1 = np.zeros((nso, nso), complex)
    for P in range(nso):
        for Q in range(nso):
            jz1[P, Q] = (lz[P // 2, Q // 2] if P % 2 == Q % 2 else 0.0) + (sz[P % 2, Q % 2] if P // 2 == Q // 2 else 0.0)
    JZ = slater(dets, jz1, np.zeros((nso,) * 4, complex))

    if gc1:
        # one-electron gauge correction, O_z = (alpha^2/2) Z sum_i r^-1 [s_z - (s.r^) z^],
        # i.e. (alpha^2/2) sum_m <p|Z r^-1 (delta_zm - z^ m^)|q> s_m, on a DFT grid
        from pyscf.dft import gen_grid
        grids = gen_grid.Grids(mol)
        grids.level = 9
        grids.build()
        phi = mol.eval_gto("GTOval_sph", grids.coords) @ Ca
        r = np.linalg.norm(grids.coords, axis=1)
        u = grids.coords / r[:, None]
        Z = mol.atom_charge(0)
        o1 = np.zeros((nso, nso), complex)
        for m in range(3):
            f = grids.weights * Z / r * ((m == 2) - u[:, 2] * u[:, m])
            Wm = np.einsum("g,gp,gq->pq", f, phi, phi)
            if os.environ.get("GC1_SAMEL"):  # debug: drop the l -> l +- 2 elements
                lo = np.array([mos[k][0] for k in cols])
                Wm = Wm * (lo[:, None] == lo[None, :])
            o1 += 0.5 * ALPHA ** 2 * Wm[np.ix_(sp, sp)] * smat[m][sgl][:, sgl]
        OZ = slater(dets, o1, np.zeros((nso,) * 4, complex))
        ngrid, col, tag = len(r), 7, "GC1e"
    elif "--rec" in sys.argv:
        # two-body recoil term O_z = sum_{i != j} (x_i p_y,j - y_i p_x,j), p = -i grad
        r = mol.intor("int1e_r", comp=3)
        pk = 1j * mol.intor("int1e_ipovlp", comp=3)  # <mu|p|nu> = -i <mu|d nu> = i <d mu|nu>
        A = [Ca.T @ r[k] @ Ca for k in range(3)]
        B = [Ca.T @ pk[k] @ Ca for k in range(3)]
        w = np.einsum("pr,qs->pqrs", A[0], B[1]) - np.einsum("pr,qs->pqrs", A[1], B[0])
        same = sgl[:, None] == sgl[None, :]
        o = w[np.ix_(sp, sp, sp, sp)] * same[:, None, :, None] * same[None, :, None, :]
        OZ = slater(dets, None, o + o.transpose(1, 0, 3, 2))
        ngrid, col, tag = 0, 9, "rec2"
    else:
        (OZ, ngrid), col, tag = gc2e_oz(mol, Ca, sp, sgl, smat, d2, parts, nso, dets), 9, "GC2e"
    return compare(E, U, JZ, OZ, ngrid, col, tag, acfs, cfptbl, prep, cfgfile, parts)


def gc2e_oz(mol, Ca, sp, sgl, smat, d2, parts, nso, dets):
    """The determinant matrix of O_z of the two-electron gauge correction, and the grid size."""
    _, W, ngrid = gc2eref.gc2e_tensors(mol, Ca, level=6, atom_grid=(400, 302),
                                       piece=os.environ.get("GC2E_PIECE", "all"))
    rank = int(os.environ.get("GC2E_RANK", "-1"))  # debug: keep one spatial rank of W_km
    if rank >= 0:
        tr = np.einsum("kk...->...", W) / 3
        iso = np.einsum("km,...->km...", np.eye(3), tr)
        anti = (W - W.transpose(1, 0, 2, 3, 4, 5)) / 2
        W = [iso, anti, W - iso - anti][rank]
    Wz = W[2]  # field along z: [m, a, b, c, d]
    o = np.zeros((nso,) * 4, complex)  # <P(1) Q(2)|o|R(1) S(2)>
    for m in range(3):
        w = Wz[m][np.ix_(sp, sp, sp, sp)]  # [P, R, Q, S] -> electron 1 P->R, electron 2 Q->S
        sm = smat[m][sgl][:, sgl]
        dd = d2[sgl][:, sgl]
        spin = (sm[:, None, :, None] * dd[None, :, None, :] if parts & 1 else 0.0) + \
               (2 * dd[:, None, :, None] * sm[None, :, None, :] if parts & 2 else 0.0)
        o += 0.5 * ALPHA ** 2 * w.transpose(0, 2, 1, 3) * spin
    GO = o + o.transpose(1, 0, 3, 2)
    return slater(dets, None, GO), ngrid


def compare(E, U, JZ, OZ, ngrid, col, tag, acfs, cfptbl, prep, cfgfile, parts):
    """Delta g = <O_z>/M of each level against acfs column col."""
    # levels
    det_levels = []
    i = 0
    while i < len(E):
        j = i
        while j + 1 < len(E) and abs(E[j + 1] - E[i]) < 1e-9:
            j += 1
        V = U[:, i:j + 1]
        mz, vz = np.linalg.eigh(V.conj().T @ JZ @ V)
        psi = V @ vz[:, -1]
        M = mz[-1]
        if M > 0.25:
            det_levels.append((E[i], j - i + 1, M, (psi.conj() @ OZ @ psi).real / M))
        i = j + 1

    opts = ["rec2"] if tag == "rec2" else ["gc2e", f"gc2epart={parts}"]
    out = subprocess.run([acfs, cfptbl, f"{prep}/fort.15", f"{prep}/fort.16", f"{prep}/fort.18",
                          f"{prep}/fort.21", cfgfile, "100000"] + opts,
                         capture_output=True, text=True, check=True).stdout
    c_levels = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 10 and re.match(r"^\d+/2$", f[0]) and f[0] != "0/2":
            c_levels.append((float(f[2]), int(f[0].split("/")[0]), float(f[col])))
    worst, nmatch, big = 0.0, 0, 0.0
    for e, deg, M, dg in det_levels:
        cand = [x for x in c_levels if abs(x[0] - e) < 1e-8 and x[1] == deg - 1]
        if not cand:
            print(f"   no acfs level at E = {e:.10f} (degeneracy {deg})")
            worst = np.inf
            continue
        nmatch += 1
        d = abs(cand[0][2] - dg)
        if d > worst and os.environ.get("V5_WORST"):
            print(f"   worst so far: J={deg - 1}/2 E={e:.10f} det {dg:+.12e} acfs {cand[0][2]:+.12e}")
        worst = max(worst, d)
        big = max(big, abs(dg))
    print(f"   {len(E)} states, {len(det_levels)} levels with J > 0, {nmatch} matched; grid {ngrid} points")
    for e, deg, M, dg in det_levels[:6]:
        cand = [x for x in c_levels if abs(x[0] - e) < 1e-8 and x[1] == deg - 1]
        print(f"   J={deg - 1}/2  E={e:.8f}  det {dg:+.10e}  acfs {cand[0][2] if cand else float('nan'):+.10e}")
    # rec2 is O(1) (GC2e ~1e-4): near-degenerate, strongly mixed excited levels
    # carry the ~1e-9 relative differences of the two Hamiltonians' mixing
    # (ground levels agree to ~5e-12), so its tolerance is relative
    tol = 1e-8 * max(1.0, big) if tag == "rec2" else 1e-10
    ok = worst < tol and nmatch == len(det_levels)
    print(f"Z3 {tag}{f' (parts {parts})' if tag == 'GC2e' else ''}: max |dg_acfs - dg_det| = "
          f"{worst:.1e} (largest |dg| {big:.2e})  {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
