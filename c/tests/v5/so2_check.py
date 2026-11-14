"""S2-M5: the independent two-electron spin-orbit operator (python/so2ref.py)
against PySCF's int2e_p1vxp1 (libhexpr2's convention, equal to GAMESS HSO2),
element by element, in the f sample's GTO basis and orbitals.

usage: so2_check.py prep_dir seq ...   (sequence orbitals of the tape)
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import atomci_pyscf  # noqa: E402
import so2ref  # noqa: E402

ALPHA = so2ref.ALPHA


def radial_functions(path18, path21, grid, r=None):
    """{(l, index): (R, dR)} of the MOs on grid.r (or at the radii r),
    normalized on the grid"""
    basis, z = atomci_pyscf.read_basis(path18)
    vec = atomci_pyscf.read_vectors(path21)

    def contracted(l, prim, x):
        R = np.zeros_like(x)
        dR = np.zeros_like(x)
        for a, c in prim:
            e = np.exp(-a * x * x)
            R += c * a ** (0.5 * l + 0.75) * x ** l * e
            dR += c * a ** (0.5 * l + 0.75) * ((l * x ** (l - 1) if l else 0.0) - 2 * a * x ** (l + 1)) * e
        return R, dR

    bf = {}
    for l, n, prim in basis:
        R, dR = contracted(l, prim, grid.r)
        s = np.sqrt(np.sum(grid.w * R * R * grid.r ** 2))
        x = grid.r if r is None else r
        R, dR = contracted(l, prim, x)
        bf.setdefault(l, []).append((R / s, dR / s))
    out = {}
    for l, C in vec.items():
        for i in range(C.shape[1]):
            R = sum(C[mu, i] * bf[l][mu][0] for mu in range(C.shape[0]))
            dR = sum(C[mu, i] * bf[l][mu][1] for mu in range(C.shape[0]))
            out[l, i + 1] = (R, dR)
    return out


def main():
    prep = sys.argv[1]
    seqs = [int(x) for x in sys.argv[2:]]
    dump = subprocess.run([os.path.join(os.path.dirname(__file__), "..", "..", "build", "nfte_dump"),
                           f"{prep}/fort.15"], capture_output=True, text=True, check=True).stdout
    nwtn = re.search(r"^NWTN (.*)$", dump, re.M).group(1).split()
    count, orbs_of = {}, {}
    for s, t in enumerate(nwtn, 1):
        l = (int(t.split(":")[1]) - 1) // 2
        count[l] = count.get(l, 0) + 1
        orbs_of[s] = (l, count[l])

    mol, C, mos = atomci_pyscf.build(f"{prep}/fort.18", f"{prep}/fort.21")
    grid = so2ref.Grid()
    rad = radial_functions(f"{prep}/fort.18", f"{prep}/fort.21", grid)

    # real orbitals (PySCF) and complex ones (so2ref) of the chosen shells
    cols, corbs, U_blocks = [], [], []
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(40, 3))
    pts *= (0.5 / np.linalg.norm(pts, axis=1))[:, None]
    ao = mol.eval_gto("GTOval_sph", pts)
    radp = radial_functions(f"{prep}/fort.18", f"{prep}/fort.21", grid, np.linalg.norm(pts, axis=1))
    for s in seqs:
        l, i = orbs_of[s]
        ks = [k for k, (ll, ii, m) in enumerate(mos) if ll == l and ii == i]
        cols += ks
        R, dR = rad[l, i]
        for m in range(-l, l + 1):
            corbs.append((l, m, R, dR))
        # U: complex = U @ real, fitted at the points
        real = ao @ C[:, ks]                                   # (pts, 2l+1)
        rr = np.linalg.norm(pts, axis=1)
        th, ph = np.arccos(pts[:, 2] / rr), np.arctan2(pts[:, 1], pts[:, 0])
        Rv = radp[l, i][0]
        cplx = np.array([Rv * so2ref.ylm(l, m, th, ph) for m in range(-l, l + 1)]).T
        U, *_ = np.linalg.lstsq(real, cplx, rcond=None)        # real @ U = cplx
        U_blocks.append(U.T)                                    # complex_m = sum U[m, mr] real_mr
    n = len(cols)
    U = np.zeros((n, n), complex)
    o = 0
    for B in U_blocks:
        k = B.shape[0]
        U[o:o + k, o:o + k] = B
        o += k
    print(f"{n} spatial orbitals; max |U U^+ - 1| = {np.abs(U @ U.conj().T - np.eye(n)).max():.1e}")

    # PySCF side, real orbitals: W[P,Q,R,S]
    Ca = C[:, cols]
    g2 = np.einsum("cijkl,ip,jq,kr,ls->cpqrs", mol.intor("int2e_p1vxp1", comp=3), Ca, Ca, Ca, Ca,
                   optimize=True) * (0.5 * ALPHA ** 2 * 1j)
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2
    nso = 2 * n
    sp, sg = np.arange(nso) // 2, np.arange(nso) % 2
    d = np.eye(2)
    Wr = np.zeros((nso,) * 4, complex)
    for c, sm in enumerate((sx, sy, sz)):
        g6 = g2[c][np.ix_(sp, sp, sp, sp)]
        M = sm[sg][:, sg]
        D = d[sg][:, sg]
        Wr += g6 * (M[:, :, None, None] * D[None, None] + 2 * D[:, :, None, None] * M[None, None])
    # to the complex basis: W'[P',Q',R',S'] = sum W U*[P',P] U[Q',Q] U*[R',R] U[S',S]
    Us = np.kron(U, np.eye(2))
    Wc = np.einsum("pqrs,Pp,Qq,Rr,Ss->PQRS", Wr, Us.conj(), Us, Us.conj(), Us, optimize=True)

    O = so2ref.so2_orbital_integrals(corbs, grid, so2ref.Sphere(), kmax=2 * max(o[0] for o in corbs))
    Wref = so2ref.so2_spin_orbital(O)
    big = np.abs(Wc).max()
    diff = np.abs(Wref - Wc).max()
    print(f"max |W_pyscf| = {big:.3e}; max |W_ref - W_pyscf| = {diff:.2e} (relative {diff / big:.1e})")
    i = np.unravel_index(np.argmax(np.abs(Wref - Wc)), Wc.shape)
    print("worst element", i, Wref[i], Wc[i])
    return 0 if diff < 1e-6 * big else 1


if __name__ == "__main__":
    sys.exit(main())
