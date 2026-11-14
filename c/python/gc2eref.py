"""Two-electron gauge correction of g by an independent route (S2-M7, Z1).

The magnetic-field derivative of H_SO2 with p -> p + A, A = (1/2) B x r
(technical note, Sec. 5.1; cf. K. Pachucki, Phys. Rev. A 69,
052502 (2004), Eq. (32)):

  O_b = (alpha^2/2) sum_{i != j} [grad_i(1/r_ij) x (b x r_i)] . (s_i + 2 s_j),

i.e. for field direction b = e_k and spin component m the spatial kernel
  X_km(1, 2) = delta_km (r1 . grad_1 g) - r1_m d_1k g,   g = 1/r12.

For real orbitals, electron 1 on (a, b) and electron 2 on (c, d), with
f = phi_a phi_b(r1) and V_cd(R) = int phi_c phi_d(r) / |r - R| dr:

  pre:  W_km(ab, cd) = int f(R) [delta_km R . grad V_cd(R) - R_m d_k V_cd(R)] dR
        (the operator as it stands; grad_1 g integrated over electron 2 is
        grad V_cd)
  post: W_km(ab, cd) = int V_cd(R) [-(2 f + R . grad f) delta_km + R_m d_k f] dR
        (after integrating by parts in electron 1, the form the C code uses)

Electron 1 is integrated on a PySCF DFT grid with the analytic GTO values
and gradients; V_cd and grad V_cd come from the one-electron integrals
int1e_grids and int1e_grids_ip at the grid points (the sign of the latter is
checked against finite differences). Neither form uses a multipole
expansion or the tensor algebra of the C code.
"""
import numpy as np
from pyscf.dft import gen_grid

ALPHA = 1 / 137.035999084


def potentials(mol, Ca, coords, batch=600):
    """V[g, c, d] and its gradient E[k, g, c, d] at the points coords."""
    ng, n = len(coords), Ca.shape[1]
    V = np.empty((ng, n, n))
    E = np.empty((3, ng, n, n))
    for s in range(0, ng, batch):
        c = coords[s:s + batch]
        v = mol.intor("int1e_grids", grids=c)
        V[s:s + batch] = np.einsum("gij,ic,jd->gcd", v, Ca, Ca, optimize=True)
        # <d_k i| 1/|r-R| |j> + <i| 1/|r-R| |d_k j> = int d_k(ij)/|r-R| = d_R_k V
        vi = mol.intor("int1e_grids_ip", grids=c)
        vi = vi + vi.transpose(0, 1, 3, 2)
        E[:, s:s + batch] = np.einsum("kgij,ic,jd->kgcd", vi, Ca, Ca, optimize=True)
    return V, E


def check_field_sign(mol, Ca, h=1e-4):
    """grad V from int1e_grids_ip against central differences of V."""
    pts = np.array([[0.3, -0.2, 0.5], [1.1, 0.4, -0.7]])
    _, E = potentials(mol, Ca, pts)
    err = 0.0
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        Vp, _ = potentials(mol, Ca, pts + d)
        Vm, _ = potentials(mol, Ca, pts - d)
        fd = (Vp - Vm) / (2 * h)
        err = max(err, np.abs(fd - E[k]).max() / max(np.abs(fd).max(), 1e-30))
    return err


def gc2e_tensors(mol, Ca, level=8, atom_grid=None, piece="all"):
    """pre and post W[k, m, a, b, c, d] (field k, spin component m).
    piece (post form only, for debugging): "dens" the -2 f delta_km term,
    "rd" the R grad terms, "all" both."""
    grids = gen_grid.Grids(mol)
    grids.level = level
    if atom_grid:
        grids.atom_grid = atom_grid  # (radial, angular) points, for diffuse orbitals
    grids.build()
    R, w = grids.coords, grids.weights
    ao = mol.eval_gto("GTOval_sph_deriv1", R)
    phi = ao[0] @ Ca
    dphi = np.array([ao[k + 1] @ Ca for k in range(3)])
    f = np.einsum("ga,gb->gab", phi, phi)
    df = np.einsum("kga,gb->kgab", dphi, phi)
    df = df + df.transpose(0, 1, 3, 2)
    V, E = potentials(mol, Ca, R)
    n = Ca.shape[1]
    pre = np.zeros((3, 3, n, n, n, n))
    post = np.zeros((3, 3, n, n, n, n))
    RE = np.einsum("gk,kgcd->gcd", R, E)
    Rdf = np.einsum("gk,kgab->gab", R, df)
    pre_iso = np.einsum("g,gab,gcd->abcd", w, f, RE, optimize=True)
    # piece (debug): "dens" keeps only -2 f delta_km, "rd" only the R grad terms
    wd, wr = {"dens": (1, 0), "rd": (0, 1)}.get(piece, (1, 1))
    post_iso = np.einsum("g,gab,gcd->abcd", w, -(2 * wd * f + wr * Rdf), V, optimize=True)
    for k in range(3):
        pre[k, k] += pre_iso
        post[k, k] += post_iso
        for m in range(3):
            pre[k, m] -= np.einsum("g,gab,gcd->abcd", w * R[:, m], f, E[k], optimize=True)
            post[k, m] += wr * np.einsum("g,gab,gcd->abcd", w * R[:, m], df[k], V, optimize=True)
    return pre, post, len(w)


def main():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "v5"))
    import atomci_pyscf

    prep = sys.argv[1]
    specs = [tuple(map(int, s.split(":"))) for s in sys.argv[2:]] or [(0, 13), (0, 12), (1, 11)]
    mol, C, mos = atomci_pyscf.build(f"{prep}/fort.18", f"{prep}/fort.21")
    cols = [k for k, (l, i, m) in enumerate(mos) if (l, i) in specs]
    Ca = C[:, cols]
    print(f"PySCF: {mol.nao} AOs; {len(cols)} MO components {specs}")
    print(f"grad V from int1e_grids_ip vs finite differences: rel {check_field_sign(mol, Ca):.1e}")
    level = int(os.environ.get("GC2E_LEVEL", "6"))
    # (radial, angular) points per atom: the default level-6 radial grid is too
    # coarse for diffuse virtual orbitals (pre - post 2e-7); 400 radial points
    # give ~1e-11
    ag = os.environ.get("GC2E_GRID", "400,302")
    pre, post, ng = gc2e_tensors(mol, Ca, level, tuple(map(int, ag.split(","))))
    d = np.abs(pre - post).max() / np.abs(pre).max()
    print(f"grid level {level} ({ng} points): max|W| {np.abs(pre).max():.6e}  max|pre - post| rel {d:.1e}")
    # spatial ranks of W_km in (k, m): 0 trace, 1 antisymmetric, 2 symmetric traceless
    tr = np.einsum("kk...->...", pre) / 3
    w0 = np.abs(tr).max()
    w1 = np.abs(pre - pre.transpose(1, 0, 2, 3, 4, 5)).max() / 2
    sym = (pre + pre.transpose(1, 0, 2, 3, 4, 5)) / 2 - np.einsum("km,...->km...", np.eye(3), tr)
    w2 = np.abs(sym).max()
    print(f"spatial ranks: max|rank 0| {w0:.3e}  max|rank 1| {w1:.3e}  max|rank 2| {w2:.3e}")
    ok = d < 1e-10 and w1 > 1e-6
    print(f"GC2e reference: pre = post (integration by parts) {'passed' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
