"""S2-M8, V5: the hyperfine and transition reduced matrix elements of acfs
("hfs", "trans"; ac_lsj_hfs_reduced, ac_lsj_trans_reduced) against a
determinant CI on PySCF integrals in the same GTO basis and orbitals.

usage: v5_props.py ACFS CFPTBL prep_dir configs

The determinant side: H = h + V_ee + H_SO1 (as v5_so1.py, v5_gc2e.py); for
each level the states of definite M within its degenerate eigenspace (J_z =
L_z + S_z), and the q = 0 component of each operator between them,
  <A M|O_0|B M> = (-1)^(J_A - M) (J_A K J_B; -M 0 M) <A||O||B>,
compared with acfs's reduced element (sign and all for A = B, the absolute
value otherwise: the phases of the eigenvectors are arbitrary). The z
components, in Cartesian form on PySCF's integrals or a DFT grid:
  orb   l_z r^-3 = i [grad(1/r) x grad]_z     (int1e_pnucxp / Z; also on the grid)
  sd    (g_s/2) sum_m s_m (3 r^_m r^_z - delta_mz) r^-3      (grid)
  con   (g_s/2) (8 pi/3) delta(r) s_z         (the AOs evaluated at r = 0)
  quad  C^2_0 r^-3 = (3 r^_z^2 - 1)/(2 r^3)   (grid)
  E1    z (int1e_r), and d/dz (int1e_ipovlp)
  M1    L_z + g_s S_z (int1e_cg_irxp)
  E2    r^2 C^2_0 = (3 z^2 - r^2)/2 (int1e_rr)
  M1r   L_z + g_s S_z + delta mu_z, and dM1 delta mu_z alone: the one-electron
        relativistic M1 part of Pachucki, PRA 69, 052502 (2004), Eqs. (59)-(62),
        mu = -2 sum_i (sigma T + T_vec + T^.sigma) without the r_ab terms:
          spin-scalar -4 alpha^2 sum_i s_z,i (w^2 r^2/12 + p^2/3 - Z/(6r))
            (int1e_r2, int1e_kin, int1e_nuc);
          orbital -alpha^2 sum_i (w^2 r^2/10 + p^2/2) l_z,i, with l_z p^2 as
            int (l_z phi_p)* (-lap phi_q) (grid, second derivatives);
          tensor -4 alpha^2 sum_i sum_m T_zm s_m, T = -(w^2/20) (r r) -
            (1/8) (p p) + (Z/4) (r r)/r^3, each the symmetric traceless part,
            p_z p_m as int (d_z phi_p)(d_m phi_q) (grid);
        w from the determinant energies
        With V5P_M1R2E set, also the spin two-electron part (acfs "m1r2e
        m1r2epart=1"), -alpha^2 sum_{a != b} s_a.(grad_a grad_a r_ab): the
        kernel int int phi_P phi_R(1) phi_Q phi_S(2) d_1z d_1j r12 by parts in
        electron 1 against Phi_QS(R) = int phi_Q phi_S |r - R| (Coulomb
        potentials of r^2 rho, r rho, rho on an auxiliary basis, see
        m1_spin2e_tensors); V5P_M1ATOMGRID (default 400,434) the electron-1 grid
        V5P_M1R2E=2: the orbital two-electron part (m1r2epart=2, the
        orbit-orbit Zeeman operator, m1_orb2e_tensor); =3 both.
        V5P_M1SEQS (more orbitals) and V5P_M1CACHE (a directory) let lists
        on one prep compute that tensor once
The s-s pairs of sd and quad are set to zero (their angular integral
vanishes; on the grid they would be 0 * a divergent radial integral). For
the transitions acfs prints S = |<lo||O||up>|^2; lines it leaves out
(|<||>| < 1e-12) are compared as zero. Nothing of the C tensor algebra or
radial integrals is used. V5 checks the implementation against a reading of
the operators, not the reading itself.
"""
import math
import os
import re
import subprocess
import sys
from fractions import Fraction as R

import numpy as np
from pyscf import ao2mo

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import atomci_pyscf  # noqa: E402
from v5_gc2e import slater  # noqa: E402
from v5_so1 import determinants, orbital_list, read_configs, sign_apply  # noqa: E402

ALPHA = 1 / 137.035999084
GS = 2.00231930436256


def onebody(dets, o1):
    """<I|sum o1|J> over the determinants (o1 on spin orbitals)."""
    index = {d: i for i, d in enumerate(dets)}
    M = np.zeros((len(dets), len(dets)), complex)
    nso = o1.shape[0]
    for J, dj in enumerate(dets):
        sj = set(dj)
        M[J, J] = sum(o1[P, P] for P in dj)
        for Q in dj:
            for P in range(nso):
                if P in sj or o1[P, Q] == 0:
                    continue
                sg, occ = sign_apply(dj, [(True, P), (False, Q)])
                I = index.get(tuple(sorted(occ)))
                if I is not None:
                    M[I, J] += sg * o1[P, Q]
    return M


def m1_spin2e_tensors(mol, Ca, level=5, atom_grid=None, batch=200):
    """W[j][P, R, Q, S] = int int phi_P phi_R(1) phi_Q phi_S(2) d_1z d_1j r12, the
    spatial kernel of the spin two-electron M1 term (z row), as
      int dR [d_z d_j (phi_P phi_R)](R) Phi_QS(R),  Phi_QS(R) = int phi_Q phi_S(r) |r - R| dr
    (by parts twice in electron 1; Phi is smooth). With |r - R| = (r^2 - 2 r.R + R^2)/|r - R|,
      Phi = V[r^2 rho] - 2 R.V[r rho] + R^2 V[rho],
    V[g](R) = int g(r)/|r - R| dr from int1e_grids. The atom is at the origin, so r_i phi and
    r^2 phi are exactly sums of Cartesian Gaussians (l+1, l+2): an uncontracted Cartesian
    auxiliary basis (l, l+1, l+2 for each primitive) represents phi, r_i phi, r^2 phi (least
    squares on a level-3 Becke grid; the residual is returned). No multipoles."""
    from pyscf import gto
    from pyscf.dft import gen_grid
    prims = set()
    for ib in range(mol.nbas):
        for a in mol.bas_exp(ib):
            for L in range(mol.bas_angular(ib), mol.bas_angular(ib) + 3):
                prims.add((L, float(a)))
    sym = mol.atom_symbol(0)
    aux = gto.M(atom=[[sym, (0, 0, 0)]], unit="Bohr", cart=True, spin=mol.spin, charge=mol.charge,
                basis={sym: [[L, [a, 1.0]] for L, a in sorted(prims)]}, verbose=0)
    # weighted on a Becke grid: random points around the origin miss a compact 1s, and the
    # fit is then wrong off the points (3% in (1s1s|1s1s) of the m5 basis)
    fg = gen_grid.Grids(mol)
    fg.level = 3
    fg.build()
    pts, sw = fg.coords, np.sqrt(np.abs(fg.weights))[:, None]
    X = aux.eval_gto("GTOval_cart", pts) * sw
    phi = mol.eval_gto("GTOval_sph", pts) @ Ca  # MOs
    r2 = np.sum(pts ** 2, axis=1)
    targets = [sw * t for t in [phi] + [pts[:, i, None] * phi for i in range(3)] + [r2[:, None] * phi]]
    M, resid = [], 0.0
    for t in targets:
        c = np.linalg.lstsq(X, t, rcond=1e-13)[0]
        resid = max(resid, np.abs(X @ c - t).max() / np.abs(t).max())
        M.append(c.T)  # [MO, aux]
    grids = gen_grid.Grids(mol)
    grids.level = level
    if atom_grid:  # (radial, angular): diffuse MOs need more radial points than a level gives
        grids.atom_grid = atom_grid
    grids.build()
    R, wt = grids.coords, grids.weights
    ao = mol.eval_gto("GTOval_sph_deriv2", R)
    v0, d1 = ao[0] @ Ca, [ao[k] @ Ca for k in (1, 2, 3)]
    d2 = {0: ao[6] @ Ca, 1: ao[8] @ Ca, 2: ao[9] @ Ca}  # d_z d_j: xz, yz, zz
    n = Ca.shape[1]
    W = np.zeros((3, n, n, n, n))
    for s in range(0, len(R), batch):
        g = slice(s, s + batch)
        v = aux.intor("int1e_grids", grids=R[g])  # [g, aux, aux]
        V0 = np.einsum("ca,gab,db->gcd", M[0], v, M[0], optimize=True)
        Vr = [np.einsum("ca,gab,db->gcd", M[1 + i], v, M[0], optimize=True) for i in range(3)]
        V2 = np.einsum("ca,gab,db->gcd", M[4], v, M[0], optimize=True)
        Rg = R[g]
        Phi = V2 - 2 * sum(Rg[:, i, None, None] * Vr[i] for i in range(3)) + \
            np.sum(Rg ** 2, axis=1)[:, None, None] * V0
        Phi = 0.5 * (Phi + Phi.transpose(0, 2, 1))
        for j in range(3):
            D = (np.einsum("gp,gr->gpr", d2[j][g], v0[g]) + np.einsum("gp,gr->gpr", v0[g], d2[j][g])
                 + np.einsum("gp,gr->gpr", d1[2][g], d1[j][g]) + np.einsum("gp,gr->gpr", d1[j][g], d1[2][g]))
            W[j] += np.einsum("g,gpr,gqs->prqs", wt[g], D, Phi, optimize=True)
    return W, resid, len(R)


def m1_orb2e_tensor(mol, Ca, level=5, atom_grid=None, batch=100):
    """X[P, R, Q, S] = sum_ij int int A_i(1) K_ij(r12) B_j(2), the spatial kernel of the
    orbital two-electron M1 term -(alpha^2/2) sum_{a != b} [r_a x (K_ab p_b)]_z (the
    orbit-orbit Zeeman operator), with A_i = eps_zki x_k phi_P phi_R, B_j = phi_Q d_j phi_S
    and the Breit kernel K = delta/r + r r/r^3 = 2 delta/r - d d r (no delta function):
      2 sum_i int A_i V[B_i] - sum_j int C_j Phi[B_j],  C_j = sum_i d_i d_j A_i
    (the second by parts twice in electron 1). V and Phi as in m1_spin2e_tensors, with
    d_j phi_S, r_k d_j phi_S, r^2 d_j phi_S on an auxiliary basis of l-1 .. l+3."""
    from pyscf import gto
    from pyscf.dft import gen_grid
    prims = set()
    for ib in range(mol.nbas):
        lb = mol.bas_angular(ib)
        for a in mol.bas_exp(ib):
            for L in range(max(0, lb - 1), lb + 4):
                prims.add((L, float(a)))
    sym = mol.atom_symbol(0)
    aux = gto.M(atom=[[sym, (0, 0, 0)]], unit="Bohr", cart=True, spin=mol.spin, charge=mol.charge,
                basis={sym: [[L, [a, 1.0]] for L, a in sorted(prims)]}, verbose=0)
    fg = gen_grid.Grids(mol)
    fg.level = 3
    fg.build()
    pts, sw = fg.coords, np.sqrt(np.abs(fg.weights))[:, None]
    X = aux.eval_gto("GTOval_cart", pts) * sw
    ao = mol.eval_gto("GTOval_sph_deriv1", pts)
    phi, dphi = ao[0] @ Ca, [ao[1 + j] @ Ca for j in range(3)]
    r2 = np.sum(pts ** 2, axis=1)
    # targets: phi; per j: d_j phi, x_k d_j phi (k = 0..2), r^2 d_j phi
    targets = [phi]
    for j in range(3):
        targets += [dphi[j]] + [pts[:, k, None] * dphi[j] for k in range(3)] + [r2[:, None] * dphi[j]]
    M, resid = [], 0.0
    for t in targets:
        c = np.linalg.lstsq(X, sw * t, rcond=1e-13)[0]
        resid = max(resid, np.abs(X @ c - sw * t).max() / np.abs(sw * t).max())
        M.append(c.T)  # [MO, aux]
    Mphi = M[0]
    Md = [M[1 + 5 * j: 6 + 5 * j] for j in range(3)]  # [d_j phi, x d_j phi, y d_j phi, z d_j phi, r^2 d_j phi]
    grids = gen_grid.Grids(mol)
    grids.level = level
    if atom_grid:
        grids.atom_grid = atom_grid
    grids.build()
    R, wt = grids.coords, grids.weights
    ao = mol.eval_gto("GTOval_sph_deriv2", R)
    v0, d1 = ao[0] @ Ca, [ao[k] @ Ca for k in (1, 2, 3)]
    ix = {(0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 1): 7, (1, 2): 8, (2, 2): 9}
    d2 = [[ao[ix[min(a, b), max(a, b)]] @ Ca for b in range(3)] for a in range(3)]
    n = Ca.shape[1]
    out = np.zeros((n, n, n, n))

    def pr(u, w):  # u_P w_R
        return np.einsum("gp,gr->gpr", u, w)

    for s in range(0, len(R), batch):
        g = slice(s, s + batch)
        x, y = R[g, 0][:, None, None], R[g, 1][:, None, None]
        f0, f1 = v0[g], [d[g] for d in d1]
        f2 = [[d2[a][b][g] for b in range(3)] for a in range(3)]
        rho = pr(f0, f0)
        drho = [pr(f1[a], f0) + pr(f0, f1[a]) for a in range(3)]

        def ddrho(a, b):
            return pr(f2[a][b], f0) + pr(f1[a], f1[b]) + pr(f1[b], f1[a]) + pr(f0, f2[a][b])

        A = [-y * rho, x * rho]  # A_x, A_y
        C = [x * ddrho(1, j) - y * ddrho(0, j) for j in range(3)]
        C[0] = C[0] + drho[1]
        C[1] = C[1] - drho[0]
        v = aux.intor("int1e_grids", grids=R[g])  # [g, aux, aux]
        T = np.einsum("ca,gab->gcb", Mphi, v, optimize=True)  # once; then each target is cheap
        Rg = R[g]
        R2 = np.sum(Rg ** 2, axis=1)[:, None, None]
        for j in range(3):
            V = [np.einsum("gcb,db->gcd", T, Md[j][t], optimize=True) for t in range(5)]
            Phi = V[4] - 2 * sum(Rg[:, k, None, None] * V[1 + k] for k in range(3)) + R2 * V[0]
            out -= np.einsum("g,gpr,gqs->prqs", wt[g], C[j], Phi, optimize=True)
            if j < 2:
                out += 2 * np.einsum("g,gpr,gqs->prqs", wt[g], A[j], V[0], optimize=True)
    return out, resid, len(R)


def f(x):
    return math.factorial(int(x))


def tri(a, b, c):
    if c < abs(a - b) or c > a + b or (a + b + c) != int(a + b + c):
        return 0.0
    return f(a + b - c) * f(a - b + c) * f(-a + b + c) / f(a + b + c + 1)


def w3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0 or tri(j1, j2, j3) == 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    s = 0.0
    for t in range(0, 60):
        d = [t, j3 - j2 + t + m1, j3 - j1 + t - m2, j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2]
        if min(d) < 0:
            continue
        p = 1.0
        for x in d:
            p *= f(x)
        s += (-1) ** t / p
    return (-1) ** int(j1 - j2 - m3) * math.sqrt(tri(j1, j2, j3) * f(j1 + m1) * f(j1 - m1) * f(j2 + m2)
                                               * f(j2 - m2) * f(j3 + m3) * f(j3 - m3)) * s


def main():
    acfs, cfptbl, prep, cfgfile = sys.argv[1:5]
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
    lo = np.array([mos[k][0] for k in cols])
    n = Ca.shape[1]
    nso = 2 * n
    sp, sgl = np.arange(nso) // 2, np.arange(nso) % 2
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2
    smat = np.array([sx, sy, sz])
    same = (sgl[:, None] == sgl[None, :]).astype(float)

    def spatial(o):  # spin-free orbital operator on spin orbitals
        return o[np.ix_(sp, sp)] * same

    def spin(o, m):  # orbital operator times s_m
        return o[np.ix_(sp, sp)] * smat[m][sgl][:, sgl]

    # H = h + V_ee + H_SO1
    Z = mol.atom_charge(0)
    h = Ca.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ Ca
    pnucxp = np.einsum("ip,cij,jq->cpq", Ca, mol.intor("int1e_pnucxp", comp=3), Ca)
    h1 = spatial(h).astype(complex)
    for c in range(3):
        h1 += spin(0.5 * ALPHA ** 2 * 1j * pnucxp[c], c)
    eri = ao2mo.restore(1, ao2mo.full(mol, Ca), n)
    G = eri[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3).astype(complex)
    G *= (sgl[:, None, None, None] == sgl[None, None, :, None]) & (sgl[None, :, None, None] == sgl[None, None, None, :])
    dets = determinants(cfgs, shell_so)
    H = slater(dets, h1, G)
    E, U = np.linalg.eigh(0.5 * (H + H.conj().T))

    mol.set_common_orig((0, 0, 0))
    lv = -1j * np.einsum("ip,cij,jq->cpq", Ca, mol.intor("int1e_cg_irxp", comp=3), Ca)
    JC = [onebody(dets, spatial(lv[c]) + spin(np.eye(n), c)) for c in range(3)]
    JZ = JC[2]
    J2 = sum(Jc @ Jc for Jc in JC)  # the determinant space is closed under J

    # operators (z components)
    from pyscf.dft import gen_grid
    grids = gen_grid.Grids(mol)
    grids.level = int(os.environ.get("V5P_GRID", "9"))
    grids.build()
    ao = mol.eval_gto("GTOval_sph_deriv2", grids.coords)
    phi = ao[0] @ Ca
    dphi = [ao[k] @ Ca for k in (1, 2, 3)]
    lap = (ao[4] + ao[7] + ao[9]) @ Ca  # xx + yy + zz
    rr = np.linalg.norm(grids.coords, axis=1)
    u = grids.coords / rr[:, None]
    wr3 = grids.weights / rr ** 3
    ss = (lo[:, None] == 0) & (lo[None, :] == 0)
    orb = (1j * pnucxp[2] / Z)
    # the same on the grid: l_z = -i (x d/dy - y d/dx)
    orb_grid = -1j * np.einsum("g,gp,gq->pq", wr3,
                               phi, grids.coords[:, 0, None] * dphi[1] - grids.coords[:, 1, None] * dphi[0])
    sd = [np.where(ss, 0.0, np.einsum("g,gp,gq->pq", wr3 * (3 * u[:, m] * u[:, 2] - (m == 2)), phi, phi))
          for m in range(3)]
    quad = np.where(ss, 0.0, np.einsum("g,gp,gq->pq", wr3 * (3 * u[:, 2] ** 2 - 1) / 2, phi, phi))
    phi0 = mol.eval_gto("GTOval_sph", np.zeros((1, 3)))[0] @ Ca
    con = np.outer(phi0, phi0)
    r = mol.intor("int1e_r", comp=3)
    r2 = mol.intor("int1e_rr", comp=9)
    ops = {
        "orb": (1, onebody(dets, spatial(orb))),
        "sd": (1, onebody(dets, sum(spin(GS / 2 * sd[m], m) for m in range(3)))),
        "con": (1, onebody(dets, spin(GS / 2 * 8 * np.pi / 3 * con, 2))),
        "quad": (2, onebody(dets, spatial(quad))),
        "E1l": (1, onebody(dets, spatial(Ca.T @ r[2] @ Ca))),
        "E1v": (1, onebody(dets, spatial(-Ca.T @ mol.intor("int1e_ipovlp", comp=3)[2] @ Ca))),
        "M1": (1, JZ + (GS - 1) * onebody(dets, spin(np.eye(n), 2))),
        "E2": (2, onebody(dets, spatial(Ca.T @ ((3 * r2[8] - r2[0] - r2[4] - r2[8]) / 2) @ Ca))),
    }
    # spin-scalar relativistic M1 parts: s_z r^2, s_z p^2, s_z Z/r
    sr2 = onebody(dets, spin(Ca.T @ mol.intor("int1e_r2") @ Ca, 2))
    sp2 = onebody(dets, spin(2 * Ca.T @ mol.intor("int1e_kin") @ Ca, 2))
    szr = onebody(dets, spin(-Ca.T @ mol.intor("int1e_nuc") @ Ca, 2))

    # orbital parts: r^2 l_z and l_z p^2 (l_z = -i (x d/dy - y d/dx))
    x0, y0 = grids.coords[:, 0, None], grids.coords[:, 1, None]
    lzphi = -1j * (x0 * dphi[1] - y0 * dphi[0])
    olr2 = np.einsum("g,gp,gq->pq", grids.weights * rr ** 2, phi, lzphi)
    olp2 = np.einsum("g,gp,gq->pq", grids.weights, lzphi.conj(), -lap)
    lr2 = onebody(dets, spatial(olr2))
    lp2 = onebody(dets, spatial(olp2))
    # tensor parts: the symmetric traceless (r r), (r r)/r^3, (p p) in the z row, with s_m
    tr2 = tzr = tpp = 0
    gdot = sum(np.einsum("g,gp,gq->pq", grids.weights, dphi[k], dphi[k]) for k in range(3))
    for m in range(3):
        d = 1.0 if m == 2 else 0.0
        qrr = grids.coords[:, 2] * grids.coords[:, m] - d * rr ** 2 / 3
        orr2 = np.einsum("g,gp,gq->pq", grids.weights * qrr, phi, phi)
        orr3 = np.einsum("g,gp,gq->pq", grids.weights * qrr / rr ** 3, phi, phi)
        opp = np.einsum("g,gp,gq->pq", grids.weights, dphi[2], dphi[m]) - d * gdot / 3
        tr2 = tr2 + onebody(dets, spin(orr2, m))
        tzr = tzr + onebody(dets, spin(Z * orr3, m))
        tpp = tpp + onebody(dets, spin(opp, m))
    # the rank-2 part of Pachucki's H6 per unit kappa: -alpha^2 sum_m s_m (p_z p_m - delta_zm p^2/3)
    # (acfs "zoff h6", its last column)
    ops["H6"] = (1, -ALPHA ** 2 * tpp)
    # isotope shift (acfs "is"): sum p^2/2 (int1e_kin), sum delta(r) (the AOs at r = 0), and the
    # two-body sum_{i<j} p_i.p_j = -sum_k sum_{i<j} d_k(i) d_k(j), <P|d_k|R> = -int1e_ipovlp
    dk = -np.einsum("ip,cij,jq->cpq", Ca, mol.intor("int1e_ipovlp", comp=3), Ca)
    gp = -sum(np.einsum("pr,qs->pqrs", dk[k], dk[k]) for k in range(3))
    gp = gp[np.ix_(sp, sp, sp, sp)] * (same[:, None, :, None] * same[None, :, None, :])
    isops = {"NMS": onebody(dets, spatial(Ca.T @ mol.intor("int1e_kin") @ Ca)),
             "SMS": slater(dets, None, gp.astype(complex)),
             "rho0": onebody(dets, spatial(con))}
    del gp
    herm = max(np.abs(olp2 - olp2.conj().T).max(), np.abs(olr2 - olr2.conj().T).max())
    # V5P_M1R2E: the spin two-electron term -alpha^2 sum_{a != b} sum_j s_j(a) d_az d_aj r_ab
    # (acfs "m1r2e m1r2epart=1")
    # (value 1; 2 the orbital two-electron term, 3 both)
    s2e, o2e, m1r2e = 0, 0, int(os.environ.get("V5P_M1R2E") or 0)
    ag = os.environ.get("V5P_M1ATOMGRID", "400,434")
    ag = tuple(map(int, ag.split(","))) if ag else None
    if m1r2e & 1:
        W, fitres, ng2 = m1_spin2e_tensors(mol, Ca, level=int(os.environ.get("V5P_M1GRID", "5")),
                                           atom_grid=ag)
        o = np.zeros((nso,) * 4, complex)  # <P(1) Q(2)|o|R(1) S(2)>
        for j in range(3):
            w = W[j][np.ix_(sp, sp, sp, sp)]  # [P, R, Q, S]
            sm = smat[j][sgl][:, sgl]
            o += -ALPHA ** 2 * w.transpose(0, 2, 1, 3) * (sm[:, None, :, None] * same[None, :, None, :])
        s2e = slater(dets, None, o + o.transpose(1, 0, 3, 2))
        print(f"   spin two-electron M1: auxiliary-basis fit residual {fitres:.1e}, electron-1 grid "
              f"{ng2} points")
    if m1r2e & 2:
        # the orbital two-electron term -(alpha^2/2) sum_{a != b} [r_a x (K_ab p_b)]_z, p = -i grad
        # V5P_M1SEQS: more orbitals (sequence numbers) to compute the tensor on, and
        # V5P_M1CACHE: a directory to keep it in, so that lists sharing a prep compute it once
        # (the tensor of a set of MOs is the slice of that of a larger set)
        xs = sorted(set(seqs) | {int(t) for t in os.environ.get("V5P_M1SEQS", "").split()})
        colsx = [k for s in xs for k, (ll, ii, m) in enumerate(mos) if (ll, ii) == orbs[s]]
        lev = int(os.environ.get("V5P_M1GRID", "5"))
        cdir, cache = os.environ.get("V5P_M1CACHE"), None
        if cdir:
            import hashlib
            key = hashlib.sha1(C[:, colsx].tobytes() + repr((ag, lev)).encode()).hexdigest()[:16]
            cache = os.path.join(cdir, f"orb2e_{key}.npz")
        if cache and os.path.exists(cache):
            z = np.load(cache)
            Xx, fitres, ng2 = z["X"], float(z["resid"]), int(z["ng"])
        else:
            Xx, fitres, ng2 = m1_orb2e_tensor(mol, C[:, colsx], level=lev, atom_grid=ag)
            if cache:
                os.makedirs(cdir, exist_ok=True)
                np.savez(cache, X=Xx, resid=fitres, ng=ng2)
        ix = [colsx.index(k) for k in cols]
        Xo = Xx[np.ix_(ix, ix, ix, ix)]
        w = Xo[np.ix_(sp, sp, sp, sp)]  # [P, R, Q, S]
        o = 0.5j * ALPHA ** 2 * w.transpose(0, 2, 1, 3) * (same[:, None, :, None] * same[None, :, None, :])
        go = o + o.transpose(1, 0, 3, 2)
        hx = np.abs(go - go.conj().transpose(2, 3, 0, 1)).max() / np.abs(go).max()
        o2e = slater(dets, None, go)
        print(f"   orbital two-electron M1: auxiliary-basis fit residual {fitres:.1e}, electron-1 grid "
              f"{ng2} points; Hermitian to {hx:.1e}")

    def dmu(w):  # Pachucki's Eqs. (60)-(62), the one-electron terms
        scalar = -4 * ALPHA ** 2 * (w * w * sr2 / 12 + sp2 / 3 - szr / 6)
        orbital = -ALPHA ** 2 * (w * w * lr2 / 10 + lp2 / 2)
        drop = os.environ.get("V5P_M1DROP", "")  # sensitivity test: leave a part out here only
        tensor = -4 * ALPHA ** 2 * (-w * w * tr2 / 20 * ("tr2" not in drop) - tpp / 8 * ("tpp" not in drop)
                                    + tzr / 4 * ("tzr" not in drop))
        return scalar + (0 if "orb" in drop else orbital) + (0 if "ten" in drop else tensor) + \
            (0 if "s2e" in drop else s2e) + (0 if "o2e" in drop else o2e)
    gdiff = np.abs(orb_grid - orb)[~ss].max() / np.abs(orb).max()

    # determinant levels: {M: psi}. Levels of different J can lie closer than
    # eigh resolves (2.6e-8 hartree in t5, eigenvectors mixed at ~1e-6), so
    # each cluster within 1e-5 hartree is split by J^2 before J_z.
    dl = []
    i = 0
    while i < len(E):
        j = i
        while j + 1 < len(E) and E[j + 1] - E[j] < 1e-5:
            j += 1
        V = U[:, i:j + 1]
        j2, w = np.linalg.eigh(V.conj().T @ J2 @ V)
        tj = np.rint(np.sqrt(4 * j2 + 1) - 1).astype(int)
        for t in sorted(set(tj)):
            W = V @ w[:, tj == t]
            if W.shape[1] % (t + 1):
                raise SystemExit(f"J = {t}/2 block of dimension {W.shape[1]} near E = {E[i]}")
            e, x = np.linalg.eigh(W.conj().T @ H @ W)
            W = W @ x
            for k in range(0, W.shape[1], t + 1):
                Wl = W[:, k:k + t + 1]
                mz, vz = np.linalg.eigh(Wl.conj().T @ JZ @ Wl)
                dl.append((e[k:k + t + 1].mean(), t + 1,
                           {R(round(2 * m), 2): Wl @ vz[:, q] for q, m in enumerate(mz)}))
        i = j + 1
    dl.sort(key=lambda x: x[0])

    out = subprocess.run([acfs, cfptbl, f"{prep}/fort.15", f"{prep}/fort.16", f"{prep}/fort.18",
                          f"{prep}/fort.21", cfgfile, "100000", "hfs", "trans", "zoff", "h6", "is"]
                         + (["m1r2e", f"m1r2epart={m1r2e}"] if m1r2e else []),
                         capture_output=True, text=True, check=True).stdout
    lev, hfs, trans, iso, sec = {}, {}, {}, {}, 0
    for line in out.splitlines():
        if line.startswith("Zeeman reduced"):
            sec = 4
        elif line.startswith("Isotope shift"):
            sec = 5
        elif line.startswith("Hyperfine"):
            sec = 1
        elif line.startswith("Transitions"):
            sec = 2
        elif line.startswith("levels relative"):
            sec = 3
        x = line.split()
        if not x or not re.match(r"^\d+/2$", x[0]):
            continue
        if sec == 0:
            lev[(int(x[0][:-2]), int(x[1]))] = float(x[2])
        elif sec == 5:
            iso[(int(x[0][:-2]), int(x[1]))] = dict(zip(("NMS", "SMS", "rho0"),
                                                        (float(x[2]), float(x[3]), float(x[5]))))
        elif sec == 4:  # the zoff table goes with the hyperfine pairs (same A, B, ranks 1)
            hfs.setdefault((int(x[0][:-2]), int(x[1]), int(x[2][:-2]), int(x[3])), {})["H6"] = float(x[-1])
        elif sec == 1:
            hfs.setdefault((int(x[0][:-2]), int(x[1]), int(x[2][:-2]), int(x[3])), {}).update(
                zip(("orb", "sd", "con", "quad"), (float(x[4]), float(x[5]), float(x[6]), float(x[8]))))
        elif sec == 2:
            k = (int(x[0][:-2]), int(x[1]), int(x[2][:-2]), int(x[3]))
            d = trans.setdefault(k, {})
            if x[4] == "E1":
                d["E1l"], d["E1v"] = math.sqrt(float(x[6])), math.sqrt(float(x[7]))
            elif x[4] == "M1r":
                d["M1r"], d["dM1"] = math.sqrt(float(x[6])), abs(float(x[7]))
            else:
                d[x[4]] = math.sqrt(float(x[6]))

    def det_level(key):
        e = lev[key]
        c = [x for x in dl if abs(x[0] - e) < 1e-8 and x[1] == key[0] + 1]
        if len(c) != 1:
            raise SystemExit(f"no determinant level for acfs {key} (E = {e})")
        return c[0]

    def reduced(name, A, B):
        """<A||O||B> from <A M|O_0|B M>, M = min(J_A, J_B) or below; None if every 3j is small."""
        K, O = ops[name] if isinstance(name, str) else name
        JA, JB = R(A[1] - 1, 2), R(B[1] - 1, 2)
        M = min(JA, JB)
        while M >= -min(JA, JB):
            w = w3j(JA, K, JB, -M, 0, M)
            if abs(w) > 0.05:
                v = A[2][M].conj() @ O @ B[2][M]
                return (-1) ** int(JA - M) * v / w
            M -= 1
        return None

    if os.environ.get("V5P_WORST"):
        par = np.array([(-1) ** sum(lo[P // 2] for P in d) for d in dets])
        leak = max(min(np.sum(np.abs(psi[par > 0]) ** 2), np.sum(np.abs(psi[par < 0]) ** 2))
                   for x in dl for psi in x[2].values())
        print(f"   largest parity admixture in the determinant states: {leak:.1e}")
        for key in sorted(lev):
            x = det_level(key)
            gap = min(abs(x[0] - y[0]) for y in dl if y is not x)
            print(f"   level {key}: E acfs - det {lev[key] - x[0]:+.1e}, nearest level (any J) {gap:.1e}")
    print(f"   {len(dets)} determinants, {len(dl)} levels ({len(lev)} acfs levels); grid "
          f"{len(rr)} points, l_z r^-3 on the grid against int1e_pnucxp: {gdiff:.1e} relative; "
          f"r^2 l_z, l_z p^2 Hermitian to {herm:.1e}")
    worst, big = {}, {}
    for (ta, na, tb, nb), d in hfs.items():
        A, B = det_level((ta, na)), det_level((tb, nb))
        for name, cv in d.items():
            v = reduced(name, A, B)
            if v is None:
                continue
            if abs(v.imag) > 1e-8 * max(1.0, abs(v)) and A is B:
                print(f"   warning: {name} {ta}/2 #{na} imaginary diagonal {v}")
            dv = abs(v.real - cv) if A is B else abs(abs(v) - abs(cv))
            if os.environ.get("V5P_WORST") and dv > worst.get(name, 0.0):
                gap = min((abs(x[0] - y[0]) for x in (A, B) for y in dl if y[1] == x[1] and y is not x),
                          default=math.inf)
                print(f"   worst so far {name} {(ta, na)} {(tb, nb)}: det {v.real:+.10e} acfs {cv:+.10e}"
                      f"  nearest same-J level {gap:.1e} hartree")
            worst[name] = max(worst.get(name, 0.0), dv)
            big[name] = max(big.get(name, 0.0), abs(cv))
    keys = sorted(lev)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            if lev[kb] < lev[ka]:
                ka, kb = kb, ka
            if abs(ka[0] - kb[0]) > 4 or ka[0] + kb[0] < 2:
                continue
            A, B = det_level(ka), det_level(kb)
            d = trans.get(ka + kb, {})
            w = B[0] - A[0]
            for name in ("E1l", "E1v", "M1", "E2", "M1r", "dM1"):
                if name == "M1r":  # L + g_s S + delta mu: also the relative sign
                    v = reduced((1, ops["M1"][1] + dmu(w)), A, B)
                    cv = d.get(name, d.get("M1", 0.0))
                elif name == "dM1":
                    v = reduced((1, dmu(w)), A, B)
                    cv = d.get(name, 0.0)
                else:
                    v = reduced(name, A, B)
                    cv = d.get(name, 0.0)
                if v is None:
                    continue
                if os.environ.get("V5P_WORST") and abs(abs(v) - cv) > worst.get(name, 0.0):
                    gap = min((abs(x[0] - y[0]) for x in (A, B) for y in dl
                               if y[1] == x[1] and y is not x), default=math.inf)
                    print(f"   worst so far {name} {ka} {kb}: det {abs(v):.10e} acfs {cv:.10e}"
                          f"  nearest same-J level {gap:.1e} hartree")
                worst[name] = max(worst.get(name, 0.0), abs(abs(v) - cv))
                big[name] = max(big.get(name, 0.0), cv)
    # isotope shift: expectation values in every level (the same for each M)
    for key, d in iso.items():
        A = det_level(key)
        for name, cv in d.items():
            vals = [(psi.conj() @ isops[name] @ psi).real for psi in A[2].values()]
            v = vals[0]
            if max(vals) - min(vals) > 1e-9 * max(1.0, abs(v)):
                print(f"   warning: {name} {key} depends on M: {min(vals)} .. {max(vals)}")
            worst[name] = max(worst.get(name, 0.0), abs(v - cv))
            big[name] = max(big.get(name, 0.0), abs(cv))
    ok = True
    for name in ("orb", "sd", "con", "quad", "H6", "E1l", "E1v", "M1", "E2", "M1r", "dM1",
                 "NMS", "SMS", "rho0"):
        if name not in worst:
            continue
        rel = worst[name] / big[name] if big[name] > 0 else worst[name]
        good = rel < float(os.environ.get("V5P_TOL", "3e-8"))
        ok &= good
        print(f"   {name:5} max |acfs - det| {worst[name]:.1e}, largest |<||>| {big[name]:.3e}, "
              f"relative {rel:.1e}  {'ok' if good else 'FAILED'}")
    print(f"P3 hyperfine, transitions and isotope shift: {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
