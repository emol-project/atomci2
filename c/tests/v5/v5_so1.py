"""S2-M4, V5: Coulomb + H_SO1 levels of acfs against a determinant CI on
PySCF (libcint) integrals in the same basis and orbitals.

usage: v5_so1.py ACFS CFPTBL prep_dir configs [nfte_dump] [--so2] [--contact] [--ss]

--ss adds the spin-spin operator (tensor part) on both sides (acfs "ss"; here
from int2e_ip1ip2).

--so2 adds the two-electron spin-orbit operator on both sides (acfs "so2";
here from int2e_p1vxp1 as in libhexpr2, checked there against GAMESS HSO2).

prep_dir holds an ATOMCI GTO run (fort.15 tape, fort.16 NFTRI, fort.18
basis, fort.21 MO coefficients); configs is an acfs configuration file
("seq*occ ..." per line, sequence numbers of the tape). The determinant
side uses nothing of ATOMCI but the basis and the MO coefficients:
  h      = <p|T + V_nuc|q>                  (int1e_kin + int1e_nuc)
  (pq|rs)                                   (int2e)
  h_SO   = (alpha^2/2) i <p|int1e_pnucxp|q> (libhexpr2's convention,
                                             equal to GAMESS HSO1)
over all determinants of the configurations, in PySCF's real spherical
orbitals (the spectrum does not depend on the orbital phases). Every acfs
level of 2J must appear 2J+1 times in the determinant spectrum.
"""
import itertools
import os
import re
import subprocess
import sys

import numpy as np
from pyscf import ao2mo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atomci_pyscf  # noqa: E402

ALPHA = 1 / 137.035999084


def orbital_list(dump):
    """seq -> (l, index within l), and the default closed shells."""
    nwtn = re.search(r"^NWTN (.*)$", dump, re.M).group(1).split()
    nwtc = [int(x) for x in re.search(r"^NWTC(.*)$", dump, re.M).group(1).split()]
    nwsq = [int(x) for x in re.search(r"^NWSQ (.*)$", dump, re.M).group(1).split()]
    if any(nwsq):
        raise ValueError("explicit NWSQ not handled")
    count, out = {}, {}
    for s, t in enumerate(nwtn, 1):
        l = (int(t.split(":")[1]) - 1) // 2
        count[l] = count.get(l, 0) + 1
        out[s] = (l, count[l])
    return out, nwtc


def read_configs(path, orbs, closed):
    cfgs = []
    for line in open(path):
        line = line.split("#")[0].split()
        if not line:
            continue
        occ = {int(a): int(b) for a, b in (t.split("*") for t in line)}
        for s in closed:
            occ.setdefault(s, 4 * orbs[s][0] + 2)
        cfgs.append(occ)
    return cfgs


def determinants(cfgs, shell_so):
    dets = set()
    for occ in cfgs:
        parts = [list(itertools.combinations(shell_so[s], n)) for s, n in sorted(occ.items())]
        for p in itertools.product(*parts):
            dets.add(tuple(sorted(x for t in p for x in t)))
    return sorted(dets)


def sign_apply(det, ops):
    """Apply ops [(create?, P), ...] (last acts first) to det (sorted
    tuple). Returns (sign, new det) or (0, None)."""
    occ = list(det)
    s = 1
    for cre, P in reversed(ops):
        if cre:
            if P in occ:
                return 0, None
            k = sum(1 for x in occ if x < P)
            occ.insert(k, P)
        else:
            if P not in occ:
                return 0, None
            k = occ.index(P)
            occ.pop(k)
        s *= -1 if k % 2 else 1
    return s, tuple(occ)


def main():
    so2 = "--so2" in sys.argv
    contact = "--contact" in sys.argv
    ss = "--ss" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    acfs, cfptbl, prep, cfgfile = args[:4]
    dump_tool = args[4] if len(args) > 4 else os.path.join(os.path.dirname(acfs), "nfte_dump")
    dump = subprocess.run([dump_tool, f"{prep}/fort.15"], capture_output=True, text=True,
                          check=True).stdout
    orbs, closed = orbital_list(dump)
    cfgs = read_configs(cfgfile, orbs, closed)

    mol, C, mos = atomci_pyscf.build(f"{prep}/fort.18", f"{prep}/fort.21")
    S = C.T @ mol.intor("int1e_ovlp") @ C
    print(f"PySCF: {mol.nao} AOs, {C.shape[1]} MO components; max |C^T S C - 1| "
          f"{np.abs(S - np.eye(len(S))).max():.1e}")

    # spatial orbitals of the active shells, spin orbitals P = 2 p + sigma
    seqs = sorted({s for occ in cfgs for s in occ})
    cols, shell_so = [], {}
    for s in seqs:
        l, i = orbs[s]
        ks = [k for k, (ll, ii, m) in enumerate(mos) if ll == l and ii == i]
        shell_so[s] = [2 * (len(cols) + j) + sg for j in range(len(ks)) for sg in (0, 1)]
        cols += ks
    Ca = C[:, cols]
    n = Ca.shape[1]
    h = Ca.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ Ca
    eri = ao2mo.restore(1, ao2mo.full(mol, Ca), n)
    hso = 0.5 * ALPHA ** 2 * 1j * np.einsum("ip,cij,jq->cpq", Ca, mol.intor("int1e_pnucxp", comp=3), Ca)
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2
    nso = 2 * n
    h1 = np.zeros((nso, nso), complex)
    for P in range(nso):
        for Q in range(nso):
            p, a, q, b = P // 2, P % 2, Q // 2, Q % 2
            v = h[p, q] if a == b else 0.0
            v += sum(hso[k, p, q] * s[a, b] for k, s in enumerate((sx, sy, sz)))
            h1[P, Q] = v

    # two-electron spin-orbital array G[P,Q,R,S] = <PQ|RS> (electron 1: P -> R,
    # electron 2: Q -> S), H2 = 1/2 sum G a+P a+Q a_S a_R
    sp = np.arange(nso) // 2
    sg = np.arange(nso) % 2
    G = eri[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3).astype(complex)  # (pr|qs)
    G *= (sg[:, None, None, None] == sg[None, None, :, None]) & \
         (sg[None, :, None, None] == sg[None, None, None, :])
    if so2:
        # H_SO2 = sum_c sum_pqrs g_c(pq|rs) {N[s_c(pq) E(rs)] + 2 N[E(pq) s_c(rs)]},
        # g_c = (alpha^2/2) i int2e_p1vxp1[c] (libhexpr2, equal to GAMESS HSO2),
        # i.e. W[P,Q,R,S] a+P a+R a_S a_Q with electron 1 on P, Q
        g2 = np.einsum("cijkl,ip,jq,kr,ls->cpqrs", mol.intor("int2e_p1vxp1", comp=3),
                       Ca, Ca, Ca, Ca, optimize=True) * (0.5 * ALPHA ** 2 * 1j)
        smat = np.array([sx, sy, sz])
        d = np.eye(2)
        W = np.zeros((nso,) * 4, complex)
        for c in range(3):
            g6 = g2[c][np.ix_(sp, sp, sp, sp)]
            part = os.environ.get("AC_SO2_PART", "")
            w_sso = 0.0 if part == "soo" else 1.0
            w_soo = 0.0 if part == "sso" else 1.0
            W += g6 * (w_sso * smat[c][sg][:, sg][:, :, None, None] * d[sg][:, sg][None, None, :, :] +
                       w_soo * 2 * d[sg][:, sg][:, :, None, None] * smat[c][sg][:, sg][None, None, :, :])
        # a+P a+R a_S a_Q = 1/2 (that + relabelled): <PR|QS> += W[P,Q,R,S] + W[R,S,P,Q]
        G += W.transpose(0, 2, 1, 3) + W.transpose(2, 0, 3, 1)

    if ss:
        # H_SS = alpha^2 sum_{i<j} s_i,a s_j,b (delta_ab r^2 - 3 x_a x_b)/r^5 (tensor part,
        # acfs "ss"). With M_ab(pr|qs) = int d_a(phi_p phi_r)(1) d_b(phi_q phi_s)(2)/r12
        # = int phi_p phi_r phi_q phi_s d1_a d2_b (1/r12) and
        # d1_a d2_b (1/r12) = (delta_ab r^2 - 3 x_a x_b)/r^5 + (4 pi/3) delta_ab delta(r12),
        # the kernel is the traceless part M_ab - delta_ab M_cc/3. M from int2e_ip1ip2
        # ((d_a i j|d_b k l)), transformed shell pair by shell pair (the AO array is
        # too large for the f sample).
        loc = mol.ao_loc_nr()
        X = np.zeros((9, n, n, n, n))
        for P in range(mol.nbas):
            for Q in range(mol.nbas):
                blk = mol.intor("int2e_ip1ip2", comp=9,
                                shls_slice=(P, P + 1, 0, mol.nbas, Q, Q + 1, 0, mol.nbas))
                X += np.einsum("cijkl,ip,jr,kq,ls->cprqs", blk, Ca[loc[P]:loc[P + 1]], Ca,
                               Ca[loc[Q]:loc[Q + 1]], Ca, optimize=True)
        M = X + X.transpose(0, 2, 1, 3, 4) + X.transpose(0, 1, 2, 4, 3) + X.transpose(0, 2, 1, 4, 3)
        M = M.reshape(3, 3, n, n, n, n)
        T = M - np.eye(3)[:, :, None, None, None, None] * np.einsum("aaprqs->prqs", M) / 3
        print(f"H_SS: max |T_ab - T_ba(electrons swapped)| "
              f"{np.abs(T - T.transpose(1, 0, 4, 5, 2, 3)).max():.1e}")
        smat = np.array([sx, sy, sz])
        sgl = np.arange(nso) % 2
        S1 = smat[:, sgl][:, :, sgl]  # [a, P, R]
        # electron 1: P -> R, electron 2: Q -> S
        Tso = T[np.ix_(range(3), range(3), sp, sp, sp, sp)]  # [a, b, P, R, Q, S]
        G += ALPHA ** 2 * np.einsum("abPRQS,aPR,bQS->PQRS", Tso, S1, S1, optimize=True)

    if contact:
        # two-electron Darwin + spin-spin contact in their spin-dependent form,
        # -pi alpha^2 delta(r12) - (8 pi/3) alpha^2 s1.s2 delta(r12), with
        # (pr|qs)_delta = int phi_p phi_r phi_q phi_s on a DFT grid
        from pyscf.dft import gen_grid
        grids = gen_grid.Grids(mol)
        grids.level = 9
        grids.build()
        phi = mol.eval_gto("GTOval_sph", grids.coords) @ Ca
        dl = np.einsum("g,gp,gr,gq,gs->prqs", grids.weights, phi, phi, phi, phi, optimize=True)
        smat = np.array([sx, sy, sz])
        sgl = np.arange(nso) % 2
        same = (sgl[:, None] == sgl[None, :]).astype(float)
        s1s2 = np.einsum("cab,cde->adbe", smat[:, sgl][:, :, sgl], smat[:, sgl][:, :, sgl])  # [P,Q,R,S]
        spin = -np.pi * ALPHA ** 2 * same[:, None, :, None] * same[None, :, None, :] \
            - 8 * np.pi / 3 * ALPHA ** 2 * s1s2
        G += dl[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3) * spin

    def gas(P, Q, R, S_):  # <PQ||RS>
        return G[P, Q, R, S_] - G[P, Q, S_, R]

    dets = determinants(cfgs, shell_so)
    nd = len(dets)
    H = np.zeros((nd, nd), complex)
    index = {d: i for i, d in enumerate(dets)}
    for J, dj in enumerate(dets):
        sj = set(dj)
        for I, di in enumerate(dets):
            si = set(di)
            pi, qj = sorted(si - sj), sorted(sj - si)
            if len(pi) > 2:
                continue
            if not pi:
                v = sum(h1[P, P] for P in di)
                v += 0.5 * sum(gas(P, Q, P, Q) for P in di for Q in di)
                H[I, J] = v
            elif len(pi) == 1:
                P, Q = pi[0], qj[0]
                sg, d = sign_apply(dj, [(True, P), (False, Q)])
                v = h1[P, Q] + sum(gas(P, R, Q, R) for R in si & sj)
                H[I, J] = sg * v
            else:
                (P1, P2), (Q1, Q2) = pi, qj
                sg, d = sign_apply(dj, [(True, P1), (True, P2), (False, Q2), (False, Q1)])
                H[I, J] = sg * gas(P1, P2, Q1, Q2)
    herm = np.abs(H - H.conj().T).max()
    Ed = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    print(f"determinants: {nd}, max |H - H^+| {herm:.1e}")

    out = subprocess.run([acfs, cfptbl, f"{prep}/fort.15", f"{prep}/fort.16", f"{prep}/fort.18",
                          f"{prep}/fort.21", cfgfile, "100000"] + (["so2"] if so2 else []) + (["contact"] if contact else [])
                         + (["ss"] if ss else []),
                         capture_output=True, text=True, check=True).stdout
    lev = [(int(m.group(1)), float(m.group(2)))
           for m in re.finditer(r"^ *(\d+)/2 +\d+ +(\S+) ", out, re.M)]
    Ec = sorted(e for tJ, e in lev for _ in range(tJ + 1))
    if len(Ec) != nd:
        print(f"FAIL: acfs gives {len(Ec)} states with multiplicity, determinants {nd}")
        sys.exit(1)
    d = np.abs(np.array(Ec) - Ed).max()
    lowest = sorted(lev, key=lambda x: x[1])
    e0 = lowest[0][1]
    print("lowest levels (acfs, cm-1 above the lowest):",
          ", ".join(f"J={tJ}/2 {(e - e0) * 219474.6313632:.4f}" for tJ, e in lowest[:6]))
    split_d = sorted(set(np.round((Ed - Ed[0]) * 219474.6313632, 4)))[:6]
    print("lowest levels (determinants, cm-1):", ", ".join(f"{x:.4f}" for x in split_d))
    ok = d < 1e-8 and herm < 1e-12
    print(f"V5 SO1{' + SO2' if so2 else ''}{' + contact' if contact else ''}{' + SS' if ss else ''}: {len(lev)} LSJ levels, {nd} states; max |E_acfs - E_det| = {d:.1e}  "
          f"{'passed' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
