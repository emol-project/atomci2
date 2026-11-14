"""hexpr-atomci with radial integrals from PySCF, against the FCI of PySCF.

The full CI in the CSFs of hexpr-atomci and in the determinants of PySCF span the
same space, so the two must give the same levels. Where the sector with fixed Ms
is small enough, every eigenvalue is compared; otherwise the lowest ones.

usage: python pyscf_atom.py path/to/libhxa_block.so
"""
import ctypes
import itertools
import sys

import numpy as np
from pyscf import ao2mo, fci, gto, scf


def radial_basis(mol):
    """[(l, shell AO offsets for m = first component)]; AO index = offset + mi."""
    out = {}
    loc = mol.ao_loc_nr()
    for s in range(mol.nbas):
        l = mol.bas_angular(s)
        for ic in range(mol.bas_nctr(s)):
            out.setdefault(l, []).append(loc[s] + ic * (2 * l + 1))
    return out


def radial_orbitals(mol):
    """Real radial orbitals, from the Fock matrix of a spherically averaged density.
    The FCI does not depend on them; this choice only helps it converge."""
    mf = scf.RHF(mol)
    fock = mf.get_fock(dm=mf.get_init_guess(key="atom"))
    s1e = mol.intor("int1e_ovlp")
    basis = radial_basis(mol)
    orbs = []
    for l, offs in basis.items():
        blocks = []
        for mi in range(2 * l + 1):
            ix = np.array(offs) + mi
            blocks.append((fock[np.ix_(ix, ix)], s1e[np.ix_(ix, ix)]))
        for f, s in blocks:
            if abs(f - blocks[0][0]).max() > 1e-8 or abs(s - blocks[0][1]).max() > 1e-10:
                sys.exit("FAILED: the averaged Fock matrix is not spherical")
        f, s = blocks[0]
        x = np.linalg.inv(np.linalg.cholesky(s))
        w, v = np.linalg.eigh(x @ f @ x.T)
        c = x.T @ v
        for i in range(len(offs)):
            orbs.append((w[i], l, c[:, i], offs))
    orbs.sort(key=lambda o: o[0])
    return orbs


def real_harmonics(element, lmax):
    """Y[l] with shape (2l+1, npoints) in the AO order of PySCF, and quadrature weights.
    From AOs of one primitive, so the radial factor is positive and constant on the sphere."""
    n = 2 * lmax + 2
    x, wx = np.polynomial.legendre.leggauss(n + lmax + 2)
    phi = 2 * np.pi * np.arange(2 * n + 2 * lmax + 4) / (2 * n + 2 * lmax + 4)
    wphi = 2 * np.pi / len(phi)
    st = np.sqrt(1 - x * x)
    coords = np.array([[st[i] * np.cos(p), st[i] * np.sin(p), x[i]] for i in range(len(x)) for p in phi])
    w = np.array([wx[i] * wphi for i in range(len(x)) for _ in phi])
    aux = gto.M(atom=f"{element} 0 0 0", spin=None, charge=0, verbose=0,
                basis={element: [[l, [1.0, 1.0]] for l in range(lmax + 1)]})
    ao = aux.eval_gto("GTOval_sph", coords)
    Y, i = {}, 0
    for l in range(lmax + 1):
        y = ao[:, i : i + 2 * l + 1].T
        Y[l] = y / np.sqrt((y[0] * y[0] * w).sum())
        i += 2 * l + 1
    return Y, w


def angular_tensors(Y, w):
    """A[k][la,lb,lc,ld][ma,mb,mc,md] of (ab|cd) = sum_k A^k R^k(ab;cd)."""
    cache = {}

    def get(la, lb, lc, ld):
        key = (la, lb, lc, ld)
        if key not in cache:
            res = {}
            for k in range(max(abs(la - lb), abs(lc - ld)), min(la + lb, lc + ld) + 1):
                if (la + lb + k) % 2 or (lc + ld + k) % 2:
                    continue
                g1 = np.einsum("ap,bp,qp,p->abq", Y[la], Y[lb], Y[k], w)
                g2 = np.einsum("cp,dp,qp,p->cdq", Y[lc], Y[ld], Y[k], w)
                res[k] = 4 * np.pi / (2 * k + 1) * np.einsum("abq,cdq->abcd", g1, g2)
            cache[key] = res
        return cache[key]

    return get


def integrals(mol, element, orbs):
    """The MO integrals of PySCF, and h(a,b), R^k(ab;cd) of the radial orbitals."""
    nao = mol.nao_nr()
    lorb = [o[1] for o in orbs]
    first = []
    columns = []
    for _, l, c, offs in orbs:
        first.append(len(columns))
        for mi in range(2 * l + 1):
            col = np.zeros(nao)
            col[np.array(offs) + mi] = c
            columns.append(col)
    C = np.array(columns).T
    norb = C.shape[1]
    h1 = C.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)

    n = len(orbs)
    h = np.zeros((n, n))
    for a, b in itertools.product(range(n), repeat=2):
        if lorb[a] == lorb[b]:
            h[a, b] = h1[first[a], first[b]]
            for mi in range(2 * lorb[a] + 1):
                if abs(h1[first[a] + mi, first[b] + mi] - h[a, b]) > 1e-9:
                    sys.exit("FAILED: h depends on m")

    lmax = max(lorb)
    kmax = 2 * lmax
    Y, w = real_harmonics(element, kmax)
    tensors = angular_tensors(Y, w)
    rk = np.zeros((kmax + 1, n, n, n, n))
    worst = 0.0
    for a, b, c, d in itertools.product(range(n), repeat=4):
        la, lb, lc, ld = lorb[a], lorb[b], lorb[c], lorb[d]
        block = eri[first[a] : first[a] + 2 * la + 1, first[b] : first[b] + 2 * lb + 1,
                    first[c] : first[c] + 2 * lc + 1, first[d] : first[d] + 2 * ld + 1]
        ang = tensors(la, lb, lc, ld)
        if not ang:
            worst = max(worst, abs(block).max())
            continue
        ks = sorted(ang)
        M = np.array([ang[k].ravel() for k in ks]).T
        sol = np.linalg.lstsq(M, block.ravel(), rcond=None)[0]
        worst = max(worst, abs(M @ sol - block.ravel()).max())
        for k, v in zip(ks, sol):
            rk[k, a, b, c, d] = v
    if worst > 1e-9:
        sys.exit(f"FAILED: the ERIs do not decompose into R^k: residual {worst:.2e}")
    return h, rk, kmax, h1, eri, lorb, worst


def configurations(lorb, nel):
    caps = [2 * (2 * l + 1) for l in lorb]

    def rec(i, left):
        if i == len(caps) - 1:
            if left <= caps[i]:
                yield [left]
            return
        for x in range(min(caps[i], left) + 1):
            for rest in rec(i + 1, left - x):
                yield [x] + rest

    return list(rec(0, nel))


def hexpr_levels(lib, h, rk, kmax, lorb, nel):
    """All levels of the full CI, as (energy, 2S, 2L, parity)."""
    cfg = configurations(lorb, nel)
    lo = np.array(lorb, dtype=np.intc)
    occ = np.array(cfg, dtype=np.intc).ravel()
    err = ctypes.create_string_buffer(256)
    x = lib.hxa_open(len(lorb), lo.ctypes.data_as(ctypes.c_void_p), len(cfg),
                     occ.ctypes.data_as(ctypes.c_void_p), err, 256)
    if not x:
        sys.exit("FAILED: " + err.value.decode())
    labels = {}
    for i in range(lib.hxa_ncsf(x)):
        ts, tl, par = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        lib.hxa_label(x, i, ctypes.byref(ts), ctypes.byref(tl), ctypes.byref(par))
        key = (ts.value, tl.value, par.value)
        labels[key] = labels.get(key, 0) + 1
    levels = []
    hh = np.ascontiguousarray(h, dtype=np.float64)
    rr = np.ascontiguousarray(rk, dtype=np.float64)
    for (ts, tl, par), dim in sorted(labels.items()):
        H = np.zeros((dim, dim))
        rc = lib.hxa_block(x, ts, tl, par, hh.ctypes.data_as(ctypes.c_void_p),
                           rr.ctypes.data_as(ctypes.c_void_p), kmax,
                           H.ctypes.data_as(ctypes.c_void_p), dim)
        if rc:
            sys.exit(f"FAILED: hxa_block returned {rc}")
        H = np.triu(H) + np.triu(H, 1).T
        levels += [(e, ts, tl, par) for e in np.linalg.eigvalsh(H)]
    n = lib.hxa_ncsf(x)
    lib.hxa_close(x)
    levels.sort()
    return levels, n, len(cfg)


def fci_dense(h1, eri, norb, nelec):
    """Every eigenvalue of the fixed-Ms sector."""
    from pyscf.fci import cistring
    from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    h2 = absorb_h1e(h1, eri, norb, nelec, 0.5)
    H = np.zeros((na * nb, na * nb))
    for i in range(na * nb):
        c = np.zeros(na * nb)
        c[i] = 1.0
        H[:, i] = contract_2e(h2, c.reshape(na, nb), norb, nelec).ravel()
    if abs(H - H.T).max() > 1e-10:
        sys.exit("FAILED: the FCI matrix is not symmetric")
    return np.linalg.eigvalsh(H)


def fci_lowest(h1, eri, norb, nelec, nroots):
    s = fci.direct_spin1.FCI()
    s.conv_tol = 1e-12
    s.max_cycle = 500
    s.max_space = 60
    s.nroots = nroots
    e, _ = s.kernel(h1, eri, norb, nelec, nroots=nroots, max_cycle=500)
    return np.sort(np.asarray(e))


def run(lib, element, basis, nelec_ab, nlow=None):
    nel = sum(nelec_ab)
    mol = gto.M(atom=f"{element} 0 0 0", basis=basis, spin=(nel % 2), verbose=0)
    orbs = radial_orbitals(mol)
    h, rk, kmax, h1, eri, lorb, resid = integrals(mol, element, orbs)
    norb = h1.shape[0]
    levels, ncsf, ncfg = hexpr_levels(lib, h, rk, kmax, lorb, nel)
    ms2 = nelec_ab[0] - nelec_ab[1]
    # a level appears (2L+1) times in the sector with this Ms, if S >= |Ms|
    expanded = sorted(e for e, ts, tl, _ in levels if ts >= ms2 for _ in range(tl + 1))
    expanded = np.array(expanded)
    if nlow is None:
        ref = fci_dense(h1, eri, norb, nelec_ab)
        if len(ref) != len(expanded):
            sys.exit(f"FAILED: {element}/{basis}: {len(ref)} FCI levels, {len(expanded)} from the CSFs")
        what = f"all {len(ref)}"
    else:
        ref = fci_lowest(h1, eri, norb, nelec_ab, nlow)
        expanded = expanded[:nlow]
        what = f"the lowest {nlow}"
    dev = abs(ref - expanded).max()
    name = "".join("spdfg"[l] for l in lorb)
    print(f"{element} {basis}: radial {name}, {ncfg} configurations, {ncsf} CSFs; "
          f"ERI fit residual {resid:.1e}")
    print(f"  {what} FCI levels with Ms = {ms2}/2 compared: largest difference {dev:.2e}")
    shown, last = 0, None
    for e, ts, tl, par in levels:
        tag = (round(e, 7), ts, tl, par)
        if tag == last:
            continue
        last = tag
        print(f"  {e:16.10f}  {ts + 1}{'SPDFGHIKLMN'[tl // 2]}{'o' if par else 'e'}")
        shown += 1
        if shown == 5:
            break
    if dev > 1e-8:
        sys.exit(f"FAILED: {element}/{basis}: difference {dev:.2e}")


def main():
    lib = ctypes.CDLL(sys.argv[1])
    lib.hxa_open.restype = ctypes.c_void_p
    lib.hxa_open.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_char_p, ctypes.c_int]
    lib.hxa_close.argtypes = [ctypes.c_void_p]
    lib.hxa_ncsf.argtypes = [ctypes.c_void_p]
    lib.hxa_label.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3
    lib.hxa_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_int]
    # every level: s and p orbitals; then with a d orbital and an odd electron count
    run(lib, "Be", "6-31g", (2, 2))
    run(lib, "Li", "cc-pvdz", (2, 1))
    # the lowest levels of open-shell carbon: 3P (3 in Ms = 0), 1D (5), 1S (1)
    run(lib, "C", "cc-pvdz", (3, 3), nlow=9)
    print("passed")


if __name__ == "__main__":
    main()
