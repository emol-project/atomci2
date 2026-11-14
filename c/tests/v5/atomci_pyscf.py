"""Bridge from an ATOMCI run to PySCF (stage 2, V5).

Reads the GTO basis from the header of NFAII (unit 18, written by AIIGT4)
and the MO coefficients from unit 21 (WRTCP), and builds
  - a PySCF Mole for the atom (spherical GTOs, one contracted shell per
    ATOMCI basis function, in the IFST/INXT order of each l), and
  - the MO coefficient matrix C (AO x MO) with an MO list (l, index, m)
    whose radial parts are ATOMCI's MOs (index 1-based within l, as NWSQ),
    copied to every real spherical component m.
CE on the file are the coefficients of normalized primitives (S2-M3), which
is also what PySCF expects; both sides normalize the contraction.
"""
import struct

import numpy as np
from pyscf import gto


def _records(path):
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b:
                return
            n, = struct.unpack("<i", b)
            yield f.read(n)
            f.read(4)


def read_basis(path18):
    """[(l, n, [(exponent, ce), ...]), ...] grouped by l, and Z."""
    hdr = next(_records(path18))
    o = 33 * 4

    def ints(k):
        nonlocal o
        v = struct.unpack("<%di" % k, hdr[o:o + 4 * k])
        o += 4 * k
        return v

    def dbls(k):
        nonlocal o
        v = struct.unpack("<%dd" % k, hdr[o:o + 8 * k])
        o += 8 * k
        return v

    lmax, nwt, _, _, _ = ints(5)
    issh = ints(lmax)
    ints(2 * nwt)
    z, _ = dbls(2)
    npqn = ints(nwt)
    dl = dbls(nwt)
    if o == len(hdr):
        raise ValueError(f"{path18}: STO basis; PySCF needs GTOs")
    ifst = ints(lmax)
    inxt = ints(nwt)
    nterm = ints(nwt)
    ce = np.array(dbls(6000)).reshape(150, 40)
    oe = np.array(dbls(6000)).reshape(150, 40)
    out = []
    for l in range(lmax):
        i = ifst[l]
        for _ in range(issh[l]):
            k = i - 1
            if npqn[k] != l + 1:
                raise ValueError(f"basis function {i}: n = {npqn[k]} is not l + 1")
            out.append((l, npqn[k], [(dl[k] * oe[k][j], ce[k][j]) for j in range(nterm[k])]))
            i = inxt[k]
    return out, z


def read_vectors(path21):
    """{l: array (nbf_l, nmo_l)} of radial MO coefficients."""
    recs = _records(path21)
    h = next(recs)
    nsym, = struct.unpack("<i", h[8:12])
    nbf = struct.unpack("<20i", h[12:92])
    nob = struct.unpack("<20i", h[92:172])
    out = {}
    for l in range(nsym):
        need, got = nbf[l] * nob[l], []
        while len(got) < need:
            r = next(recs)
            nt, = struct.unpack("<i", r[:4])
            got += struct.unpack("<%dd" % nt, r[4:4 + 8 * nt])
        out[l] = np.array(got).reshape(nob[l], nbf[l]).T  # basis fastest
    return out


def build(path18, path21, element=None):
    """(mol, C, mos): mos[k] = (l, index, m) of column k of C."""
    basis, z = read_basis(path18)
    vec = read_vectors(path21)
    sym = element or gto.mole._symbol(int(round(z)))
    shells = [[l] + [[a, c] for a, c in prim] for l, _, prim in basis]
    nel = int(round(z))
    mol = gto.M(atom=f"{sym} 0 0 0", basis={sym: shells}, spin=nel % 2, cart=False,
                unit="Bohr", verbose=0)
    # AO offsets: shells in the order given, components 2l+1 each
    ao0, off = [], 0
    for l, _, _ in basis:
        ao0.append(off)
        off += 2 * l + 1
    assert off == mol.nao
    first = {}
    for k, (l, _, _) in enumerate(basis):
        first.setdefault(l, k)
    cols, mos = [], []
    for l in sorted(vec):
        nb, nm = vec[l].shape
        for i in range(nm):
            for m in range(2 * l + 1):
                c = np.zeros(mol.nao)
                for mu in range(nb):
                    c[ao0[first[l] + mu] + m] = vec[l][mu, i]
                cols.append(c)
                mos.append((l, i + 1, m))
    return mol, np.array(cols).T, mos
