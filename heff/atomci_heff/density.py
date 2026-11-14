"""One-particle density from ATOMCI's density-expression file (CI unit NFCID, default 24).

Records [cici.fpl CALEMS]: first IDIAG (I4); then NDN (I4), NDCOL(LNIX),
NDROW(LNIX), NDIX(LNIX) (I4) and DENE(LNIX) (R8), with entries 1..NDN valid.
NDCOL/NDROW are the CSF pair (1-based, row >= col), NDIX the one-electron
integral address (INT1AD) and DENE the coefficient of that integral in
H_row,col. The last record has NDN = 0. With a CI vector c:

    D_label = sum_I c_I^2 DENE_II + 2 sum_{I>J} c_I c_J DENE_IJ

D_label is gamma_aa for a = b and gamma_ab + gamma_ba = 2 gamma_ab for a != b.
"""
import struct

import numpy as np

from .reader import records


def label_density(path, c):
    rec = records(path)
    D = {}
    for r in rec[1:]:
        ndn = struct.unpack('<i', r[:4])[0]
        if ndn == 0:
            break
        lnix = (len(r) - 4) // 20
        col = np.frombuffer(r[4:4 + 4 * lnix], '<i4')[:ndn] - 1
        row = np.frombuffer(r[4 + 4 * lnix:4 + 8 * lnix], '<i4')[:ndn] - 1
        ix = np.frombuffer(r[4 + 8 * lnix:4 + 12 * lnix], '<i4')[:ndn]
        de = np.frombuffer(r[4 + 12 * lnix:4 + 20 * lnix], '<f8')[:ndn]
        f = np.where(row == col, 1.0, 2.0) * c[row] * c[col] * de
        for a, v in zip(ix, f):
            D[int(a)] = D.get(int(a), 0.0) + float(v)
    return D


def gamma_per_l(D, addressing):
    """gamma[l] in the per-l orbital index of the coupling sequence (nwsq order, 0-based)."""
    nwlf, nwsq = addressing.nwlf, addressing.nwsq
    n = len(nwlf)
    gam = {}
    for l1 in sorted(set(nwlf)):
        m = sum(1 for x in nwlf if x == l1)
        gam[l1 - 1] = np.zeros((m, m))
    for a in range(1, n + 1):
        for b in range(1, a + 1):
            if nwlf[a - 1] != nwlf[b - 1]:
                continue
            v = D.get(addressing.int1ad(a, b), 0.0)
            l = nwlf[a - 1] - 1
            i, j = nwsq[a - 1] - 1, nwsq[b - 1] - 1
            if i == j:
                gam[l][i, i] = v
            else:
                gam[l][i, j] = gam[l][j, i] = v / 2
    return gam
