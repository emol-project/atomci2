"""Energy-expression (EE) tape: H_IJ = sum_i WT_i * XINT[addr_i] as a linear map on integral labels.

Record layout after the header and CSF records [ntisqt.fpl X_STOREH, read in
cici.fpl CALEMS]:
    ITW, JTW, NOE, N1OE, N2OE, NIJ, NII, LTRI   (I4 x 8)
    WT(NOE) (R8), ISQN(LTRI) (I4), INDEXI(NIJ) (I4), INDEXJ(NII) (I4)
Pair k of the record uses the shared coefficients WT and the addresses
ISQN(NOE*k + 1 .. NOE*k + NOE). If NII = 0 the pairs are diagonal
(INDEXI(k), INDEXI(k)); otherwise (INDEXI(k), INDEXJ(k)) with INDEXI >
INDEXJ. Records with NOE = 0 are block separators. The first N1OE addresses
of a record are one-electron labels (INT1AD), the rest two-electron
(INT2AD); addresses index the ATOMCI in-core integral buffer XINT.
"""
import struct

import numpy as np

from .reader import records


def _skip_header(rec):
    h = np.frombuffer(rec[0][120:], '<i4')
    intmax = int(h[10])
    k, got = 2, 0
    while got < intmax:
        got += struct.unpack('<i', rec[k][:4])[0]
        k += 1
    return k


def read_ee(path):
    """Yield (row, col, coeffs, addresses, n1) per matrix-element contribution (0-based CSF indices)."""
    rec = records(path)
    k0 = _skip_header(rec)
    out = []
    for r in rec[k0:]:
        itw, jtw, noe, n1, n2, nij, nii, ltri = struct.unpack('<8i', r[:32])
        if noe == 0:
            continue
        off = 32
        wt = np.frombuffer(r[off:off + 8 * noe], '<f8')
        off += 8 * noe
        isqn = np.frombuffer(r[off:off + 4 * ltri], '<i4')
        off += 4 * ltri
        ii = np.frombuffer(r[off:off + 4 * nij], '<i4')
        off += 4 * nij
        jj = np.frombuffer(r[off:off + 4 * nii], '<i4')
        assert ltri == noe * nij
        for p in range(nij):
            row = ii[p] - 1
            col = row if nii == 0 else jj[p] - 1
            out.append((row, col, wt, isqn[noe * p:noe * (p + 1)], n1))
    return out


def design(entries, n):
    """Design matrix A (rows: unique (I>=J) elements, cols: distinct labels) with H_IJ = A g.

    Returns A, the list of (I, J) per row, the sorted label addresses, and
    the number of one-electron-label columns (from the N1OE split).
    """
    labels = sorted({int(a) for _, _, _, ad, _ in entries for a in ad})
    col = {a: c for c, a in enumerate(labels)}
    one = sorted({int(a) for _, _, _, ad, n1 in entries for a in ad[:n1]})
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    rowof = {p: r for r, p in enumerate(pairs)}
    A = np.zeros((len(pairs), len(labels)))
    for i, j, wt, ad, n1 in entries:
        r = rowof[(max(i, j), min(i, j))]
        for w, a in zip(wt, ad):
            A[r, col[int(a)]] += w
    return A, pairs, labels, set(one)


def sym_vec(Hm, pairs):
    return np.array([Hm[i, j] for i, j in pairs])
