"""Readers for ATOMCI scratch files (gfortran unformatted sequential, little endian).

Layouts, checked against the source:

- H diagonal, unit NFCIH1 (default 38): one record ``NKEEP, NKEEP`` (I4) and
  ``HDIAG(NKEEP)`` (R8)  [cici.fpl CALEMS, STORE_AND_CLOSE_FILES].
- H off-diagonal, unit NFCIH2 (default 39): one header record, then records
  ``HOFF(LEMBF)`` R8, ``IROW(LEMBF)`` I4, ``ICOL(LEMBF)`` I4 with
  LEMBF=2382. Slot LEMBF is a control slot: ICOL(LEMBF) is the entry count
  of the record, IROW(LEMBF)=0 marks the last record. Only the lower
  triangle (row > col) is stored; there is no value threshold.
- EE tape, unit NFTE (default 15): record 1 holds NAME(30 I4) and then
  ISPIN, IORB, IPAR, NWT, NWF, ITSTV, ITNEL, LIJB, IDIAG, LMAX, INTMAX, LWFL,
  NTYPE, NOREF, MXRPS. Record 2 holds ISSH(LMAX), NWTN(NWT), NWTC(NWF),
  NWSQ(NWT). Then come ``KT, NWFSG(KT)`` chunks (I4 + I2*KT) totalling
  INTMAX I2 words [wfetap.fpl COPY_NFWFW_TO_NFWFE]. NWFSG is a sequence of
  CSF types: ns, krps, nwen(ns), nweq(ns), nwesl(ns), then krps orbital
  tuples of ns shell indices. CSF numbering, which is H's row/column
  order, is the order of these types and tuples.
- Orbital vectors (unit 27, WTVEC): a header record per set, then one
  record per symmetry ``isym-1, C(NBF*NOB)`` with column-major columns =
  orbitals, coefficients over the normalized contracted basis.
"""
import struct

import numpy as np

LEMBF = 2382
LLETTER = 'SPDFGHIKLMNOQ'
llower = 'spdfghiklmnoq'


def records(path):
    b = open(path, 'rb').read()
    i, out = 0, []
    while i < len(b):
        n = struct.unpack('<i', b[i:i + 4])[0]
        if struct.unpack('<i', b[i + 4 + n:i + 8 + n])[0] != n:
            raise ValueError(f'{path}: bad record marker at byte {i}')
        out.append(b[i + 4:i + 4 + n])
        i += 8 + n
    return out


def iter_records(path):
    """Records one at a time (for files too large to read whole, e.g. the full-CI EE tape)."""
    with open(path, 'rb') as f:
        while True:
            head = f.read(4)
            if not head:
                return
            n = struct.unpack('<i', head)[0]
            body = f.read(n)
            if struct.unpack('<i', f.read(4))[0] != n:
                raise ValueError(f'{path}: bad record marker')
            yield body


def read_h_sparse(diag_path, off_path):
    """Lower triangle of H from units 38/39 without densifying.

    Returns (diag, rows, cols, vals, nent) with rows > cols, 0-based. Unit 39
    is memory-mapped record by record (all records after the header have the
    same length 16*LEMBF).
    """
    rd = records(diag_path)
    n1, n2 = struct.unpack('<ii', rd[0][:8])
    assert n1 == n2
    d = np.frombuffer(rd[0][8:8 + 8 * n1], '<f8').copy()
    with open(off_path, 'rb') as f:
        hlen = struct.unpack('<i', f.read(4))[0]
        f.seek(4 + hlen)
        assert struct.unpack('<i', f.read(4))[0] == hlen
        nocsf, lembf = struct.unpack('<ii', open(off_path, 'rb').read(12)[4:12])
    assert nocsf == n1 and lembf == LEMBF, (nocsf, lembf)
    dt = np.dtype([('m1', '<i4'), ('v', '<f8', LEMBF), ('r', '<i4', LEMBF), ('c', '<i4', LEMBF), ('m2', '<i4')])
    mm = np.memmap(off_path, dtype=dt, mode='r', offset=8 + hlen)
    assert np.all(mm['m1'] == 16 * LEMBF) and np.all(mm['m2'] == 16 * LEMBF)
    ends = np.nonzero(mm['r'][:, -1] == 0)[0]
    assert len(ends) >= 1, 'no terminating record in ' + off_path
    nrec = int(ends[0]) + 1
    counts = np.asarray(mm['c'][:nrec, -1], dtype=np.int64)
    nent = int(counts.sum())
    rows = np.empty(nent, np.int32)
    cols = np.empty(nent, np.int32)
    vals = np.empty(nent, np.float64)
    o = 0
    for k in range(nrec):
        m = int(counts[k])
        rec = mm[k]
        rows[o:o + m] = rec['r'][:m] - 1
        cols[o:o + m] = rec['c'][:m] - 1
        vals[o:o + m] = rec['v'][:m]
        o += m
    assert np.all(rows > cols)
    return d, rows, cols, vals, nent


def read_h(diag_path, off_path):
    """Dense symmetric H from units 38/39 (or 32/33 for the PT stage)."""
    d, rows, cols, vals, nent = read_h_sparse(diag_path, off_path)
    H = np.diag(d)
    H[rows, cols] = vals
    H[cols, rows] = vals
    return H, nent


def sparse_h(npz):
    """scipy.sparse CSR symmetric H from a results/<run>.npz (dense or sparse export)."""
    import scipy.sparse as sp
    if 'H' in npz.files:
        return sp.csr_matrix(npz['H'])
    n = int(npz['N'])
    L = sp.coo_matrix((npz['H_vals'], (npz['H_rows'], npz['H_cols'])), shape=(n, n)).tocsr()
    return (L + L.T - sp.diags(L.diagonal())).tocsr()


def _ls(code):
    return f"{code >> 8}{LLETTER[(code & 255) // 2]}"


def read_csfs(ee_path):
    """CSF list in H order from the EE tape. Returns (header dict, list of CSFs)."""
    it = iter_records(ee_path)      # only the header and CSF records are read
    rec = [next(it), next(it)]
    h = np.frombuffer(rec[0][120:], '<i4')
    keys = ['ISPIN', 'IORB', 'IPAR', 'NWT', 'NWF', 'ITSTV', 'ITNEL', 'LIJB', 'IDIAG',
            'LMAX', 'INTMAX', 'LWFL', 'NTYPE', 'NOREF', 'MXRPS']
    hd = {k: int(v) for k, v in zip(keys, h)}
    lm, nwt, nwf = hd['LMAX'], hd['NWT'], hd['NWF']
    h2 = np.frombuffer(rec[1], '<i4')
    nwtn = h2[lm:lm + nwt]
    orbs = [f"{x >> 8}{llower[(x & 255) // 2]}" for x in nwtn]
    hd['orbitals'] = orbs
    hd['closed'] = [orbs[i - 1] for i in h2[lm + nwt:lm + nwt + nwf]]
    w = []
    while len(w) < hd['INTMAX']:
        r = next(it)
        kt = struct.unpack('<i', r[:4])[0]
        w.extend(np.frombuffer(r[4:4 + 2 * kt], '<i2').tolist())
    assert len(w) == hd['INTMAX']
    csfs, i = [], 0
    while i < len(w):
        ns, kr = w[i], w[i + 1]
        nen = w[i + 2:i + 2 + ns]
        neq = w[i + 2 + ns:i + 2 + 2 * ns]
        nesl = w[i + 2 + 2 * ns:i + 2 + 3 * ns]
        for j in range(kr):
            o = w[i + 2 + 3 * ns + ns * j:i + 2 + 3 * ns + ns * (j + 1)]
            shells = [(orbs[a - 1], q) for a, q in zip(o, neq)]
            csfs.append({
                'label': ' '.join(f'{s}{q}' for s, q in shells),
                'shells': shells,
                'shell_states': [hex(x) for x in nen],
                'couplings': [_ls(x) for x in nesl],
            })
        i += 2 + ns * (3 + kr)
    assert len(csfs) == hd['ITSTV'], (len(csfs), hd['ITSTV'])
    return hd, csfs


def read_vectors(path, nbf):
    """All orbital sets in a WTVEC file: list of {l: C (nbf x nob)}."""
    rec = records(path)
    sets, i = [], 0
    nsym = len(nbf)
    while i < len(rec):
        cur = {}
        for l in range(nsym):
            r = rec[i + 1 + l]
            assert struct.unpack('<i', r[:4])[0] == l
            n = nbf[l]
            cur[l] = np.frombuffer(r[4:], '<f8').reshape(n, n).T.copy()
        sets.append(cur)
        i += 1 + nsym
    return sets


def load_h(npz):
    """Dense H from a results/<run>.npz (dense 'H' or sparse lower triangle H_rows/H_cols/H_vals/N)."""
    if 'H' in npz.files:
        return npz['H']
    n = int(npz['N'])
    H = np.zeros((n, n))
    r, c, v = npz['H_rows'], npz['H_cols'], npz['H_vals']
    H[r, c] = v
    H[c, r] = v
    return H
