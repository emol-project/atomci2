import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
from atomci_heff import heff  # noqa: E402
from atomci_heff.reader import LEMBF, read_h  # noqa: E402


def model(n=12, seed=0, coupling=0.05):
    """Symmetric matrix with a well-separated 2-dim P space; the EN series converges."""
    rng = np.random.default_rng(seed)
    d = np.concatenate([[0.0, 0.3], 2.0 + np.sort(rng.random(n - 2)) * 3])
    V = rng.normal(scale=coupling, size=(n, n))
    V = (V + V.T) / 2
    np.fill_diagonal(V, 0)
    return np.diag(d) + V


def test_bloch_heff_reproduces_eigenvalues():
    H = model()
    e, C, wP, _ = heff.spectrum(H, [0, 1])
    states, amb = heff.select_states(wP, 2)
    assert list(states) == [0, 1] and amb < 0.1
    r = heff.exact_heff(H, [0, 1], states)
    assert np.allclose(np.sort(np.linalg.eigvals(r['bloch']).real), r['E'], atol=1e-12)
    assert np.allclose(r['hermitian'], r['hermitian'].T, atol=1e-13)
    assert np.allclose(np.linalg.eigvalsh(r['hermitian']), r['E'], atol=1e-12)


def test_exact_chi_solves_bloch_equation():
    H = model()
    P, Q = heff.split(H.shape[0], [0, 1])
    r = heff.exact_heff(H, P, [0, 1])
    chi = r['chi']
    # Q H (P + chi) = chi P H (P + chi): the Bloch equation in matrix form
    lhs = H[np.ix_(Q, P)] + H[np.ix_(Q, Q)] @ chi
    rhs = chi @ (H[np.ix_(P, P)] + H[np.ix_(P, Q)] @ chi)
    assert np.abs(lhs - rhs).max() < 1e-12
    # and H_eff = P H Omega P
    assert np.allclose(H[np.ix_(P, P)] + H[np.ix_(P, Q)] @ chi, r['bloch'], atol=1e-12)


def test_en_series_converges_to_exact_heff():
    H = model()
    r = heff.exact_heff(H, [0, 1], [0, 1])
    terms, chis = heff.en_series(H, [0, 1], 60)
    sums = heff.partial_sums(terms)
    assert np.abs(sums[-1] - r['bloch']).max() < 1e-10
    assert np.abs(np.sort(np.linalg.eigvals(sums[-1]).real) - r['E']).max() < 1e-10
    chi = sum(chis[1:])
    assert np.abs(chi - r['chi']).max() < 1e-10


def test_en_low_orders_match_closed_forms():
    H = model(coupling=0.2, seed=3)
    terms, _ = heff.en_series(H, [0, 1], 3)
    d = np.diag(H)
    P, Q = [0, 1], list(range(2, H.shape[0]))
    # 2nd order: sum_q V_pq V_qp' / (E_p' - E_q)
    h2 = np.array([[sum(H[p, q] * H[q, pp] / (d[pp] - d[q]) for q in Q) for pp in P] for p in P])
    assert np.allclose(terms[2], h2, atol=1e-14)
    assert np.allclose(terms[1], H[np.ix_(P, P)] - np.diag(d[P]), atol=1e-15)


def test_series_diverges_with_intruder():
    # an intruder Q state between the two P diagonals with strong coupling
    H = np.diag([0.0, 1.0, 0.5, 5.0])
    H[0, 2] = H[2, 0] = 0.4
    H[1, 2] = H[2, 1] = 0.4
    H[0, 3] = H[3, 0] = 0.1
    terms, _ = heff.en_series(H, [0, 1], 40)
    assert np.abs(terms[-1]).max() > np.abs(terms[5]).max()


def _write_record(f, payload):
    f.write(struct.pack('<i', len(payload)))
    f.write(payload)
    f.write(struct.pack('<i', len(payload)))


def test_reader_roundtrip(tmp_path):
    H = model(n=7)
    n = H.shape[0]
    with open(tmp_path / 'd', 'wb') as f:
        _write_record(f, struct.pack('<ii', n, n) + np.diag(H).astype('<f8').tobytes())
    rows, cols = np.tril_indices(n, -1)
    with open(tmp_path / 'o', 'wb') as f:
        _write_record(f, struct.pack('<ii', n, LEMBF) + b'\0' * 100)
        per = LEMBF - 1
        for s in range(0, len(rows), per):
            r, c = rows[s:s + per], cols[s:s + per]
            k = len(r)
            hv = np.zeros(LEMBF)
            ir = np.zeros(LEMBF, '<i4')
            ic = np.zeros(LEMBF, '<i4')
            hv[:k] = H[r, c]
            ir[:k] = r + 1
            ic[:k] = c + 1
            ic[-1] = k
            ir[-1] = 0 if s + per >= len(rows) else 1
            _write_record(f, hv.astype('<f8').tobytes() + ir.tobytes() + ic.tobytes())
    H2, nent = read_h(tmp_path / 'd', tmp_path / 'o')
    assert nent == n * (n - 1) // 2
    assert np.array_equal(H, H2)


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'results', 'A.npz')),
                    reason='results/A.npz not generated')
def test_run_A_ground_state_matches_atomci_output():
    z = np.load(os.path.join(os.path.dirname(__file__), '..', 'results', 'A.npz'))
    assert z['H'].shape == (42, 42)
    assert list(z['labels'][:2]) == ['2s2', '2p2']
    assert abs(np.linalg.eigvalsh(z['H'])[0] - (-14.617330)) < 1e-6


def test_load_h_sparse_roundtrip(tmp_path):
    from atomci_heff.reader import load_h
    H = model(n=9)
    r, c = np.nonzero(np.tril(H))
    np.savez(tmp_path / 'h.npz', H_rows=r, H_cols=c, H_vals=H[r, c], N=9)
    assert np.array_equal(load_h(np.load(tmp_path / 'h.npz')), H)
    np.savez(tmp_path / 'd.npz', H=H)
    assert np.array_equal(load_h(np.load(tmp_path / 'd.npz')), H)
