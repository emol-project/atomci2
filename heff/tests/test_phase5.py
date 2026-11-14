import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
from atomci_heff import approx, heff  # noqa: E402
from test_heff import model  # noqa: E402

P = [0, 1]


def blocks(H):
    Q = list(range(2, H.shape[0]))
    return (H[np.ix_(P, P)], H[np.ix_(P, Q)], H[np.ix_(Q, P)], H[np.ix_(Q, Q)]), Q


def test_parse_and_classify():
    refs = [{'1s': 2, '2s': 2}, {'1s': 2, '2p': 2}]
    lev, holes, conf = approx.classify(['2s2 1s2', '1s2 2p2', '11p1 1s1 2p2', '15s2 14s1 13s1', '11d4'], refs)
    assert list(lev) == [0, 0, 1, 4, 4] and list(holes) == [0, 0, 1, 2, 2]
    assert approx.parse_label('11p1 1s2 2p1') == {'11p': 1, '1s': 2, '2p': 1}


def test_heff_from_states_matches_exact():
    H = model()
    r = heff.exact_heff(H, P, [0, 1])
    s = approx.exact(H, P, [0, 1])
    assert np.allclose(s['bloch'], r['bloch'], atol=1e-12)
    assert np.allclose(s['dc'], r['hermitian'], atol=1e-12)
    t = approx.heff_from_chi(H[np.ix_(P, P)], H[np.ix_(P, range(2, 12))], r['chi'])
    assert np.allclose(t['dc'], r['hermitian'], atol=1e-11)


def test_solve_g_one_is_bloch_and_zero_is_linear():
    H = model(coupling=0.05)
    (HPP, HPQ, HQP, HQQ), Q = blocks(H)
    ref = heff.exact_heff(H, P, [0, 1])['chi']
    chi1, it, ok = approx.solve_g(HPP, HPQ, HQP, HQQ, 1.0, np.zeros_like(ref))
    assert ok and np.abs(chi1 - ref).max() < 1e-10
    chi0, cond, smin, sep = approx.sylvester_linear(HPP, HQP, HQQ)
    assert np.abs(approx.g_residual(HPP, HPQ, HQP, HQQ, chi0, 0.0)).max() < 1e-12
    chi0n, _, ok = approx.solve_g(HPP, HPQ, HQP, HQQ, 0.0, np.zeros_like(ref))
    assert ok and np.abs(chi0n - chi0).max() < 1e-10
    assert sep > 0 and cond >= 1


def test_blocked_extremes():
    H = model()
    Q = list(range(2, H.shape[0]))
    full = approx.blocked(H, Q, [Q])
    diag = approx.blocked(H, Q, [[q] for q in Q])
    assert np.array_equal(full, H)
    off = diag[np.ix_(Q, Q)] - np.diag(np.diag(diag[np.ix_(Q, Q)]))
    assert np.abs(off).max() == 0 and np.array_equal(diag[np.ix_(P, Q)], H[np.ix_(P, Q)])


def test_bw_solve_lands_on_lowest_pair():
    H = model(coupling=0.05)
    res, ids = approx.bw_solve(H, P)
    assert res['converged'] and sorted(i[1] for i in ids) == [0, 1]


def test_hierarchical_full_is_exact_and_empty_R_is_HMM():
    H = model(n=14, coupling=0.08, seed=4)
    M = list(range(6))
    R = list(range(6, 14))
    e, C = np.linalg.eigh(H)
    w, U = np.linalg.eigh(H[np.ix_(M, M)])
    out = approx.hierarchical(H, M, R, U[:, :2], mode='full')
    assert abs(out[0][0] - e[0]) < 1e-10 and abs(out[1][0] - e[1]) < 1e-10
    v = out[0][1] / np.linalg.norm(out[0][1])
    assert abs(abs(v @ C[:, 0]) - 1) < 1e-10
    out0 = approx.hierarchical(H, M, [], U[:, :2], mode='diag')
    assert abs(out0[0][0] - w[0]) < 1e-12


def test_hierarchical_block_with_one_group_equals_full():
    H = model(n=12, coupling=0.08, seed=2)
    M, R = list(range(5)), list(range(5, 12))
    w, U = np.linalg.eigh(H[np.ix_(M, M)])
    a = approx.hierarchical(H, M, R, U[:, :1], mode='block', groups=[R])
    b = approx.hierarchical(H, M, R, U[:, :1], mode='full')
    assert abs(a[0][0] - b[0][0]) < 1e-12


def test_davidson_q_limits():
    assert approx.davidson_q(-1.0, 1.0, -0.9) == -1.0
    assert abs(approx.davidson_q(-1.0, 0.9, -0.9) - (-1.01)) < 1e-15
