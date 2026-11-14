"""Phase 9 recon: d-shell pair weights (atomci_heff/cfp_l.py)."""
import numpy as np

from atomci_heff import cfp, cfp_l as C

FK = {0: 0.83, 2: 0.37, 4: 0.23}


def test_p3_matches_p_module():
    for L, S in [(0, 3), (2, 1), (1, 1)]:
        W, W0 = C.pair_weights(1, 3, L, S), cfp.pair_weights(3, L, S)
        assert all(abs(W[T][0, 0] - W0[T]) < 1e-12 for T in W0)


def test_d2_racah():
    A, B, Cc = C.racah_abc(FK[0], FK[2], FK[4])
    r = C.racah_d2(A, B, Cc)
    for T, nm in C.D2_TERMS.items():
        assert abs(C.slater_condon(2, 2, *T, FK)[0, 0] - r[nm]) < 1e-12


def test_d3_pair_prediction_and_racah():
    A, B, Cc = C.racah_abc(FK[0], FK[2], FK[4])
    r3 = C.racah_d3(A, B, Cc)
    e2 = {T: C.slater_condon(2, 2, *T, FK)[0, 0] for T in C.D2_TERMS}
    for T, nm in C.D3_TERMS.items():
        sc = C.slater_condon(2, 3, *T, FK)
        W = C.pair_weights(2, 3, *T)
        pred = 3 * sum(W[t] * e2[t] for t in W)
        assert np.abs(sc - pred).max() < 1e-12
        assert np.allclose(sum(W.values()), np.eye(sc.shape[0]))
        ref = np.atleast_1d(r3[nm])
        assert np.allclose(np.linalg.eigvalsh(sc), sorted(ref) if len(ref) > 1 else [ref[0]] * sc.shape[0])


def test_d3_2D_seniority_basis():
    St, q = C.stretched(2, 3, 2, 1)
    assert St.shape[1] == 2 and q[0] > 0.5 and abs(q[1]) < 1e-12     # seniority 1 first, then 3
