import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
from atomci_heff import heff, partition, resum, singular, solvers  # noqa: E402
from test_heff import model  # noqa: E402

P = [0, 1]


def exact(H):
    return heff.exact_heff(H, P, [0, 1])


def rot_Q(H, seed=1):
    n = H.shape[0]
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(n - 2, n - 2)))
    O = np.eye(n)
    O[2:, 2:] = R
    return O.T @ H @ O


def test_all_partitionings_converge_to_exact():
    H = model(coupling=0.05)
    E = exact(H)['E']
    for mode in partition.MODES:
        terms, *_ = partition.series(H, P, mode, 40)
        err = partition.eig_errors(terms, E)
        assert err[-1].max() < 1e-10, mode


def test_q_rotation_invariance():
    H = model(coupling=0.1)
    H2 = rot_Q(H)
    E = exact(H)['E']
    for mode, invariant in [('EN', False), ('P', False), ('Q', True), ('PQ', True)]:
        e1 = partition.eig_errors(partition.series(H, P, mode, 6)[0], E)
        e2 = partition.eig_errors(partition.series(H2, P, mode, 6)[0], E)
        same = np.allclose(e1, e2, atol=1e-12, rtol=1e-8)
        assert same == invariant, mode


def test_pq_partition_v_has_only_offdiagonal_blocks():
    H = model()
    Ht, U, Pt = partition.transform(H, P, 'PQ')
    V = Ht - np.diag(np.diag(Ht))
    assert np.abs(V[:2, :2]).max() < 1e-13 and np.abs(V[2:, 2:]).max() < 1e-13


def test_ep_two_level_analytic():
    # H0 = diag(0,1), V = v sigma_x: E = (1 +- sqrt(1 + 4 z^2 v^2))/2, EP at z = i/(2v)
    v = 0.7
    d = np.array([0.0, 1.0])
    V = np.array([[0, v], [v, 0]])
    eps = singular.find_eps(d, V, zmax=2.0, nre=9, nim=9)
    assert len(eps) == 1
    assert abs(eps[0]['z'] - 1j / (2 * v)) < 1e-10
    assert eps[0]['gap'] < 1e-6


def test_ep_found_in_random_matrix_are_coalescences():
    H = model(n=8, coupling=0.3, seed=5)
    d = np.diag(H)
    V = H - np.diag(d)
    eps = singular.find_eps(d, V, zmax=1.5, nre=13, nim=7)
    assert eps
    for ep in eps:
        assert ep['gap'] < 1e-5


def test_solvers_reach_exact_chi():
    H = model(coupling=0.05)
    ref = exact(H)['chi']
    z = np.zeros((H.shape[0] - 2, 2))
    for res in [solvers.fixed_point(H, P, z), solvers.fixed_point(H, P, z, alpha=0.5),
                solvers.fixed_point(H, P, z, diis=6), solvers.newton(H, P, z),
                solvers.bw_iteration(H, P, z)]:
        assert res['converged']
        assert np.abs(res['chi'] - ref).max() < 1e-9
    ids = solvers.identify(H, P, ref)
    assert [i[1] for i in ids] == [0, 1]


def test_newton_can_land_on_other_pair():
    # start from the exact chi of states (0, 2): Newton must stay on that solution
    H = model(coupling=0.05)
    r = heff.exact_heff(H, P, [0, 2])
    res = solvers.newton(H, P, r['chi'] + 1e-4)
    assert res['converged']
    assert sorted(i[1] for i in solvers.identify(H, P, res['chi'])) == [0, 2]


def test_pade_on_convergent_and_branch_point_series():
    H = model(coupling=0.05)
    terms, *_ = partition.series(H, P, 'EN', 30)
    tab = resum.pade_table(terms, Ls=[8])
    assert np.abs(tab[8]['matrix'] - exact(H)['bloch']).max() < 1e-10
    # 2-level model with the EP inside |z|<1: partial sums diverge, Pade converges
    v = 0.8
    H2 = np.array([[0.0, v], [v, 1.0]])
    # P = {0}: scalar H_eff series = lower eigenvalue series
    terms2, _ = heff.en_series(H2, [0], 30)
    c = [t[0, 0] for t in terms2]
    ex = np.linalg.eigvalsh(H2)[0]
    assert abs(sum(c) - ex) > 1.0
    errs = [abs(resum.pade_at(c, L) - ex) for L in (2, 4, 6, 8, 10)]
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-5


def test_eigen_series_matches_partial_sum_eigs():
    H = model(coupling=0.05)
    terms, *_ = partition.series(H, P, 'EN', 30)
    lo, hi, _ = resum.eigen_series(terms)
    assert abs(lo.sum() - exact(H)['E'][0]) < 1e-10
    assert abs(hi.sum() - exact(H)['E'][1]) < 1e-10


def test_radius_matches_nearest_pq_ep_for_even_series():
    # 3-level model, P = {0}, partition 'PQ': H_eff(z) is even in z
    H = np.array([[0.0, 0.0, 0.6], [0.0, 3.0, 0.2], [0.6, 0.2, 0.5]])
    terms, _, Ht, _, Pt = partition.series(H, [0], 'PQ', 40)
    d = np.diag(Ht).copy()
    eps = singular.find_eps(d, Ht - np.diag(d), zmax=3.0, nre=15, nim=15)
    pq = [ep for ep in eps if singular.classify(d, Ht - np.diag(d), ep, Pt, nsteps=400)[0][0] == 0]
    R_ep = min(abs(ep['z']) for ep in pq)
    assert abs(partition.radius_from_terms(terms, n0=20) - R_ep) / R_ep < 0.1
