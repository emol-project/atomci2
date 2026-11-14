import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import integrals as I  # noqa: E402

# Be run C coupling sequence: 5 s, 4 p, 3 d (outer), then 2s, 2p, 1s
BE_NWLF = [1] * 5 + [2] * 4 + [3] * 3 + [1, 2, 1]


def test_int2ad_coulomb_symmetries():
    ad = I.Addressing(BE_NWLF)
    rng = np.random.default_rng(0)
    n = len(BE_NWLF)
    checked = 0
    for _ in range(400):
        a, b, c, d = (int(x) for x in rng.integers(1, n + 1, 4))
        la, lb, lc, ld = (BE_NWLF[i - 1] - 1 for i in (a, b, c, d))
        for k in range(0, 5):
            if (la + lb + k) % 2 or (lc + ld + k) % 2:
                continue
            if not (abs(la - lb) <= k <= la + lb and abs(lc - ld) <= k <= lc + ld):
                continue
            x = ad.int2ad(2 * k + 1, a, b, c, d)
            assert x == ad.int2ad(2 * k + 1, b, a, c, d) == ad.int2ad(2 * k + 1, a, b, d, c)
            assert x == ad.int2ad(2 * k + 1, c, d, a, b)
            checked += 1
    assert checked > 50


def test_label_table_is_injective_and_dense():
    ad = I.Addressing(BE_NWLF)
    tab = ad.label_table()
    n1 = sum(1 for v in tab.values() if v[0] == '1e')
    assert n1 == ad.n1e  # one-electron addresses 1..n1e, all used
    assert sorted(a for a, v in tab.items() if v[0] == '1e') == list(range(1, ad.n1e + 1))
    assert max(tab) == ad.ntot  # last two-electron address = total length
    assert len(tab) == ad.ntot  # every address in 1..ntot decodes to one label


def test_slater_against_closed_form():
    # two normalized 1s-type Gaussians: P = N r exp(-a r^2); R^0(aa; bb) has a closed form
    x, r = I.radial_grid(n=20001)
    a, b = 0.7, 2.3
    Pa = (2 * (2 * a) ** 1.5 / I._gamma(1.5)) ** 0.5 * r * np.exp(-a * r * r)
    Pb = (2 * (2 * b) ** 1.5 / I._gamma(1.5)) ** 0.5 * r * np.exp(-b * r * r)
    # F0 between spherical Gaussian densities: 2/sqrt(pi) * sqrt(2ab/(a+b)) for unit-normalized densities
    exact = 2 / np.sqrt(np.pi) * np.sqrt(2 * a * b / (a + b))
    assert abs(I.slater(Pa, Pa, Pb, Pb, 0, x, r) - exact) < 1e-10
    # symmetry R^k(ab;cd) = R^k(cd;ab)
    Pc = Pb * r
    assert abs(I.slater(Pa, Pb, Pc, Pa, 0, x, r) - I.slater(Pc, Pa, Pa, Pb, 0, x, r)) < 1e-10


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'runs', 'C', 'fort.15')),
                    reason='runs/C not present (gitignored run directory)')
def test_be_valence_tape_reproduces_hmm():
    _C.setup_path()
    import phase6_step0
    res = phase6_step0.verify_tape('C')
    assert res['undecoded'] == 0 and res['max_diff'] < 1e-10


def test_counting_known_cases():
    from atomci_heff import counting as K
    # textbook terms: p3 -> 4S, 2P, 2D (one 2P); d2 -> 1S,1D,1G,3P,3F; p2 -> 1S,1D,3P
    assert K.n_csf([(1, 3)], 1, 1) == 1 and K.n_csf([(1, 3)], 0, 3) == 1 and K.n_csf([(1, 3)], 2, 1) == 1
    assert K.n_csf([(2, 2)], 3, 2) == 1 and K.n_csf([(2, 2)], 4, 0) == 1 and K.n_csf([(2, 2)], 2, 0) == 1
    assert K.n_csf([(1, 2)], 0, 0) == 1 and K.n_csf([(1, 1), (1, 1)], 0, 0) == 1
    # d4 has two 1S states (seniority 0 and 4); Phase 4 per-configuration count
    assert K.n_csf([(2, 4)], 0, 0) == 2
    # Be 1S even, 4 electrons in 7s5p3d: 2221 CSFs in 1288 configurations (Phase 4, = ATOMCI)
    be = [(f'{i}s', 0) for i in range(1, 8)] + [(f'{i}p', 1) for i in range(2, 7)] + [(f'{i}d', 2) for i in range(3, 6)]
    c = K.count_space(be, 4, 0, 0, 0)
    assert sum(c.values()) == 2221 and len(c) == 1288


def test_boron_counts_match_atomci_runs():
    from atomci_heff import counting as K
    o = [(f'{i}s', 0) for i in range(1, 5)] + [(f'{i}p', 1) for i in range(2, 5)] + [(f'{i}d', 2) for i in range(3, 5)] + [('4f', 3)]
    assert sum(K.count_space(o, 5, 1, 1, 1).values()) == 9433        # b_full_4321 (ATOMCI N)
    val = [x for x in o if x[0] != '1s']
    assert sum(K.count_space(val, 3, 1, 1, 1).values()) == 169      # b_val_4321


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'runs', 'b_sdc_7532', 'fort.24')),
                    reason='runs/b_sdc_7532 not present (gitignored run directory)')
def test_density_from_unit24_has_correct_trace():
    from atomci_heff import density
    from atomci_heff.reader import load_h, read_csfs
    H = load_h(np.load(os.path.join(ROOT, 'results', 'b_sdc_7532.npz')))
    c = np.linalg.eigh(H)[1][:, 0]
    hd, _ = read_csfs(os.path.join(ROOT, 'runs', 'b_sdc_7532', 'fort.15'))
    ad = I.Addressing(['spdf'.index(o[-1]) + 1 for o in hd['orbitals']])
    gam = density.gamma_per_l(density.label_density(os.path.join(ROOT, 'runs', 'b_sdc_7532', 'fort.24'), c), ad)
    assert abs(sum(np.trace(g) for g in gam.values()) - 5.0) < 1e-10
    for g in gam.values():
        w = np.linalg.eigvalsh(g)
        assert w.min() > -1e-10 and w.max() < 2 + 1e-10


def test_symlower_and_davidson_on_random_matrix():
    import scipy.sparse as sp
    from atomci_heff.sparse import SymLower, davidson
    rng = np.random.default_rng(1)
    n = 400
    A = sp.random(n, n, density=0.05, random_state=2).toarray()
    H = np.tril(A, -1) * 0.1
    H = H + H.T + np.diag(np.sort(rng.uniform(-5, 50, n)))
    op = SymLower(sp.csr_matrix(np.tril(H, -1)), np.diag(H))
    x = rng.normal(size=(n, 3))
    assert np.abs(op.matvec(x) - H @ x).max() < 1e-12
    idx = np.arange(0, n, 3)
    assert np.abs(op.sub(idx).dense() - H[np.ix_(idx, idx)]).max() == 0
    E, C, rn, it, nmv = davidson(op, 5, tol=1e-9)
    assert np.abs(E - np.linalg.eigvalsh(H)[:5]).max() < 1e-12


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'runs', 'b_full_4321', 'fort.39')),
                    reason='runs/b_full_4321 not present (gitignored run directory)')
def test_sparse_reader_matches_export():
    from atomci_heff.reader import load_h, read_h_sparse
    H = load_h(np.load(os.path.join(ROOT, 'results', 'b_full_4321.npz')))
    d, r, c, v, n = read_h_sparse(os.path.join(ROOT, 'runs', 'b_full_4321', 'fort.38'),
                                  os.path.join(ROOT, 'runs', 'b_full_4321', 'fort.39'))
    assert n == np.count_nonzero(np.tril(H, -1)) and np.all(H[r, c] == v) and np.all(np.diag(H) == d)


def test_ion_ladder_counts_match_atomci_runs():
    # Phase 7 decks (scripts/gen_ion.py): independent counts = ATOMCI N of each run
    from atomci_heff import counting as K
    o = ([(f'{i}s', 0) for i in range(1, 8)] + [(f'{i}p', 1) for i in range(2, 7)]
         + [(f'{i}d', 2) for i in range(3, 6)] + [('4f', 3)])
    val = [x for x in o if x[0] != '1s']
    atomci = {(2, 0, 0, 0): 50, (3, 0, 1, 0): 473, (3, 1, 1, 1): 825, (3, 2, 1, 0): 838, (3, 3, 1, 1): 613,
              (4, 0, 0, 0): 2737, (4, 1, 2, 1): 7914, (4, 1, 0, 1): 5601, (4, 1, 2, 0): 6753, (4, 2, 0, 0): 6608}
    frozen = {(3, 0, 1, 0): 6, (3, 1, 1, 1): 5, (3, 2, 1, 0): 3, (3, 3, 1, 1): 1,
              (4, 0, 0, 0): 43, (4, 1, 2, 1): 48, (4, 1, 0, 1): 48, (4, 1, 2, 0): 22, (4, 2, 0, 0): 45}
    for (n, L, s2, p), N in atomci.items():
        assert sum(K.count_space(o, n, L, s2, p).values()) == N
    for (n, L, s2, p), N in frozen.items():
        assert sum(K.count_space(val, n - 2, L, s2, p).values()) == N


def test_p3_pair_weights_and_slater_condon():
    # Phase 8: two-particle CFP weights derived numerically; checked against Gaunt-coefficient Slater-Condon
    from fractions import Fraction
    from atomci_heff import cfp
    want = {(0, 3): {(1, 2): Fraction(1)}, (2, 1): {(1, 2): Fraction(1, 2), (2, 0): Fraction(1, 2)},
            (1, 1): {(1, 2): Fraction(1, 2), (2, 0): Fraction(5, 18), (0, 0): Fraction(2, 9)}}
    F0, F2 = 0.7, 0.013
    e2 = {T: cfp.slater_condon(2, *T, F0, F2) for T in [(1, 2), (2, 0), (0, 0)]}
    assert [round((e2[T] - F0) / F2, 9) for T in [(1, 2), (2, 0), (0, 0)]] == [-5, 1, 10]
    for term, ws in want.items():
        got = cfp.pair_weights(3, *term)
        for T in e2:
            assert abs(got[T] - float(ws.get(T, 0))) < 1e-12
        e3 = cfp.slater_condon(3, *term, F0, F2)
        assert abs(3 * sum(got[T] * e2[T] for T in e2) - e3) < 1e-12
