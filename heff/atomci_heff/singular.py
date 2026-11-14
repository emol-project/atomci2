"""Exceptional points of H(z) = H0 + z V (H0 diagonal, V real symmetric).

At an exceptional point (EP) two eigenvalues coalesce and the shared
eigenvector x is self-orthogonal (x^T x = 0, unconjugated). We solve the
bordered system with Newton's method:

    (H0 + z V - E) x = 0,   x^T x = 0,   c^T x = 1

The unknowns are (x, E, z), all complex; the EP is a regular root. Start
points come from near-degenerate eigenpairs on a complex z grid, with
x0 = x_i + i x_j, which is self-orthogonal when x_i, x_j are
complex-orthonormal.

H(z) is real symmetric for real z, so there are no EPs on the real axis,
and they come in conjugate pairs z, conj(z).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cnorm(x):
    return x / np.sqrt(x @ x)


def newton_ep(d, V, z0, E0, x0, c, tol=1e-12, maxit=60):
    n = len(d)
    x, E, z = x0.astype(complex), complex(E0), complex(z0)
    for _ in range(maxit):
        A = np.diag(d) + z * V - E * np.eye(n)
        F = np.concatenate([A @ x, [x @ x, c @ x - 1.0]])
        if np.abs(F).max() < tol:
            return z, E, x, True
        J = np.zeros((n + 2, n + 2), complex)
        J[:n, :n] = A
        J[:n, n] = -x
        J[:n, n + 1] = V @ x
        J[n, :n] = 2 * x
        J[n + 1, :n] = c
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return z, E, x, False
        x = x + dx[:n]
        E = E + dx[n]
        z = z + dx[n + 1]
        if abs(z) > 5:
            return z, E, x, False
    A = np.diag(d) + z * V - E * np.eye(n)
    ok = np.abs(A @ x).max() < 1e-9 and abs(x @ x) < 1e-9
    return z, E, x, ok


def ep_gap(d, V, z, E):
    """Distance between the two eigenvalues of H(z) nearest E (should be ~sqrt(eps))."""
    ev = np.linalg.eigvals(np.diag(d) + z * V)
    s = np.sort(np.abs(ev - E))
    return s[1]


def find_eps(d, V, zmax=1.6, nre=33, nim=17, seed=0):
    """All EPs with Im z > 0 and |z| < zmax that are reachable from a grid of start points."""
    rng = np.random.default_rng(seed)
    n = len(d)
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    found = []
    for zr in np.linspace(-zmax, zmax, nre):
        for zi in np.linspace(0.02, zmax, nim):
            z0 = zr + 1j * zi
            if abs(z0) > zmax:
                continue
            ev, X = np.linalg.eig(np.diag(d) + z0 * V)
            for i in range(n):
                dist = np.abs(ev - ev[i])
                dist[i] = np.inf
                j = int(np.argmin(dist))
                if j < i:
                    continue
                xi, xj = _cnorm(X[:, i]), _cnorm(X[:, j])
                x0 = xi + 1j * xj
                x0 = x0 / (c @ x0)
                z, E, x, ok = newton_ep(d, V, z0, 0.5 * (ev[i] + ev[j]), x0, c)
                if not ok or abs(z) >= zmax:
                    continue
                if z.imag < 0:
                    z, E, x = np.conj(z), np.conj(E), np.conj(x)
                if abs(z.imag) < 1e-8:
                    continue
                if all(abs(z - f['z']) > 1e-7 for f in found):
                    found.append(dict(z=z, E=E, x=x, gap=ep_gap(d, V, z, E)))
    return sorted(found, key=lambda f: abs(f['z']))


def track(d, V, zend, nsteps=4000, tfinal=0.995):
    """Follow all n branches along the straight ray z = t*zend, t in [0, tfinal].

    Returns perm with perm[k] = index (into the eigenvalues at t=tfinal) of
    the branch that starts at d[k] at z = 0, plus those final eigenvalues.
    """
    n = len(d)
    prev = d.astype(complex)
    ts = np.concatenate([np.linspace(0, 0.9, nsteps // 2)[1:],
                         0.9 + (tfinal - 0.9) * np.linspace(0, 1, nsteps // 2)[1:]])
    for t in ts:
        ev = np.linalg.eigvals(np.diag(d) + t * zend * V)
        r, cidx = linear_sum_assignment(np.abs(prev[:, None] - ev[None, :]))
        new = np.empty(n, complex)
        new[r] = ev[cidx]
        prev = new
    return prev


def classify(d, V, ep, P, nsteps=4000):
    """Which z=0 branches (H0 states) coalesce at this EP? Returns sorted pair (k1, k2)."""
    final = track(d, V, ep['z'], nsteps=nsteps)
    k = np.argsort(np.abs(final - ep['E']))[:2]
    return tuple(sorted(int(i) for i in k)), np.abs(final[k[0]] - final[k[1]])
