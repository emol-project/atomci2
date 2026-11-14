"""lowdin_n2.py -- S2-M9 (B4): an independent reference for acatsp "bio" at
n = 2 (single-CSF Hartree-Fock of each term, separately optimized), from
the nonorthogonal-orbital (Loewdin cofactor) formula, without any
biorthonormal transformation and without the C code.

Bra 1s^2 2s (orbitals X, from 2S_2.w), ket 1s'^2 2p (orbitals Y, 2P_2.w).
Determinants |1s a, 2s a, 1s b| and |1s' a, 2p a, 1s' b|: the alpha block
of the overlap is [[s11, 0], [s21, 0]] (s11 = <1s|1s'>, s21 = <2s|1s'>,
<ns|2p> = 0), the beta block [[s11]], and a one-electron operator o
(rank 1, s -> p) gives by the cofactor rule
    <D|o|D'> = s11 (s11 <2s|o|2p> - s21 <1s|o|2p>).
With orthonormal orbitals this is <2s|o|2p>, and the LS multiplet line
strength of one s -> p electron is S = 2 x (radial factor)^2 (checked on
the averaged orbitals of P6 against acatsp in the one-list run, below).
Radial factors: length int P_a r P_b dr; velocity int P_a (P_b' + P_b/r) dr
(the l = 0 <- 1 gradient factor, R_a (R_b' + 2 R_b / r) r^2 = P_a (P_b' + P_b/r)).
Integrals on ATSP's own grid r_j = exp(-4 + j/16)/Z (trapezoid in ln r
with the power-law part below the first point added, P' by sixth-order
differences in ln r).

usage: lowdin_n2.py 2S_2.w 2P_2.w [li_2.w]
prints S_l and S_v (a.u.); with li_2.w also the one-orbital-set values.
"""
import struct
import sys

import numpy as np


def read_w(path):
    """{label: (r, P)} from an ATSP .w file (gfortran unformatted)"""
    out, data, pos = {}, open(path, "rb").read(), 0
    while pos < len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        rec = data[pos + 4:pos + 4 + n]
        pos += 8 + n
        el = rec[12:15].decode().strip()
        m, z = struct.unpack_from("<id", rec, 15)
        # ATOM*6 TERM*6 EL*3, M, then Z, E(nl), EK, AZ, then F(1..M)
        f = np.frombuffer(rec, dtype="<f8", count=m, offset=15 + 4 + 8 * 4)
        j = np.arange(m)
        r = np.exp(-4.0 + j / 16.0) / z
        out[el] = (r, f * np.sqrt(r))
    return out


def on_grid(w, el, m):
    r, p = w[el]
    q = np.zeros(m)
    q[:len(p)] = p
    return q


def dx(p):
    """dP/dx on the uniform x grid, h = 1/16 (sixth order, interior)"""
    h, d = 1.0 / 16.0, np.zeros_like(p)
    c = [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]
    pp = np.concatenate([np.zeros(3), p, np.zeros(3)])
    for k, ck in enumerate(c):
        d += ck * pp[k:k + len(p)]
    return d / h


def factors(wx, wy, z=3.0):
    m = max(len(v[0]) for v in list(wx.values()) + list(wy.values()))
    r = np.exp(-4.0 + np.arange(m) / 16.0) / z
    h = 1.0 / 16.0

    def integ(f):
        """int f dr = int g dx, g = f r: trapezoid in x from the first point,
        plus the part below it, where g ~ r^k = e^(k x) (k from the first
        two points): g_0 / k (without it the 1s norm is short by ~6e-6)"""
        g = f * r
        k = np.log(g[1] / g[0]) / h
        return h * (np.sum(g) - g[0] / 2) + g[0] / k
    g = lambda w, el: on_grid(w, el, m)
    s11 = integ(g(wx, "1s") * g(wy, "1s"))
    s21 = integ(g(wx, "2s") * g(wy, "1s"))
    p2p = g(wy, "2p")
    dp = dx(p2p) / r + p2p / r  # P' + P/r
    rl = lambda el: integ(g(wx, el) * r * p2p)
    rv = lambda el: integ(g(wx, el) * dp)
    return s11, s21, rl("2s"), rl("1s"), rv("2s"), rv("1s")


def main():
    wx, wy = read_w(sys.argv[1]), read_w(sys.argv[2])
    s11, s21, r22, r12, v22, v12 = factors(wx, wy)
    ml = s11 * (s11 * r22 - s21 * r12)
    mv = s11 * (s11 * v22 - s21 * v12)
    print(f"lowdin: <1s|1s'> {s11:.10f} <2s|1s'> {s21:.10f}"
          f"  S_l {2 * ml ** 2:.10e}  S_v {2 * mv ** 2:.10e}")
    if "-v" in sys.argv:
        print(f"  r22 {r22:.10f} r12 {r12:.10f} v22 {v22:.10f} v12 {v12:.10f}")
        for name, a in (("no s21 term", s11 * s11 * r22), ("s21 term +", s11 * (s11 * r22 + s21 * r12)),
                        ("s11 once", s11 * r22 - s21 * r12), ("r22 only", r22)):
            print(f"  {name:12s} S_l {2 * a * a:.10e}")
    if len(sys.argv) > 3 and sys.argv[3] != "-v":
        w = read_w(sys.argv[3])
        _, _, r22, _, v22, _ = factors(w, w)
        print(f"lowdin-same: S_l {2 * r22 ** 2:.10e}  S_v {2 * v22 ** 2:.10e}")


if __name__ == "__main__":
    main()
