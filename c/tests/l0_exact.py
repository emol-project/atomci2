#!/usr/bin/env python3
"""l0_exact.py -- check ac_wigner's 6j and 9j against exact arithmetic.

Independent of the Fortran: each 6j is computed from the Racah formula as
sign * sqrt(rational) with Fractions, then evaluated to 50 digits with
Decimal; a 9j is the 6j sum of METECC eq. (A40) in the same precision.

Usage: l0_exact.py path/to/wigner_dump [--tj6 N] [--tj9 N]

Tolerance. Both symbols are alternating sums, so rounding error scales with
the size of the summands, not with the (possibly much smaller) result: a 9j
of 1e-3 built from 6j products of 1e-2 legitimately keeps ~1e-16 absolute
error. Each value is therefore judged against its scale, the same sum taken
over absolute values; exit status 1 if |C - exact| > 1e-14 * scale.
"""
import argparse
import itertools
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from math import factorial

getcontext().prec = 50


def tri(a, b, c):
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def delta2(a, b, c):
    """Delta(abc)^2 = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!, args in 2j."""
    return Fraction(
        factorial((a + b - c) // 2) * factorial((a - b + c) // 2)
        * factorial((-a + b + c) // 2),
        factorial((a + b + c) // 2 + 1))


def frac_dec(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def sixj(a, b, c, d, e, f):
    """(value, scale): scale is the Racah sum over |terms| times the root."""
    if not (tri(a, b, c) and tri(a, e, f) and tri(d, b, f) and tri(d, e, c)):
        return Decimal(0), Decimal(0)
    al = [(a + b + c) // 2, (a + e + f) // 2, (d + b + f) // 2, (d + e + c) // 2]
    be = [(a + b + d + e) // 2, (b + c + e + f) // 2, (c + a + f + d) // 2]
    s = Fraction(0)
    sabs = Fraction(0)
    for t in range(max(al), min(be) + 1):
        den = 1
        for x in al:
            den *= factorial(t - x)
        for x in be:
            den *= factorial(x - t)
        term = Fraction(factorial(t + 1), den)
        s += term if t % 2 == 0 else -term
        sabs += term
    d2 = delta2(a, b, c) * delta2(a, e, f) * delta2(d, b, f) * delta2(d, e, c)
    root = frac_dec(d2).sqrt()
    return frac_dec(s) * root, frac_dec(sabs) * root


def ninej(a, b, c, d, e, f, g, h, i):
    """(value, scale): scale is the 6j sum over |products of 6j scales|."""
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    s = Decimal(0)
    scale = Decimal(0)
    for x in range(lo, hi + 1, 2):
        sign = 1 if x % 2 == 0 else -1
        v1, s1 = sixj(a, b, c, f, i, x)
        v2, s2 = sixj(d, e, f, b, x, h)
        v3, s3 = sixj(g, h, i, x, a, d)
        s += sign * (x + 1) * v1 * v2 * v3
        scale += (x + 1) * s1 * s2 * s3
    return s, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tj6", type=int, default=10)
    ap.add_argument("--tj9", type=int, default=6)
    args = ap.parse_args()

    cases = []
    r6 = range(args.tj6 + 1)
    for t in itertools.product(r6, repeat=6):
        a, b, c, d, e, f = t
        if tri(a, b, c) and tri(a, e, f) and tri(d, b, f) and tri(d, e, c):
            cases.append((6,) + t)
    r9 = range(args.tj9 + 1)
    for a, b, c in itertools.product(r9, repeat=3):
        if not tri(a, b, c):
            continue
        for d, e, f in itertools.product(r9, repeat=3):
            if not tri(d, e, f):
                continue
            for g, h, i in itertools.product(r9, repeat=3):
                if tri(g, h, i) and tri(a, d, g) and tri(b, e, h) and tri(c, f, i):
                    cases.append((9, a, b, c, d, e, f, g, h, i))

    inp = "\n".join(" ".join(map(str, c)) for c in cases) + "\n"
    out = subprocess.run([args.dump], input=inp, capture_output=True, text=True,
                         check=True).stdout.split("\n")

    worst_scaled = {6: 0.0, 9: 0.0}
    worst_abs = {6: 0.0, 9: 0.0}
    count = {6: 0, 9: 0}
    bad = 0
    for line in out:
        if not line:
            continue
        f = line.split()
        kind = int(f[0])
        t = list(map(int, f[1:-1]))
        vc = Decimal(f[-1])
        ve, scale = sixj(*t) if kind == 6 else ninej(*t)
        diff = abs(vc - ve)
        scaled = float(diff / scale) if scale > 0 else float(diff)
        count[kind] += 1
        worst_scaled[kind] = max(worst_scaled[kind], scaled)
        worst_abs[kind] = max(worst_abs[kind], float(diff))
        if scaled > 1e-14:
            bad += 1
            if bad <= 5:
                print(f"mismatch {kind}j {t}: C {vc} exact {ve:.20e} "
                      f"scale {scale:.3e}", file=sys.stderr)
    for k in (6, 9):
        print(f"{k}j exact  {count[k]:10d} compared   max |C-exact| = "
              f"{worst_abs[k]:.3e}   max |C-exact|/scale = {worst_scaled[k]:.3e}")
    print("L0 exact", "FAILED" if bad else "passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
