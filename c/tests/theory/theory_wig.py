# 6j and 9j with doubled arguments, for the theory_check_*.py scripts.
import math

F_ = math.factorial


def _tri2(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def _delta2(a, b, c):
    return math.sqrt(F_((a + b - c) // 2) * F_((a - b + c) // 2) * F_((-a + b + c) // 2) / F_((a + b + c) // 2 + 1))


def sixj2(a, b, c, d, e, f):
    """6j with doubled arguments (Racah formula)."""
    if not (_tri2(a, b, c) and _tri2(a, e, f) and _tri2(d, b, f) and _tri2(d, e, c)):
        return 0.0
    pre = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    s = 0.0
    lo = max(a + b + c, a + e + f, d + b + f, d + e + c) // 2
    hi = min(a + b + d + e, b + c + e + f, a + c + d + f) // 2
    for t in range(lo, hi + 1):
        s += (-1) ** t * F_(t + 1) / (F_(t - (a + b + c) // 2) * F_(t - (a + e + f) // 2) * F_(t - (d + b + f) // 2)
                                      * F_(t - (d + e + c) // 2) * F_((a + b + d + e) // 2 - t)
                                      * F_((b + c + e + f) // 2 - t) * F_((a + c + d + f) // 2 - t))
    return pre * s


def ninej2(a, b, c, d, e, f, g, h, i):
    """9j (doubled) as a sum of 6j products (checked in theory_check_conv.py)."""
    s = 0.0
    for x in range(0, 60):
        s += (-1) ** x * (x + 1) * sixj2(a, d, g, h, i, x) * sixj2(b, e, h, d, x, f) * sixj2(c, f, i, x, a, b)
    return s


def sgn(x2):
    """(-1)^(x2/2) for an even doubled exponent."""
    assert x2 % 2 == 0, x2
    return -1.0 if (x2 // 2) % 2 else 1.0
