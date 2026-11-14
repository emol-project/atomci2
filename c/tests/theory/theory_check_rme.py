# rme (technical note, Sec. 3.1): the special cases of Edmonds (7.1.5) -- the reduced
# elements of the two parts of a coupled product and of a scalar product -- against the
# 9j formula, and the 9j with a zero argument.
from itertools import product
from sympy import Rational as R, sqrt, simplify
from sympy.physics.wigner import wigner_6j, wigner_9j

def tri(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c).is_integer

def d(j):
    return 2 * j + 1

half = [R(n, 2) for n in range(0, 5)]
nbad = 0
ntest = 0
for j1, j1p, j2, j2p, k in product(half, repeat=5):
    if not k.is_integer:
        continue
    for J in half:
        for Jp in half:
            if not (tri(j1, j2, J) and tri(j1p, j2, Jp) and tri(J, Jp, k) and tri(j1, j1p, k)):
                continue
            # part 1: k2 = 0, <j2||1||j2> = sqrt(2 j2 + 1)
            g = sqrt(d(J) * d(Jp) * d(k)) * wigner_9j(j1, j1p, k, j2, j2, 0, J, Jp, k, prec=None) * sqrt(d(j2))
            f = (-1) ** int(j1 + j2 + Jp + k) * sqrt(d(J) * d(Jp)) * wigner_6j(j1, J, j2, Jp, j1p, k)
            ntest += 1
            if abs(float(g - f)) > 1e-12:
                nbad += 1
                print("part1", j1, j1p, j2, J, Jp, k, float(g), float(f))
for j1, j2, j2p, k in product(half, repeat=4):
    if not k.is_integer:
        continue
    for J in half:
        for Jp in half:
            if not (tri(j1, j2, J) and tri(j1, j2p, Jp) and tri(J, Jp, k) and tri(j2, j2p, k)):
                continue
            g = sqrt(d(J) * d(Jp) * d(k)) * wigner_9j(j1, j1, 0, j2, j2p, k, J, Jp, k, prec=None) * sqrt(d(j1))
            f = (-1) ** int(j1 + j2p + J + k) * sqrt(d(J) * d(Jp)) * wigner_6j(j2, J, j1, Jp, j2p, k)
            ntest += 1
            if abs(float(g - f)) > 1e-12:
                nbad += 1
                print("part2", j1, j2, j2p, J, Jp, k, float(g), float(f))
# scalar product: <J M| T.U |J M> = (-1)^k sqrt(2k+1) <(TU)^0> ; <JM|X^0|JM> = rme / sqrt(2J+1)
for j1, j1p, j2, j2p, k in product(half, repeat=5):
    if not k.is_integer:
        continue
    for J in half:
        if not (tri(j1, j2, J) and tri(j1p, j2p, J) and tri(j1, j1p, k) and tri(j2, j2p, k)):
            continue
        rme0 = sqrt(d(J) * d(J)) * wigner_9j(j1, j1p, k, j2, j2p, k, J, J, 0, prec=None)
        g = (-1) ** int(k) * sqrt(d(k)) * rme0 / sqrt(d(J))
        f = (-1) ** int(j1p + j2 + J) * wigner_6j(J, j2, j1, k, j1p, j2p)
        ntest += 1
        if abs(float(g - f)) > 1e-12:
            nbad += 1
            print("scalar", j1, j1p, j2, j2p, J, k, float(g), float(f))
print("tests", ntest, "bad", nbad)
