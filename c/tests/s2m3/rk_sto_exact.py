"""S2-M3: STO R^k(ab;cd) of MOs independent of quadrature. The closed form
for integer powers, each primitive term in exact rational arithmetic
(exponents taken as exact binary fractions), combined with the
coefficients in 40-digit decimal.

  rk_sto_exact.py fort.18 fort.21 k a b c d        one value
  rk_sto_exact.py fort.18 fort.21 dump nworst nrand

The second form takes an acrad dump ("k a b c d C F" per line), checks the
nworst entries with the largest |C - F| and nrand random others, and
fails when |C - exact| exceeds 1e-11. MO indices are acrad's."""
import struct, sys
from decimal import Decimal, getcontext
from fractions import Fraction
from math import factorial
getcontext().prec = 40

def records(path):
    f = open(path, 'rb')
    while True:
        b = f.read(4)
        if not b:
            return
        n, = struct.unpack('<i', b)
        yield f.read(n)
        f.read(4)

hdr = next(records(sys.argv[1]))
off = 33 * 4
lmax, nwt = struct.unpack('<2i', hdr[off:off + 8]); off += 20
issh = struct.unpack('<%di' % lmax, hdr[off:off + 4 * lmax]); off += 4 * lmax + 8 * nwt + 16
npqn = struct.unpack('<%di' % nwt, hdr[off:off + 4 * nwt]); off += 4 * nwt
dl = struct.unpack('<%dd' % nwt, hdr[off:off + 8 * nwt])

recs = records(sys.argv[2])
h = next(recs)
nsym = struct.unpack('<i', h[8:12])[0]
nbf = struct.unpack('<20i', h[12:92]); nob = struct.unpack('<20i', h[92:172])
coef = []
for l in range(nsym):
    need, got = nbf[l] * nob[l], []
    while len(got) < need:
        r = next(recs)
        nt, = struct.unpack('<i', r[:4])
        got += struct.unpack('<%dd' % nt, r[4:4 + 8 * nt])
    coef.append(got)

first = [sum(issh[:l]) for l in range(lmax)]
mos = []  # per MO: [(C as Decimal, power m of r in R, exponent as Fraction)]
for l in range(lmax):
    for i in range(nob[l]):
        t = []
        for mu in range(issh[l]):
            j = first[l] + mu
            n, a = npqn[j], Decimal(dl[j])
            norm = ((2 * a) ** (2 * n + 1) / factorial(2 * n)).sqrt()
            t.append((norm * Decimal(coef[l][mu + i * nbf[l]]), n - 1, Fraction(dl[j])))
        mos.append(t)

def rho(a, b):  # R_a R_b r^2 = sum C r^m e^{-x r}
    return [(c1 * c2, m1 + m2 + 2, x1 + x2) for c1, m1, x1 in mos[a] for c2, m2, x2 in mos[b]]

def mom(n, x):  # int_0^inf r^n e^{-x r} dr
    return Fraction(factorial(n)) / x ** (n + 1)

def term(k, m1, x, m2, y):
    """int r^m2 e^{-y r} [r^-(k+1) int_0^r s^(m1+k) e^{-xs} ds
                          + r^k int_r^inf s^(m1-k-1) e^{-xs} ds] dr"""
    p, q = m1 + k, m1 - k - 1
    s = mom(m2 - k - 1, y) * Fraction(factorial(p)) / x ** (p + 1)
    for j in range(p + 1):
        s -= Fraction(factorial(p)) / x ** (p + 1) * x ** j / factorial(j) * mom(m2 - k - 1 + j, x + y)
    for j in range(q + 1):
        s += Fraction(factorial(q)) / x ** (q + 1) * x ** j / factorial(j) * mom(m2 + k + j, x + y)
    return s

def rk(k, a, b, c, d):
    val = Decimal(0)
    for C1, m1, x in rho(a, b):
        for C2, m2, y in rho(c, d):
            t = term(k, m1, x, m2, y)
            val += C1 * C2 * Decimal(t.numerator) / Decimal(t.denominator)
    return val

if len(sys.argv) == 8:
    k, a, b, c, d = map(int, sys.argv[3:8])
    print('R^%d(%d %d; %d %d) = %s' % (k, a, b, c, d, rk(k, a, b, c, d)))
    sys.exit()

# acrad dump: the nworst entries with the largest |C - F| and nrand random ones
import random
rows = [l.split() for l in open(sys.argv[3])]
rows = [(tuple(map(int, r[:5])), float(r[5]), float(r[6])) for r in rows]
nworst, nrand = int(sys.argv[4]), int(sys.argv[5])
rows.sort(key=lambda r: -abs(r[1] - r[2]))
random.seed(1)
pick = rows[:nworst] + random.sample(rows[nworst:], nrand)
ec = ef = 0
for key, cv, fv in pick:
    ex = rk(*key)
    ec, ef = max(ec, abs(Decimal(cv) - ex)), max(ef, abs(Decimal(fv) - ex))
print('%d entries (%d worst |C-F|, %d random): max |C - exact| %.2e, max |F - exact| %.2e'
      % (len(pick), nworst, nrand, ec, ef))
sys.exit(0 if ec < Decimal('1e-11') else 1)
