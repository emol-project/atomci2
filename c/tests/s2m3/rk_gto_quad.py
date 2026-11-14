"""S2-M3: GTO R^k(ab;cd) of MOs by another route than ac_radial. Inner
integrals in closed form through the regularized incomplete gamma
functions, outer integral by adaptive Gauss-Kronrod (scipy.integrate.quad).

  rk_gto_quad.py fort.18 fort.21 dump nworst nrand

Takes an acrad dump ("k a b c d C F" per line), checks the nworst entries
with the largest |C - F| and nrand random others, and fails when
|C - quad| exceeds 1e-11."""
import random, struct, sys, warnings
import numpy as np
from scipy import integrate, special
# quad reports roundoff when it reaches double precision before epsrel 1e-14
warnings.simplefilter('ignore', integrate.IntegrationWarning)

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
o = 33 * 4
def ints(k):
    global o
    v = struct.unpack('<%di' % k, hdr[o:o + 4 * k]); o += 4 * k; return v
def dbls(k):
    global o
    v = struct.unpack('<%dd' % k, hdr[o:o + 8 * k]); o += 8 * k; return v
lmax, nwt, _, _, _ = ints(5)
issh = ints(lmax); ints(2 * nwt); dbls(2)
npqn = ints(nwt); dl = dbls(nwt)
ifst = ints(lmax); inxt = ints(nwt); nterm = ints(nwt)
ce = np.array(dbls(6000)).reshape(150, 40); oe = np.array(dbls(6000)).reshape(150, 40)

order = []
for l in range(lmax):
    i = ifst[l]
    for _ in range(issh[l]):
        order.append(i - 1); i = inxt[i - 1]

def gint(q, a):
    return special.gamma((q + 1) / 2) / (2 * a ** ((q + 1) / 2))

bfs = []  # per l: list of (n, [(coef, alpha)]) normalized
pos = 0
for l in range(lmax):
    fl = []
    for _ in range(issh[l]):
        i = order[pos]; pos += 1
        n = npqn[i]
        prim = [(ce[i][j] * (dl[i] * oe[i][j]) ** ((2 * n + 1) / 4), dl[i] * oe[i][j]) for j in range(nterm[i])]
        s = sum(c1 * c2 * gint(2 * n, a1 + a2) for c1, a1 in prim for c2, a2 in prim)
        fl.append((n, [(c / np.sqrt(s), a) for c, a in prim]))
    bfs.append(fl)

recs = records(sys.argv[2])
h = next(recs)
nsym = struct.unpack('<i', h[8:12])[0]
nbf = struct.unpack('<20i', h[12:92]); nob = struct.unpack('<20i', h[92:172])
mos = []  # per MO: [(C, m power of r in R, alpha)]
for l in range(nsym):
    need, got = nbf[l] * nob[l], []
    while len(got) < need:
        r = next(recs)
        nt, = struct.unpack('<i', r[:4])
        got += struct.unpack('<%dd' % nt, r[4:4 + 8 * nt])
    for i in range(nob[l]):
        t = []
        for mu, (n, prim) in enumerate(bfs[l]):
            c = got[mu + i * nbf[l]]
            t += [(c * cp, n - 1, a) for cp, a in prim]
        mos.append(t)

def rho(a, b):  # R_a R_b r^2
    return [(c1 * c2, m1 + m2 + 2, x1 + x2) for c1, m1, x1 in mos[a] for c2, m2, x2 in mos[b]]

def rk(k, a, b, c, d):
    r1 = rho(a, b); r2 = rho(c, d)
    C1 = np.array([t[0] for t in r1]); M1 = np.array([t[1] for t in r1]); X1 = np.array([t[2] for t in r1])
    C2 = np.array([t[0] for t in r2]); M2 = np.array([t[1] for t in r2]); X2 = np.array([t[2] for t in r2])
    p, q = M1 + k, M1 - k - 1
    ga, gb = gint(p, X1), gint(q, X1)
    def f(r):
        z = X1 * r * r
        y = np.sum(C1 * (r ** (-(k + 1)) * ga * special.gammainc((p + 1) / 2, z)
                         + r ** k * gb * special.gammaincc((q + 1) / 2, z)))
        return np.sum(C2 * r ** M2 * np.exp(-X2 * r * r)) * y
    pts = [1e-3, 1e-2, 0.05, 0.2, 0.5, 1, 2, 4, 8]
    v = 0.0
    for lo, hi in zip([0] + pts, pts + [40.0]):
        v += integrate.quad(f, lo, hi, epsabs=1e-15, epsrel=1e-14, limit=400)[0]
    return v

rows = [l.split() for l in open(sys.argv[3])]
rows = [(tuple(map(int, r[:5])), float(r[5]), float(r[6])) for r in rows]
nworst, nrand = int(sys.argv[4]), int(sys.argv[5])
rows.sort(key=lambda r: -abs(r[1] - r[2]))
random.seed(1)
pick = rows[:nworst] + random.sample(rows[nworst:], nrand)
ec = ef = 0
for key, cv, fv in pick:
    ex = rk(*key)
    ec, ef = max(ec, abs(cv - ex)), max(ef, abs(fv - ex))
print('%d entries (%d worst |C-F|, %d random): max |C - quad| %.2e, max |F - quad| %.2e'
      % (len(pick), nworst, nrand, ec, ef))
sys.exit(0 if ec < 1e-11 else 1)
