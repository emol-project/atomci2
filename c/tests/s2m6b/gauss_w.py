"""S2-M6b: Gaussian orbitals as an ATSP .w file, with exact orbit-orbit energies.

usage: gauss_w.py a b Z out.w

Orbitals (on ATSP's grid r_j = exp(-4 + j/16)/Z):
  1s = N_s exp(-a r^2),
  2s = the normalised part of exp(-b r^2) orthogonal to 1s,
  2p = N_p z exp(-b r^2) (radial part).
Printed exact values (alpha = 1/137.036):
  E_OO(1s 2p 3P): the orbit-orbit energy of |1s up 2p_z up|. Only exchange
    contributes; with A = p grad s, only its transverse part survives the
    kernel 8 pi/q^2 (1 - q^ q^), and for Gaussians
    A~_i(q) = -2a N_s N_p G(q) (delta_iz/(2c) - q_i q_z/(4c^2)),
    G = (pi/c)^(3/2) exp(-q^2/(4c)), c = a + b, so
    E = (alpha^2/2) 8 pi (a N_s N_p/c)^2 (pi/c)^3 (8 pi/3) sqrt(pi c/2) / (2 pi)^3.
  E_OO(1s 2s 3S) = 0: the s-s transition current is radial, hence
    longitudinal, and has no transverse part.
"""
import math
import struct
import sys

a, b, Z, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
alpha = 1 / 137.036
Ns = (2 * a / math.pi) ** 0.75
Np = (2 * b / math.pi) ** 0.75 * 2 * math.sqrt(b)
c = a + b
E = alpha ** 2 / 2 * 8 * math.pi * (a * Ns * Np / c) ** 2 * (math.pi / c) ** 3 * (8 * math.pi / 3) \
    * math.sqrt(math.pi * c / 2) / (2 * math.pi) ** 3
print(f"exact E_OO(1s2p 3P) = {E:.15e}")
print("exact E_OO(1s2s 3S) = 0")
# 2s: exp(-b r^2) - S 1s, normalised (S = <1s|exp(-b r^2)>)
Nb = (2 * b / math.pi) ** 0.75
S = Ns * Nb * (math.pi / c) ** 1.5  # overlap of the normalised Gaussians
N2 = 1 / math.sqrt(1 - S * S)
fs = math.sqrt(4 * math.pi)
funcs = {
    "1s": (lambda x: x * fs * Ns * math.exp(-a * x * x), fs * Ns),
    "2s": (lambda x: x * fs * N2 * (Nb * math.exp(-b * x * x) - S * Ns * math.exp(-a * x * x)),
           fs * N2 * (Nb - S * Ns)),
    "2p": (lambda x: x * x * Np * math.sqrt(4 * math.pi / 3) * math.exp(-b * x * x),
           Np * math.sqrt(4 * math.pi / 3)),
}
m = 220
r = [math.exp(-4 + j / 16) / Z for j in range(m)]
recs = []
for lab, (P, az) in funcs.items():
    vals = [P(x) / math.sqrt(x) for x in r]  # ATSP stores P/sqrt(r)
    mr = max(j for j in range(m) if abs(vals[j]) > 1e-14) + 1
    body = struct.pack("<6s6s3si4d", b"G     ", b"      ", lab.rjust(3).encode(), mr, Z, 0.0, 0.0, az)
    body += struct.pack(f"<{mr}d", *vals[:mr])
    recs.append(struct.pack("<i", len(body)) + body + struct.pack("<i", len(body)))
with open(out, "wb") as f:
    f.write(b"".join(recs))
