"""recoil_x.py -- the two-body part of the nuclear-recoil correction to the
orbital g factor of an atom with an open 2p shell (report only, PySCF).

With a nucleus of mass M, the orbital moment is (Phillips; Hegstrom)
    mu_L = -mu_B [ L - (m/M) sum_{i,j} (r_i x p_j) ]
so g_L = 1 - (m/M) (1 + X / <L_z>),  X = <sum_{i != j} (r_i x p_j)_z>.
For a single determinant the direct part vanishes by parity and
    X = -sum_sigma [ Tr(x G p_y G) - Tr(y G p_x G) ],  G = C_occ C_occ^+
(exchange between the 2p and the core s orbitals). The determinant is built
from spherically averaged (fractional occupation) ROHF orbitals, the p
orbitals combined into p_{+1}, p_0, p_{-1}.

usage: recoil_x.py ATOM NP BASIS     e.g.  recoil_x.py F 5 unc-aug-cc-pwcvqz
NP electrons in 2p; the determinant has the Hund's-rule M_S and M_L = L.
"""
import sys

import numpy as np
from pyscf import gto, scf

atom, nps, basis = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spin = {1: 1, 2: 2, 3: 3, 4: 2, 5: 1}[nps]
mol = gto.M(atom=f"{atom} 0 0 0", basis=basis, spin=spin, verbose=0)
mf = scf.addons.frac_occ(scf.ROHF(mol))
mf.kernel()
C, e = mf.mo_coeff, mf.mo_energy
order = np.argsort(e)
core = [C[:, k].astype(complex) for k in order[:2]]   # 1s, 2s
P = C[:, order[2:5]]                                   # the three 2p
lab = mol.ao_labels(fmt=False)
comp = {c: [i for i, l in enumerate(lab) if l[2].endswith("p") and l[3] == c] for c in "xyz"}
S = mol.intor("int1e_ovlp")
pure = {}
for c in "xyz":
    Q = P[comp[c]].T @ P[comp[c]]
    v = P @ np.linalg.eigh(Q)[1][:, -1]
    pure[c] = v
istar = int(np.argmax(np.abs(pure["x"][comp["x"]])))
for c in "xyz":
    v = pure[c]
    if v[comp[c][istar]] < 0:
        v = -v
    pure[c] = v / np.sqrt(v @ S @ v)
px, py, pz = pure["x"], pure["y"], pure["z"]
orb = {1: -(px + 1j * py) / np.sqrt(2), 0: pz.astype(complex), -1: (px - 1j * py) / np.sqrt(2)}
# Hund's rule, M_L = L: alpha fills m = 1, 0, -1 first, then beta from m = 1
ma = [1, 0, -1][:min(nps, 3)]
mb = [1, 0, -1][:max(nps - 3, 0)]
G = {}
for sig, ms in (("a", ma), ("b", mb)):
    O = np.array(core + [orb[m] for m in ms]).T
    G[sig] = O @ O.conj().T
r = mol.intor("int1e_r")                 # <mu| r_k |nu>
pm = 1j * mol.intor("int1e_ipovlp")      # <mu| p_k |nu> = -i <mu|d_k nu> = +i <d_k mu|nu>
mol.set_common_orig((0, 0, 0))
Lm = -1j * mol.intor("int1e_cg_irxp")    # <mu| (r x p)_k |nu> = -i <mu| r x nabla |nu>
# Gs_{mu nu} = sum_a c_{mu a} c_{nu a}^*:  sum_a <a|A|a> = Tr(A Gs),
# sum_ab <a|A|b><b|B|a> = Tr(A Gs B Gs) (not Gs^T, which is Gs^*)
def ex(Gs, A, B):
    return np.trace(A @ Gs @ B @ Gs)
Gt = G["a"] + G["b"]
Lz = np.trace(Lm[2] @ Gt)
Ntot = np.trace(S @ Gt)
X = -sum(ex(Gs, r[0], pm[1]) - ex(Gs, r[1], pm[0]) for Gs in G.values())
X += np.trace(r[0] @ Gt) * np.trace(pm[1] @ Gt) - np.trace(r[1] @ Gt) * np.trace(pm[0] @ Gt)
# Beltran-Lopez et al. (Phys. Rev. 172, 44) Eqs. (3), (4): per core s orbital,
# a = <ns|x|p_x> = R/sqrt(3), b = <ns|d/dx|p_x>; X = -2 sum a b (direct form,
# one net same-spin pair), = -(2/3) sum (e_2p - e_ns) R^2 for a local potential
grad = -mol.intor("int1e_ipovlp")       # <mu| d_k |nu>
ep = e[order[2]]
for k, nm in zip(order[:2], ("1s", "2s")):
    s = C[:, k]
    a = s @ r[0] @ px
    b = s @ grad[0] @ px
    print(f"   {nm}: R = <ns|r|2p> = {abs(a) * np.sqrt(3):.5f}  e_2p - e_ns = {ep - e[k]:.5f}"
          f"  -2ab = {-2 * a * b:+.5f}  -(2/3) dE R^2 = {-2 * (ep - e[k]) * a * a:+.5f}")
print(f"{atom} 2p^{nps} {basis}: N = {Ntot.real:.6f}  <L_z> = {Lz.real:+.6f}{Lz.imag:+.1e}i"
      f"  X = {X.real:+.6f}{X.imag:+.1e}i  1 + X/<L_z> = {1 + X.real / Lz.real:.6f}")
