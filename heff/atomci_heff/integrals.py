"""Decode ATOMCI integral addresses and compute the integrals independently.

The addressing below is a Python port of ATOMCI's src/setiad.f and is therefore a derivative work of ATOMCI:
(c) Copyright Fukashi Sasaki, Masahiro Sekiya, Takeshi Noro, Kazumasa Ohtsuki, You Osanai, Nobuaki Miura, and
Hiroaki Honda, 1990-2003, 2026. Distributed under the ATOMCI terms (non-commercial research and educational use;
disclaimer and full terms in LEGENDS.txt at the top of this repository, Sections 1-3).

Addressing: a Python port of src/setiad.f (SETIAD, INT1AD, INT2AD). Orbitals
are numbered 1..NWT in coupling-sequence order. NWLF(i) = l+1 and NWSQ(i) is
the index of orbital i among the orbitals with the same l.

    one-electron  (a|h|b)                                  address INT1AD(a, b)
    two-electron  R^k(ab; cd) = int int P_a P_b (1) r<^k/r>^(k+1) P_c P_d (2)
                  KXX = 2k+1 in the tape's convention, address INT2AD(KXX, a, b, c, d)

INT2AD gives the same address under a<->b, c<->d and (ab)<->(cd): the
Coulomb-integral symmetries of real radial functions.
"""
import itertools

import numpy as np
from scipy.integrate import cumulative_simpson, simpson


class Addressing:
    def __init__(self, nwlf, nwsq=None):
        self.nwlf = list(nwlf)            # l+1 per orbital, 1-based values
        nwt = len(self.nwlf)
        self.lmax = max(self.nwlf)
        L = self.lmax
        self.issh = [0] * (L + 2)
        sq = []
        for i in range(nwt):
            k = self.nwlf[i]
            self.issh[k] += 1
            sq.append(self.issh[k])
        self.nwsq = list(nwsq) if nwsq else sq
        issh = self.issh
        itad = [0] * (L + 2)
        ioead = [0] * (L + 2)
        for i in range(2, L + 1):
            ioead[i] = ioead[i - 1] + issh[i - 1] * (issh[i - 1] + 1) // 2
            itad[i] = (L - i + 2) * (i - 1) + itad[i - 1]
        kmax = 2 * L - 1
        igtad = ioead[L] + issh[L] * (issh[L] + 1) // 2
        self.n1e = igtad
        iscd, igcd = {}, {}
        for km in range(1, kmax + 1):
            kpt = (km + 1) % 2 + 1
            isad = 0
            for ida in range(kpt, L + 1, 2):
                if ida > km:
                    break
                for ita in range(ida, kmax + 1, 2):
                    if ita < km:
                        continue
                    ln1 = (ida + ita) // 2
                    if ln1 > L:
                        break
                    ln2 = (ita - ida) // 2 + 1
                    kt = (ita - km) // 2
                    iad = itad[ln2] + kt * (L - ln2 + 1) + ida
                    iscd[iad] = isad
                    idima = issh[ln1] * issh[ln2]
                    if ida == 1:
                        idima = issh[ln1] * (issh[ln1] + 1) // 2
                    isad += idima
                    igcd[iad] = igtad
                    for idb in range(kpt, ida + 1, 2):
                        itbg = ita if idb == ida else kmax
                        stop = False
                        for itb in range(idb, itbg + 1, 2):
                            if itb < km:
                                continue
                            ln3 = (idb + itb) // 2
                            if ln3 > L:
                                stop = True
                                break
                            ln4 = (itb - idb) // 2 + 1
                            idimb = issh[ln3] * issh[ln4]
                            if idb == 1:
                                idimb = issh[ln3] * (issh[ln3] + 1) // 2
                            idim = idima * idimb
                            if ida == idb and ita == itb:
                                idim = idima * (idima + 1) // 2
                            igtad += idim
                        if stop:
                            continue
        self.itad, self.ioead, self.iscd, self.igcd = itad, ioead, iscd, igcd
        self.ntot = igtad

    def int1ad(self, m1, m2):
        L = self.nwlf[m1 - 1]
        i1, i2 = self.nwsq[m1 - 1], self.nwsq[m2 - 1]
        ia, ib = min(i1, i2), max(i1, i2)
        return self.ioead[L] + ib * (ib - 1) // 2 + ia

    def int2ad(self, kxx, m1, m2, m3, m4):
        k = kxx // 2 + 1
        i1, i2, i3, i4 = m1, m2, m3, m4
        L = self.lmax
        issh, nwsq, nwlf = self.issh, self.nwsq, self.nwlf
        while True:
            l1, l2, l3, l4 = (nwlf[i - 1] for i in (i1, i2, i3, i4))
            ida, idb = abs(l1 - l2) + 1, abs(l3 - l4) + 1
            ita, itb = l1 + l2 - 1, l3 + l4 - 1
            if ida > idb or (ida == idb and ita >= itb):
                break
            i1, i2, i3, i4 = i3, i4, i1, i2
        s1, s2, s3, s4 = (nwsq[i - 1] for i in (i1, i2, i3, i4))
        idima = issh[l2] * issh[l1]
        if l1 < l2:
            idia, l12l = issh[l2] * (s1 - 1) + s2, l1
        elif l1 == l2:
            kt = max(s1, s2)
            l12l = l1
            idia = kt * (kt - 1) // 2 + min(s1, s2)
            idima = issh[l1] * (issh[l1] + 1) // 2
        else:
            idia, l12l = issh[l1] * (s2 - 1) + s1, l2
        idimb = issh[l3] * issh[l4]
        if l3 < l4:
            idib, l34l = issh[l4] * (s3 - 1) + s4, l3
        elif l3 == l4:
            kt = max(s3, s4)
            l34l = l3
            idib = kt * (kt - 1) // 2 + min(s3, s4)
            idimb = issh[l3] * (issh[l3] + 1) // 2
        else:
            idib, l34l = issh[l3] * (s4 - 1) + s3, l4
        if ida == idb and ita == itb:
            kt = max(idia, idib)
            idta = kt * (kt - 1) // 2 + min(idia, idib)
        else:
            idta = (idia - 1) * idimb + idib
        kt = (ita - k) // 2
        iada = self.itad[l12l] + kt * (L - l12l + 1) + ida
        kt = (itb - k) // 2
        iadb = self.itad[l34l] + kt * (L - l34l + 1) + idb
        return self.igcd[iada] + self.iscd[iadb] * idima + idta

    def label_table(self):
        """address -> ('1e', a, b) or ('2e', k, a, b, c, d), 1-based coupling-sequence orbital indices."""
        nwt = len(self.nwlf)
        tab = {}
        for a in range(1, nwt + 1):
            for b in range(1, a + 1):
                if self.nwlf[a - 1] == self.nwlf[b - 1]:
                    tab.setdefault(self.int1ad(a, b), ('1e', a, b))
        orbs = range(1, nwt + 1)
        for a, b in itertools.combinations_with_replacement(orbs, 2):
            la, lb = self.nwlf[a - 1] - 1, self.nwlf[b - 1] - 1
            for c, d in itertools.combinations_with_replacement(orbs, 2):
                lc, ld = self.nwlf[c - 1] - 1, self.nwlf[d - 1] - 1
                for k in range(0, 2 * (self.lmax - 1) + 1):
                    if (la + lb + k) % 2 or (lc + ld + k) % 2:
                        continue
                    if not (abs(la - lb) <= k <= la + lb and abs(lc - ld) <= k <= lc + ld):
                        continue
                    tab.setdefault(self.int2ad(2 * k + 1, a, b, c, d), ('2e', k, a, b, c, d))
        return tab


# ---------------------------------------------------------------- radial integrals


def radial_grid(rmin=1e-7, rmax=150.0, n=40001):
    x = np.linspace(np.log(rmin), np.log(rmax), n)
    return x, np.exp(x)


def radial_functions(fns, vecs, r):
    """P_i(r) = r R_i(r) for orbitals given as {l: C (nbf x nob)} over gto.read_bas functions."""
    out = {}
    for l, C in vecs.items():
        F = [(e, c) for (ll, e, c) in fns if ll == l]
        chi = []
        for e, c in F:
            prim = np.array([cc * (2 * (2 * a) ** (l + 1.5) / _gamma(l + 1.5)) ** 0.5 *
                             r ** l * np.exp(-a * r * r) for a, cc in zip(e, c)]).sum(axis=0)
            chi.append(prim)
        chi = np.array(chi)
        # normalize contracted functions numerically (int chi^2 r^2 dr = 1)
        x = np.log(r)
        nrm = np.sqrt(simpson(chi ** 2 * r ** 3, x=x, axis=1))
        chi = chi / nrm[:, None]
        out[l] = (C.T @ chi) * r  # P = r R
    return out


def _gamma(z):
    from math import gamma
    return gamma(z)


def slater(Pa, Pb, Pc, Pd, k, x, r):
    """R^k(ab; cd) = int int Pa Pb (1) r<^k / r>^(k+1) Pc Pd (2) dr1 dr2 on the log grid x."""
    rho2 = Pc * Pd
    inner = cumulative_simpson(rho2 * r ** k * r, x=x, initial=0.0)
    outer_tot = simpson(rho2 * r ** (-k - 1) * r, x=x)
    outer = outer_tot - cumulative_simpson(rho2 * r ** (-k - 1) * r, x=x, initial=0.0)
    Y = inner * r ** (-k - 1) + outer * r ** k
    return simpson(Pa * Pb * Y * r, x=x)
