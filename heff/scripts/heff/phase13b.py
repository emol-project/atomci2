#!/usr/bin/env python3
"""Phase 13b: N dependence of the d-shell effective operators.

Step a (algebra only, no ATOMCI): Judd-Leavitt (1986) operators as Fock-space operators.
  - e1..e4 as pure two-body operators (d^2 matrices fitted to the JL d^3 vectors), t1..t4 as pure three-body
    operators (JL d^3 vectors, Phase 12a); lifted to d^4/d^5 and compared with JL Tables 7/8 (refs/JL1986).
  - The relative signs of the repeated-term seniority states of d^4/d^5 are fixed by this comparison.
  - N dependence of the e-part of lifted 2- and 3-body operators (d^2..d^10).

Usage: phase13b.py a   -> results/phase13/phase13b_a.txt/.json
"""
import itertools
import json
import os
import sys
from fractions import Fraction
from math import comb, sqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import cfp_l, dshell as S  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
EK = ['e0', 'e1', 'e2', 'e3', 'e4']
TK = ['t1', 't2', 't3', 't4']


def frac(x, dmax=10 ** 6):
    f = Fraction(x).limit_denominator(dmax)
    return str(f) if abs(float(f) - x) < 1e-9 * max(1, abs(x)) else f'{x:.10g}'


def fsurd(x):
    """x as p/q or p/q sqrt(r) (small r)."""
    if abs(x) < 1e-12:
        return '0'
    for r in (1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 21, 22, 33, 35, 42, 55, 66, 70, 77, 105, 165, 189):
        f = Fraction(x / sqrt(r)).limit_denominator(5000)
        if abs(float(f) * sqrt(r) - x) < 1e-9 * max(1, abs(x)):
            return str(f) if r == 1 else f'{f} s{r}'
    return f'{x:.10g}'


def e_twobody():
    """d^2 matrices (pure 2-body) of e1..e4 and the d^3 fit residuals; e0 = identity (0-body)."""
    T3 = S.jl_table(3)
    terms2 = S.terms(2)
    basis = []                          # pure 2-body operators: projector onto each d^2 term
    for i, (L, twoS, vs) in enumerate(terms2):
        v = np.zeros(len(S.components(2)))
        v[i] = 1.0
        basis.append(S.from_vec(2, v))
    lifted = np.array([S.to_vec(3, S.lift(B, 2, 3)) for B in basis]).T      # 9 x 5
    out, res = {}, {}
    for k in EK[1:]:
        c, *_ = np.linalg.lstsq(lifted, T3[k], rcond=None)
        res[k] = float(np.abs(lifted @ c - T3[k]).max())
        out[k] = (c, sum(ci * B for ci, B in zip(c, basis)))
    return out, res


def ops_fock():
    """{name: (pure n-body d^n matrix, n)} for e0..e4, t1..t4."""
    E2, res = e_twobody()
    T3 = S.jl_table(3)
    ops = {'e0': (np.ones((1, 1)), 0)}
    for k in EK[1:]:
        ops[k] = (E2[k][1], 2)
    for k in TK:
        ops[k] = (S.from_vec(3, T3[k]), 3)
    return ops, E2, res


def fix_signs(N, ops):
    """Choose the relative signs of the repeated-term seniority states of d^N so that the lift of every e_k and t_i
    equals the JL table operator (e and f criteria alone leave the d^5 2F, 2G 3|5 signs open: only t2..t4 connect
    them). Returns signs, the score and the number of patterns attaining it."""
    rep = [(L, twoS, len(vs)) for L, twoS, vs in S.terms(N) if len(vs) > 1]
    lifted = {k: S.lift(M, n, N) for k, (M, n) in ops.items()}
    best, scores = None, []
    keep = {k: v for k, v in S.SIGN.items() if k[0] != N}
    for pat in itertools.product(*[list(itertools.product((1, -1), repeat=k - 1)) for _, _, k in rep]):
        S.SIGN.clear()
        S.SIGN.update(keep)
        for (L, twoS, k), p in zip(rep, pat):
            S.SIGN[(N, L, twoS)] = (1,) + p
        tab = S.jl_table(N)
        score = 0.0
        for k, X in lifted.items():
            v = S.to_vec(N, X)
            score += float(np.abs(v - tab[k]).max())
        scores.append(score)
        if best is None or score < best[0]:
            best = (score, {k: v for k, v in S.SIGN.items() if k[0] == N})
    S.SIGN.clear()
    S.SIGN.update(keep)
    S.SIGN.update(best[1])
    assert best[1] == {k: v for k, v in S.SIGN_JL.items() if k[0] == N}, 'dshell.SIGN_JL is out of date'
    return best[0], best[1], sum(x < 1e-9 for x in scores), sorted(scores)[1]


def step_a():
    w = []
    a = w.append
    os.makedirs(OUT, exist_ok=True)
    js = {}
    a('# Phase 13b step a: JL operators as Fock-space operators; lift to d^4 / d^5 (no ATOMCI)')
    a('Term states: cfp_l stretched states (seniority ordered), full multiplets by L-, S- lowering. Components: '
      'diagonal per term, off-diagonal <v|X|v\'> (v < v\'); metric (2L+1)(2S+1), off-diagonal twice (JL Eq. (1)).')
    ops, E2, res = ops_fock()
    a('\n## 1. e1..e4 as pure two-body operators')
    a('d^2 term values (1S 3P 1D 3F 1G, i.e. the d^2 matrix; e0 = identity, a 0-body operator) fitted so that the '
      'lift to d^3 equals the JL d^3 vector; max residual of the fit:')
    a('| op | 1S | 3P | 1D | 3F | 1G | d^3 fit residual |')
    a('|---|---|---|---|---|---|---|')
    for k in EK[1:]:
        c = E2[k][0]
        a(f'| {k} | ' + ' | '.join(frac(x) for x in c) + f' | {res[k]:.1e} |')
    js['e_d2'] = {k: E2[k][0].tolist() for k in EK[1:]}
    js['e_fit_residual'] = res
    norms = S.jl_norms()
    a('Norms of the d^2 operators (metric (2L+1)(2S+1)) vs JL Table 9 (d2): ' + ', '.join(
        f'{k} {frac(S.ip(2, S.to_vec(2, E2[k][1]), S.to_vec(2, E2[k][1])))}/{norms[k][2]}' for k in EK[1:]))

    a('\n## 2. Lift to d^4, d^5 and comparison with JL Tables 7, 8')
    sg = {}
    for N in (4, 5):
        score, signs, nbest, second = fix_signs(N, ops)
        sg[N] = signs
        tab = S.jl_table(N)
        a(f'\n### d^{N}: relative signs of the seniority states (cfp_l sign = largest determinant coefficient positive), '
          f'chosen over all patterns: max |lift - table| summed over e0..t4 = {score:.1e}; patterns reaching < 1e-9: {nbest}; '
          f'next best score {second:.2f}')
        for (NN, L, twoS), s in signs.items():
            vs = dict(((x, y), z) for x, y, z in S.terms(N))[(L, twoS)]
            a(f'- {twoS + 1}{S.LCH[L]} v = {vs}: signs {s}')
        a(f'\nLift of each operator, decomposed in the JL d^{N} basis (coefficients on the unnormalized table operators; '
          f'f = largest |coefficient| on f1..f12):')
        a('| op | ' + ' | '.join(EK + TK) + ' | max |f| |')
        a('|---|' + '---|' * (len(EK + TK) + 1))
        jn = {}
        for k, (M, n) in ops.items():
            v = S.to_vec(N, S.lift(M, n, N))
            c = S.jl_coeffs(N, v, tab)
            fmax = max(abs(c[f]) for f in tab if f.startswith('f'))
            a(f'| {k} | ' + ' | '.join(frac(c[x]) for x in EK + TK) + f' | {fmax:.1e} |')
            jn[k] = {x: c[x] for x in tab}
        js[f'lift_d{N}'] = jn
    js['signs'] = {f'{N},{L},{twoS}': s for N in sg for (_, L, twoS), s in sg[N].items()}

    # 3. N dependence of the e-part: lifted e-basis in d^N (N = 2..10), projection
    a('\n## 3. N dependence (d^2 .. d^10)')
    a('e_k^(N) = lift of the pure 2-body e_k (e0 = identity). For a lifted operator X the e-part is its projection on '
      'span{e0..e4}^(N) (Gram solve in the component metric); for N <= 5 the JL t_i^(N) are orthogonal to it.')
    Ns = list(range(2, 11))
    eN = {N: [S.to_vec(N, S.lift(*ops[k], N)) for k in EK] for N in Ns}

    def epart(N, x):
        E = eN[N] if N > 2 else eN[N][:]
        G = np.array([[S.ip(N, p, q) for q in E] for p in E])
        c = np.linalg.lstsq(G, [S.ip(N, p, x) for p in E], rcond=None)[0]
        r = x - sum(ci * e for ci, e in zip(c, E))
        return c, np.sqrt(max(S.ip(N, r, r), 0.0) / max(S.ip(N, x, x), 1e-300))

    cases = {
        'Coulomb F2 (F^2 = 49, i.e. B = 1 per F2 = F^2/49)': (cfp_l.coulomb(2, 2, {2: 49.0}), 2),
        'Coulomb F4 (F^4 = 441)': (cfp_l.coulomb(2, 2, {4: 441.0}), 2),
    }
    for k in TK:
        cases[f'{k} (pure 3-body)'] = ops[k]
    jd = {}
    for name, (M, n) in cases.items():
        a(f'\n{name}: e-coefficients of the lift; "rest" = relative norm outside span{{e}}')
        a('| N | ' + ' | '.join(EK) + ' | rest |')
        a('|---|' + '---|' * 6)
        rows = {}
        for N in Ns:
            if N < n:
                continue
            x = S.to_vec(N, S.lift(M, n, N))
            c, rest = epart(N, x)
            rows[N] = c.tolist() + [float(rest)]
            a(f'| {N} | ' + ' | '.join(frac(ci) for ci in c) + f' | {rest:.2e} |')
        jd[name] = rows
    js['ndep'] = jd

    a('\n3-body operators whose d^3 matrix is an e_k (k = 0..4; e0 = the d^3 identity): e-coefficients of the lift, '
      'divided by the expected factor C(N,3) for e0 and (N - 2) for e1..e4 (other coefficients must vanish)')
    a('| d^3 matrix | ' + ' | '.join(f'N={N}' for N in range(3, 9)) + ' |')
    a('|---|' + '---|' * 6)
    T3 = S.jl_table(3)
    jx = {}
    for i, k in enumerate(EK):
        M = S.from_vec(3, T3[k])
        cells = []
        for N in range(3, 9):
            c, rest = epart(N, S.to_vec(N, S.lift(M, 3, N)))
            fac = comb(N, 3) if i == 0 else N - 2
            other = max(abs(c[j]) for j in range(5) if j != i)
            cells.append(f'{frac(c[i] / fac)} (other {other:.0e}, rest {rest:.0e})')
        jx[k] = cells
        a(f'| {k} | ' + ' | '.join(cells) + ' |')
    js['threebody_e'] = jx
    a('\nNorm of the lifted pure 3-body t_i in d^N divided by the JL d^3 norm (expected C(4, N - 3) from Table 9):')
    a('| op | ' + ' | '.join(f'N={N}' for N in range(3, 11)) + ' |')
    a('|---|' + '---|' * 8)
    for k in TK:
        cells = []
        for N in range(3, 11):
            x = S.to_vec(N, S.lift(*ops[k], N))
            cells.append(frac(S.ip(N, x, x) / norms[k][3]))
        a(f'| {k} | ' + ' | '.join(cells) + ' |')
    a('\nParticle-hole: diagonal elements (term by term, seniority states) of the lifted t_i in d^(10-N) vs d^N; '
      'odd quasispin rank implies diag(10-N) = -diag(N), hence 0 at N = 5:')
    jp = {}
    for n1, n2 in ((3, 7), (4, 6), (5, 5)):
        cells = []
        for k in TK:
            x1, x2 = S.to_vec(n1, S.lift(*ops[k], n1)), S.to_vec(n2, S.lift(*ops[k], n2))
            d2 = {S.comp_label(n2, key): y for (key, _), y in zip(S.components(n2), x2) if key[2] == key[3]}
            pairs = [(x, d2[S.comp_label(n1, key)]) for (key, _), x in zip(S.components(n1), x1) if key[2] == key[3]]
            dev = max(abs(x + y) for x, y in pairs)
            cells.append(f'{k} max|diag(N) + diag(10-N)| {dev:.1e} (max|diag| {max(abs(x) for x, _ in pairs):.3g})')
            jp[f'{n1},{n2},{k}'] = dev
        a(f'- N = {n1} vs {n2}: ' + '; '.join(cells))
    js['particle_hole'] = jp
    json.dump(js, open(os.path.join(OUT, 'phase13b_a.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, 'phase13b_a.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


ATOMCI_REPO = os.path.dirname(_C.ATOMCI_SCRIPTS)


def core_runs():
    """[(name, N, term, multiplicity)] of the core minimal space, d^0 .. d^9."""
    out = []
    for N in range(10):
        for L, twoS, vs in S.terms(N):
            t = f'{twoS + 1}{S.LCH[L]}'
            out.append((f'vcm_d{N}_{t}e', N, t, len(vs)))
    return out


def gen_core():
    """Decks inputs/vcm_d<N>_<term>e: [Ne] frozen, vq_orb orbitals (record 2), configurations 3s2 3p6 3d^N (P) and
    3s 3p6 3d^(N+1) (Q). Deck patches as scripts/phase10_recon/trial.py."""
    _C.setup_path()
    import re
    import counts as CN
    import gen_v
    import trial
    Lmap = CN.orbitals(trial.V)
    for name, nd, term, mult in core_runs():
        Lt = S.LCH.index(term[1])
        deck, csf = gen_v.run_atinputs('vq', trial.inp_text(term[1], int(term[0]), nd, 2), ATOMCI_REPO)
        header = csf.split(' ))\n', 1)[0] + ' ))\n'
        seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
        nob, mpob = [], []
        for let in trial.LET:
            orbs = [o for o in seq if o.endswith(let)]
            nob.append(len(orbs))
            mpob += [gen_v.vec(o) for o in orbs]
        assert nob == list(trial.NBF), nob
        P = frozenset({'3s': 2, '3p': 6, **({'3d': nd} if nd else {})}.items())
        cfgs = {P} | ({frozenset({'3s': 1, '3p': 6, '3d': nd + 1}.items())} if nd < 10 else set())
        tot, ncfg, per = CN.count(Lmap, cfgs, term)
        assert P in per and per[P] == mult, (name, per)
        order = [P] + [c for c in per if c != P]
        lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)) + '  ))'
                 for c in order]
        deck = gen_v.sub_all(deck, 'itnel', 18 + nd, 14 + nd)
        deck = gen_v.sub_first(deck, 'nrnfg', '1,0,1,1,0,1,1,1')
        wf = deck.index(' &wf\n')
        head, tail = deck[:wf], deck[wf:]
        tail = gen_v.sub_first(tail, 'ispin', int(term[0]))
        tail = gen_v.sub_first(tail, 'iorb', 2 * Lt + 1)
        tail = gen_v.sub_first(tail, 'ipar', 0)
        deck = head + tail
        deck = re.sub(r'^rule=1,$', 'rule=0,', deck, count=1, flags=re.M)
        deck = gen_v.sub_first(deck, 'npflg', 0)
        deck = deck.replace(' &wf\n', ' &wf\nlrps=4000000,\n', 1)
        deck = deck.replace(' &ci\n', ' &ex\nretry=0,\n /\n &ci\n', 1)
        deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
        deck = gen_v.sub_all(deck, 'nstate', mult)
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nob)))
        deck = gen_v.sub_block(deck, 'mpob', ','.join(map(str, mpob)))
        deck = gen_v.sub_all(deck, 'nind', len(nob))
        deck = gen_v.sub_block(deck, 'ind', ',\n'.join(f'2,{l},1,{n}' for l, n in enumerate(nob)))
        deck = re.sub(r"^title='.*',$", f"title=' {name} ',", deck, count=1, flags=re.M)
        out = os.path.join(ROOT, 'inputs', name)
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, 'be.deck'), 'w').write(deck)
        open(os.path.join(out, 'be.csf'), 'w').write(header + '\n'.join(lines) + '\n )\n')
        print(f'{name}: {tot} CSFs ({per[P]} P), {ncfg} configurations')


# ---------------------------------------------------------------- Fock-space pipeline (steps b, c)

def term_key(N, term):
    return S.LCH.index(term[1]), int(term[0]) - 1


def vec_from_blocks(N, blocks):
    """Component vector of d^N from {(L, 2S): m x m H_eff block in the seniority basis}."""
    return np.array([blocks[(L, twoS)][i, j] for (L, twoS, i, j), _ in S.components(N)])


def blocks_from_vec(N, v):
    out = {}
    for ((L, twoS, i, j), _), x in zip(S.components(N), v):
        m = len(dict(((a, b), c) for a, b, c in S.terms(N))[(L, twoS)])
        B = out.setdefault((L, twoS), np.zeros((m, m)))
        B[i, j] = B[j, i] = x
    return out


def extract(vecs):
    """Pure n-body parts V0..V3 (d^n matrices) from the component vectors of d^0..d^3."""
    return S.nbody_parts([S.from_vec(N, vecs[N]) for N in range(4)])


def predict(V, N):
    return S.to_vec(N, S.apply_parts(V, N))


def eig_resid(N, v_act, v_pred):
    """{term label: [eigenvalue differences actual - predicted]} (sorted eigenvalues per (L, S) block)."""
    A, B = blocks_from_vec(N, v_act), blocks_from_vec(N, v_pred)
    return {f'{twoS + 1}{S.LCH[L]}': (np.linalg.eigvalsh(A[(L, twoS)]) - np.linalg.eigvalsh(B[(L, twoS)])).tolist()
            for (L, twoS) in A}


def jl_split(N, x, tab=None):
    """Coefficients of x on the d^N JL operators (N = 3, 4, 5), and normalized coefficients <x, H_i>/sqrt(A_i)."""
    tab = tab or S.jl_table(N)
    c = S.jl_coeffs(N, x, tab)
    nrm = {k: S.ip(N, x, h) / np.sqrt(S.ip(N, h, h)) for k, h in tab.items()}
    return c, nrm


def v3_variant(V, kind):
    """V3 variants: 'full'; 'ut4' (t-part of the d^3 matrix of V3 projected on span{u, t4}, e-part kept); 'zero'."""
    if kind == 'full':
        return V
    if kind == 'zero':
        return V[:3]
    T3 = S.jl_table(3)
    x = S.to_vec(3, V[3])
    c = S.jl_coeffs(3, x, T3)
    u = 3 * T3['t1'] + 2 * T3['t2']
    tpart = sum(c[k] * T3[k] for k in TK)
    proj = sum(S.ip(3, tpart, b) / S.ip(3, b, b) * b for b in (u, T3['t4']))
    return V[:3] + [S.from_vec(3, x - tpart + proj)]


def eav(N, v):
    return float(np.sum(S.weights(N) * v * np.array([1.0 if i == j else 0.0 for (L, tS, i, j), _ in S.components(N)]))
                 / comb(10, N))


def lifted_t(N):
    """{t_i: component vector of the lifted pure 3-body JL t_i in d^N}."""
    ops, _, _ = ops_fock()
    return {k: S.to_vec(N, S.lift(*ops[k], N)) for k in TK}


def t_coords(N, x, tN):
    """Coefficients of x on the lifted t_i^(N) (orthogonal to the e^(N) and to each other for N <= 7)."""
    G = np.array([[S.ip(N, p, q) for q in tN.values()] for p in tN.values()])
    return dict(zip(tN, np.linalg.solve(G, [S.ip(N, p, x) for p in tN.values()])))


def load_run(name):
    from atomci_heff.reader import load_h
    d = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
    H = load_h(d)
    lab = [str(x) for x in d['labels']]
    return H, lab


def signed_maps(m):
    for perm in itertools.permutations(range(m)):
        for sg in itertools.product((1, -1), repeat=m):
            U = np.zeros((m, m))
            for i, (p, s) in enumerate(zip(perm, sg)):
                U[p, i] = s
            yield U


def step_b():
    from atomci_heff import heff, sdmodel as SM
    w = []
    a = w.append
    js = {}
    runs = core_runs()
    data = {}
    for name, N, term, mult in runs:
        H, lab = load_run(name)
        Pi = [i for i, l in enumerate(lab) if l.startswith('3s2')]
        Qi = [i for i, l in enumerate(lab) if not l.startswith('3s2')]
        L, twoS = term_key(N, term)
        m, nq, ops = SM.block(N, L, twoS)
        assert m == len(Pi) == mult and nq == len(Qi), (name, m, nq, len(Pi), len(Qi))
        data[name] = dict(N=N, term=term, L=L, twoS=twoS, H=H, P=Pi, Q=Qi, m=m, nq=nq, ops=ops)
    a('# Phase 13b step b: core minimal space (P = 3s2 3p6 3d^N, Q = 3s 3p6 3d^(N+1)), d^0 .. d^9')
    a(f'{len(runs)} ATOMCI runs (inputs/vcm_*, [Ne] frozen, vq_orb orbitals), 1-6 CSFs each.')

    # 1. radial parameters of the determinant model from basis-invariant traces; eigenvalue check
    lin = [k for k in SM.PARAMS if k != 'R2']
    rows, rhs = [], []
    num = den = 0.0
    for d in data.values():
        H, Pi, Qi, m, ops = d['H'], d['P'], d['Q'], d['m'], d['ops']
        rows.append([np.trace(ops[k][:m, :m]) for k in lin])
        rhs.append(np.trace(H[np.ix_(Pi, Pi)]))
        if Qi:
            rows.append([np.trace(ops[k][m:, m:]) for k in lin])
            rhs.append(np.trace(H[np.ix_(Qi, Qi)]))
            num += np.sum(H[np.ix_(Pi, Qi)] ** 2)
            den += np.sum(ops['R2'][:m, m:] ** 2)
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    p = dict(zip(lin, sol))
    p['R2'] = float(np.sqrt(num / den))
    tr_res = float(np.abs(np.array(rows) @ sol - rhs).max())
    a('\n## 1. Determinant model of the space (atomci_heff/sdmodel.py) and the ATOMCI blocks')
    a(f'Radial parameters fitted to the P and Q block traces of all runs (linear, max trace residual {tr_res:.1e} Eh); '
      f'R2 = R^2(3s3d;3d3d) from sum ||H_PQ||^2: |R2| = {p["R2"]:.8f} Eh.')
    a('F2(dd) = {:.8f}, F4(dd) = {:.8f}, G2(sd) = {:.8f}, F0(sd) = {:.8f}, F0(dd) = {:.8f} Eh (Racah B = {:.4f}, '
      'C = {:.4f} mEh)'.format(p['F2dd'], p['F4dd'], p['G2sd'], p['F0sd'], p['F0dd'],
                               1e3 * (p['F2dd'] / 49 - 5 * p['F4dd'] / 441), 1e3 * 35 * p['F4dd'] / 441))
    ev_dev = 0.0
    maps = {}
    map_dev = 0.0
    for name, d in data.items():
        Hm = sum(p[k] * d['ops'][k] for k in SM.PARAMS)
        H = d['H']
        ev_dev = max(ev_dev, float(np.abs(np.linalg.eigvalsh(H) - np.linalg.eigvalsh(Hm)).max()))
        Pi, Qi, m = d['P'], d['Q'], d['m']
        A1 = H[np.ix_(Pi, Pi)]
        B1 = Hm[:m, :m]
        A2 = H[np.ix_(Pi, Qi)] @ H[np.ix_(Qi, Pi)] if Qi else np.zeros((m, m))
        B2 = Hm[:m, m:] @ Hm[m:, :m] if Qi else np.zeros((m, m))
        A3 = H[np.ix_(Pi, Qi)] @ H[np.ix_(Qi, Qi)] @ H[np.ix_(Qi, Pi)] if Qi else np.zeros((m, m))
        B3 = Hm[:m, m:] @ Hm[m:, m:] @ Hm[m:, :m] if Qi else np.zeros((m, m))

        def mis(U):
            return float(max(np.abs(U.T @ A @ U - B).max() for A, B in ((A1, B1), (A2, B2), (A3, B3))))
        best = min(((mis(U), U) for U in signed_maps(m)), key=lambda x: x[0])
        cands = sum(1 for U in signed_maps(m) if mis(U) < 1e-8)
        maps[name] = best[1]
        map_dev = max(map_dev, best[0])
        d['ndeg'] = cands // 2              # the global sign is free
    a(f'All {len(data)} blocks: max |eigenvalue(ATOMCI) - eigenvalue(model)| = {ev_dev:.1e} Eh.')
    a(f'P-CSF -> seniority map: signed permutation U with U^T X U equal to the model for X = H_PP, H_PQ H_QP and '
      f'H_PQ H_QQ H_QP (Q-basis invariant) '
      f'(max deviation {map_dev:.1e} Eh); runs with more than one valid map (global sign aside): '
      f'{[n for n, d in data.items() if d["ndeg"] > 1]}')
    a('ATOMCI P-CSF order for the repeated terms (seniority of each CSF): ' + '; '.join(
        f'{d["N"]}:{d["term"]} ' + str([S.terms(d["N"])[[(x[0], x[1]) for x in S.terms(d["N"])].index((d["L"], d["twoS"]))][2][
            int(np.argmax(np.abs(maps[n][r])))] for r in range(d['m'])]) for n, d in data.items() if d['m'] > 1))
    js['params'] = p
    js['model_eig_dev'] = ev_dev
    js['map_dev'] = map_dev
    js['maps'] = {n: U.tolist() for n, U in maps.items()}

    # 2. H_eff per term: closure with fixed Delta and exact des Cloizeaux
    d3 = [d for d in data.values() if d['N'] == 3]
    gq = sum((2 * d['L'] + 1) * (d['twoS'] + 1) * np.trace(d['H'][np.ix_(d['Q'], d['Q'])]) for d in d3)
    nqw = sum((2 * d['L'] + 1) * (d['twoS'] + 1) * len(d['Q']) for d in d3)
    gp = sum((2 * d['L'] + 1) * (d['twoS'] + 1) * np.trace(d['H'][np.ix_(d['P'], d['P'])]) for d in d3) / comb(10, 3)
    Delta = float(gq / nqw - gp)
    a(f'\nFixed Delta (Phase 11 definition at d^3: g-weighted mean diagonal over the Q CSFs minus E_av(3d^3)) = '
      f'{Delta:.6f} Eh, used for every N.')
    blocks = {'closure': {}, 'exact': {}}
    flags = []
    for name, d in data.items():
        H, Pi, Qi, U, m = d['H'], d['P'], d['Q'], maps[name], d['m']
        A = H[np.ix_(Pi, Pi)] - (H[np.ix_(Pi, Qi)] @ H[np.ix_(Qi, Pi)] / Delta if Qi else 0)
        blocks['closure'].setdefault(d['N'], {})[(d['L'], d['twoS'])] = U.T @ A @ U
        e, C, wP, _ = heff.spectrum(H, Pi)
        st, amb = heff.select_states(wP, m)
        r = heff.exact_heff(H, Pi, st) if Qi else dict(hermitian=H[np.ix_(Pi, Pi)], condX=1.0)
        blocks['exact'].setdefault(d['N'], {})[(d['L'], d['twoS'])] = U.T @ r['hermitian'] @ U
        d['wP'] = float(wP[st].min())
        d['amb'] = float(amb)
        d['cond'] = float(r['condX'])
        if d['wP'] < 0.9 or amb > 0.3 or (m > 1 and r['condX'] > 1.2):
            flags.append(name)
    a(f'Exact targets: min P weight {min(d["wP"] for d in data.values()):.5f}, max ambiguity '
      f'{max(d["amb"] for d in data.values()):.1e}, max cond(X) {max(d["cond"] for d in data.values()):.5f}; '
      f'flagged: {flags or "none"}')
    vecs = {k: {N: vec_from_blocks(N, b[N]) for N in range(10)} for k, b in blocks.items()}

    # 3. extraction and prediction
    T3 = S.jl_table(3)
    u = 3 * T3['t1'] + 2 * T3['t2']
    res = {}
    for kind in ('closure', 'exact'):
        V = extract(vecs[kind])
        c3, n3 = jl_split(3, S.to_vec(3, V[3]))
        a(f'\n## 2{"ab"[kind == "exact"]}. {kind}: V3 (pure 3-body, d^3 matrix) in the JL d^3 basis')
        a('| | ' + ' | '.join(EK + TK) + ' |')
        a('|---|' + '---|' * 9)
        a('| coefficient (mEh, table operators) | ' + ' | '.join(f'{1e3 * c3[k]:+.6e}' for k in EK + TK) + ' |')
        if kind == 'closure':
            ref = {'t1': -3, 't2': -2, 't3': 0, 't4': -2}
            sc = p['R2'] ** 2 / Delta / 4410
            a('| Phase 11 form -(3t1+2t2+2t4)/4410 R2^2/Delta | | | | | | ' + ' | '.join(
                f'{1e3 * ref[k] * sc:+.6e}' for k in TK) + ' |')
            dev = max(abs(c3[k] - ref[k] * sc) for k in TK) / (2 * sc)
            a(f'Relative deviation of the t-part from the Phase 11 form: {dev:.1e}')
            js['closure_t_dev'] = dev
        rows_k = {}
        for var in ('full', 'ut4', 'zero'):
            if kind == 'closure' and var != 'full':
                continue
            Vv = v3_variant(V, var)
            per = {}
            for N in range(4, 10):
                vp = predict(Vv, N)
                per[N] = dict(eig=eig_resid(N, vecs[kind][N], vp), vec=(vecs[kind][N] - vp).tolist(),
                              eav=eav(N, vecs[kind][N]) - eav(N, vp))
            rows_k[var] = per
        res[kind] = dict(V=V, rows=rows_k, c3=c3)
        a(f'\nResidual actual - predicted (lift of V0..V3), sorted eigenvalues per (L, S) block, mEh; rows N = 4..9')
        for var, per in rows_k.items():
            a(f'\nV3 variant: {var}')
            for N, r in per.items():
                ev = [x for v in r['eig'].values() for x in v]
                a(f'- d^{N}: max |resid| {1e3 * max(map(abs, ev)):.3e}, rms (per state, (2L+1)(2S+1)-weighted) '
                  f'{1e3 * np.sqrt(S.ip(N, np.array(r["vec"]), np.array(r["vec"])) / comb(10, N)):.3e}; '
                  f'E_av resid {1e3 * r["eav"]:+.3e}; per term: ' + ', '.join(
                      f'{t} ' + '/'.join(f'{1e3 * x:+.4f}' for x in v) for t, v in r['eig'].items()))
    # d^4 / d^5 JL decomposition of the exact residual
    a('\n## 3. Exact CI: residual (actual - lift of V0..V3, V3 full) of d^4 and d^5 in the JL basis')
    a('Normalized coefficients <r, H_i>/sqrt(A_i) in mEh (A_i = JL Table 9 norm; sum of squares = weighted norm^2 of r).')
    js_jl = {}
    for N in (4, 5):
        r = np.array(res['exact']['rows']['full'][N]['vec'])
        c, nr = jl_split(N, r)
        js_jl[N] = nr
        tab = S.jl_table(N)
        a(f'\nd^{N}: |r| = {1e3 * np.sqrt(S.ip(N, r, r)):.4e} mEh;  delta-e: ' + ', '.join(
            f'{k} {1e3 * nr[k]:+.3e}' for k in EK) + ';  delta-t: ' + ', '.join(
            f'{k} {1e3 * nr[k]:+.3e}' for k in TK) + ';  f: ' + ', '.join(
            f'{k} {1e3 * nr[k]:+.3e}' for k in tab if k.startswith('f')))
        a(f'  parts: |e| {1e3 * np.sqrt(sum(nr[k] ** 2 for k in EK)):.3e}, |t| '
          f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in TK)):.3e}, |f| '
          f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in tab if k.startswith("f"))):.3e} mEh')
    # E_av absolute
    a('\n## 4. Absolute test: E_av(N) = sum_k C(N, k) h_k, h_k = tr V_k / C(10, k), from d^0..d^3')
    a('| N | ' + ' | '.join(f'{kind}: actual (Eh) | pred - actual (mEh)' for kind in ('closure', 'exact')) + ' |')
    a('|---|---|---|---|---|')
    for N in range(4, 10):
        cells = []
        for kind in ('closure', 'exact'):
            V = res[kind]['V']
            h = [np.trace(Vk) / comb(10, k) for k, Vk in enumerate(V)]
            pred = sum(comb(N, k) * h[k] for k in range(4))
            act = eav(N, vecs[kind][N])
            cells += [f'{act:.9f}', f'{1e3 * (pred - act):+.3e}']
        a(f'| {N} | ' + ' | '.join(cells) + ' |')
    # particle-hole: t-coordinates on the lifted t_i^(N)
    a('\n## 5. Three-body (t) part of the actual H_eff(N) on the lifted t_i^(N), N = 3..7 (lift predicts N-independent '
      'coefficients = those of V3); mEh')
    a('| H_eff | N | t1 | t2 | t3 | t4 | ratio to N = 3 (t-vector, metric-weighted) |')
    a('|---|---|---|---|---|---|---|')
    jph = {}
    for kind in ('closure', 'exact'):
        base = None
        for N in range(3, 8):
            tN = lifted_t(N)
            c = t_coords(N, vecs[kind][N], tN)
            x = np.array([c[k] for k in TK])
            if base is None:
                base = x
                G3 = np.array([[S.ip(3, T3[p_], T3[q_]) for q_ in TK] for p_ in TK])
            ratio = float(base @ G3 @ x / (base @ G3 @ base))
            jph.setdefault(kind, {})[N] = dict(t=c, ratio=ratio)
            a(f'| {kind} | {N} | ' + ' | '.join(f'{1e3 * c[k]:+.5e}' for k in TK) + f' | {ratio:+.6f} |')
    js['t_by_N'] = jph
    # 4-body: V4 from d^4, predict d^5..d^9
    a('\n## 6. Exact CI: V4 = H_eff(4) - lift(V0..V3) (pure 4-body), then predict d^5 .. d^9 (residual = 5-body and up)')
    V = res['exact']['V']
    V4 = V + [S.from_vec(4, vecs['exact'][4]) - S.apply_parts(V, 4)]
    js_v4 = {}
    for N in range(5, 10):
        vp = predict(V4, N)
        r = vecs['exact'][N] - vp
        er = eig_resid(N, vecs['exact'][N], vp)
        ev = [x for v in er.values() for x in v]
        r0 = np.array(res['exact']['rows']['full'][N]['vec'])
        js_v4[N] = er
        a(f'- d^{N}: max |resid| {1e3 * max(map(abs, ev)):.3e} mEh (V0..V3 only: '
          f'{1e3 * max(abs(x) for v in res["exact"]["rows"]["full"][N]["eig"].values() for x in v):.3e}); weighted norm '
          f'{1e3 * np.sqrt(S.ip(N, r, r)):.3e} (V0..V3: {1e3 * np.sqrt(S.ip(N, r0, r0)):.3e}); per term: ' + ', '.join(
              f'{t} ' + '/'.join(f'{1e3 * x:+.5f}' for x in v) for t, v in er.items()))
    if True:
        r = vecs['exact'][5] - predict(V4, 5)
        c, nr = jl_split(5, r)
        a('  d^5 residual after V4 in the JL basis (normalized, mEh): ' + ', '.join(
            f'{k} {1e3 * nr[k]:+.2e}' for k in nr if abs(nr[k]) > 1e-9))
        c4, n4 = jl_split(4, S.to_vec(4, V4[4]))
        a('  V4 (d^4 matrix) in the JL d^4 basis (normalized, mEh): ' + ', '.join(
            f'{k} {1e3 * n4[k]:+.2e}' for k in n4 if abs(n4[k]) > 1e-9))
    js['v4_pred'] = js_v4
    js['jl_resid'] = js_jl
    js['flags'] = flags
    js['Delta'] = Delta
    js['runs'] = {n: dict(wP=d['wP'], amb=d['amb'], cond=d['cond']) for n, d in data.items()}
    js['res'] = {k: {var: {N: dict(eig=r['eig'], eav=r['eav']) for N, r in per.items()} for var, per in v['rows'].items()}
                 for k, v in res.items()}
    js['c3'] = {k: v['c3'] for k, v in res.items()}
    json.dump(js, open(os.path.join(OUT, 'phase13b_b.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, 'phase13b_b.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


# ---------------------------------------------------------------- step c: s-free valence spaces

ECORE = -936.77483589       # [Ar] core energy of the vq valence runs (Phase 13a, phase9.core_energy)
SPACES = {'QZP': (7, 6, 6, 4, 3, 2), 'QZP-s': (0, 6, 6, 4, 3, 2), 'QZP-s-p': (0, 0, 6, 4, 3, 2),
          '6d': (0, 0, 6, 0, 0, 0), '3d2g': (0, 0, 3, 0, 2, 0), '1p3d1f1g': (0, 1, 3, 1, 1, 0), '3d1g': (0, 0, 3, 0, 1, 0)}
COST_D5 = {'3d1g': 0.14, '1p3d1f1g': 1.2, '3d2g': 2.1, '6d': 4.1}      # Phase 13a estimates, d^5 total wall (h)


def space_orbitals(v):
    names = []
    for l, n in enumerate(v):
        names += (['3d'] if l == 2 else []) + [f'{10 + k}{"spdfgh"[l]}' for k in range(1, n - (1 if l == 2 else 0) + 1)]
    return set(names)


def cfg_orbitals(label):
    import re
    return [m.group(1) for m in re.finditer(r'(\d+[spdfgh])(\d+)', label)]


def core_map(N, term):
    js = json.load(open(os.path.join(OUT, 'phase13b_b.json')))
    return np.array(js['maps'][f'vcm_d{N}_{term}e'])


def valence_heff(run, N, term, orbs, mult):
    """Exact des Cloizeaux H_eff (seniority basis) of the d^N targets in the configuration subset of results/<run>
    whose orbitals lie in orbs. Returns (H_eff block, info)."""
    from atomci_heff import heff
    from atomci_heff.reader import load_h
    z = np.load(os.path.join(ROOT, 'results', run + '.npz'), allow_pickle=True)
    lab = [str(x) for x in z['labels']]
    keep = [i for i, l in enumerate(lab) if set(cfg_orbitals(l)) <= orbs]
    H = load_h(z)[np.ix_(keep, keep)]
    lab = [lab[i] for i in keep]
    Pi = [i for i, l in enumerate(lab) if l == f'3d{N}']
    assert len(Pi) == mult, (run, len(Pi), mult)
    e, C, wP, _ = heff.spectrum(H, Pi)
    st, amb = heff.select_states(wP, mult)
    r = heff.exact_heff(H, Pi, st) if len(Pi) < len(lab) else dict(hermitian=H, condX=1.0)
    U = core_map(N, term)
    below = int(sum(1 for k in range(len(e)) if k not in st and e[k] < e[st].max()))
    return U.T @ r['hermitian'] @ U, dict(ncsf=len(lab), wP=float(wP[st].min()), amb=float(amb),
                                         cond=float(r['condX']), below=below, E=e[st].tolist())


def valence_vecs(orbs, N_list=(1, 2, 3), runs=None):
    """{N: component vector} from the existing d^1..d^3 QZP exports restricted to orbs (d^0 = ECORE)."""
    runs = runs or {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo'}
    vecs, info = {0: np.array([ECORE])}, {}
    for N in N_list:
        blocks = {}
        for L, twoS, vs in S.terms(N):
            t = f'{twoS + 1}{S.LCH[L]}'
            B, inf = valence_heff(runs[N].format(t), N, t, orbs, len(vs))
            blocks[(L, twoS)] = B
            info[f'd{N} {t}'] = inf
        vecs[N] = vec_from_blocks(N, blocks)
    return vecs, info


def tpart(N, x):
    """t-part (vector) and coordinates on (u, t4, w, t3) of a d^3 vector x."""
    T3 = S.jl_table(3)
    c = S.jl_coeffs(3, x, T3)
    t = sum(c[k] * T3[k] for k in TK)
    bas = {'u': 3 * T3['t1'] + 2 * T3['t2'], 't4': T3['t4'], 'w': T3['t1'] - T3['t2'], 't3': T3['t3']}
    return t, {k: S.ip(3, t, b) / np.sqrt(S.ip(3, b, b)) for k, b in bas.items()}, c


def step_c1():
    w = []
    a = w.append
    a('# Phase 13b step c1: d^3 three-body vector in reduced s-free spaces (configuration subsets of the QZP exports)')
    a('V3 = pure 3-body part from exact des Cloizeaux H_eff of d^0..d^3 (d^0 = [Ar] core energy); t-part = its '
      'component on t1..t4 (JL d^3); coordinates on the orthogonal (u, t4, w, t3), normalized (mEh, metric sqrt(ip)); '
      'angle and norm ratio vs full QZP; "e-part" = V3 component on e1..e4 (a 3-body operator that is (N-2) x 2-body).')
    js = {}
    ref = None
    a('| space | d^3 CSFs (max) | |t| (mEh) | ratio | angle (deg) | u | t4 | w | t3 | in span{u,t4} | |e1..e4 part| (mEh) | '
      'min P weight | max amb | max cond | d^5 cost (h, 13a) |')
    a('|---|' + '---|' * 14)
    for name, v in SPACES.items():
        orbs = space_orbitals(v)
        vecs, info = valence_vecs(orbs)
        V = extract(vecs)
        x = S.to_vec(3, V[3])
        t, co, c = tpart(3, x)
        nt = np.sqrt(S.ip(3, t, t))
        T3 = S.jl_table(3)
        ep = sum(c[k] * T3[k] for k in EK[1:])
        ne = np.sqrt(S.ip(3, ep, ep))
        if ref is None:
            ref = t
        ratio = nt / np.sqrt(S.ip(3, ref, ref))
        ang = np.degrees(np.arccos(np.clip(S.ip(3, t, ref) / nt / np.sqrt(S.ip(3, ref, ref)), -1, 1)))
        inside = (co['u'] ** 2 + co['t4'] ** 2) / nt ** 2
        js[name] = dict(v=v, t=t.tolist(), coords=co, ratio=ratio, angle=ang, inside=inside, e_norm=ne,
                        info=info, c=c)
        a(f'| {name} | {max(i["ncsf"] for k, i in info.items() if k.startswith("d3"))} | {1e3 * nt:.4f} | {ratio:.4f} | '
          f'{ang:.2f} | ' + ' | '.join(f'{1e3 * co[k]:+.4f}' for k in ('u', 't4', 'w', 't3')) +
          f' | {100 * inside:.1f}% | {1e3 * ne:.4f} | {min(i["wP"] for i in info.values()):.4f} | '
          f'{max(i["amb"] for i in info.values()):.2e} | {max(i["cond"] for i in info.values()):.4f} | '
          f'{COST_D5.get(name, "-")} |')
    ok = [n for n in COST_D5 if js[n]['angle'] <= 5 and COST_D5[n] <= 2.5]
    best = min(ok, key=lambda n: js[n]['angle']) if ok else None
    a(f'\nCandidates with angle <= 5 deg and d^5 cost <= 2.5 h: {ok or "none"}; chosen: {best}')
    js['chosen'] = best
    json.dump(js, open(os.path.join(OUT, 'phase13b_c1.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, 'phase13b_c1.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


VRUN = {4: 'vq_v1p_{}e_ivo_r{}', 5: 'vq_v0p_{}e_ivo_r{}'}


def gap_check(spec, N=5, terms=None):
    """Targets and the lowest non-target state of d^N runs in space spec (existing exports): per (L, S) block, the
    target P weights / ambiguity / cond and the gap = lowest non-target root - highest target root (eV); also the lowest
    non-target root relative to the d^N ground state."""
    from atomci_heff import heff
    from atomci_heff.reader import load_h
    out = {}
    for L, twoS, vs in S.terms(N):
        t = f'{twoS + 1}{S.LCH[L]}'
        if terms and t not in terms:
            continue
        run = VRUN[N].format(t, spec)
        z = np.load(os.path.join(ROOT, 'results', run + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        H = load_h(z)
        Pi = [i for i, l in enumerate(lab) if l == f'3d{N}']
        e, C, wP, _ = heff.spectrum(H, Pi)
        st, amb = heff.select_states(wP, len(vs))
        other = [k for k in range(len(e)) if k not in st]
        r = heff.exact_heff(H, Pi, st) if other else dict(condX=1.0)
        lowest = e[other[0]] if other else np.inf
        # dominant configuration of the lowest non-target root
        dom = ''
        if other:
            c2 = {}
            for i, l in enumerate(lab):
                c2[l] = c2.get(l, 0.0) + C[i, other[0]] ** 2
            dom = max(c2, key=c2.get)
        out[t] = dict(wP=wP[st].tolist(), amb=float(amb), cond=float(r['condX']), Et=e[st].tolist(),
                      E_other=float(lowest), gap_eV=float((lowest - e[st].max()) * 27.211386), dom=dom,
                      ncsf=len(lab), below=int(sum(1 for k in other if e[k] < e[st].max())))
    return out


def step_c2(spec, extra=()):
    """d^4/d^5 analysis in the reduced space spec (+ extra spaces as further size points)."""
    w = []
    a = w.append
    js = {}
    for sp in (spec,) + tuple(extra):
        orbs = space_orbitals(SPACES[{'3d1g': '3d1g', '1p3d1f1g': '1p3d1f1g', '3d2g': '3d2g', '6d': '6d'}[sp]])
        a(f'\n# Phase 13b step c2: space {sp} (orbitals {sorted(orbs)})')
        vecs, info = valence_vecs(orbs)
        for N in (4, 5):
            gc = gap_check(sp, N)
            blocks = {}
            for L, twoS, vs in S.terms(N):
                t = f'{twoS + 1}{S.LCH[L]}'
                B, inf = valence_heff(VRUN[N].format(t, sp), N, t, orbs, len(vs))
                blocks[(L, twoS)] = B
                info[f'd{N} {t}'] = inf
            vecs[N] = vec_from_blocks(N, blocks)
            flag = [t for t, g in gc.items() if min(g['wP']) < 0.9 or g['amb'] > 0.3 or g['cond'] > 1.2 or g['below']]
            a(f'\nd^{N} targets: min P weight {min(min(g["wP"]) for g in gc.values()):.4f}, max ambiguity '
              f'{max(g["amb"] for g in gc.values()):.2e}, max cond {max(g["cond"] for g in gc.values()):.4f}, '
              f'min gap (lowest non-target - highest target, same block) {min(g["gap_eV"] for g in gc.values()):.2f} eV '
              f'({min(gc, key=lambda t: gc[t]["gap_eV"])}, {gc[min(gc, key=lambda t: gc[t]["gap_eV"])]["dom"]}); '
              f'flagged blocks: {flag or "none"}')
            js[f'{sp} gap d{N}'] = gc
        V = extract(vecs)
        res = {}
        for var in ('full', 'ut4', 'zero'):
            Vv = v3_variant(V, var)
            for N in (4, 5):
                vp = predict(Vv, N)
                res[(var, N)] = dict(eig=eig_resid(N, vecs[N], vp), vec=vecs[N] - vp, eav=eav(N, vecs[N]) - eav(N, vp))
        a('\nResidual actual - predicted (lift of V0..V3) per term, mEh (sorted eigenvalues per (L, S) block)')
        for N in (4, 5):
            terms = list(res[('full', N)]['eig'])
            a(f'\n| d^{N} term | ' + ' | '.join(f'V3 {v}' for v in ('full', 'ut4', 'zero')) + ' |')
            a('|---|---|---|---|')
            for t in terms:
                a(f'| {t} | ' + ' | '.join('/'.join(f'{1e3 * x:+.4f}' for x in res[(v, N)]['eig'][t])
                                         for v in ('full', 'ut4', 'zero')) + ' |')
            a('| weighted rms per state | ' + ' | '.join(
                f'{1e3 * np.sqrt(S.ip(N, res[(v, N)]["vec"], res[(v, N)]["vec"]) / comb(10, N)):.4f}'
                for v in ('full', 'ut4', 'zero')) + ' |')
            a('| E_av residual | ' + ' | '.join(f'{1e3 * res[(v, N)]["eav"]:+.4f}' for v in ('full', 'ut4', 'zero')) + ' |')
            if N == 5:
                single = [t for t in terms if len(res[('full', N)]['eig'][t]) == 1]
                dev = max(abs(res[('full', 5)]['eig'][t][0] - res[('ut4', 5)]['eig'][t][0]) for t in single)
                a(f'Singly occurring d^5 terms {single}: max |full - ut4| = {1e3 * dev:.1e} mEh (t-part of V3 has no '
                  f'diagonal d^5 element, JL fact (ii)); these residuals test the e-part and the 4-, 5-body parts only.')
        for N in (4, 5):
            r = res[('full', N)]['vec']
            c, nr = jl_split(N, r)
            tab = S.jl_table(N)
            a(f'\nd^{N} residual (V3 full) in the JL basis, normalized <r, H_i>/sqrt(A_i), mEh: |r| '
              f'{1e3 * np.sqrt(S.ip(N, r, r)):.4f}; |e| {1e3 * np.sqrt(sum(nr[k] ** 2 for k in EK)):.4f}, |t| '
              f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in TK)):.4f}, |f| '
              f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in tab if k.startswith("f"))):.4f}')
            a('  ' + ', '.join(f'{k} {1e3 * nr[k]:+.4f}' for k in tab))
            js[f'{sp} jl d{N}'] = nr
        # 4-body: V4 from d^4, predict d^5
        V4 = V + [S.from_vec(4, vecs[4]) - S.apply_parts(V, 4)]
        vp = predict(V4, 5)
        er = eig_resid(5, vecs[5], vp)
        r5 = vecs[5] - vp
        a(f'\nd^5 predicted from d^0..d^4 (V4 = d^4 residual as a pure 4-body operator): max |resid| '
          f'{1e3 * max(abs(x) for v in er.values() for x in v):.4f} mEh, weighted rms '
          f'{1e3 * np.sqrt(S.ip(5, r5, r5) / comb(10, 5)):.4f} mEh (V0..V3: '
          f'{1e3 * np.sqrt(S.ip(5, res[("full", 5)]["vec"], res[("full", 5)]["vec"]) / comb(10, 5)):.4f}); per term: ' +
          ', '.join(f'{t} ' + '/'.join(f'{1e3 * x:+.4f}' for x in v) for t, v in er.items()))
        c3 = S.jl_coeffs(3, S.to_vec(3, V[3]), S.jl_table(3))
        js[sp] = dict(res={f'{v} {N}': dict(eig=r['eig'], eav=r['eav']) for (v, N), r in res.items()}, v4_d5=er,
                      info=info, c3=c3)
    tag = '_'.join((spec,) + tuple(extra))
    json.dump(js, open(os.path.join(OUT, f'phase13b_c2.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, f'phase13b_c2.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'gap':
        g = gap_check(sys.argv[2], int(sys.argv[3]), sys.argv[4].split(',') if len(sys.argv) > 4 else None)
        for t, x in g.items():
            print(t, json.dumps({k: (np.round(v, 5).tolist() if isinstance(v, list) else v) for k, v in x.items()}))
    elif cmd == 'c2':
        step_c2(sys.argv[2], tuple(sys.argv[3:]))
    else:
        {'a': step_a, 'gen_core': gen_core, 'b': step_b, 'c1': step_c1}[cmd]()
