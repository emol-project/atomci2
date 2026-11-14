#!/usr/bin/env python3
"""Run one example of heff/examples/<name>/ from scratch with ATOMCI: orbitals -> decks -> ATOMCI (stop after H) -> H_eff
-> projection. Reads only the example's inputs/ (never HEFF_DATA) and the ATOMCI of ATOMCI_BIN / ATOMCI_CFPTBL.

  run_example.py NAME [--check]          packaged code; work tree HEFF_WORK (default heff/work/examples/NAME)
  run_example.py NAME --research ROOT    the same stages with the research code in a copy of the research
                                         repository at ROOT (for the comparison in the example's README)

--check compares the result with NAME/expected.json (tolerances in the file) and exits 1 on a difference.
The stages call the research functions that the production runs used: the orbital recipe (phase15s_nd orb /
phase18 coorb), the HF P-block runs (d3), the configuration rule (phase15s.ras_cfgs_x on phase15t._virt, or a named
space), N15.prep, run_atomci.py per term block, N15.analyze (phase15k.analyze3 for d3, phase15j.analyze2 with
phase17.own_hf for d6), and for d6 the projection phase18_co.three6 of the dumped exact H_eff. The ATOMCI runs of one
example go 3 at a time (as the production queues), each on one thread."""
import json
import os
import shutil
import subprocess
import sys
import time

HEFF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(HEFF, 'examples')
NPROC = int(os.environ.get('HEFF_EXAMPLE_PROCS', '3'))
D3_TERMS = ('2P', '4P', '2D', '2F', '4F', '2G', '2H')


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def spec(name):
    return json.load(open(os.path.join(EXAMPLES, name, 'example.json')))


class Engine:
    """Where the code and the work tree are: the packaged heff/ or a research-repository copy."""

    def __init__(self, name):
        self.name, self.research = name, arg('--research')
        if self.research:
            self.work = os.path.abspath(self.research)
            self.paths = [self.work, os.path.join(self.work, 'scripts'), os.path.join(self.work, 'scripts', 'phase10_recon')]
            self.env = {}
        else:
            self.work = os.path.abspath(os.environ.get('HEFF_WORK') or os.path.join(HEFF, 'work', 'examples', name))
            self.paths = [HEFF]
            self.env = dict(HEFF_WORK=self.work, HEFF_DATA=os.path.join(EXAMPLES, name, 'inputs'))

    def script(self, n):
        if not self.research:
            sys.path.insert(0, HEFF)
            from atomci_heff import config as C
            return C.script(n)
        for d in ('scripts', os.path.join('results', 'phase19'), os.path.join('results', 'phase21')):
            p = os.path.join(self.work, d, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(n)

    def environ(self, ex, **kw):
        return dict(os.environ, ION=ex['ion'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    HEFF_SOLVER=ex.get('solver', 'dense'), HEFF_SPARSE_ABOVE=str(ex.get('sparse_above', 500)),
                    HEFF_DUMP=os.path.join(self.work, 'results', 'heff_dump'), **self.env, **kw)

    def py(self, ex, args, cwd=None):
        t = time.time()
        subprocess.run([sys.executable] + args, cwd=cwd or self.work, env=self.environ(ex), check=True,
                       stdout=subprocess.DEVNULL)
        print(f'ok {time.time() - t:6.1f} s  {os.path.basename(args[0])} {" ".join(args[1:])[:70]}', flush=True)

    def stage(self, ex, st):
        args = [os.path.abspath(__file__), self.name, '--stage', st] + (['--research', self.research] if self.research else [])
        self.py(ex, args)

    def atomci(self, ex, names, tag):
        """run_atomci.py per block, NPROC at a time, one thread each."""
        t, todo, run = time.time(), list(names), []
        while todo or run:
            while todo and len(run) < NPROC:
                n = todo.pop(0)
                run.append((n, subprocess.Popen([sys.executable, self.script('run_atomci.py'), n, '--vectors-from',
                                                 f'vq_orb_{tag}'], cwd=self.work, env=self.environ(ex),
                                                stdout=subprocess.DEVNULL)))
            time.sleep(0.2)
            for x in [x for x in run if x[1].poll() is not None]:
                if x[1].returncode:
                    raise SystemExit(f'heff: ATOMCI run {x[0]} failed (rc {x[1].returncode})')
                run.remove(x)
        print(f'ok {time.time() - t:6.1f} s  run_atomci.py x {len(names)} ({names[0]} ...)', flush=True)


# ---------------------------------------------------------------- stages inside one process (ION set)

def modules(eng, ex):
    sys.path[:0] = eng.paths
    if not eng.research:
        from atomci_heff import config as C
        C.setup_path()
    if ex['N'] == 3:
        import phase15u as U
        return U, U.T, U.S, U.N15, U.Q
    import functools  # noqa: F401
    import phase17 as P
    import phase15j as J
    import phase18_write_deck as W
    J.write_deck = W.write_deck                        # the Z-generic writer (phase18.prep)
    return P, P.U.T, P.U.S, P.N15, P.Q


def inject(ex, T, S, N15, Q):
    """The example's space as one more named space (configuration rule of the production spaces)."""
    r = ex.get('rule')
    if not r:
        return
    virt = T._virt(**r['virt'])
    cf = S.ras_cfgs_x(virt, maxh=r['maxh'], maxv=r.get('maxv', 2))
    prev = N15.spaces

    def spaces():
        sp = prev()
        sp[ex['space']] = ('allcore', cf)
        return sp
    N15.spaces = spaces
    Q.spaces = spaces


def run_stage(eng, ex, st):
    M, T, S, N15, Q = modules(eng, ex)
    inject(ex, T, S, N15, Q)
    sp, tag = ex['space'], ex['tag']
    if st == 'prep':
        N15.prep([sp])
    elif st == 'names':
        import phase13d as D
        print(json.dumps([N15.rname(sp, t) for L, twoS, m, t in D.term_list(ex['N'])]))
    elif st == 'analyze':
        if ex['N'] == 6:
            M.own_hf([sp])
        N15.analyze(tag, [sp])
    elif st == 'project':
        import numpy as np
        if ex['N'] == 3:
            r = json.load(open(os.path.join(eng.work, 'results', 'phase15', f'phase15k_{tag}.json')))[sp]
            p, dims = r['exact']['proj'], r['dims']
        else:
            import phase18_co as CO
            z = np.load(os.path.join(eng.work, 'results', 'heff_dump', f'{tag}_{sp}_{ex.get("solver", "dense")}.npz'))
            p = CO.three6(CO.x_of(z, 'exact'), ex['zeta_ion'])['proj']
            r = json.load(open(os.path.join(eng.work, 'results', 'phase15', f'phase15j_{tag}.json')))[sp]
            dims = r['dims']
        keys = ['T1', 'T2', 'T3', 'T4', 'B', 'C', 'alpha', 'beta'] + [f'f{i}' for i in range(1, 13) if f'f{i}' in p]
        out = dict(example=eng.name, ion=ex['ion'], space=sp, dims=dims, values={k: float(p[k]) for k in keys})
        json.dump(out, open(os.path.join(eng.work, 'results', f'example_{eng.name}.json'), 'w'), indent=1)


# ---------------------------------------------------------------- driver

def init(eng):
    """Copy the example's inputs/ (research layout) into the work tree; the work tree must be new or a former run of it."""
    src = os.path.join(EXAMPLES, eng.name, 'inputs')
    for p in range(9, 22):              # the research scripts expect their result directories to exist
        os.makedirs(os.path.join(eng.work, 'results', f'phase{p}'), exist_ok=True)
    for sub in ('15j_terms', '15k_terms'):
        os.makedirs(os.path.join(eng.work, 'results', 'phase15', sub), exist_ok=True)
    for d, _, fs in os.walk(src):
        for f in fs:
            s = os.path.join(d, f)
            t = os.path.join(eng.work, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copyfile(s, t)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if not a:
        raise SystemExit(__doc__)
    name = a[0]
    eng, ex = Engine(name), spec(name)
    if '--stage' in sys.argv:
        run_stage(eng, ex, arg('--stage'))
        return
    t0 = time.time()
    if not eng.research:
        sys.path.insert(0, HEFF)
        from atomci_heff import config as C
        C.atomci_paths()
        if os.path.isdir(os.path.join(eng.work, 'runs')) or os.path.isdir(os.path.join(eng.work, 'results')):
            raise SystemExit(f'heff: work tree {eng.work} is not empty (remove it, or set HEFF_WORK to a new directory)')
        os.makedirs(eng.work, exist_ok=True)
        init(eng)
    if ex['N'] == 3:
        eng.py(ex, [eng.script('phase15s_nd.py'), 'orb'])
        eng.py(ex, [eng.script('phase15s_nd.py'), 'vcm'])
        eng.atomci(ex, [f'vcm_d3_{t}e_crnd3' for t in D3_TERMS], ex['orbitals'])
    else:
        eng.py(ex, [eng.script('phase18.py'), 'coorb'])
        if eng.research:
            eng.py(ex, ['step1.py'], cwd=os.path.join(eng.work, 'results', 'phase15', '15i_trees'))
            eng.py(ex, ['step2.py'], cwd=os.path.join(eng.work, 'results', 'phase15', '15i_trees'))
            eng.py(ex, ['step4.py'], cwd=os.path.join(eng.work, 'results', 'phase15', '15i_trees'))
        else:
            eng.py(ex, [eng.script('make_operators.py')])
    eng.stage(ex, 'prep')
    r = subprocess.run([sys.executable, os.path.abspath(__file__), name, '--stage', 'names']
                       + (['--research', eng.research] if eng.research else []),
                       cwd=eng.work, env=eng.environ(ex), check=True, capture_output=True, text=True)
    eng.atomci(ex, json.loads(r.stdout.strip().split('\n')[-1]), ex['orbitals'])
    eng.stage(ex, 'analyze')
    eng.stage(ex, 'project')
    res = json.load(open(os.path.join(eng.work, 'results', f'example_{name}.json')))
    res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(eng.work, 'results', f'example_{name}.json'), 'w'), indent=1)
    print(f'{name}: {res["wall_s"]:.0f} s; ' + ', '.join(f'{k} {v:.6f}' for k, v in res['values'].items()))
    print('result:', os.path.join(eng.work, 'results', f'example_{name}.json'))
    if '--check' in sys.argv:
        sys.exit(1 if check(name, res, eng, ex) else 0)


def check(name, res, eng, ex):
    import numpy as np
    exp = json.load(open(os.path.join(EXAMPLES, name, 'expected.json')))
    bad = []
    z = np.load(os.path.join(eng.work, 'results', 'heff_dump', f'{ex["tag"]}_{ex["space"]}_{ex.get("solver", "dense")}.npz'))
    ze = np.load(os.path.join(EXAMPLES, name, 'expected_heff.npz'))
    if sorted(z.files) != sorted(ze.files):
        bad.append('H_eff dump: different blocks')
    else:
        dh = max(float(np.abs(z[k] - ze[k]).max()) for k in z.files)
        print(f'max |dH_eff| against expected_heff.npz: {dh:.3e} Eh ({len(z.files)} blocks)')
        if dh > exp.get('tol_heff_Eh', 1e-9):
            bad.append(f'H_eff: max |d| {dh:.3e} Eh > {exp.get("tol_heff_Eh", 1e-9)}')
    for k, v in exp['values'].items():
        tol = exp['tol_mEh'] if k[0] == 'T' else exp['tol_cm']
        if abs(res['values'][k] - v) > tol:
            bad.append(f'{k}: {res["values"][k]:.9f} vs expected {v:.9f} (tol {tol})')
    if res['dims'] != exp['dims']:
        bad.append(f'dims {res["dims"]} vs expected {exp["dims"]}')
    print('check:', 'OK' if not bad else 'DIFFERENT\n  ' + '\n  '.join(bad))
    return bad


if __name__ == '__main__':
    main()
