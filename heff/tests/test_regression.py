"""Regression tests of the packaged paper path.

  test_fixture_*        bundled Cr3+ fixture (tests/fixtures/cr3_fam, 33 kB), no external data
  test_level1_tables    rebuilds paper_tables.md, phase19_tables.md and the other Level 1 tables from the H_eff dumps
                        found via HEFF_DATA and compares them byte for byte with checks/reference/ (about 7 min);
                        skipped, with the reason, when HEFF_DATA does not hold the Level 1 inputs
Each test runs the scripts in a fresh HEFF_WORK under pytest's tmp_path (a subprocess, so that the environment
variables are read by config.py as in normal use)."""
import json
import os
import subprocess
import sys

import pytest

HEFF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HEFF)
from atomci_heff import config as C  # noqa: E402

PROBE = r'''
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], 'scripts', 'tables'))
import init_work
init_work.init('fixture')
from atomci_heff import config as C
import numpy as np
import phase13d as D
import phase15k as K
import phase17c_rule_tables as RT
z = np.load(os.path.join(C.WORK, 'results', 'phase17', 'heff_dump', 'q13_crnd_fam.npz'))
tl = {t: (L, twoS) for L, twoS, m, t in D.term_list(3)}
x = K.B.vec_from_blocks(3, {tl[k.split('|')[1]]: z[k] for k in z.files if k.startswith('X-|')})
p = K.decompose3(x)
print(json.dumps(dict(dump={k: float(p[k]) for k in ('T1', 'T2', 'T3', 'T4', 'B', 'C', 'alpha', 'beta')},
                      rule=list(RT.exact(RT.RUNS['crnd_fam'])))))
'''


def run_py(code_or_args, work, data=None, timeout=1800):
    env = dict(os.environ, HEFF_WORK=str(work), OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
    if data:
        env['HEFF_DATA'] = str(data)
    args = ['-c', code_or_args, HEFF] if isinstance(code_or_args, str) else code_or_args
    r = subprocess.run([sys.executable] + args, env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-4000:]
    return r.stdout


@pytest.fixture(scope='module')
def fixture_out(tmp_path_factory):
    return json.loads(run_py(PROBE, tmp_path_factory.mktemp('work')).strip().split('\n')[-1])


def test_fixture_dump_reproduces_stored_decomposition(fixture_out):
    """exact H_eff of Cr3+ fam (dump) -> A..beta, T1..T4 = the stored phase15k result (written by another code path)."""
    ref = json.load(open(os.path.join(HEFF, 'tests', 'fixtures', 'cr3_fam', 'results', 'phase15',
                                      'phase15k_17_crnd_fam_d5f2g2h1.json')))['fam_d5f2g2h1']['exact']['proj']
    d = fixture_out['dump']
    for k in ('T1', 'T2', 'T3', 'T4'):              # mEh
        assert abs(d[k] - ref[k]) < 1e-7, (k, d[k], ref[k])
    for k in ('B', 'C', 'alpha', 'beta'):           # cm-1
        assert abs(d[k] - ref[k]) < 1e-5, (k, d[k], ref[k])
    assert (f"{d['T1']:.3f}", f"{d['T2']:.3f}") == ('12.227', '1.683')


def test_fixture_rule_table_exact_values(fixture_out):
    """phase17c_rule_tables reads the Cr3+ fam exact values from the JSON; they equal the values once written there."""
    assert fixture_out['rule'] == ['Cr³⁺ fam', 12.227, 1.683, 992.9, 3754.5, 78.8]


def test_missing_input_is_an_error(tmp_path):
    """A table script stops with a clear error on a missing input (no silent 'pending')."""
    env = dict(os.environ, HEFF_WORK=str(tmp_path))
    r = subprocess.run([sys.executable, C.script('phase18_tables.py')], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'heff: missing input' in r.stderr, r.stderr[-2000:]


def test_init_refuses_missing_data(tmp_path):
    env = dict(os.environ, HEFF_WORK=str(tmp_path / 'w'), HEFF_DATA=str(tmp_path / 'empty'))
    r = subprocess.run([sys.executable, C.script('init_work.py'), 'level1'], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'level1 inputs missing' in r.stderr


def _level1_data():
    man = os.path.join(HEFF, 'checks', 'manifest_level1.txt')
    fs = [x.strip() for x in open(man) if x.strip() and not x.startswith('#')]
    return [f for f in fs if not os.path.isfile(os.path.join(C.DATA, f))]


@pytest.mark.skipif(bool(_level1_data()), reason=f'HEFF_DATA={C.DATA} does not hold the Level 1 inputs '
                                                  '(checks/manifest_level1.txt)')
def test_level1_tables(tmp_path):
    out = run_py([C.script('rebuild_level1.py'), '--check'], tmp_path, data=C.DATA, timeout=3600)
    assert 'DIFFERENT' not in out, out
    assert out.count('identical ') == 6, out
