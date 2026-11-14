"""The three examples of heff/examples/ (orbitals -> ATOMCI -> H_eff -> projection, from the repository alone).

The runs are marked slow (minutes each, they need the ATOMCI of this repository: `make && make install`): skipped by
`make heff-test`, run by `make heff-examples` or `python3 -m pytest -m slow tests/test_examples.py`. The fast tests
only check that each example is complete and does not depend on HEFF_DATA."""
import json
import os
import subprocess
import sys

import pytest

HEFF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HEFF)
from atomci_heff import config as C  # noqa: E402

EXAMPLES = ('cr3_n3h1', 'cr3_mrci', 'co3_mrci')


@pytest.mark.parametrize('name', EXAMPLES)
def test_example_is_complete(name):
    d = os.path.join(HEFF, 'examples', name)
    for f in ('example.json', 'expected.json', 'README.md'):
        assert os.path.isfile(os.path.join(d, f)), f
    exp = json.load(open(os.path.join(d, 'expected.json')))
    assert {'T1', 'T2', 'T3', 'T4', 'B', 'C', 'alpha', 'beta'} <= set(exp['values'])
    if json.load(open(os.path.join(d, 'example.json')))['N'] == 6:
        assert {f'f{i}' for i in range(1, 13)} <= set(exp['values'])
    n = sum(len(fs) for _, _, fs in os.walk(os.path.join(d, 'inputs')))
    assert n > 0


@pytest.mark.slow
@pytest.mark.parametrize('name', EXAMPLES)
def test_example_run(name, tmp_path):
    """Full run in a new work tree, compared with expected.json (run_example.py --check)."""
    C.atomci_paths()
    env = dict(os.environ, HEFF_WORK=str(tmp_path / name), HEFF_DATA=str(tmp_path / 'no-data'))
    r = subprocess.run([sys.executable, C.script('run_example.py'), name, '--check'], env=env, capture_output=True,
                       text=True, timeout=1800)
    print(r.stdout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert 'check: OK' in r.stdout
    assert not os.path.exists(tmp_path / 'no-data')
