"""Paths of the heff/ code: one place, read from environment variables.

  ATOMCI_BIN     the ATOMCI binary                 default <repo>/local/bin/atomci   (`make install`)
  ATOMCI_CFPTBL  the CFP table                     default <repo>/local/CFPTBL       (`make install`)
  ATOMCI_SCRIPTS directory holding atinputs/      default <repo>/scripts            (the atinpgen package)
  HEFF_DATA      the data repository (read only)  default <repo>/heff/data
  HEFF_WORK      the work tree (all writes)       default <repo>/heff/work

<repo> is the atomci2 checkout that contains this heff/ directory. Every script works inside HEFF_WORK with
the research layout: HEFF_WORK/results/..., HEFF_WORK/runs/<run>/, HEFF_WORK/inputs/<run>/. HEFF_DATA has the same
layout and is never written: `scripts/tables/init_work.py` copies the files a step needs from HEFF_DATA into
HEFF_WORK, and stops with an error if one of them is missing.

Code-side reference data shipped with heff/ (read only): basis/ (basis sets), refdata/levels/ (NIST ASD level lists),
refdata/JL1986/ (Judd & Leavitt 1986 tables, transcribed).
"""
import os
import sys

HEFF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../heff
REPO = os.path.dirname(HEFF)                                                 # the atomci2 checkout


def _env(name, default):
    v = os.environ.get(name)
    return os.path.abspath(os.path.expanduser(v)) if v else default


ATOMCI_BIN = _env('ATOMCI_BIN', os.path.join(REPO, 'local', 'bin', 'atomci'))
ATOMCI_CFPTBL = _env('ATOMCI_CFPTBL', os.path.join(REPO, 'local', 'CFPTBL'))
ATOMCI_SCRIPTS = _env('ATOMCI_SCRIPTS', os.path.join(REPO, 'scripts'))
DATA = _env('HEFF_DATA', os.path.join(HEFF, 'data'))
WORK = _env('HEFF_WORK', os.path.join(HEFF, 'work'))

BASIS = os.path.join(HEFF, 'basis')
REFDATA = os.path.join(HEFF, 'refdata')          # levels/, JL1986/
SCRIPTS = os.path.join(HEFF, 'scripts')
SCRIPT_DIRS = [os.path.join(SCRIPTS, d) for d in ('orbitals', 'decks', 'heff', 'fits', 'tables',
                                                   os.path.join('fits', 'trees'))]
TREES = os.path.join(WORK, 'results', 'phase15', '15i_trees')    # generated operator files (scripts/fits/trees)


def root():
    """The work tree (research layout); created on first use."""
    os.makedirs(WORK, exist_ok=True)
    return WORK


def setup_path():
    """heff/ and every heff/scripts/* directory on sys.path (the research scripts import each other by name)."""
    for d in [HEFF] + SCRIPT_DIRS:
        if d not in sys.path:
            sys.path.insert(1, d)


def script(name):
    """Full path of a packaged script by file name (they keep their research names; the names are unique)."""
    for d in SCRIPT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'heff: no script {name!r} under {SCRIPTS}')


def need(path, what=None):
    """path if it exists, else a clear error (no silent fallback)."""
    if not os.path.exists(path):
        hint = what or 'run scripts/tables/init_work.py (copies inputs from HEFF_DATA) or the step that writes it'
        raise FileNotFoundError(f'heff: missing input {path}\n  HEFF_WORK={WORK}\n  HEFF_DATA={DATA}\n  {hint}')
    return path


def atomci_paths():
    """ATOMCI binary and CFPTBL, checked."""
    need(ATOMCI_BIN, 'build and install ATOMCI (`make && make install` at the repository top) or set ATOMCI_BIN')
    need(ATOMCI_CFPTBL, 'build and install ATOMCI (`make && make install`) or set ATOMCI_CFPTBL')
    return ATOMCI_BIN, ATOMCI_CFPTBL


def summary():
    return dict(ATOMCI_BIN=ATOMCI_BIN, ATOMCI_CFPTBL=ATOMCI_CFPTBL, ATOMCI_SCRIPTS=ATOMCI_SCRIPTS,
                HEFF_DATA=DATA, HEFF_WORK=WORK)


setup_path()

if __name__ == '__main__':
    for k, v in summary().items():
        print(f'{k}={v}')
