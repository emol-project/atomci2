# need.sh -- let a test skip cleanly when an external program it needs is
# absent. Each function prints "SKIPPED <script>: <reason>" and returns 1; a
# script sources this file and writes `need_... || exit 0`, so a skipped test
# ends with status 0 and run_all_checks.sh reports it as skipped.
_need_script=$(basename "$0")
_need_skip() { echo "SKIPPED $_need_script: $*"; }

# need_dir VAR description: the variable is set and names a directory
need_dir() {
    local v=$1 d=${!1:-}
    if [ -z "$d" ]; then _need_skip "$v is not set ($2)"; return 1; fi
    if [ ! -d "$d" ]; then _need_skip "$v=$d is not a directory ($2)"; return 1; fi
}

# need_cmd description command...: every command is on PATH
need_cmd() {
    local desc=$1 c; shift
    for c in "$@"; do
        command -v "$c" > /dev/null 2>&1 || { _need_skip "$c not found on PATH ($desc)"; return 1; }
    done
}

# need_exe path description: an executable file
need_exe() {
    [ -x "$1" ] || { _need_skip "$1 not found ($2)"; return 1; }
}

# need_py module...: $PYTHON (default python3) imports every module
need_py() {
    local py=${PYTHON:-python3} m
    for m in "$@"; do
        "$py" -c "import $m" > /dev/null 2>&1 || { _need_skip "$py cannot import $m (set PYTHON to a Python with $m)"; return 1; }
    done
}
