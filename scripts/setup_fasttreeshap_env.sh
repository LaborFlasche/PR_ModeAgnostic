#!/usr/bin/env bash
# Provisions the dedicated venv that FastTreeShapBackend shells out to
# (fasttreeshap needs numpy<2, incompatible with this project's numpy>=2 stack).
# Defaults to ~/.cache rather than inside the repo: OneDrive/iCloud sync can
# silently mangle a venv's bin/python symlink into a plain text file.
set -euo pipefail

VENV_DIR="${FASTTREESHAP_VENV_DIR:-$HOME/.cache/pr-modeagnostic/.venv-fasttreeshap}"
PYTHON_BIN="${FASTTREESHAP_PYTHON_BIN:-python3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: $PYTHON_BIN not found. fasttreeshap needs numpy<2, which needs Python <=3.11." >&2
    echo "Install one (e.g. 'uv python install 3.10') and re-run, or set FASTTREESHAP_PYTHON_BIN." >&2
    exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/fasttreeshap_requirements.txt"

echo "fasttreeshap venv ready at $VENV_DIR"
echo "Set FASTTREESHAP_VENV_PYTHON=$VENV_DIR/bin/python if it differs from the backend's default."
