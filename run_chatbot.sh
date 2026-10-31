#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
APP_ENTRY="$SCRIPT_DIR/contextual2.py"

if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Virtual environment not found at $VENV_DIR (or python missing). Run ./setup.sh first."
  exit 1
fi

if [ ! -f "$APP_ENTRY" ]; then
  echo "App entrypoint not found at $APP_ENTRY"
  exit 1
fi

# Warn early if Python 3.15 or newer is in play.
# pip falls over building chonkie-core on 3.15 (PyO3 still caps at 3.14 as of
# 2026-10), which silently leaves the venv without any dependencies.
PY_VER="$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
case "$PY_VER" in
  3.15|3.16|3.17|3.18|3.19|3.2[0-9])
    echo "Warning: $VENV_DIR uses Python $PY_VER; dependencies are known to fail to install." >&2
    echo "         Rebuild the venv with Python 3.14 (see ./setup.sh)." >&2
    ;;
esac

if ! "$VENV_DIR/bin/python" -c 'import ollama' >/dev/null 2>&1; then
  echo "Dependencies missing from $VENV_DIR (cannot import ollama)." >&2
  echo "Run ./setup.sh to repair the environment." >&2
  exit 1
fi

exec "$VENV_DIR/bin/python" "$APP_ENTRY" "$@"
