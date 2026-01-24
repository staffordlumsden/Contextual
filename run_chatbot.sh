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

exec "$VENV_DIR/bin/python" "$APP_ENTRY" "$@"
