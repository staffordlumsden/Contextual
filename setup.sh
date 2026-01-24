#!/bin/bash
# setup.sh — create venv, install deps, and prepare a runner for contextual2.py

set -eo pipefail

# --- Paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_DIR="$SCRIPT_DIR"
APP_ENTRY="contextual2.py"
APP_PATH="$APP_DIR/$APP_ENTRY"

# Allow overrides via environment, with safe defaults
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"

# Prefer user-specified interpreter, otherwise auto-detect a sensible python3
if [ -z "${PYTHON_CMD:-}" ]; then
  for candidate in /opt/homebrew/bin/python3.13 python3.13 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_CMD="$(command -v "$candidate")"
      break
    fi
  done
fi

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$APP_DIR/requirements.txt}"
RUNNER="$APP_DIR/run_chatbot.sh"

# --- Helpers ---
echo_green() { printf "\033[0;32m%s\033[0m\n" "$1"; }
echo_red()   { printf "\033[0;31m%s\033[0m\n" "$1"; }

echo_green "Starting setup for Contextual2…"

# 1) Check Python
if [ -z "${PYTHON_CMD:-}" ] || ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo_red "Error: Could not locate a python3 interpreter. Install Python 3 and/or set PYTHON_CMD."
  exit 1
fi

# 2) Check files
if [ ! -f "$APP_PATH" ]; then
  echo_red "Missing $APP_ENTRY at: $APP_PATH"
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo_red "Missing requirements file at: $REQUIREMENTS_FILE"
  exit 1
fi

# 3) Create (or repair) venv
if [ ! -d "$VENV_DIR" ]; then
  echo_green "Creating virtual environment at: $VENV_DIR…"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR."
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo_red "Warning: $VENV_DIR/bin/python not found or not executable. Recreating venv..."
  rm -rf "$VENV_DIR"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# 4) Install dependencies
echo_green "Upgrading pip and installing dependencies from: $REQUIREMENTS_FILE..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"

# 5) Create runner (keeps same VENV_DIR logic)
cat > "$RUNNER" << 'EOF'
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
EOF

chmod +x "$RUNNER"

echo_green "Setup complete! ✅"
echo
echo "Run the app with:"
echo "  \"$RUNNER\""
