#!/usr/bin/env bash
set -euo pipefail

PYVER="3.12.11"
PROJECT_VENV=".venv"

echo "==> Checking for apt-based system..."
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script targets Debian/Ubuntu (apt). Exiting."
  exit 1
fi

echo "==> Installing build prerequisites..."
sudo apt-get update -y
sudo apt-get install -y build-essential curl git \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev libncursesw5-dev libgdbm-dev tk-dev uuid-dev \
  liblzma-dev xz-utils libxml2-dev libxmlsec1-dev

# Decide which shell rc to edit (bash as default)
SHELL_NAME="$(basename "${SHELL:-bash}")"
if [ "$SHELL_NAME" = "zsh" ]; then
  RC_FILE="$HOME/.zshrc"
else
  RC_FILE="$HOME/.bashrc"
fi

# Install pyenv if not present
if [ ! -d "$HOME/.pyenv" ]; then
  echo "==> Installing pyenv..."
  curl -fsSL https://pyenv.run | bash
else
  echo "==> pyenv already installed."
fi

# Ensure pyenv is available in current session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found on PATH after install. Check your environment." >&2
  exit 1
fi

# Add pyenv init to shell rc (idempotent)
if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' "$RC_FILE" 2>/dev/null; then
  {
    echo ''
    echo '# >>> pyenv init >>>'
    echo 'export PYENV_ROOT="$HOME/.pyenv"'
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
    echo 'eval "$(pyenv init -)"'
    echo '# <<< pyenv init <<<'
  } >> "$RC_FILE"
  echo "==> Appended pyenv init to $RC_FILE"
else
  echo "==> pyenv init already present in $RC_FILE"
fi

# Initialize pyenv in this script
eval "$(pyenv init -)"

echo "==> Updating pyenv definitions..."
pyenv update || true

echo "==> Installing Python $PYVER (this may take a few minutes)..."
# -s means "skip if already installed"
CFLAGS="-O2" pyenv install -s "$PYVER"

echo "==> Setting local Python version to $PYVER"
pyenv local "$PYVER"

echo "==> Python in use:"
python -V
which python

# Create virtualenv
if [ -d "$PROJECT_VENV" ]; then
  echo "==> Virtualenv $PROJECT_VENV already exists."
else
  echo "==> Creating virtualenv $PROJECT_VENV..."
  python -m venv "$PROJECT_VENV"
fi

# Activate venv
# shellcheck disable=SC1090
source "$PROJECT_VENV/bin/activate"

echo "==> Upgrading pip tooling..."
python -m pip install -U pip setuptools wheel

# Install requirements if present
if [ -f "requirements.txt" ]; then
  echo "==> Installing project requirements..."
  pip install -r requirements.txt
else
  echo "==> No requirements.txt found; skipping dependency install."
fi

echo "==> Verifying installed packages..."
pip check || {
  echo "pip check found issues. Review the messages above." >&2
  exit 1
}

echo ""
echo "✅ All set."
echo "• Python: $(python -V)"
echo "• Venv:   $(realpath "$PROJECT_VENV")"
echo "• To use later: source $PROJECT_VENV/bin/activate"
echo "• New shells will auto-load pyenv via $RC_FILE (restart your terminal to apply)."

