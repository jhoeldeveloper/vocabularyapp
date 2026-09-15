#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path of the script directory
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$APP_DIR"

echo "🚀 Setting up FastAPI app in $APP_DIR..."

# 1. Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "📦 installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. kokoro-onnx phonemizes text with misaki, which shells out to espeak-ng for
#    several languages (and as an English out-of-dictionary fallback). Install it
#    as a system dependency so g2p never fails. Best-effort: skip if we can't
#    (e.g. no root / unknown package manager) — English still works on most setups.
echo "🔊 Ensuring espeak-ng (kokoro-onnx phonemizer backend)..."
if command -v espeak-ng >/dev/null 2>&1; then
  echo "   (espeak-ng already present)"
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y espeak-ng || \
    echo "   ⚠ apt install of espeak-ng failed; continuing without it"
elif command -v pacman >/dev/null 2>&1; then
  sudo pacman -S --noconfirm espeak-ng || \
    echo "   ⚠ pacman install of espeak-ng failed; continuing without it"
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y espeak-ng || \
    echo "   ⚠ dnf install of espeak-ng failed; continuing without it"
else
  echo "   ⚠ no supported package manager found; install espeak-ng manually if TTS fails"
fi

# 2b. Download the Kokoro-82M ONNX weights + voices bundle (skips if present).
echo "📥 Ensuring Kokoro ONNX model files..."
bash "$APP_DIR/download_models.sh"

# 3. Create the Python 3.12 venv only if it's missing or built with the
#    wrong interpreter — otherwise reuse it so we don't reinstall all
#    dependencies (torch, kokoro, ...) on every start.
echo "📦 Ensuring Python 3.12 venv..."
if [ ! -f "venv/bin/python" ] || ! "venv/bin/python" --version 2>&1 | grep -q "3.12"; then
    echo "   (rebuilding venv)"
    rm -rf venv
    uv venv --python 3.12 venv
fi

# 4. Install dependencies into the venv.
echo "📥 Installing dependencies..."
export PATH="$HOME/.local/bin:$PATH"
source venv/bin/activate
uv pip install -r requirements.txt

# 5. Start the server.
#    --host 0.0.0.0 makes it accessible outside the container/host.
#    --port 8000 is the standard FastAPI port.
#    PYTHONUNBUFFERED=1 makes TTS progress prints appear in real time
#    instead of being buffered until the request completes.
echo "✅ Starting FastAPI with Uvicorn..."
PYTHONUNBUFFERED=1 uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
