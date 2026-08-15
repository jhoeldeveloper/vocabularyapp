#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path of the script directory
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$APP_DIR"

echo "🚀 Setting up FastAPI app in $APP_DIR..."

# 1. Check for the activation script specifically
# If venv exists but is broken/empty, this will trigger a fresh install
if [ ! -f "venv/bin/activate" ]; then
    echo "📦 Virtual environment missing or broken. Rebuilding..."
    rm -rf venv/
    python3 -m venv venv
fi

# 2. Activate venv
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "📥 Ensuring dependencies are up to date..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Fallback: install FastAPI and Uvicorn if requirements.txt is missing
    pip install fastapi uvicorn
fi

# 4. Start the server
# --host 0.0.0.0 makes it accessible outside the container/host
# --port 8000 is the standard FastAPI port
echo "✅ Starting FastAPI with Uvicorn..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
