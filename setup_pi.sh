#!/bin/bash
# Setup script for Raspberry Pi 3 Model B + Camera Module 3
# Run on Pi after flashing Raspberry Pi OS 64-bit Lite/Desktop
# Per blueprint Sec 5 software stack

set -e

echo "=== Smart Bird Feeder Pi 3 Setup ==="

# Update
sudo apt update

# System deps for Picamera2 + OpenCV + TFLite
# Trixie (Debian 13 / Pi OS trixie) dropped libatlas-base-dev - now uses openblas automatically
echo "Installing system packages..."
# Core packages - must succeed
sudo apt install -y python3-picamera2 python3-opencv python3-pip python3-venv \
    libcamera-apps \
    sqlite3 git || true

# Optional legacy ATLAS (for old buster/bullseye/bookworm) - ignore failure on trixie
sudo apt install -y libatlas-base-dev 2>/dev/null || \
sudo apt install -y libopenblas0 libatlas3-base 2>/dev/null || \
echo "libatlas not available on trixie - OK, openblas used automatically by numpy (expected on Debian 13)"

# Extra helpful packages, ignore if missing
sudo apt install -y python3-numpy python3-libcamera 2>/dev/null || true

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv --system-site-packages
    # --system-site-packages allows picamera2 from apt to be seen
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip wheel

echo "Installing Python deps..."
# Core deps - opencv via pip may conflict with apt, but try pip first for newer
pip install numpy Pillow Flask Flask-Cors requests httpx python-dotenv || true
# opencv may already be available via apt python3-opencv inside --system-site-packages venv, but try pip wheel
pip install opencv-python || echo "opencv-python pip failed, using apt python3-opencv"

# TFLite interpreter - Python 3.13 trixie broke tflite-runtime wheels
# Preferred new runtime: ai-edge-litert (official successor), then tflite-runtime, then tensorflow (heavy)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    echo "Installing TFLite interpreter for $ARCH on Python $(python --version)..."
    # Try new official runtime first (has py3.13 wheels on piwheels)
    pip install ai-edge-litert 2>&1 | tail -20 || true
    # Legacy tflite-runtime (works on py 3.9-3.11, fails on 3.13 trixie - expected)
    pip install tflite-runtime 2>&1 | tail -20 || \
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime 2>&1 | tail -20 || \
    echo "tflite-runtime no wheel for $(python --version) on trixie - will use ai-edge-litert if installed"
    
    # Verify which interpreter is available
    python -c "import sys; print('Checking interpreters...'); \
try:
    from ai_edge_litert.interpreter import Interpreter; print('✔ ai-edge-litert available'); \
except Exception as e: print(f'✘ ai-edge-litert: {e}'); \
try:
    from tflite_runtime.interpreter import Interpreter; print('✔ tflite-runtime available'); \
except Exception as e: print(f'✘ tflite-runtime: {e}'); \
try:
    import tensorflow as tf; print('✔ tensorflow available'); \
except Exception as e: print(f'✘ tensorflow: {e}')" || true
else
    echo "x86 architecture - installing tflite-runtime generic"
    pip install tflite-runtime || pip install ai-edge-litert || true
fi

echo "Downloading models..."
python scripts/download_models.py

echo "Testing camera..."
# rpicam-jpeg may be newer name; fallback to libcamera-jpeg
if command -v rpicam-jpeg &> /dev/null; then
    rpicam-jpeg -o test_bird.jpg --timeout 2000 || true
elif command -v libcamera-jpeg &> /dev/null; then
    libcamera-jpeg -o test_bird.jpg --timeout 2000 || true
else
    echo "No rpicam-jpeg found, will test via python"
fi

python scripts/test_camera.py || true

echo ""
echo "=== Setup done ==="
echo "To run pipeline:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
echo "In another terminal:"
echo "  source venv/bin/activate"
echo "  python src/dashboard/app.py"
echo ""
echo "Dashboard will be at http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "For systemd autostart, see systemd/ folder (create service files)"
echo "Wi-Fi test at fence: python scripts/wifi_test.py --duration 120"
