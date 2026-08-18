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
# tflite-runtime wheel for aarch64: needs specific version
# Try tflite-runtime first, fallback note
pip install opencv-python numpy Pillow Flask Flask-Cors requests httpx python-dotenv || true

# tflite-runtime install attempt
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    echo "Installing tflite-runtime for $ARCH..."
    pip install tflite-runtime || pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime || echo "tflite-runtime failed, will try tensorflow lite via tensorflow"
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
