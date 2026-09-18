#!/bin/bash
# Install + enable systemd autostart for vision pipeline + dashboard
# Run ONCE on the Pi from the project root: ./scripts/install_services.sh
# After this, just plug in power and both start on boot.
set -e

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(whoami)"
PROJECT_DIR="$SRC_DIR"

echo "Project: $PROJECT_DIR (user: $USER_NAME)"
echo "Venv python: $PROJECT_DIR/venv/bin/python"

if [ ! -x "$PROJECT_DIR/venv/bin/python" ]; then
  echo "ERROR: $PROJECT_DIR/venv/bin/python not found."
  echo "Run ./setup_pi.sh first."
  exit 1
fi

# Ensure user can access camera (picamera2 / libcamera needs video group)
if ! groups | grep -q video; then
  echo "Adding $USER_NAME to video group (for camera)..."
  sudo usermod -aG video "$USER_NAME" || true
  echo "NOTE: log out/in once for video group to apply, then re-run this script."
fi

# Patch User/WorkingDirectory/ExecStart to match actual location
for svc in bird-feeder bird-dashboard; do
  src="$SRC_DIR/systemd/${svc}.service"
  tmp="/tmp/${svc}.service"
  if [ "$svc" = "bird-feeder" ]; then
    ENTRY="src/main.py"
  else
    ENTRY="src/dashboard/app.py"
  fi
  sed -e "s|^User=.*|User=${USER_NAME}|" \
      -e "s|^WorkingDirectory=.*|WorkingDirectory=${PROJECT_DIR}|" \
      -e "s|^ExecStart=.*|ExecStart=${PROJECT_DIR}/venv/bin/python ${ENTRY}|" \
      "$src" > "$tmp"
  echo "--- $svc.service (installed version) ---"
  cat "$tmp"
  sudo cp "$tmp" "/etc/systemd/system/${svc}.service"
done

sudo systemctl daemon-reload
sudo systemctl enable --now bird-feeder.service
sudo systemctl enable --now bird-dashboard.service

echo ""
echo "Done. Status:"
sudo systemctl is-enabled bird-feeder.service bird-dashboard.service || true
sudo systemctl --no-pager --full status bird-feeder.service bird-dashboard.service || true
echo ""
echo "Logs:"
echo "  journalctl -u bird-feeder -f"
echo "  journalctl -u bird-dashboard -f"
echo "Dashboard: http://\$(hostname -I | awk '{print \$1}'):5000"
echo "Test reboot with: sudo reboot"
