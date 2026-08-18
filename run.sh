#!/bin/bash
# Simple tmux runner for pipeline + dashboard
# Usage: ./run.sh on Pi
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

if [ ! -d "venv" ]; then
  echo "venv not found, running setup..."
  ./setup_pi.sh
fi

source venv/bin/activate

echo "Starting bird feeder in tmux..."
tmux new-session -d -s birdfeeder "python src/main.py"
tmux new-window -t birdfeeder:1 -n dashboard "python src/dashboard/app.py"
tmux attach -t birdfeeder || tmux ls

echo "Dashboard at http://$(hostname -I | awk '{print $1}'):5000"
