# Backyard AI Bird Feeder 

**Pi 3 Model B + Arducam 12MP Camera Module 3 (IMX708, AF, standard FOV) • Clean 5-stage local pipeline • No cloud**

```
[Arducam 12MP Module 3] → Motion Gate (320x240 diff) → Bird Detector (SSD MBv2 COCO, 65%) → IoU Tracker (grace 4s) → Best-Frame (Laplacian sharpness + size + edge) → Species (AIY birds_V1_3, 964 classes, 65%) → SQLite + Flask Dashboard
```

Two-stage gate kept: species classifier has no "not bird" class, would label squirrels/branches as birds. SSD gate filters.

---

## Hardware – Current

- **Pi 3B v1.2** – 1.2GHz Quad A53, 1GB RAM, Wi-Fi + Ethernet
- **Camera: Arducam 12MP Camera Module 3** – Sony IMX708, 12MP, phase-detect AF, standard FOV (not wide). Compatible with Pi Camera Module 3 driver stack (`picamera2`). **Cable: often ships 22-pin for Pi5, Pi3 needs 15-pin CSI or 22-to-15 adapter** – verified for this build.
  - Autofocus critical for feeder perch 15-40cm close focus
  - Trixie libcamera returns BGR already for RGB888 – handled in `src/camera.py` (no RGB→BGR double swap)
- **Power** – 5V/2.5A Micro-USB window-side. Fence mount needs weatherproof run or solar (out of scope).
- **Housing** – PETG/ASA, drainage, sealed electronics, optical dome.

Mounting:
- **Window (Option A):** easy power, direct sightline
- **Fence (Option B):** short perch distance, test Wi-Fi RSSI ≥ -67 dBm via `wifi_test.py`, power/weatherproofing harder

---

## Software Stack – Current (cleaned)

- OS: Raspberry Pi OS **Trixie (Debian 13) 64-bit**, Python 3.13
- Venv with `--system-site-packages` to see apt `python3-picamera2` + `python3-opencv`
- Runtime: `ai-edge-litert` (official TFLite successor, has py3.13 aarch64 wheels) – **not `tflite-runtime`** (no py3.13 wheel). TensorFlow heavy fallback removed.
- Libs: `numpy`, `Pillow`, `Flask`, `Flask-Cors`, `sqlite3`
- Dashboard: Tailwind CDN + Chart.js
- Models: `models/` gitignored, fetched via Kaggle Models (Coral archive 404 since Apr 2026)

### Project structure (cleaned src)

```
.
├── config.py                 # only used thresholds – cloud / Atlanta list removed
├── requirements.txt          # ai-edge-litert only
├── setup_pi.sh
├── run.sh                    # tmux launcher
├── src/
│   ├── camera.py             # picamera2 primary, opencv fallback only
│   ├── database.py           # visits table only – settings table removed
│   ├── main.py
│   ├── pipeline/
│   │   ├── motion_detector.py
│   │   ├── bird_detector.py  # SSD only, no YOLO placeholder, uint8 only
│   │   ├── tracker.py        # IoU + centroid, _last_cleanup removed
│   │   ├── best_frame_selector.py # evaluate_frame() removed
│   │   └── species_classifier.py # 965->964 background drop, no cloud, no dummy model
│   └── dashboard/
│       ├── app.py            # /api/live/mjpeg and /data/captures removed
│       └── templates/
├── scripts/
│   ├── download_models.py    # GCS SSD + KaggleHub for AIY
│   ├── download_from_kaggle.py # kagglehub handle fixed
│   ├── create_fallback_species_model.py # fallback only if needed
│   ├── test_camera.py
│   └── wifi_test.py
├── models/.gitkeep
├── data/.gitkeep
└── assets/{best,captures}/.gitkeep + latest.jpg
```

Bloat removed: cloud fallback env keys, `ATLANTA_EXPECTED` list, `MOTION_HISTORY_SECONDS`, `CAMERA_AUTOFOCUS_MODE`, `YOLOBirdDetector`, `evaluate_frame`, MJPEG stream, settings DB table, placeholder label generator, dummy TFLite creator, 8+ dead Coral URLs.

---

## Quickstart Pi 3B – Arducam 12MP Module 3

```bash
ssh pi@10.0.0.241 # or pi@bird-feeder-raspberry-pi.local

cd ~/bird-feeder-raspberry-pi
chmod +x setup_pi.sh
./setup_pi.sh  # apt: picamera2 opencv + venv --system-site-packages + ai-edge-litert + models + rpicam-jpeg test

# manual fallback:
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install ai-edge-litert numpy Pillow Flask Flask-Cors python-dotenv
python scripts/download_models.py

python scripts/test_camera.py   # writes test_picam2.jpg, verifies Arducam AF
python scripts/wifi_test.py --duration 120

# Pipeline:
source venv/bin/activate
python src/main.py

# Dashboard (2nd terminal):
source venv/bin/activate
python src/dashboard/app.py
# http://10.0.0.241:5000
```

**SCP from Windows (cleaned code only):**
```powershell
scp -r .\src .\config.py pi@10.0.0.241:~/bird-feeder-raspberry-pi/
# or full:
scp -r .\src .\systemd .\scripts .\config.py .\requirements.txt .\run.sh .\setup_pi.sh .\blueprint.md pi@10.0.0.241:~/bird-feeder-raspberry-pi/
```

---

## Pipeline Details – Current

### Stage 1 Motion Gate
- 320x240 resize, grayscale, GaussianBlur(21), thresh 25, dilate 2x, min area 500
- Background `accumulateWeighted(0.15)`
- Triggers Stage2 only on motion – low CPU

### Stage 2 Bird Detector
- `ssd_mobilenet_v2_coco_quant.tflite` 300x300 uint8, COCO
- Threshold 65% (tuned down from 70%), bird ID via label string scan + fallback [1,14,15,16]
- ~600-1200ms on Pi3B CPU

### Stage 3 Visit Tracking
- IoU >=0.3 OR centroid move <100px = same session
- Grace 4.0s ends session, MAX 60s force close
- Returns closed session to finalize when new bird arrives (low IoU)

### Stage 4 Best-Frame
- Candidate every 0.25s while bird present
- Sharpness `Laplacian(gray).var()` on bbox crop
- Size: bbox area/frame*10 capped 1.0
- Position: penalize if within 5% edge, else center preference
- Weights 0.5/0.35/0.15, blur thresh 20

### Stage 5 Species – AIY birds_V1_3 (Kaggle v3)
- **Model:** `birds_V1_3.tflite` – MobileNet, 964 species iNaturalist, **965 outputs actually (964 + background at idx0)** – code drops background idx0, dequantizes (scale/zp)
- Input 224x224 BGR→RGB, uint8 (no float path)
- Labels: `aiy_birds_V1_3_labels.txt`
- Confidence 65% (was 85%), else "unknown"
- Top-K 3 stored, background >50% => unknown

### Storage
- `data/bird_feeder.db` visits: session_id, timestamp, species, raw_species, confidence, best_image_path, bbox_json, sharpness, top_k, inference_ms, duration
- `assets/best/` JPEGs, `assets/latest.jpg` for dashboard polling

### Dashboard
- `0.0.0.0:5000`
- `/` live + recent 10, `/gallery?species=`, `/analytics` Chart.js, `/settings` display only
- `/api/visits`, `/api/stats`, `/api/gallery`, `/assets/<file>`, `/api/live/frame` (best or latest), `/api/settings` GET, `/api/health`
- MJPEG `/api/live/mjpeg` removed (conflicts with pipeline camera lock)

---

## Models – Kaggle Era (Coral archived Apr 2026)

Old GitHub `google-coral/test_data` raw URLs 404. Now:

- **SSD:** still GCS `https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip` – handled by `download_models.py`

- **AIY birds:** moved to **Kaggle Models** `google/aiy-vision-classifier-birds-v1` Version 3 (2416 downloads)
  - Browser: https://www.kaggle.com/models/google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3/download → extract tar → contains `model.tflite` → rename to `birds_V1_3.tflite`
    ```powershell
    scp .\birds_V1_3.tflite pi@10.0.0.241:~/bird-feeder-raspberry-pi/models/
    ```
  - CLI (new):
    ```bash
    pip install kagglehub
    python scripts/download_from_kaggle.py  # tries handle ("google/aiy-vision-classifier-birds-v1","tfLite/birds-v1/3")
    python scripts/download_models.py       # verifies
    ```
  - Mirror: `https://github.com/gallaugher-bmhs/WhosAtMyFeeder/raw/main/models/birds_V1_3.tflite`

Pipeline runs even if species model missing – logs as unknown but tracking/best-frame/dashboard work.

---

## Testing / Tuning

```bash
rpicam-jpeg -o test_bird.jpg
python scripts/test_camera.py

python scripts/wifi_test.py --duration 120
# -50 Excellent, -60 Good, -67 Good/fair, -70 Marginal, -80 Weak

python scripts/download_models.py
python scripts/debug_species.py --image assets/best/xxx_best.jpg
```

Threshold tuning:
- Grace 2-3s if high-traffic merges
- `MOTION_MIN_AREA` up if foliage wind triggers
- `SPECIES_CONFIDENCE_THRESHOLD` 0.65 for more IDs, 0.85 for fewer false positives

---

## Autostart on Plug-In (systemd)

`systemd/bird-feeder.service` (vision `src/main.py`) + `systemd/bird-dashboard.service`
(Flask `src/dashboard/app.py :5000`) are independent, restart on crash, start on boot.

One-time install on Pi:

```bash
cd ~/bird-feeder-raspberry-pi
chmod +x scripts/install_services.sh
./scripts/install_services.sh
# enables + starts both now, and on every boot after power plug
```

Verify:

```bash
sudo systemctl is-enabled bird-feeder bird-dashboard
sudo systemctl status bird-feeder bird-dashboard --no-pager
journalctl -u bird-feeder -f
journalctl -u bird-dashboard -f
# dashboard at http://<pi-ip>:5000
sudo reboot  # unplug/replug test – both should come back up
```

Useful:

```bash
sudo systemctl restart bird-feeder bird-dashboard
sudo systemctl stop bird-feeder bird-dashboard
sudo systemctl disable bird-feeder bird-dashboard  # turn autostart off
```

Note: installer patches `User=` / `WorkingDirectory=` / `ExecStart=` to your actual
checkout path, adds you to `video` group for Arducam Module 3 access.

---

## License

MIT – Atlanta backyard, Arducam 12MP Module 3.
