# Smart AI Window/Fence Bird Feeder v2

**Pi 3 Model B + Camera Module 3 (standard FOV, AF) • Two-stage local vision pipeline • No cloud**

Implements blueprint.md v2 in full:

```
[Camera Stream] → Motion Gate (320x240 frame diff) → Bird Detector (TFLite SSD MBv2, >70% bird) → IoU Visit Tracking (grace 3-5s debounce) → Best-Frame Selection (sharpness Laplacian, bbox size, edge-clip) → Species Classifier (AIY birds_V1/3 MobileNet 964 classes, iNaturalist) → Confidence >85% else unknown → SQLite + Flask Dashboard
```

### Why two-stage?

Species classifier has no "not bird" output – it will label squirrels/branches/shadows as a bird species. Object detection gate prevents flooding dashboard with false sightings.

---

## Hardware

- **Pi 3B v1.2** – 1.2GHz Quad A53, 1GB RAM, Wi-Fi + Ethernet
- **Camera Module 3** – 12MP IMX708, standard FOV (not wide), phase-detect AF. **Check cable: Module 3 often ships 22-pin for Pi5, Pi3 needs 15-pin CSI or adapter.**
- **Power** – 5V/2.5A Micro-USB only valid if window-side (Option A). Fence Option B needs weatherproof cable or battery+solar (not scoped).
- **Housing** – PETG/ASA custom, drainage, sealed electronics chamber, optical dome.

Mounting decision open (blueprint §6):
- **Option A window:** easy power, human sightline, one-way film hides indoor motion
- **Option B fence:** very short cam-to-perch distance → verify AF close focus, Wi-Fi test at fence (100-150ft typical LOS), power/weatherproofing harder, loses direct viewing unless feeder faces window

---

## Software Stack

- OS: Raspberry Pi OS 64-bit Lite/Desktop
- Python 3.10+ venv with `--system-site-packages` (needs picamera2 from apt)
- `opencv-python`, `tflite-runtime` (Pi3 CPU inference 600ms-1.2s), `Flask`, `sqlite3`
- Dashboard: Tailwind CDN + Chart.js – live view, gallery, analytics, settings
- Models: `models/` – download via script

### Project structure

```
.
├── blueprint.md
├── config.py                 # all thresholds & paths
├── requirements.txt
├── setup_pi.sh               # Pi first-time setup
├── src/
│   ├── camera.py             # Picamera2 + OpenCV fallback
│   ├── database.py           # SQLite visits, stats
│   ├── main.py               # pipeline loop
│   ├── pipeline/
│   │   ├── motion_detector.py   # Stage1 320x240 diff
│   │   ├── bird_detector.py     # Stage2 SSD MBv2 bird >70%
│   │   ├── tracker.py           # Stage3 IoU + grace debounce
│   │   ├── best_frame_selector.py # Stage4 Laplacian sharpness/size/pos
│   │   └── species_classifier.py # Stage5 AIY 964 local + threshold 85%
│   └── dashboard/
│       ├── app.py            # Flask
│       └── templates/
│           ├── base.html
│           ├── index.html (live)
│           ├── gallery.html
│           ├── analytics.html
│           └── settings.html
├── scripts/
│   ├── download_models.py
│   ├── test_camera.py
│   └── wifi_test.py          # for fence mounting decision
├── models/                   # TFLite files (gitignored or downloaded)
├── data/                     # bird_feeder.db
└── assets/
    ├── captures/             # all candidates if enabled
    ├── best/                 # best per visit (served)
    └── latest.jpg            # last frame for dashboard
```

---

## Quickstart on Pi 3B

You said you're SSHed into Pi with camera tested (`rpicam-jpeg -o test_bird.jpg` worked).

```bash
ssh pi@bird-feeder-raspberry-pi.local
# lick raspberrypi - your note

cd ~/bird-feeder-raspberry-pi   # or where you cloned/scp'd
chmod +x setup_pi.sh
./setup_pi.sh   # apt + venv + models + camera test

# Manual steps if setup_pi fails partially:
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy Pillow Flask Flask-Cors tflite-runtime requests

python scripts/download_models.py
python scripts/test_camera.py
python scripts/wifi_test.py --duration 120   # if considering fence mount

# Run pipeline (terminal 1):
source venv/bin/activate
python src/main.py

# Dashboard (terminal 2):
source venv/bin/activate
python src/dashboard/app.py
# Then open http://pi-ip:5000 from laptop
```

`scp` code from dev machine to Pi:
```bash
scp -r C:\Users\camer\Documents\bird-feeder-raspberry-pi pi@10.0.0.241:~/
```

---

## Pipeline details

### Stage 1 – Motion Gate
- Low-res 320x240, Gaussian blur 21, threshold 25, dilate 2 iterations, min area 500px²
- Running average background `accumulateWeighted(0.15)` ~1 sec history
- Keeps CPU low baseline; only triggers inference when motion

### Stage 2 – Bird Detector Gate
- Quantized SSD MobileNetV2 COCO 300x300, >70% threshold
- Parses `bird` label via label file string match, fallback IDs [1,14,15,16] covers COCO indexing variants
- Reports bbox `xmin,ymin,xmax,ymax` normalized → pixel
- Expected ~600-1200ms on Pi3 bare CPU (blueprint revised estimate). Not using Coral USB TPU (deprecated driver)

### Stage 3 – Visit Tracking
- IoU-based: high overlap + similar size = same bird continuation
- Grace 4.0 sec ends session after no detection → doubles as debounce/cooldown
- Centroid distance + IoU mitigates high-traffic merge (departing+arriving merged if within grace)
- `MAX_SESSION_DURATION 60s` forces close stuck
- Multi-bird note: if multiple birds frequent simultaneously, extend to simple centroid multi-tracker

### Stage 4 – Best-Frame Selection
- Candidate every 250ms while bird present
- Sharpness via `cv2.Laplacian(gray).var()` on bbox crop
- Size score: bbox area / frame area scaled
- Position: penalize if within 5% edge (clipped), else slight center preference
- Weights: sharpness 0.5, size 0.35, position 0.15, blur threshold 20
- Runs species classifier **once** on winning frame (cost equivalent to single-shot design)

### Stage 5 – Species Classifier Local
- Google AIY `birds_V1_3.tflite` – MobileNet backbone, 964 species, iNaturalist training
- Input 224x224 RGB, preprocess `(pixel-127.5)/127.5` → [-1,1] for float model; direct uint8 for quant model
- Label map pulled and Atlanta coverage verified in logs:
  Northern Cardinal, Carolina Chickadee, Carolina Wren, Tufted Titmouse, Mourning Dove, Blue Jay, Downy/Red-bellied Woodpecker, American Goldfinch, White-breasted Nuthatch, European Starling, American Crow, House/ Purple Finch, American Robin, Northern Mockingbird – all expected to be in 964 set
- **Action item**: spot-test with reference photos of top 15-20 visitors, especially look-alike pairs (House vs Purple Finch)

### Confidence Check
- Only log as confirmed species if >=85% (tunable 85-90% per spec). Below logs as "unknown" rather than wrong label
- Cloud fallback optional, disabled by default (`ENABLE_CLOUD_FALLBACK=False`). Needs GEMINI_API_KEY etc.

### Storage
- `data/bird_feeder.db` SQLite: `visits` table stores session_id, timestamp, species, raw, confidence, best_image_path, bbox_json, sharpness, top_k, inference_ms, duration, num candidates
- `assets/best/` best frame per visit, `assets/latest.jpg` for dashboard live
- `SAVE_ALL_CANDIDATES=False` by default to save SD wear – set True for debugging

### Dashboard
- Flask `0.0.0.0:5000`
- Routes: `/` live feed + recent, `/gallery?species=..`, `/analytics` Chart.js hourly/daily/species doughnut, `/settings` thresholds, `/api/*` JSON, `/assets/<file>` images, `/api/live/mjpeg` debug MJPEG (conflicts with main pipeline if both try to open camera – prefer `latest.jpg` polling)

---

## Testing / Tuning

**Camera:**
```bash
rpicam-jpeg -o test_bird.jpg   # you did this
python scripts/test_camera.py
```

**Wi-Fi at fence:**
```bash
python scripts/wifi_test.py --duration 120
# RSSI >= -67 dBm excellent, -67 to -70 fair/marginal, < -75 weak -> window mount or mesh/extender/Ethernet
```

**Models:**
```bash
python scripts/download_models.py
# If coral test_data URL fails, manual download from:
# - https://github.com/google-coral/test_data
# - Community mirror: WhosAtMyFeeder project
# Place into models/birds_V1_3.tflite + aiy_birds_V1_3_labels.txt
# SSD model zip from https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

**Tune thresholds after running:**
- Shorten grace to 2-3s if high-traffic merging observed
- Raise `MOTION_MIN_AREA` if wind foliage triggers too many inferences
- Adjust `SPECIES_CONFIDENCE_THRESHOLD` 85-90% based on false "unknown" vs wrong label tradeoff
- Logs show coverage: look for `Atlanta species coverage check` in `SpeciesClassifier`

---

## Frigate rejected (§7)

Frigate + sidecar (WhosAtMyFeeder pattern) evaluated, not adopted:
- Needs NVR stack (MQTT, continuous decode), best on N100 miniPC or Pi5+Hailo, not bare Pi3B
- Coral USB TPU deprecated (driver archived, unsupported modern kernels)
- Security dual-purpose dropped – so NVR benefits not needed
- Useful piece kept: AIY birds TFLite model directly in pipeline Stage 5, keeping Pi3B viable

---

## Open items checklist (blueprint §9)

- [ ] 15-pin vs 22-pin cable adapter for Module 3 + Pi3
- [ ] Verify min focus distance vs perch distance (embedded case)
- [ ] Pull label map + confirm Atlanta species (automated log, still spot-test)
- [ ] Window vs fence mounting decision – Wi-Fi test + viewing preference
- [ ] If fence, scope power + weatherproofing
- [ ] Tune grace + confidence thresholds empirically once hardware running

---

## Systemd (optional autostart)

Create `/etc/systemd/system/bird-feeder.service`:

```
[Unit]
Description=Smart Bird Feeder Pipeline
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/bird-feeder-raspberry-pi
ExecStart=/home/pi/bird-feeder-raspberry-pi/venv/bin/python src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

And dashboard service similarly executing `src/dashboard/app.py`. Enable via `sudo systemctl enable --now bird-feeder`.

---

## License

MIT – personal build, Atlanta backyard.
