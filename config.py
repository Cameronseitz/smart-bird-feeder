"""
Central config - Pi 3B + Camera Module 3
Only values actually used by src/
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
CAPTURES_DIR = ASSETS_DIR / "captures"
BEST_DIR = ASSETS_DIR / "best"

DB_PATH = DATA_DIR / "bird_feeder.db"

# Camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 10

# Camera Module 3 focus / exposure tuning (feeder close-range fix, no mechanical change)
# AfMode: 0=manual (uses CAMERA_LENS_POSITION), 1=auto, 2=continuous
CAMERA_AF_MODE = 2
# AfSpeed: 0=normal, 1=fast (fewer hunting frames on moving bird)
CAMERA_AF_SPEED = 1
# Feeder-zone AF window. DISABLED by default: libcamera v0.7.1 on Trixie
# expects integer sensor-pixel Rectangles, not normalized floats - passing
# floats kills Picamera2 init and forces a dead OpenCV/V4L fallback.
# Leave None (full-frame AF). Only set to pixel ints e.g. [(500, 300, 800, 600)]
# after verifying with rpicam-hello --autofocus-windows.
CAMERA_AF_WINDOWS = None
# Manual focus override: None=autofocus, else dioptre e.g. 6.0.
# 7" rig intent: lock to calibrated 7" plane as default (continuous AF is
# fallback only — full-frame AF hunts to background trees/fence at 127-178mm).
# Calibrate with: rpicam-hello --autofocus-mode manual --lens-position X
# sweep 2.0-9.0 step 0.5, or: venv/bin/python scripts/test_camera.py --calibrate-lens
# then set via Settings page (writes config_overrides.json, restart to apply).
# Keep EV default 0.6 here; Pi override 1.2 kept for now (birds too dark below it).
CAMERA_LENS_POSITION = None
# AeMeteringMode: 0=centre-weighted, 1=spot, 2=matrix. Centre/spot exposes
# the bird, not the bright background (silhouette fix).
CAMERA_AE_METERING_MODE = 0
# AeExposureMode: 0=normal, 1=short/sports (freeze motion), 2=long, 3=custom
CAMERA_AE_EXPOSURE_MODE = 1
# ExposureValue -8.0..+8.0, lifts dark foreground bird. 0.5-1.0 typical.
CAMERA_EXPOSURE_VALUE = 0.6
# Upper bound on auto shutter - DISABLED by default. FrameDurationLimits is
# (min, max) frame interval, not exposure: (100000, 10000) for 10fps is
# inverted and fails configure. Leave None and rely on AeExposureMode=short.
CAMERA_MAX_EXPOSURE_US = None

# Stage 1 Motion
MOTION_RESIZE_W = 320
MOTION_RESIZE_H = 240
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 500
MOTION_BLUR_KERNEL = 21
MOTION_DILATE_ITERS = 2

# Stage 2 Detection
DETECTION_MODEL_PATH = MODELS_DIR / "ssd_mobilenet_v2_coco_quant.tflite"
DETECTION_LABELS_PATH = MODELS_DIR / "coco_labels.txt"
DETECTION_THRESHOLD = 0.65
DETECTION_INPUT_SIZE = 300
DETECTION_BIRD_CLASS_IDS = [1, 14, 15, 16]
DETECTION_MAX_INFERENCE_MS = 1500

# Stage 3 Tracking
IOU_THRESHOLD = 0.3
GRACE_PERIOD_SEC = 4.0
MAX_SESSION_DURATION_SEC = 60
CENTER_MOVE_THRESHOLD = 100

# Stage 4 Best Frame
CAPTURE_INTERVAL_SEC = 0.25
# Sharpness dominates: blurry-close frames must not beat sharp-far frames.
SHARPNESS_WEIGHT = 0.7
SIZE_WEIGHT = 0.2
POSITION_WEIGHT = 0.1
# Observed: blurry cards 100-460, good Blue Jay 1500+. 20 accepted everything.
SHARPNESS_BLUR_THRESHOLD = 120.0
# Normalization knee: sharpness >= this scores 1.0 on sharp term.
SHARPNESS_GOOD_REF = 500.0
# Bbox larger than this fraction of frame is "too close" (below min focus
# distance) - heavily penalized even if detector loves it.
MAX_BBOX_FRACTION = 0.55
EDGE_CLIP_MARGIN = 0.05

# Stage 5 Species
SPECIES_MODEL_PATH = MODELS_DIR / "birds_V1_3.tflite"
SPECIES_FALLBACK_MODEL_PATH = MODELS_DIR / "mobilenet_v2_1.0_224_quant.tflite"
SPECIES_LABELS_PATH = MODELS_DIR / "aiy_birds_V1_3_labels.txt"
SPECIES_INPUT_SIZE = 224
SPECIES_CONFIDENCE_THRESHOLD = 0.65
SPECIES_TOP_K = 3

# Dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = False
MAX_RECENT_VISITS = 100
SAVE_ALL_CANDIDATES = True

LOG_LEVEL = "INFO"

# ---- Persistent dashboard overrides (settings page) ----
# Dashboard POST /api/settings writes only these keys to config_overrides.json.
# Loaded at import so pipeline + dashboard pick them up after restart/reboot.
# config.py defaults stay as fallback; delete the JSON to reset.
OVERRIDES_PATH = BASE_DIR / "config_overrides.json"
WRITABLE_KEYS = {
    "DETECTION_THRESHOLD",
    "SPECIES_CONFIDENCE_THRESHOLD",
    "GRACE_PERIOD_SEC",
    "MOTION_THRESHOLD",
    "CAMERA_EXPOSURE_VALUE",
    "CAMERA_AE_METERING_MODE",
    "CAMERA_AE_EXPOSURE_MODE",
    "CAMERA_AF_MODE",
    "CAMERA_LENS_POSITION",
    "SHARPNESS_BLUR_THRESHOLD",
    "SHARPNESS_GOOD_REF",
    "MAX_BBOX_FRACTION",
}
OVERRIDES_LOADED: dict = {}


def _apply_overrides() -> None:
    try:
        import json

        if not OVERRIDES_PATH.exists():
            return
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            return
        g = globals()
        for k, v in data.items():
            if k in WRITABLE_KEYS and k in g:
                g[k] = v
                OVERRIDES_LOADED[k] = v
    except Exception:
        pass


_apply_overrides()
