"""
Central config for Smart Bird Feeder v2
Tuned for Pi 3B + Camera Module 3 (standard FOV, autofocus)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
CAPTURES_DIR = ASSETS_DIR / "captures"
BEST_DIR = ASSETS_DIR / "best"

DB_PATH = DATA_DIR / "bird_feeder.db"

# --- Camera ---
# Picamera2 preferred, OpenCV fallback
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 10  # lower FPS saves CPU on Pi3
# For autofocus Module 3, picamera2 handles. Set lens position: None = continuous AF
CAMERA_AUTOFOCUS_MODE = "continuous"  # or "auto" or None

# --- Stage 1: Motion Gate ---
MOTION_RESIZE_W = 320
MOTION_RESIZE_H = 240
MOTION_THRESHOLD = 25          # pixel diff threshold
MOTION_MIN_AREA = 500          # min contour area to count as motion
MOTION_BLUR_KERNEL = 21
MOTION_DILATE_ITERS = 2
MOTION_HISTORY_SECONDS = 1.0   # keep background adapt slow

# --- Stage 2: Bird Detection Gate ---
# MobileNetV2 SSD quantized (COCO) or YOLOv8n
DETECTION_MODEL_PATH = MODELS_DIR / "ssd_mobilenet_v2_coco_quant.tflite"
DETECTION_LABELS_PATH = MODELS_DIR / "coco_labels.txt"
DETECTION_THRESHOLD = 0.70   # 70% per spec
DETECTION_INPUT_SIZE = 300   # SSD 300x300
# COCO bird class handling: COCO 90 includes bird as id 16 (1-indexed) or 14 zero-indexed depending on model
# We'll scan label file for 'bird' string match
DETECTION_BIRD_CLASS_IDS = [1, 14, 15, 16]  # fallback set if labels missing
DETECTION_MAX_INFERENCE_MS = 1500  # warn if exceeds

# --- Stage 3: Visit Session Tracking ---
IOU_THRESHOLD = 0.3            # min IoU to consider same bird continuation
GRACE_PERIOD_SEC = 4.0         # session ends after X sec no detection (debounce)
MAX_SESSION_DURATION_SEC = 60  # force close long stuck session
CENTER_MOVE_THRESHOLD = 100    # max centroid movement in px (scaled) to stay same

# --- Stage 4: Best-Frame Selection ---
CAPTURE_INTERVAL_SEC = 0.25    # ~200-300ms candidate sampling
# weights for scoring: sharpness highest weight per spec
SHARPNESS_WEIGHT = 0.5
SIZE_WEIGHT = 0.35
POSITION_WEIGHT = 0.15
# sharpness via Laplacian variance
SHARPNESS_BLUR_THRESHOLD = 20  # below this is considered too blurry, deprioritize heavily
EDGE_CLIP_MARGIN = 0.05        # if bbox within 5% of frame edge, penalize

# --- Stage 5: Species Classifier ---
# Google AIY birds_V1/3 : MobileNet, 964 classes, iNaturalist
SPECIES_MODEL_PATH = MODELS_DIR / "birds_V1_3.tflite"
SPECIES_LABELS_PATH = MODELS_DIR / "aiy_birds_V1_3_labels.txt"
SPECIES_INPUT_SIZE = 224
SPECIES_CONFIDENCE_THRESHOLD = 0.85  # 85-90% per spec, start 85
SPECIES_TOP_K = 3

# Optional cloud fallback
ENABLE_CLOUD_FALLBACK = False
CLOUD_FALLBACK_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
CLOUD_FALLBACK_MAX_PER_DAY = 20

# --- Storage / Dashboard ---
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = False
MAX_RECENT_VISITS = 100
SAVE_ALL_CANDIDATES = False  # if True saves every candidate frame; if False only best

# Atlanta expected species for label verification (subset)
ATLANTA_EXPECTED = [
    "Northern Cardinal", "Carolina Chickadee", "Carolina Wren", "Tufted Titmouse",
    "Mourning Dove", "Blue Jay", "Downy Woodpecker", "Red-bellied Woodpecker",
    "American Goldfinch", "White-breasted Nuthatch", "European Starling",
    "American Crow", "House Finch", "Purple Finch", "American Robin",
    "Northern Mockingbird"
]

# Logging
LOG_LEVEL = "INFO"
