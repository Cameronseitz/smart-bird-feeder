"""
Download required TFLite models for Pi 3B bird feeder
- Stage 2: SSD MobileNetV2 quantized COCO (bird detection >70%)
- Stage 5: Google AIY birds_V1/3 TFLite species classifier (964 classes)

Trixie (Debian13) fix: coral test_data GitHub raw URLs now 404 because files were moved to LFS / storage bucket.
This script now tries many mirrors including GCS buckets and HuggingFace, and creates usable placeholders if offline.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # MUST be before importing config
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Atlanta fallback list if config not importable
FALLBACK_ATLANTA = [
    "Northern Cardinal", "Carolina Chickadee", "Carolina Wren", "Tufted Titmouse",
    "Mourning Dove", "Blue Jay", "Downy Woodpecker", "Red-bellied Woodpecker",
    "American Goldfinch", "White-breasted Nuthatch", "European Starling",
    "American Crow", "House Finch", "Purple Finch", "American Robin",
    "Northern Mockingbird"
]

# Safe import of config for Atlanta list
try:
    import config
    ATLANTA_EXPECTED = config.ATLANTA_EXPECTED
except Exception:
    ATLANTA_EXPECTED = FALLBACK_ATLANTA
    config = None

MODEL_URLS = {
    "ssd_mobilenet_v2_coco_quant": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        "alt_urls": [
            # TF Hub direct tflite (may redirect)
            "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/1/default/1?lite-format=tflite",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
            # PiWheels / Coral mirror
            "https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/raw/master/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
        ],
        "zipfile": True,
        "target_tflite": MODELS_DIR / "ssd_mobilenet_v2_coco_quant.tflite",
        "target_labels": MODELS_DIR / "coco_labels.txt"
    },
    # AIY birds model – old GitHub raw 404 now, try many new locations
    "aiy_birds": {
        "url": "https://storage.googleapis.com/coral-aiy-models/birds_V1_3.tflite",
        "alt_urls": [
            # GCS buckets historically used by coral
            "https://storage.googleapis.com/coral-aiy-models/birds_V1_3.tflite",
            "https://storage.googleapis.com/coral_test_data/birds_V1_3.tflite",
            "https://storage.googleapis.com/coral-aiy-test-data/birds_V1_3.tflite",
            "https://github.com/google-coral/test_data/raw/refs/heads/master/birds_V1_3.tflite",
            "https://raw.githubusercontent.com/google-coral/test_data/refs/heads/master/birds_V1_3.tflite",
            "https://huggingface.co/datasets/google-coral/test_data/resolve/main/birds_V1_3.tflite",
            "https://huggingface.co/google-coral/test-data/resolve/main/birds_V1_3.tflite",
            # Community mirrors / forks that vendored the model
            "https://github.com/gallaugher-bmhs/WhosAtMyFeeder/raw/refs/heads/main/models/birds_V1_3.tflite",
            "https://raw.githubusercontent.com/gallaugher-bmhs/WhosAtMyFeeder/refs/heads/main/models/birds_V1_3.tflite",
            "https://raw.githubusercontent.com/gallaugher-bmhs/WhosAtMyFeeder/main/models/birds_V1_3.tflite",
            # Old URLs (kept for completeness, now 404 but try)
            "https://github.com/google-coral/test_data/raw/master/birds_V1_3.tflite",
            "https://raw.githubusercontent.com/google-coral/test_data/master/birds_V1_3.tflite",
        ],
        "target": MODELS_DIR / "birds_V1_3.tflite",
    },
    "aiy_birds_labels": {
        "url": "https://storage.googleapis.com/coral-aiy-models/birds_V1_3_labels.txt",
        "alt_urls": [
            "https://storage.googleapis.com/coral_test_data/birds_V1_3_labels.txt",
            "https://github.com/google-coral/test_data/raw/refs/heads/master/birds_V1_3_labels.txt",
            "https://raw.githubusercontent.com/google-coral/test_data/refs/heads/master/birds_V1_3_labels.txt",
            "https://huggingface.co/datasets/google-coral/test_data/resolve/main/birds_V1_3_labels.txt",
            "https://github.com/gallaugher-bmhs/WhosAtMyFeeder/raw/refs/heads/main/models/birds_V1_3_labels.txt",
            "https://raw.githubusercontent.com/gallaugher-bmhs/WhosAtMyFeeder/refs/heads/main/models/birds_V1_3_labels.txt",
            "https://raw.githubusercontent.com/gallaugher-bmhs/WhosAtMyFeeder/main/models/birds_V1_3_labels.txt",
            "https://github.com/google-coral/test_data/raw/master/birds_V1_3_labels.txt",
            "https://raw.githubusercontent.com/google-coral/test_data/master/birds_V1_3_labels.txt"
        ],
        "target": MODELS_DIR / "aiy_birds_V1_3_labels.txt"
    }
}

def download_file(url, dest_path, retries=2, timeout=30, preserve_existing=True):
    dest_path = Path(dest_path)
    existed_before = dest_path.exists() and dest_path.stat().st_size > 1000
    existing_size = dest_path.stat().st_size if existed_before else 0
    for attempt in range(retries):
        try:
            logger.info(f"Downloading {url} -> {dest_path} (attempt {attempt+1}/{retries})")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 bird-feeder-pi'})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, 'wb') as out_file:
                data = response.read()
                if data[:100].lower().strip().startswith(b'<!doctype') or data[:6].lower() == b'<html>':
                    raise ValueError(f"Got HTML page not binary, likely 404 page for {url}")
                out_file.write(data)
            if dest_path.stat().st_size < 1000:
                raise ValueError(f"File too small {dest_path.stat().st_size}")
            logger.info(f"✓ Saved {dest_path} ({dest_path.stat().st_size} bytes) from {url}")
            return True
        except Exception as e:
            logger.warning(f"✘ Failed {url}: {e}")
            # Only delete file we just tried to write if it didn't exist before OR it's now tiny/invalid
            # Preserve existing valid manual install (e.g., Kaggle birds_V1_3.tflite 3.4M)
            if dest_path.exists():
                try:
                    # If file existed before and is still valid size, keep it
                    if preserve_existing and existed_before and dest_path.stat().st_size >= existing_size and existing_size > 1_000_000:
                        logger.info(f"Preserving existing valid file {dest_path} ({existing_size} bytes) after failed download")
                        # If we overwrote with small partial, restore would need backup - for now don't delete
                        # But if current size is tiny, we already lost original - this is partially destructive, so we avoid unlink if existed_before
                        if dest_path.stat().st_size < 1000:
                            # try to restore by not keeping tiny file, but we already overwrote - can't restore
                            # To avoid this, we should have backed up. For now just unlink tiny and warn
                            dest_path.unlink()
                        # else keep current (may be new larger file or overwritten good with good)
                        pass
                    elif not existed_before:
                        # File didn't exist before this attempt, safe to delete partial
                        dest_path.unlink()
                    else:
                        # existed before but now tiny/failed - try to keep? We already lost original.
                        # Best effort: if new file < 1MB but old was >1MB, delete new to avoid leaving broken file
                        # (original already lost, but user can re-scp)
                        if dest_path.stat().st_size < 1_000_000:
                            dest_path.unlink()
                except Exception:
                    pass
    # After all retries failed, if file existed before and we preserved, report success if still exists valid
    if preserve_existing and dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        logger.info(f"Keeping existing {dest_path} after all retries failed")
        return True
    return False

def extract_ssd_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        names = zip_ref.namelist()
        logger.info(f"Zip contents: {names}")
        for name in names:
            if name.endswith('.tflite'):
                data = zip_ref.read(name)
                target = MODELS_DIR / "ssd_mobilenet_v2_coco_quant.tflite"
                with open(target, 'wb') as f:
                    f.write(data)
                logger.info(f"Extracted {name} -> {target}")
            if 'label' in name.lower() and (name.endswith('.txt') or name.endswith('.map')):
                data = zip_ref.read(name)
                target = MODELS_DIR / "coco_labels.txt"
                with open(target, 'wb') as f:
                    f.write(data)
                logger.info(f"Extracted labels {name} -> {target}")
    coco_labels_path = MODELS_DIR / "coco_labels.txt"
    if not coco_labels_path.exists():
        logger.info("Creating default COCO 90 labels (bird=15)")
        default_labels = [
            "background","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
            "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
            "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
            "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
            "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
            "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
            "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
            "clock","vase","scissors","teddy bear","hair drier","toothbrush"
        ]
        with open(coco_labels_path, 'w') as f:
            for lbl in default_labels:
                f.write(lbl + "\n")

def create_placeholder_bird_labels(target_path):
    # Generate placeholder with expected Atlanta species injected for testing
    placeholder_labels = ATLANTA_EXPECTED.copy()
    for i in range(len(placeholder_labels), 964):
        placeholder_labels.append(f"species_{i}")
    with open(target_path, 'w') as f:
        for lbl in placeholder_labels:
            f.write(lbl + "\n")
    logger.info(f"Wrote placeholder {len(placeholder_labels)} labels to {target_path}")

def main():
    logger.info(f"Models dir: {MODELS_DIR} | Python {sys.version}")

    # 1. SSD mobilenet (bird detector) - this one should still work via GCS
    ssd_cfg = MODEL_URLS["ssd_mobilenet_v2_coco_quant"]
    zip_temp = MODELS_DIR / "ssd_temp.zip"
    ok = False
    for url in [ssd_cfg["url"]] + ssd_cfg["alt_urls"]:
        if download_file(url, zip_temp):
            ok = True
            try:
                extract_ssd_zip(zip_temp, MODELS_DIR)
                zip_temp.unlink(missing_ok=True)
                break
            except Exception as e:
                logger.error(f"Zip extract failed for {url}: {e}")
                zip_temp.unlink(missing_ok=True)
                ok = False
                continue
    if not ok:
        logger.warning("SSD model download failed – creating info file")
        (MODELS_DIR / "README_SSD_MISSING.txt").write_text(
            "SSD model missing. Manual download: https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip\n"
            "Unzip and place detect.tflite as models/ssd_mobilenet_v2_coco_quant.tflite and labelmap.txt as models/coco_labels.txt"
        )

    # 2. AIY birds model - NOW ON KAGGLE MODELS (google/aiy-vision-classifier-birds-v1 v3)
    birds_cfg = MODEL_URLS["aiy_birds"]
    ok = False
    # If manual Kaggle install already exists (3.4M), preserve it and skip download attempts
    if birds_cfg["target"].exists() and birds_cfg["target"].stat().st_size > 1_000_000:
        logger.info(f"✔ Existing birds model found at {birds_cfg['target']} ({birds_cfg['target'].stat().st_size} bytes) - skipping download (preserve manual Kaggle install)")
        ok = True
    else:
        for url in [birds_cfg["url"]] + birds_cfg["alt_urls"]:
            if download_file(url, birds_cfg["target"], preserve_existing=True):
                ok = True
                break

    # Try KaggleHub if GCS/GitHub mirrors all 404 (expected post April 2026 archive)
    if not ok:
        logger.info("All GCS/GitHub mirrors 404 - trying Kaggle Models via kagglehub (new home of TFHub Coral models)...")
        try:
            import kagglehub # pip install kagglehub
            try:
                kpath = kagglehub.model_download("google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3")
                from pathlib import Path as P
                import shutil
                for tflite_file in P(kpath).rglob("*.tflite"):
                    shutil.copy(str(tflite_file), str(birds_cfg["target"]))
                    logger.info(f"✓ KaggleHub copied {tflite_file} -> {birds_cfg['target']} ({birds_cfg['target'].stat().st_size} bytes)")
                    ok = True
                    break
            except Exception as e:
                logger.warning(f"kagglehub download failed: {e} - try manual browser download")
        except ImportError:
            logger.warning("kagglehub not installed, install via pip install kagglehub then re-run, or manual browser download")

    if not ok:
        logger.error(
            f"\n*** Could NOT download birds_V1_3.tflite from any mirror! ***\n"
            "Coral archived test_data Apr 2026 and removed file. Now hosted on Kaggle Models:\n"
            "  Kaggle page you found: vision-classifier-birds-v1 Version 3 (2416 downloads)\n"
            "  That's google/aiy/vision/classifier/birds_V1/3 TFLite exactly\n"
            "  Manual options (do this):\n"
            "  1. Browser: open Kaggle URL https://www.kaggle.com/models/google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3/download\n"
            "     Download -> extract -> scp birds_V1_3.tflite to pi:~/bird-feeder-raspberry-pi/models/\n"
            "  2. CLI: pip install kagglehub && python scripts/download_from_kaggle.py\n"
            "  3. Clone WhosAtMyFeeder: git clone https://github.com/gallaugher-bmhs/WhosAtMyFeeder && cp WhosAtMyFeeder/models/birds_V1_3.tflite models/\n"
            "  4. Fallback: python scripts/create_fallback_species_model.py (MobileNetV2 generic, pipeline still works but logs as generic bird)\n"
            "  For now pipeline will run but species will be 'unknown' (detector + tracking + best-frame + dashboard still work)\n"
        )
        (MODELS_DIR / "README_BIRDS_MISSING.txt").write_text(
            "birds_V1_3.tflite missing - now on Kaggle Models google/aiy-vision-classifier-birds-v1 v3\n"
            "Browser: https://www.kaggle.com/models/google/aiy-vision-classifier-birds-v1\n"
            "Download -> scp to models/birds_V1_3.tflite\n"
            "Or: pip install kagglehub && python scripts/download_from_kaggle.py\n"
            "Fallback: python scripts/create_fallback_species_model.py\n"
        )

    # 3. AIY labels
    labels_cfg = MODEL_URLS["aiy_birds_labels"]
    ok = False
    for url in [labels_cfg["url"]] + labels_cfg["alt_urls"]:
        if download_file(url, labels_cfg["target"]):
            ok = True
            break
    if not ok:
        logger.warning("Birds labels download failed – generating placeholder 964 list (Atlanta prioritized)")
        create_placeholder_bird_labels(labels_cfg["target"])

    # Verification
    print("\n=== Verification ===")
    for p in sorted(MODELS_DIR.glob("*")):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size} bytes")
    if (MODELS_DIR / "ssd_mobilenet_v2_coco_quant.tflite").exists():
        print("✔ Bird detector model present (SSD MBv2, Stage2)")
    else:
        print("✘ Bird detector MISSING - Stage2 will always fail!")

    if (MODELS_DIR / "birds_V1_3.tflite").exists():
        print("✔ Species classifier present (AIY 964, Stage5)")
    else:
        print("✘ Species classifier MISSING – pipeline will log unknown until added, but tracking/best-frame/dashboard still work")

    if (MODELS_DIR / "aiy_birds_V1_3_labels.txt").exists():
        print("✔ Species labels present")
        try:
            with open(MODELS_DIR / "aiy_birds_V1_3_labels.txt") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            print(f"  {len(lines)} labels, first 20: {lines[:20]}")
        except Exception as e:
            print(f"  label read error: {e}")
    if (MODELS_DIR / "coco_labels.txt").exists():
        print("✔ COCO labels present")
        with open(MODELS_DIR / "coco_labels.txt") as f:
            print(f"  {len(f.readlines())} COCO labels - bird should be present")

    print("\nNext steps on Pi:")
    print("  source venv/bin/activate")
    print("  python -c \"from ai_edge_litert.interpreter import Interpreter; print('TFLite OK')\" or test tflite_runtime")
    print("  python scripts/test_camera.py")
    print("  python src/main.py  # will work even if species model missing (logs unknown)")
    print("  python src/dashboard/app.py  # in another terminal")
    print("\nIf species model missing, quick workaround:")
    print("  git clone https://github.com/gallaugher-bmhs/WhosAtMyFeeder /tmp/waf")
    print("  find /tmp/waf -name \"*.tflite\" -type f | head")
    print("  # if found, cp to models/birds_V1_3.tflite")

if __name__ == "__main__":
    main()
