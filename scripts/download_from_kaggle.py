"""
Download AIY birds_V1_3.tflite from Kaggle Models (new home of TFHub Coral models)

You found: vision-classifier-birds-v1 Version 3 (2416 downloads)
This is google/aiy/vision/classifier/birds_V1/3 TFLite variation - exactly what blueprint needs.

Two ways:
1. Browser (easiest since you already found it): Click Download on Kaggle UI, you get a .tar.gz or .tflite, then scp to Pi
2. CLI via kagglehub (new) or kaggle (old) package

Usage:
  python scripts/download_from_kaggle.py
  # or manual browser download, then:
  scp birds_V1_3.tflite pi@bird-feeder:~//bird-feeder-raspberry-pi/models/

After placement:
  ls -lh ~/bird-feeder-raspberry-pi/models/birds_V1_3.tflite
  python scripts/test_camera.py
"""

import sys
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def try_kagglehub():
    try:
        import kagglehub
        print("Trying kagglehub.model_download('google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3')...")
        # kagglehub API: model_download(handle, variation)
        # Handle from your find: google/aiy-vision-classifier-birds-v1, variation tfLite/birds-v1/3?
        # Try a few guesses
        handles = [
            ("google/aiy-vision-classifier-birds-v1", "tfLite/birds-v1/3"),
            ("google/aiy-vision-classifier-birds-v1", "tfLite"),
            ("google/aiy-vision-classifier-birds-v1", "birds-v1/3"),
            "google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3",
            "google/aiy-vision-classifier-birds-v1",
        ]
        for h in handles:
            try:
                if isinstance(h, tuple):
                    path = kagglehub.model_download(h[0], h[1])
                else:
                    path = kagglehub.model_download(h)
                print(f"kagglehub downloaded to: {path}")
                # Find .tflite inside
                p = Path(path)
                for tflite in p.rglob("*.tflite"):
                    print(f"Found tflite: {tflite}")
                    dest = MODELS_DIR / "birds_V1_3.tflite"
                    shutil.copy(str(tflite), str(dest))
                    print(f"Copied to {dest} ({dest.stat().st_size} bytes)")
                    return True
                for tflite in p.rglob("*.tflite"):  # second search
                    pass
            except Exception as e:
                print(f"kagglehub attempt {h} failed: {e}")
                continue
        return False
    except ImportError:
        print("kagglehub not installed, run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"kagglehub failed: {e}")
        return False

def try_old_kaggle_cli():
    import subprocess, shlex
    try:
        print("\nTrying old kaggle CLI: kaggle models download google/aiy-vision-classifier-birds-v1 ...")
        # Need kaggle.json credentials
        # This is best-effort
        cmd = "kaggle models versions download google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3"
        # Actually correct CLI may be: kaggle models instances versions download ...
        # Try common pattern
        cmds = [
            "kaggle models download google/aiy-vision-classifier-birds-v1",
            "kaggle models versions download google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3 -p /tmp/kaggle_birds",
            "kaggle models instances versions download google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3 --untar",
        ]
        for c in cmds:
            try:
                print(f"Running: {c}")
                subprocess.run(shlex.split(c), check=False, timeout=60)
            except Exception as e:
                print(f"  failed: {e}")
        return False
    except Exception as e:
        print(f"kaggle CLI not available: {e}")
        return False

if __name__ == "__main__":
    print("=== Kaggle AIY birds model downloader ===")
    print("You found: vision-classifier-birds-v1 v3 (2416 downloads) on Kaggle UI")
    print("That's exactly google/aiy/vision/classifier/birds_V1/3 TFLite")
    print("\n1. Browser method (recommended since you already see it):")
    print("   - On Kaggle page, click 'Download' -> you get a .tar.gz or folder")
    print("   - Extract: should contain model.tflite or birds_V1_3.tflite and labels")
    print("   - Rename/copy to:")
    print(f"     {MODELS_DIR / 'birds_V1_3.tflite'}")
    print(f"     {MODELS_DIR / 'aiy_birds_V1_3_labels.txt'}")
    print("   - Then on Windows PowerShell:")
    print("     scp .\\birds_V1_3.tflite pi@10.0.0.241:~/bird-feeder-raspberry-pi/models/")
    print("\n2. CLI method:")
    print("   pip install kagglehub")
    print("   python scripts/download_from_kaggle.py")
    print("\nTrying CLI now...\n")

    ok = try_kagglehub()
    if not ok:
        try_old_kaggle_cli()

    # Check if file now exists (maybe user manually placed)
    target = MODELS_DIR / "birds_V1_3.tflite"
    if target.exists():
        print(f"\n✔ SUCCESS: {target} exists ({target.stat().st_size} bytes)")
        print("Run: python scripts/download_models.py to verify all models")
    else:
        print(f"\n✘ Still missing: {target}")
        print("Please manually download from Kaggle UI and scp as shown above.")
        print("\nKaggle direct link guess:")
        print("  https://www.kaggle.com/models/google/aiy-vision-classifier-birds-v1/tfLite/birds-v1/3/download")
