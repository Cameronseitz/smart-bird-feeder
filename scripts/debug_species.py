"""
Debug species classifier on saved best frames / test images
Helps diagnose low confidence uniform issue

Usage on Pi:
  python scripts/debug_species.py --image assets/best/20260820_205928_306_visit_..._best.jpg
  python scripts/debug_species.py --image test_picam2.jpg
  python scripts/debug_species.py --all-best
"""

import sys
from pathlib import Path
import argparse
import cv2
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.species_classifier import SpeciesClassifier
import config

def test_image(path):
    print(f"\n=== Testing {path} ===")
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Failed to read {path}")
        return
    print(f"Image shape {frame.shape}")
    clf = SpeciesClassifier()
    result = clf.classify(frame)
    print(f"Result: species={result['species']} raw={result.get('raw_species')} conf={result['confidence']:.4f} is_conf={result['is_confident']}")
    print(f"Top3: {result['top_k']}")
    print(f"Inference ms: {result.get('inference_ms')}")
    # Also test with bounding box full frame
    h,w = frame.shape[:2]
    fake_det = {"bbox_px": [0,0,w,h]}
    result2 = clf.classify(frame, fake_det)
    print(f"With full bbox: {result2['species']} conf={result2['confidence']:.4f} top3={result2['top_k']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="image path")
    parser.add_argument("--all-best", action="store_true", help="test all best/")
    args = parser.parse_args()

    if args.image:
        test_image(args.image)
    elif args.all_best:
        best_dir = BASE_DIR / "assets" / "best"
        if not best_dir.exists():
            print(f"No {best_dir}")
            return
        for img in sorted(best_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            test_image(img)
    else:
        # Default: test latest best + test_picam2
        for p in [BASE_DIR/"test_picam2.jpg", BASE_DIR/"assets"/"latest.jpg"]:
            if p.exists():
                test_image(p)

if __name__ == "__main__":
    main()
