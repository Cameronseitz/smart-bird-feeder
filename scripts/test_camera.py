"""
Test Pi Camera Module 3 on Pi 3B
Verifies picamera2 backend and OpenCV fallback.

Usage on Pi:
  rpicam-jpeg -o test_bird.jpg   # native libcamera test (you already did)
  python scripts/test_camera.py

Should save test capture and show AF working.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_camera")

def test_picamera2_direct():
    try:
        from picamera2 import Picamera2
        import time
        import cv2
        picam2 = Picamera2()
        cfg = picam2.create_still_configuration(main={"size": (1920,1080)}, lores={"size": (640,480)}, controls={"AfMode":2})
        picam2.configure(cfg)
        picam2.start()
        time.sleep(2)  # AF settle
        # capture
        arr = picam2.capture_array("main")
        logger.info(f"Picamera2 captured array shape {arr.shape} dtype {arr.dtype}")
        # convert RGB to BGR for cv2 save
        cv2.imwrite(str(BASE_DIR / "test_picam2.jpg"), cv2.cvtColor(arr, __import__("cv2").COLOR_RGB2BGR))
        logger.info("Saved test_picam2.jpg")
        picam2.stop()
        return True
    except Exception as e:
        logger.error(f"Picamera2 direct test failed: {e}")
        return False

def test_opencv_backend():
    try:
        import cv2
        from src.camera import CameraInterface
        cam = CameraInterface(width=1280, height=720, fps=5)
        logger.info(f"Camera backend: {cam.backend}")
        for i in range(3):
            frame = cam.read()
            if frame is not None:
                logger.info(f"Frame {i}: shape {frame.shape}")
                cv2.imwrite(str(BASE_DIR / f"test_cam_{i}.jpg"), frame)
            else:
                logger.warning(f"Frame {i} None")
        cam.close()
        return True
    except Exception as e:
        logger.error(f"OpenCV backend test failed: {e}")
        import traceback; traceback.print_exc()
        return False

def test_detection():
    try:
        import cv2
        from src.pipeline import BirdDetector
        import sys; sys.path.insert(0, str(BASE_DIR))
        import config
        det = BirdDetector()
        if det.interpreter is None:
            logger.warning("Detector model not loaded, skipping inference test")
            return False
        # Try on latest test image if exists
        img_path = BASE_DIR / "test_picam2.jpg"
        if not img_path.exists():
            img_path = BASE_DIR / "test_cam_0.jpg"
        if not img_path.exists():
            logger.warning("No test image for detector")
            return False
        img = cv2.imread(str(img_path))
        present, dets = det.detect(img)
        logger.info(f"Detector test: bird_present={present} dets={dets}")
        return True
    except Exception as e:
        logger.error(f"Detector test failed: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Pi Camera Module 3 Test ===")
    ok1 = test_picamera2_direct()
    ok2 = test_opencv_backend()
    ok3 = test_detection()
    print(f"\nResults: picamera2_direct={ok1} opencv_backend={ok2} detector={ok3}")
    if ok1 or ok2:
        print("Camera OK – proceed with pipeline")
    else:
        print("Camera FAILED – check cable (15-pin vs 22-pin adapter), libcamera stack, and permissions")
