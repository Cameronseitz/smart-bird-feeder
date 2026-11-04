"""
Test Pi Camera Module 3 on Pi 3B
Verifies picamera2 backend and OpenCV fallback.

Usage on Pi:
  rpicam-jpeg -o test_bird.jpg   # native libcamera test
  python scripts/test_camera.py
  python scripts/test_camera.py --sweep  # focus/sharpness sweep at tray distances

Should save test capture and show AF working.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_camera")

import config


def sharpness_of(bgr) -> float:
    try:
        import cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def test_picamera2_direct():
    try:
        from picamera2 import Picamera2
        from src.camera import CameraInterface
        import time
        import cv2
        picam2 = Picamera2()
        # Use the same feeder-tuned controls as production (AF window,
        # centre metering, sports exposure). Validates config on this stack.
        tmp = CameraInterface.__new__(CameraInterface)
        tmp.width, tmp.height, tmp.fps = 1920, 1080, config.CAMERA_FPS
        controls = CameraInterface._build_controls(tmp)
        cfg = picam2.create_still_configuration(
            main={"size": (1920, 1080)}, lores={"size": (640, 480)},
            controls=controls,
        )
        picam2.configure(cfg)
        picam2.start()
        try:
            picam2.set_controls(controls)
        except Exception as e:
            logger.warning(f"set_controls fallback needed: {e}")
        time.sleep(2)  # AF + AEC settle
        # capture
        arr = picam2.capture_array("main")
        logger.info(f"Picamera2 captured array shape {arr.shape} dtype {arr.dtype}")
        # Trixie stack returns BGR already for RGB888; handle both
        bgr = arr
        cv2.imwrite(str(BASE_DIR / "test_picam2.jpg"), bgr)
        logger.info(f"Saved test_picam2.jpg sharpness={sharpness_of(bgr):.1f} "
                    f"(blurry<120, good>500) controls={controls}")
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

def focus_sweep():
    """Manual ruler test: put target at 10/15/20/30cm, captures one frame each ENTER."""
    from src.camera import CameraInterface
    import cv2
    cam = CameraInterface()
    logger.info(f"Backend {cam.backend}. Move target, press ENTER per distance.")
    try:
        for dist in ("10cm", "15cm", "20cm", "30cm"):
            try:
                input(f"Place target at {dist} and press ENTER...")
            except EOFError:
                break
            frame = cam.read()
            if frame is None:
                logger.warning(f"{dist}: no frame")
                continue
            s = sharpness_of(frame)
            out = BASE_DIR / f"focus_sweep_{dist}.jpg"
            cv2.imwrite(str(out), frame)
            logger.info(f"{dist}: sharpness={s:.1f} saved {out.name} "
                        f"(want rising to >500 by 15-20cm)")
    finally:
        cam.close()


def calibrate_lens():
    """Lens-position sweep at the fixed 7in perch plane (no birds needed).

    Tape-mark 7in, place printed feather-detail card there in outdoor daylight,
    leave the camera untouched, then run:
        venv/bin/python scripts/test_camera.py --calibrate-lens
    Sweeps LensPosition 2.0-9.0 step 0.5 with AfMode=0 manual, captures one
    frame per step, logs Laplacian sharpness, saves cal_7in_XpX.jpg.
    Pick max sharpness + readable barbs; keep top-2. Set via Settings page
    (Focus mode 0 manual + Lens position X) -> Save + Restart pipeline.
    Revert: Focus mode 2 continuous + blank lens position.
    """
    from picamera2 import Picamera2
    import time
    import cv2
    picam2 = Picamera2()
    base = {
        "FrameRate": float(config.CAMERA_FPS),
        "AeMeteringMode": config.CAMERA_AE_METERING_MODE,
        "AeExposureMode": config.CAMERA_AE_EXPOSURE_MODE,
        "ExposureValue": float(config.CAMERA_EXPOSURE_VALUE),
        "AfMode": 0,
    }
    cfg = picam2.create_still_configuration(
        main={"size": (1920, 1080)}, lores={"size": (640, 480)},
        controls=base,
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(2.0)
    results = []
    try:
        steps = [round(x * 0.5, 1) for x in range(4, 19)]  # 2.0-9.0
        for pos in steps:
            try:
                picam2.set_controls({"AfMode": 0, "LensPosition": float(pos)})
            except Exception as e:
                logger.warning(f"LensPosition {pos} rejected: {e}")
                continue
            time.sleep(0.8)  # let manual focus + AEC settle
            arr = picam2.capture_array("main")
            s = sharpness_of(arr)
            name = f"cal_7in_{str(pos).replace('.', 'p')}.jpg"
            cv2.imwrite(str(BASE_DIR / name), arr)
            logger.info(f"lens {pos:.1f}: sharpness={s:.1f} saved {name}")
            results.append((pos, s))
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
    results.sort(key=lambda t: t[1], reverse=True)
    if results:
        top = ", ".join(f"{p:.1f} ({s:.0f})" for p, s in results[:3])
        logger.info(f"Top lens positions (pos/sharp): {top} — verify barbs visually, keep top-2")


if __name__ == "__main__":
    print("=== Pi Camera Module 3 Test ===")
    if "--calibrate-lens" in sys.argv:
        calibrate_lens()
    elif "--sweep" in sys.argv:
        focus_sweep()
    else:
        ok1 = test_picamera2_direct()
        ok2 = test_opencv_backend()
        ok3 = test_detection()
        print(f"\nResults: picamera2_direct={ok1} opencv_backend={ok2} detector={ok3}")
        if ok1 or ok2:
            print("Camera OK – proceed with pipeline")
        else:
            print("Camera FAILED – check cable (15-pin vs 22-pin adapter), libcamera stack, and permissions")
