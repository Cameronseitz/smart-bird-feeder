"""
Camera for Pi Camera Module 3 on Pi 3B
Picamera2 primary (AF), OpenCV fallback
"""
import sys
import time
import logging
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)


class CameraInterface:
    def __init__(self, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT, fps=config.CAMERA_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self._picam2 = None
        self._cap = None
        self.backend = None
        self._init_camera()

    def _build_controls(self, include_optional: bool = True) -> dict:
        """Feeder-tuned libcamera controls. Missing attrs fall back to safe defaults."""
        af_mode = getattr(config, "CAMERA_AF_MODE", 2)
        af_speed = getattr(config, "CAMERA_AF_SPEED", 1)
        af_windows = getattr(config, "CAMERA_AF_WINDOWS", None)
        lens_pos = getattr(config, "CAMERA_LENS_POSITION", None)
        ae_metering = getattr(config, "CAMERA_AE_METERING_MODE", 0)
        ae_exposure = getattr(config, "CAMERA_AE_EXPOSURE_MODE", 1)
        ev = getattr(config, "CAMERA_EXPOSURE_VALUE", 0.6)

        controls = {
            "FrameRate": float(self.fps),
            "AeMeteringMode": ae_metering,
            "AeExposureMode": ae_exposure,
            "ExposureValue": float(ev),
        }
        if lens_pos is not None:
            # Fixed-distance feeder: lock focus, skip AF hunting entirely.
            controls["AfMode"] = 0
            controls["LensPosition"] = float(lens_pos)
        else:
            controls["AfMode"] = af_mode
            if include_optional:
                controls["AfSpeed"] = af_speed
                if af_windows:
                    # libcamera v0.7.1 Rectangle() takes ints only. Normalized
                    # 0..1 floats kill init -> skip them loudly instead.
                    try:
                        flat = [float(v) for w in af_windows for v in w]
                        if all(v <= 1.0 for v in flat):
                            logger.warning("CAMERA_AF_WINDOWS looks normalized (0..1), "
                                           "this stack needs sensor-pixel ints - skipping")
                        else:
                            controls["AfWindows"] = tuple(tuple(int(v) for v in w) for w in af_windows)
                    except Exception as we:
                        logger.warning(f"Bad CAMERA_AF_WINDOWS {af_windows}: {we} - skipping")
        return controls

    def _start_picamera2(self, controls: dict):
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        cam_config = self._picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            lores={"size": (320, 240), "format": "YUV420"},
            controls=controls,
        )
        self._picam2.configure(cam_config)
        self._picam2.start()
        # Re-apply one-by-one: older libcamera builds reject unknown
        # keys wholesale at configure time but accept the rest via set_controls.
        try:
            self._picam2.set_controls(controls)
        except Exception as e:
            logger.warning(f"Bulk set_controls rejected ({e}), retrying per-key")
            for k, v in controls.items():
                try:
                    self._picam2.set_controls({k: v})
                except Exception as ke:
                    logger.warning(f"Control {k} unsupported: {ke}")
        time.sleep(2.0)  # let AF + AEC settle
        logger.info(f"Picamera2 {self.width}x{self.height} @{self.fps}fps controls={controls}")
        self.backend = "picamera2"

    def _init_camera(self):
        try:
            controls = self._build_controls(include_optional=True)
            try:
                self._start_picamera2(controls)
                return
            except Exception as e:
                logger.warning(f"Picamera2 full controls failed ({e}), retrying minimal")
                try:
                    if self._picam2 is not None:
                        try:
                            self._picam2.close()
                        except Exception:
                            pass
                        self._picam2 = None
                except Exception:
                    pass
                minimal = self._build_controls(include_optional=False)
                self._start_picamera2(minimal)
                return
        except Exception as e:
            logger.warning(f"Picamera2 failed ({e}), trying OpenCV")
            try:
                if self._picam2 is not None:
                    try:
                        self._picam2.close()
                    except Exception:
                        pass
            except Exception:
                pass
            self._picam2 = None

        for idx in (0, 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self._cap = cap
                self.backend = f"opencv:{idx}"
                logger.info(f"OpenCV backend index {idx}")
                return
        raise RuntimeError("No camera via picamera2 nor OpenCV")

    def read(self) -> Optional[np.ndarray]:
        if self.backend == "picamera2":
            try:
                # Trixie libcamera returns BGR already for RGB888 on this stack
                arr = self._picam2.capture_array("main")
                if arr.ndim == 3 and not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                return arr
            except Exception as e:
                logger.error(f"Picamera2 capture failed: {e}")
                return None
        else:
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
            return frame if ok else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        interval = 1.0 / max(self.fps, 1)
        while True:
            start = time.time()
            frame = self.read()
            if frame is not None:
                yield frame
            else:
                time.sleep(0.1)
                continue
            elapsed = time.time() - start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def close(self):
        try:
            if self._picam2:
                self._picam2.stop()
        except Exception:
            pass
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam = CameraInterface()
    for i, f in enumerate(cam.frames()):
        print(f"frame {i}: {f.shape}")
        cv2.imwrite(f"test_{i}.jpg", f)
        if i >= 2:
            break
    cam.close()
