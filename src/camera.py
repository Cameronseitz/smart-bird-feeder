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

    def _init_camera(self):
        try:
            from picamera2 import Picamera2

            self._picam2 = Picamera2()
            cam_config = self._picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                lores={"size": (320, 240), "format": "YUV420"},
                controls={"AfMode": 2},
            )
            self._picam2.configure(cam_config)
            self._picam2.start()
            time.sleep(1.5)
            logger.info(f"Picamera2 {self.width}x{self.height} @{self.fps}fps AF continuous")
            self.backend = "picamera2"
            return
        except Exception as e:
            logger.warning(f"Picamera2 failed ({e}), trying OpenCV")
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
