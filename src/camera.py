"""
Camera abstraction for Pi Camera Module 3 on Pi 3B
- Tries picamera2 first (native libcamera stack, supports AF Module 3)
- Falls back to OpenCV V4L2 (/dev/video0)
- Exposes simple iterator: frames() -> np array BGR
"""
import time
import logging
from typing import Generator, Optional
import cv2
import numpy as np

try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
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
        # Try picamera2
        try:
            from picamera2 import Picamera2
            # Picamera2 may not be available on dev machine, so try
            self._picam2 = Picamera2()
            # Configure: main stream BGR/RGB for processing
            # Use preview size larger, but we capture at configured size
            cam_config = self._picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                lores={"size": (320, 240), "format": "YUV420"},
                controls={
                    # Autofocus handling for Module 3
                    "AfMode": 2,  # 0 manual, 1 auto, 2 continuous per libcamera
                    # "AfRange": 0,
                },
            )
            self._picam2.configure(cam_config)
            self._picam2.start()
            # Warmup AF
            time.sleep(1.5)
            logger.info(f"Picamera2 initialized {self.width}x{self.height} @ {self.fps}fps, AF continuous")
            self.backend = "picamera2"
            return
        except Exception as e:
            logger.warning(f"Picamera2 not available or failed ({e}), falling back to OpenCV")
            self._picam2 = None

        # OpenCV fallback
        for idx in (0, 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self._cap = cap
                self.backend = f"opencv:{idx}"
                logger.info(f"OpenCV camera backend opened index {idx}")
                return
        raise RuntimeError("No camera found via picamera2 nor OpenCV")

    def read(self) -> Optional[np.ndarray]:
        """Return single BGR frame or None - trixie libcamera v0.7.1 fix: RGB888 returns BGR order"""
        if self.backend == "picamera2":
            try:
                # On Pi OS trixie libcamera v0.7.1+rpt20260609, main RGB888 actually returns BGR
                # Your tests: person right + yellow→blue / green stays green confirms double-swap
                # So return directly without RGB2BGR conversion - already BGR for OpenCV
                # If colors look wrong, toggle the commented conversion below
                arr = self._picam2.capture_array("main")
                if arr.ndim == 3:
                    # Fixed: don't convert, arr is already BGR on this stack
                    bgr = arr
                    # bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # use this if red↔blue swapped
                    if not bgr.flags['C_CONTIGUOUS']:
                        bgr = np.ascontiguousarray(bgr)
                    return bgr
                return arr
            except Exception as e:
                logger.error(f"Picamera2 capture failed: {e}")
                return None
        else:
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
            if not ok:
                return None
            return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Infinite generator yielding frames throttled to target FPS"""
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
        except:
            pass
        try:
            if self._cap:
                self._cap.release()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# quick test helper
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam = CameraInterface()
    for i, f in enumerate(cam.frames()):
        print(f"frame {i}: {f.shape}")
        cv2.imwrite(f"test_{i}.jpg", f)
        if i >= 2:
            break
    cam.close()
