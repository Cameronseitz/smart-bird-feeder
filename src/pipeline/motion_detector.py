"""
Stage 1: Motion Gate - 320x240 diff to gate inference
"""
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(
        self,
        resize_w=config.MOTION_RESIZE_W,
        resize_h=config.MOTION_RESIZE_H,
        thresh=config.MOTION_THRESHOLD,
        min_area=config.MOTION_MIN_AREA,
        blur_kernel=config.MOTION_BLUR_KERNEL,
        dilate_iters=config.MOTION_DILATE_ITERS,
    ):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.thresh = thresh
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.dilate_iters = dilate_iters
        self.avg_frame = None
        self.last_trigger = 0

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.resize_w, self.resize_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return gray

    def detect(self, frame: np.ndarray) -> bool:
        gray = self._preprocess(frame)

        if self.avg_frame is None:
            self.avg_frame = gray.astype("float")
            return False

        cv2.accumulateWeighted(gray, self.avg_frame, 0.15)
        background = cv2.convertScaleAbs(self.avg_frame)

        delta = cv2.absdiff(background, gray)
        _, thresh = cv2.threshold(delta, self.thresh, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=self.dilate_iters)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            motion_found = True
            break

        if motion_found:
            self.last_trigger = time.time()

        return motion_found

    def reset(self):
        self.avg_frame = None

    def time_since_motion(self):
        return time.time() - self.last_trigger
