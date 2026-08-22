"""
Stage 3: Visit Tracking - IoU + centroid
"""
import sys
import time
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return 0.0 if union == 0 else interArea / union


def centroid(box: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centroid_distance(boxA, boxB) -> float:
    cA = centroid(boxA)
    cB = centroid(boxB)
    return math.hypot(cA[0] - cB[0], cA[1] - cB[1])


@dataclass
class CandidateFrame:
    timestamp: float
    frame: np.ndarray = field(repr=False)
    detection: Dict
    sharpness: float = 0.0
    bbox_size: float = 0.0


@dataclass
class VisitSession:
    session_id: str
    start_time: float
    last_detection_time: float
    last_bbox: List[int]
    candidates: List[CandidateFrame] = field(default_factory=list)
    closed: bool = False

    def add_candidate(self, frame_idx: np.ndarray, detection: Dict, sharpness: float):
        x1, y1, x2, y2 = detection["bbox_px"]
        area = (x2 - x1) * (y2 - y1)
        cf = CandidateFrame(
            timestamp=time.time(),
            frame=frame_idx.copy(),
            detection=detection,
            sharpness=sharpness,
            bbox_size=area,
        )
        self.candidates.append(cf)
        self.last_detection_time = cf.timestamp
        self.last_bbox = [x1, y1, x2, y2]

    def is_expired(self, grace=config.GRACE_PERIOD_SEC, max_duration=config.MAX_SESSION_DURATION_SEC) -> bool:
        now = time.time()
        return (now - self.last_detection_time > grace) or (now - self.start_time > max_duration)

    def duration(self) -> float:
        return self.last_detection_time - self.start_time


class VisitTracker:
    def __init__(self, iou_threshold=config.IOU_THRESHOLD, grace_period=config.GRACE_PERIOD_SEC):
        self.iou_threshold = iou_threshold
        self.grace_period = grace_period
        self.active_session: Optional[VisitSession] = None
        self.session_counter = 0

    def _new_session_id(self) -> str:
        self.session_counter += 1
        return f"visit_{int(time.time())}_{self.session_counter}"

    def update(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[Optional[VisitSession], bool]:
        now = time.time()

        if not detections:
            if self.active_session and self.active_session.is_expired(self.grace_period):
                closed = self.active_session
                closed.closed = True
                self.active_session = None
                logger.info(f"Session {closed.session_id} expired {closed.duration():.1f}s {len(closed.candidates)} cand")
                return closed, False
            return self.active_session, False

        best_det = max(detections, key=lambda d: d["score"])
        current_bbox = best_det["bbox_px"]

        if self.active_session is None:
            sess = VisitSession(
                session_id=self._new_session_id(),
                start_time=now,
                last_detection_time=now,
                last_bbox=current_bbox,
            )
            self.active_session = sess
            logger.info(f"New session {sess.session_id}")
            return sess, True

        iou = compute_iou(self.active_session.last_bbox, current_bbox)
        dist = centroid_distance(self.active_session.last_bbox, current_bbox)
        logger.debug(f"IoU={iou:.2f} dist={dist:.1f}")

        if iou >= self.iou_threshold or dist < config.CENTER_MOVE_THRESHOLD:
            self.active_session.last_detection_time = now
            self.active_session.last_bbox = current_bbox
            return self.active_session, False
        else:
            closed = self.active_session
            closed.closed = True
            logger.info(f"Session {closed.session_id} ended IoU {iou:.2f} move {dist:.0f}px")
            new_sess = VisitSession(
                session_id=self._new_session_id(),
                start_time=now,
                last_detection_time=now,
                last_bbox=current_bbox,
            )
            self.active_session = new_sess
            return closed, True

    def force_close(self) -> Optional[VisitSession]:
        if self.active_session:
            closed = self.active_session
            closed.closed = True
            self.active_session = None
            logger.info(f"Force closed {closed.session_id}")
            return closed
        return None
