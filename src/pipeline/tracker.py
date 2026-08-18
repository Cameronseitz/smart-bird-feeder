"""
Stage 3: Visit Session Tracking
- Determines same bird still present vs new visit via IoU
- Session concept: continuous presence; ends after grace period of no detection
- Doubles as cooldown/debounce.

Per blueprint Section 4:
- IoU tracking
- Short grace 3-5 sec
- High-traffic merge caveat: shorten grace, consider centroid tracking (we do both)

"""
import time
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    box: [x1,y1,x2,y2] pixel
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

def centroid(box: List[int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def centroid_distance(boxA, boxB) -> float:
    cA = centroid(boxA)
    cB = centroid(boxB)
    return math.hypot(cA[0]-cB[0], cA[1]-cB[1])

@dataclass
class CandidateFrame:
    timestamp: float
    frame: np.ndarray = field(repr=False)  # full frame for later crop scoring
    detection: Dict
    sharpness: float = 0.0
    bbox_size: float = 0.0
    # computed score later

@dataclass
class VisitSession:
    session_id: str
    start_time: float
    last_detection_time: float
    last_bbox: List[int]  # x1,y1,x2,y2
    candidates: List[CandidateFrame] = field(default_factory=list)
    closed: bool = False

    def add_candidate(self, frame_idx: np.ndarray, detection: Dict, sharpness: float):
        # bbox size normalized (area relative)
        x1,y1,x2,y2 = detection["bbox_px"]
        area = (x2-x1)*(y2-y1)
        cf = CandidateFrame(
            timestamp=time.time(),
            frame=frame_idx.copy(),
            detection=detection,
            sharpness=sharpness,
            bbox_size=area
        )
        self.candidates.append(cf)
        self.last_detection_time = cf.timestamp
        self.last_bbox = [x1,y1,x2,y2]

    def is_expired(self, grace=config.GRACE_PERIOD_SEC, max_duration=config.MAX_SESSION_DURATION_SEC) -> bool:
        now = time.time()
        if now - self.last_detection_time > grace:
            return True
        if now - self.start_time > max_duration:
            return True
        return False

    def duration(self) -> float:
        return self.last_detection_time - self.start_time

class VisitTracker:
    def __init__(self, iou_threshold=config.IOU_THRESHOLD, grace_period=config.GRACE_PERIOD_SEC):
        self.iou_threshold = iou_threshold
        self.grace_period = grace_period
        self.active_session: Optional[VisitSession] = None
        self.session_counter = 0
        self._last_cleanup = time.time()

    def _new_session_id(self) -> str:
        self.session_counter += 1
        return f"visit_{int(time.time())}_{self.session_counter}"

    def update(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[Optional[VisitSession], bool]:
        """
        Update tracker with new detections
        Returns (active_or_just_closed_session, is_new_session_bool)
        If no detections -> check expiry; if expired returns closed session to finalize.
        If detections but no match to active -> old session should be closed (return it) and new started.
        """
        now = time.time()

        if not detections:
            if self.active_session and self.active_session.is_expired(self.grace_period):
                closed = self.active_session
                closed.closed = True
                self.active_session = None
                logger.info(f"Session {closed.session_id} expired after {closed.duration():.1f}s with {len(closed.candidates)} candidates")
                return closed, False
            return self.active_session, False

        # pick best detection (highest score)
        best_det = max(detections, key=lambda d: d["score"])
        current_bbox = best_det["bbox_px"]

        if self.active_session is None:
            # start new session
            sess = VisitSession(
                session_id=self._new_session_id(),
                start_time=now,
                last_detection_time=now,
                last_bbox=current_bbox
            )
            self.active_session = sess
            logger.info(f"New session started: {sess.session_id}")
            return sess, True
        else:
            # compare IoU and centroid distance to last bbox
            iou = compute_iou(self.active_session.last_bbox, current_bbox)
            dist = centroid_distance(self.active_session.last_bbox, current_bbox)
            logger.debug(f"IoU={iou:.2f} dist={dist:.1f} between {self.active_session.last_bbox} and {current_bbox}")

            # High overlap OR small movement -> same session
            # Use both metrics to handle high-traffic caveat
            if iou >= self.iou_threshold or dist < config.CENTER_MOVE_THRESHOLD:
                # continue
                self.active_session.last_detection_time = now
                self.active_session.last_bbox = current_bbox
                return self.active_session, False
            else:
                # possible new bird arrived - but need to respect grace? 
                # If gap short, we still may be merging departing+arriving per blueprint caveat.
                # Mitigation: if time since last detection < grace and IoU low and distance large -> treat as NEW visit
                # Close old session now
                closed = self.active_session
                closed.closed = True
                logger.info(f"Session {closed.session_id} ended due to low IoU ({iou:.2f}) / large move ({dist:.0f}px), duration {closed.duration():.1f}s")
                # start new
                new_sess = VisitSession(
                    session_id=self._new_session_id(),
                    start_time=now,
                    last_detection_time=now,
                    last_bbox=current_bbox
                )
                self.active_session = new_sess
                return closed, True  # return closed old session, but active is new
                # Caller should handle closed first, then new will be active on next loop

    def force_close(self) -> Optional[VisitSession]:
        if self.active_session:
            closed = self.active_session
            closed.closed = True
            self.active_session = None
            logger.info(f"Force closed session {closed.session_id}")
            return closed
        return None
