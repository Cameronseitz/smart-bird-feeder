"""
Stage 4: Best-Frame Selection within session

Per blueprint Section 4:
- Capture candidate frames every 200-300ms while bird present
- Score on:
  - Sharpness: Laplacian variance cv2.Laplacian(...).var()
  - BBox size: prefer fills more of frame
  - BBox position: prefer not clipped at edges

Run species classifier once on single winning frame.

"""
import cv2
import numpy as np
import logging
from typing import List, Optional
import time

try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

def compute_sharpness(frame: np.ndarray, bbox_px: List[int] = None) -> float:
    """
    Laplacian variance; higher = sharper
    If bbox provided, compute on cropped region (bird only) for better signal
    """
    try:
        if bbox_px:
            x1,y1,x2,y2 = bbox_px
            # expand slightly?
            h,w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(score)
    except Exception as e:
        logger.debug(f"Sharpness calc failed: {e}")
        return 0.0

class BestFrameSelector:
    def __init__(
        self,
        sharpness_weight=config.SHARPNESS_WEIGHT,
        size_weight=config.SIZE_WEIGHT,
        position_weight=config.POSITION_WEIGHT,
        edge_margin=config.EDGE_CLIP_MARGIN,
        blur_thresh=config.SHARPNESS_BLUR_THRESHOLD,
    ):
        self.w_sharp = sharpness_weight
        self.w_size = size_weight
        self.w_pos = position_weight
        self.edge_margin = edge_margin
        self.blur_thresh = blur_thresh

    def score_candidate(self, candidate, frame_shape) -> float:
        """
        candidate: CandidateFrame from tracker
        frame_shape: (h,w)
        """
        # Sharpness: raw Laplacian; normalize-ish
        # typical sharpness ranges 0-1000; log scale helps?
        sharp = candidate.sharpness
        # if below blur threshold, heavily penalize but not zero
        if sharp < self.blur_thresh:
            sharp_norm = (sharp / self.blur_thresh) * 0.2  # 0-0.2
        else:
            # normalize to roughly 0-1 assuming top 500+ sharp is excellent
            sharp_norm = min(1.0, sharp / 500.0)

        # BBox size score: area / frame area, bigger is better but not 100%
        h,w = frame_shape[:2]
        frame_area = h*w
        size_norm = min(1.0, (candidate.bbox_size / frame_area) * 10)  # multiply to make reasonable size give high score; e.g., bird 10% frame => 1.0

        # Position score: penalize if near edge
        det = candidate.detection
        x1,y1,x2,y2 = det["bbox_px"]
        # distance to edge fractions
        left = x1 / w
        right = (w - x2) / w
        top = y1 / h
        bottom = (h - y2) / h
        # if any edge closer than margin, penalize
        min_edge_dist = min(left, right, top, bottom)
        if min_edge_dist < self.edge_margin:
            pos_score = min_edge_dist / self.edge_margin  # 0-1 linear
        else:
            # prefer center - compute centroid distance to image center
            cx = (x1+x2)/2.0
            cy = (y1+y2)/2.0
            dx = abs(cx - w/2) / (w/2)
            dy = abs(cy - h/2) / (h/2)
            # center distance normalized 0-1, invert
            center_dist = (dx + dy)/2
            pos_score = 1.0 - center_dist*0.5  # slight preference for center, but not huge

        pos_score = max(0.0, min(1.0, pos_score))

        total = (self.w_sharp * sharp_norm) + (self.w_size * size_norm) + (self.w_pos * pos_score)
        return total

    def select_best(self, session) -> Optional[object]:
        """
        session: VisitSession with candidates list
        Returns best CandidateFrame or None if empty
        """
        if not session.candidates:
            logger.warning(f"Session {session.session_id} has no candidates")
            return None

        if len(session.candidates) == 1:
            return session.candidates[0]

        best = None
        best_score = -1
        # need frame shape from any candidate
        frame_shape = session.candidates[0].frame.shape

        for cand in session.candidates:
            score = self.score_candidate(cand, frame_shape)
            if score > best_score:
                best_score = score
                best = cand

        logger.info(f"Best frame for {session.session_id}: sharp={best.sharpness:.1f} size={best.bbox_size} score={best_score:.3f} from {len(session.candidates)} candidates")
        return best

    def evaluate_frame(self, frame: np.ndarray, detection: dict) -> float:
        """Utility to compute sharpness of current frame given detection"""
        return compute_sharpness(frame, detection["bbox_px"])
