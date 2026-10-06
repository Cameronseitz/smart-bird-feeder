"""
Stage 4: Best-Frame Selection
"""
import sys
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)


def compute_sharpness(frame: np.ndarray, bbox_px: List[int] = None) -> float:
    try:
        if bbox_px:
            x1, y1, x2, y2 = bbox_px
            h, w = frame.shape[:2]
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
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        logger.debug(f"Sharpness failed: {e}")
        return 0.0


class BestFrameSelector:
    def __init__(
        self,
        sharpness_weight=config.SHARPNESS_WEIGHT,
        size_weight=config.SIZE_WEIGHT,
        position_weight=config.POSITION_WEIGHT,
        edge_margin=config.EDGE_CLIP_MARGIN,
        blur_thresh=config.SHARPNESS_BLUR_THRESHOLD,
        good_ref=getattr(config, "SHARPNESS_GOOD_REF", 500.0),
        max_bbox_frac=getattr(config, "MAX_BBOX_FRACTION", 0.55),
    ):
        self.w_sharp = sharpness_weight
        self.w_size = size_weight
        self.w_pos = position_weight
        self.edge_margin = edge_margin
        self.blur_thresh = blur_thresh
        self.good_ref = good_ref
        self.max_bbox_frac = max_bbox_frac

    def score_candidate(self, candidate, frame_shape) -> float:
        sharp = candidate.sharpness
        if sharp < self.blur_thresh:
            # Below blur floor: heavy penalty, linear 0..0.2
            sharp_norm = (sharp / max(self.blur_thresh, 1e-6)) * 0.2
        else:
            # 0.2 at threshold -> 1.0 at good_ref
            sharp_norm = 0.2 + 0.8 * min(1.0, sharp / max(self.good_ref, 1e-6))

        h, w = frame_shape[:2]
        size_frac = candidate.bbox_size / max(h * w, 1)
        size_norm = min(1.0, size_frac * 10)
        # Too-close penalty: bbox filling >55% of frame is inside the
        # Module 3 min-focus distance - almost certainly out of focus.
        close_penalty = 1.0
        if size_frac > self.max_bbox_frac:
            over = (size_frac - self.max_bbox_frac) / max(1.0 - self.max_bbox_frac, 1e-6)
            close_penalty = max(0.2, 1.0 - over * 0.8)

        det = candidate.detection
        x1, y1, x2, y2 = det["bbox_px"]
        left = x1 / w
        right = (w - x2) / w
        top = y1 / h
        bottom = (h - y2) / h
        min_edge_dist = min(left, right, top, bottom)
        if min_edge_dist < self.edge_margin:
            pos_score = min_edge_dist / self.edge_margin
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dx = abs(cx - w / 2) / (w / 2)
            dy = abs(cy - h / 2) / (h / 2)
            pos_score = 1.0 - ((dx + dy) / 2) * 0.5

        pos_score = max(0.0, min(1.0, pos_score))
        base = (self.w_sharp * sharp_norm) + (self.w_size * size_norm) + (self.w_pos * pos_score)
        return base * close_penalty

    def select_best(self, session) -> Optional[object]:
        if not session.candidates:
            logger.warning(f"Session {session.session_id} no candidates")
            return None
        if len(session.candidates) == 1:
            return session.candidates[0]

        best = None
        best_score = -1
        frame_shape = session.candidates[0].frame.shape
        for cand in session.candidates:
            score = self.score_candidate(cand, frame_shape)
            if score > best_score:
                best_score = score
                best = cand

        logger.info(f"Best for {session.session_id}: sharp={best.sharpness:.1f} score={best_score:.3f} n={len(session.candidates)}")
        return best
