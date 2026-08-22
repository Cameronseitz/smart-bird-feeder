"""
Main pipeline: Motion -> Bird Detector -> Tracker -> Best Frame -> Species -> DB
"""
import sys
import time
import logging
import signal
from pathlib import Path

import cv2
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.camera import CameraInterface
from src.pipeline import MotionDetector, BirdDetector, VisitTracker, BestFrameSelector, SpeciesClassifier
from src.pipeline.best_frame_selector import compute_sharpness
from src.database import BirdDatabase

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


class BirdFeederSystem:
    def __init__(self):
        # Components
        self.camera = None
        self.motion_detector = MotionDetector()
        self.bird_detector = BirdDetector()
        self.tracker = VisitTracker()
        self.frame_selector = BestFrameSelector()
        self.species_classifier = SpeciesClassifier()
        self.db = BirdDatabase()

        # Ensure dirs
        config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        config.BEST_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "assets").mkdir(exist_ok=True)

        self.running = False
        self.frame_count = 0
        self.last_best_save = None

        # capturing candidates timing
        self.last_candidate_capture = 0

    def _handle_shutdown(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def save_frame(self, frame, is_best=False, session_id=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if is_best:
            path = config.BEST_DIR / f"{ts}_{session_id}_best.jpg"
        else:
            if config.SAVE_ALL_CANDIDATES:
                path = config.CAPTURES_DIR / f"{ts}_{session_id}.jpg"
            else:
                return None
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # Also update latest.jpg for dashboard
        latest = BASE_DIR / "assets" / "latest.jpg"
        cv2.imwrite(str(latest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # relative path for DB (assets/best/xxx)
        rel = path.relative_to(BASE_DIR)
        return str(rel).replace("\\", "/")

    def finalize_session(self, session):
        """
        Called when a visit session has ended (expired or new bird arrived)
        - selects best frame
        - runs species classifier
        - checks confidence
        - saves DB
        """
        if not session or not session.candidates:
            logger.info(f"Session {session.session_id if session else 'unknown'} ended with 0 candidates – skipping")
            return

        best_cand = self.frame_selector.select_best(session)
        if not best_cand:
            logger.warning(f"No best frame for session {session.session_id}")
            return

        # Classify best frame
        result = self.species_classifier.classify(best_cand.frame, best_cand.detection)

        # Threshold check
        if result["confidence"] < config.SPECIES_CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence {result['confidence']:.2%} < {config.SPECIES_CONFIDENCE_THRESHOLD:.0%} for {result['raw_species']} – logging as unknown per spec")
            # result["species"] already set to unknown inside classifier if not confident
        else:
            logger.info(f"Confident ID: {result['species']} {result['confidence']:.2%}")

        # Save best frame image
        best_path = self.save_frame(best_cand.frame, is_best=True, session_id=session.session_id)

        # Optional: save all candidates if enabled debug
        if config.SAVE_ALL_CANDIDATES:
            for cand in session.candidates:
                self.save_frame(cand.frame, is_best=False, session_id=session.session_id)

        # Save to DB
        try:
            self.db.add_visit(
                session=session,
                best_candidate=best_cand,
                classification_result=result,
                best_image_path=best_path
            )
        except Exception as e:
            logger.error(f"DB save failed: {e}")

        self.last_best_save = best_path
        logger.info(f"Session {session.session_id} finalized: {result['species']} ({result['confidence']:.1%}) duration {session.duration():.1f}s candidates {len(session.candidates)} best={best_path}")

    def run(self):
        self.running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=== Smart Bird Feeder v2 starting ===")
        logger.info(f"Config: detection_thresh={config.DETECTION_THRESHOLD} species_thresh={config.SPECIES_CONFIDENCE_THRESHOLD} grace={config.GRACE_PERIOD_SEC}s motion_thresh={config.MOTION_THRESHOLD}")
        logger.info(f"Models: detection exists={config.DETECTION_MODEL_PATH.exists()} species exists={config.SPECIES_MODEL_PATH.exists()}")
        if not config.DETECTION_MODEL_PATH.exists() or not config.SPECIES_MODEL_PATH.exists():
            logger.warning("Models missing! Run: python scripts/download_models.py")

        try:
            self.camera = CameraInterface()
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            return

        logger.info("Entering main loop – waiting for motion...")
        idle_log_last = time.time()

        try:
            for frame in self.camera.frames():
                if not self.running:
                    break
                self.frame_count += 1

                # Stage 1: Motion gate
                motion = self.motion_detector.detect(frame)
                if not motion:
                    # Check if active session expired while no motion (maybe bird left)
                    if self.tracker.active_session:
                        closed = None
                        if self.tracker.active_session.is_expired():
                            closed = self.tracker.force_close()
                            if closed:
                                self.finalize_session(closed)
                    # periodic idle log
                    if time.time() - idle_log_last > 30:
                        logger.info(f"Idle – no motion, frame #{self.frame_count}, last motion {self.motion_detector.time_since_motion():.1f}s ago")
                        idle_log_last = time.time()
                    continue

                # Stage 2: Bird detector – expensive ~600-1200ms on Pi3, gated by motion
                bird_present, detections = self.bird_detector.detect(frame)

                # Stage 3: Visit tracking – IoU
                result_session, is_new = self.tracker.update(frame, detections if bird_present else [])

                # result_session can be:
                # - active continuing session (closed=False)
                # - closed old session when new bird arrived (closed=True) or expired
                active_for_candidates = None

                if not bird_present:
                    # no bird this frame, but tracker may have returned expired closed session
                    if result_session and getattr(result_session, "closed", False):
                        self.finalize_session(result_session)
                    continue

                # bird present path
                if result_session is None:
                    # should not happen when bird_present, but guard
                    continue

                if getattr(result_session, "closed", False):
                    # Old session closed due to low IoU (new bird arrived)
                    # Finalize old
                    self.finalize_session(result_session)
                    # New active session is in tracker.active_session
                    active_for_candidates = self.tracker.active_session
                else:
                    active_for_candidates = result_session

                if active_for_candidates is None:
                    continue

                # Stage 4: Best-Frame capture every 200-300ms while session active
                now = time.time()
                if now - self.last_candidate_capture >= config.CAPTURE_INTERVAL_SEC:
                    self.last_candidate_capture = now
                    best_det = max(detections, key=lambda d: d["score"]) if detections else None
                    if best_det:
                        sharp = compute_sharpness(frame, best_det["bbox_px"])
                        active_for_candidates.add_candidate(frame, best_det, sharp)
                        logger.debug(f"Added candidate to {active_for_candidates.session_id}: sharp={sharp:.1f} bbox={best_det['bbox_px']}")
                        latest_path = BASE_DIR / "assets" / "latest.jpg"
                        cv2.imwrite(str(latest_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                # Also check if current active session expired? (handled at next loop or no-motion)
                # Optional immediate expiry check for long sessions handled in tracker.is_expired()

        except Exception as e:
            logger.exception(f"Main loop crashed: {e}")
        finally:
            # Close any open session
            if self.tracker.active_session:
                closing = self.tracker.force_close()
                if closing and closing.candidates:
                    self.finalize_session(closing)
            if self.camera:
                self.camera.close()
            self.db.close()
            logger.info("Bird Feeder system stopped.")

if __name__ == "__main__":
    system = BirdFeederSystem()
    system.run()
