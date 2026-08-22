"""
Stage 2: Bird detector - SSD MobileNetV2 COCO quantized
"""
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)

# TFLite: prefer ai-edge-litert (py3.13 trixie), fallback tflite_runtime
Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
    Interpreter = LiteInterpreter
    logger.info("Using ai_edge_litert")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as LiteInterpreter
        Interpreter = LiteInterpreter
        logger.info("Using tflite_runtime")
    except ImportError:
        logger.warning("No TFLite runtime - install ai-edge-litert")
        Interpreter = None


class BirdDetector:
    def __init__(
        self,
        model_path=config.DETECTION_MODEL_PATH,
        labels_path=config.DETECTION_LABELS_PATH,
        threshold=config.DETECTION_THRESHOLD,
        input_size=config.DETECTION_INPUT_SIZE,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = threshold
        self.input_size = input_size
        self.labels = self._load_labels()
        self.bird_class_ids = self._find_bird_ids()
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()

    def _load_labels(self) -> List[str]:
        if not self.labels_path.exists():
            logger.warning(f"Labels missing {self.labels_path}, using fallback IDs")
            return []
        try:
            with open(self.labels_path) as f:
                labels = [l.strip() for l in f if l.strip()]
            logger.info(f"Loaded {len(labels)} labels")
            return labels
        except Exception as e:
            logger.error(f"Label load failed: {e}")
            return []

    def _find_bird_ids(self) -> List[int]:
        ids = []
        if self.labels:
            for idx, name in enumerate(self.labels):
                if "bird" in name.lower():
                    ids.append(idx)
        if not ids:
            ids = config.DETECTION_BIRD_CLASS_IDS
        logger.info(f"Bird IDs: {ids}")
        return ids

    def _load_model(self):
        if Interpreter is None:
            return
        if not self.model_path.exists():
            logger.error(f"Model missing {self.model_path}")
            return
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Detection model loaded {self.model_path}")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Dict]]:
        if self.interpreter is None:
            return False, []

        start = time.time()
        input_data = self._preprocess(frame)

        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
            num = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0]) if len(self.output_details) > 3 else len(scores)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return False, []

        h, w = frame.shape[:2]
        detections = []

        for i in range(min(num, len(scores))):
            score = float(scores[i])
            if score < self.threshold:
                continue
            class_id = int(classes[i])

            is_bird = False
            if self.labels and class_id < len(self.labels) and "bird" in self.labels[class_id].lower():
                is_bird = True
            elif class_id in self.bird_class_ids:
                is_bird = True
            else:
                for bid in self.bird_class_ids:
                    if abs(class_id - bid) <= 1 and bid < len(self.labels) and "bird" in self.labels[bid].lower():
                        is_bird = True
                        break

            if not is_bird:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(max(0, xmin * w))
            y1 = int(max(0, ymin * h))
            x2 = int(min(w, xmax * w))
            y2 = int(min(h, ymax * h))
            if x2 <= x1 or y2 <= y1:
                continue

            det = {
                "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "bbox_px": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
                "class_id": class_id,
                "label": self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}",
            }
            detections.append(det)

        ms = (time.time() - start) * 1000
        if ms > config.DETECTION_MAX_INFERENCE_MS:
            logger.warning(f"Slow inference {ms:.0f}ms")
        if detections:
            logger.info(f"Bird {len(detections)} dets {ms:.0f}ms scores {[d['score'] for d in detections]}")

        return (len(detections) > 0), detections
