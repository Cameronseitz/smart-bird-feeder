"""
Stage 2: Local Object Detector Gate (TFLite)
- Confirms "bird" class before passing to tracking/classifier
- Per spec: MobileNetV2-SSD or YOLOv8 Nano, >70% threshold, ~600ms-1.2s on Pi3

This implementation uses TFLite quantized SSD MobileNet V2 COCO.
Bird class detection is via label file scan for "bird" OR fallback IDs.

Falls back gracefully if model missing -> always returns False (safe) but logs.
Supports both tflite-runtime and tensorflow.lite Interpreter.
"""
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# TFLite import with fallbacks - trixie/py3.13 prefers ai-edge-litert
Interpreter = None
try:
    # New official runtime for 3.13 trixie (has aarch64 wheels on piwheels)
    from ai_edge_litert.interpreter import Interpreter as NewInterpreter
    Interpreter = NewInterpreter
    logger.info("Using ai_edge_litert.Interpreter (trixie/py3.13 preferred)")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        Interpreter = TFLiteInterpreter
        logger.info("Using tflite_runtime.Interpreter")
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            logger.info("Using tensorflow.lite.Interpreter (heavy, fallback)")
        except ImportError:
            try:
                # older import path for ai-edge-litert
                from ai_edge_litert import Interpreter as NewInterpreter2
                Interpreter = NewInterpreter2
                logger.info("Using ai_edge_litert.Interpreter (alt import)")
            except:
                logger.warning("No TFLite interpreter found! Install ai-edge-litert (pip install ai-edge-litert) or tflite-runtime")
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
        self.bird_class_ids = self._find_bird_class_ids()
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()

    def _load_labels(self) -> List[str]:
        if not self.labels_path.exists():
            logger.warning(f"Label file not found: {self.labels_path}, using fallback IDs {config.DETECTION_BIRD_CLASS_IDS}")
            return []
        try:
            with open(self.labels_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(labels)} labels from {self.labels_path}")
            return labels
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return []

    def _find_bird_class_ids(self) -> List[int]:
        ids = []
        if self.labels:
            for idx, name in enumerate(self.labels):
                if "bird" in name.lower():
                    ids.append(idx)
        if not ids:
            logger.info(f"No 'bird' in label file, using fallback {config.DETECTION_BIRD_CLASS_IDS}")
            ids = config.DETECTION_BIRD_CLASS_IDS
        logger.info(f"Bird class IDs: {ids} -> labels: {[self.labels[i] if i < len(self.labels) else str(i) for i in ids]}")
        return ids

    def _load_model(self):
        if Interpreter is None:
            logger.error("No Interpreter available")
            return
        if not self.model_path.exists():
            logger.error(f"Detection model missing at {self.model_path}. Run scripts/download_models.py")
            return
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Detection model loaded: {self.model_path}")
            logger.info(f"Input details: {self.input_details}")
            logger.info(f"Output details: {self.output_details}")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # SSD expects 300x300, RGB, normalized? quantized model expects uint8
        # Check input dtype
        input_detail = self.input_details[0] if self.input_details else None
        h, w = self.input_size, self.input_size
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if input_detail and input_detail['dtype'] == np.uint8:
            # quantized -> keep 0-255 uint8
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            # float model expects float32 normalized to [-1,1] or 0-1 depending
            # Mobilenet SSD float often expects normalized to 0-1 or -1,1
            # We'll normalize to 0-1 then typical SSD scaling
            input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
            # Some models expect 0-255 but float; heuristic: if input mean around quantization, handle
        return input_data

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Dict]]:
        """
        Returns (is_bird_present, detections_list)
        detections_list: [{"bbox":[ymin,xmin,ymax,xmax] normalized, "score":float, "class":int, "label":str}]
        BBox in pixel coordinates also added as "bbox_px": [x,y,w,h] or [x1,y1,x2,y2]
        """
        if self.interpreter is None:
            return False, []

        start = time.time()
        input_data = self._preprocess(frame)

        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return False, []

        # Parse outputs - SSD models typically 4 outputs: locations, classes, scores, num
        # But order varies. We'll inspect output details.
        try:
            # Common TF Lite SSD output order:
            # 0: locations [1,10,4]
            # 1: classes [1,10]
            # 2: scores [1,10]
            # 3: num_detections [1]
            # Some quantized have same
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            num = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0]) if len(self.output_details) > 3 else len(scores)

            # Some models swap boxes/classes - heuristic: boxes shape last dim 4
            # Ensure boxes is [N,4]
            if boxes.ndim == 1:
                # maybe output order different, try re-order
                # try to find tensor with shape [..4]
                for i, detail in enumerate(self.output_details):
                    t = self.interpreter.get_tensor(detail['index'])[0]
                    if t.ndim >= 2 and t.shape[-1] == 4:
                        boxes = t
                    elif t.ndim == 1 and len(t) <= 100:
                        # could be counts
                        pass
                # re-fetch
                # fallback
                pass

        except Exception as e:
            logger.error(f"Output parsing failed: {e}")
            # Try generic parsing: find 4 arrays
            boxes, classes, scores, num = None, None, None, 10
            for detail in self.output_details:
                tensor = self.interpreter.get_tensor(detail['index'])[0]
                if tensor.ndim == 2 and tensor.shape[1] == 4 or tensor.ndim == 3 and tensor.shape[2] == 4:
                    boxes = tensor[0] if tensor.ndim == 3 else tensor
                elif tensor.size <= 20 and np.issubdtype(tensor.dtype, np.floating):
                    # might be scores or classes size
                    pass
            return False, []

        h, w = frame.shape[:2]
        detections = []
        bird_present = False

        count = min(num, len(scores)) if isinstance(num, int) else len(scores)
        for i in range(count):
            try:
                score = float(scores[i])
                if score < self.threshold:
                    continue
                class_id = int(classes[i])
                # Handle 1-indexed vs 0-indexed mismatch
                # Check both raw and raw-1 for bird match if needed
                is_bird = False
                if self.labels:
                    if class_id < len(self.labels) and "bird" in self.labels[class_id].lower():
                        is_bird = True
                    # also check surrounding due to indexing quirks
                    for bid in self.bird_class_ids:
                        if class_id == bid or class_id == bid+1 or class_id == bid-1:
                            # double-check label if available
                            if bid < len(self.labels) and "bird" in self.labels[bid].lower():
                                is_bird = True
                                break
                else:
                    if class_id in self.bird_class_ids or (class_id-1) in self.bird_class_ids or (class_id+1) in self.bird_class_ids:
                        is_bird = True

                if not is_bird:
                    continue

                # BBox: SSD returns [ymin, xmin, ymax, xmax] normalized 0-1
                ymin, xmin, ymax, xmax = boxes[i]
                # convert to pixel
                x1 = int(xmin * w)
                y1 = int(ymin * h)
                x2 = int(xmax * w)
                y2 = int(ymax * h)
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

                det = {
                    "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                    "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_xywh": [int(x1), int(y1), int(bw), int(bh)],
                    "score": score,
                    "class_id": class_id,
                    "label": label,
                }
                detections.append(det)
                bird_present = True
            except Exception as e:
                logger.debug(f"Det loop error idx {i}: {e}")
                continue

        inference_ms = (time.time() - start) * 1000
        if inference_ms > config.DETECTION_MAX_INFERENCE_MS:
            logger.warning(f"Slow inference {inference_ms:.0f}ms > {config.DETECTION_MAX_INFERENCE_MS}ms (expected on Pi3)")

        if bird_present:
            logger.info(f"Bird detected: {len(detections)} boxes, inf {inference_ms:.0f}ms, scores {[d['score'] for d in detections]}")

        return bird_present, detections

# Optional YOLOv8 Nano wrapper if user prefers ultralytics
class YOLOBirdDetector:
    def __init__(self, model_path="models/yolov8n_bird.tflite", threshold=config.DETECTION_THRESHOLD):
        self.threshold = threshold
        self.model_path = Path(model_path)
        logger.info("YOLO detector not implemented yet - use BirdDetector primary. Placeholder")
