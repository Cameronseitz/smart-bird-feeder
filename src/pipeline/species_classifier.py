"""
Stage 5: Species Classifier - AIY birds_V1_3 964 + 1 background = 965 outputs
"""
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

logger = logging.getLogger(__name__)

Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter

    Interpreter = LiteInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as LiteInterpreter

        Interpreter = LiteInterpreter
    except ImportError:
        Interpreter = None


class SpeciesClassifier:
    def __init__(
        self,
        model_path=config.SPECIES_MODEL_PATH,
        labels_path=config.SPECIES_LABELS_PATH,
        input_size=config.SPECIES_INPUT_SIZE,
        threshold=config.SPECIES_CONFIDENCE_THRESHOLD,
        top_k=config.SPECIES_TOP_K,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.input_size = input_size
        self.threshold = threshold
        self.top_k = top_k
        self.labels: List[str] = []
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_labels()
        self._load_model()

    def _load_labels(self):
        if not self.labels_path.exists():
            logger.warning(f"Labels missing {self.labels_path}")
            self.labels = [f"species_{i}" for i in range(964)]
            return
        try:
            with open(self.labels_path, encoding="utf-8") as f:
                raw = [l.strip() for l in f if l.strip()]
            cleaned = []
            for line in raw:
                if ":" in line and line.split(":", 1)[0].strip().isdigit():
                    cleaned.append(line.split(":", 1)[1].strip())
                else:
                    cleaned.append(line)
            self.labels = cleaned
            logger.info(f"Loaded {len(self.labels)} labels")
        except Exception as e:
            logger.error(f"Label load failed: {e}")
            self.labels = [f"species_{i}" for i in range(964)]

    def _load_model(self):
        if Interpreter is None:
            logger.error("No TFLite runtime")
            return
        # primary + ImageNet fallback path
        candidates = [
            self.model_path,
            BASE_DIR / "models" / "mobilenet_v2_1.0_224_quant.tflite",
            BASE_DIR / "models" / "birds_V1_3.tflite",
        ]
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            logger.error(f"Species model missing {self.model_path}")
            return
        if chosen != self.model_path:
            logger.warning(f"Using fallback {chosen}")
            self.model_path = chosen
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Species model loaded {self.model_path}")
        except Exception as e:
            logger.error(f"Load failed {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray, bbox_px: List[int] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        if bbox_px:
            x1, y1, x2, y2 = bbox_px
            pad_w = int((x2 - x1) * 0.2)
            pad_h = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame
        resized = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

    def classify(self, frame: np.ndarray, detection: Dict = None) -> Dict:
        if self.interpreter is None:
            return {"species": "unknown", "confidence": 0.0, "top_k": [], "is_confident": False}

        bbox = detection["bbox_px"] if detection and "bbox_px" in detection else None
        input_data = self._preprocess(frame, bbox)

        start = time.time()
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"species": "unknown", "confidence": 0.0, "top_k": [], "is_confident": False}

        probs = output.astype(np.float32)
        out_detail = self.output_details[0]
        if out_detail["dtype"] in (np.uint8, np.int8):
            scale, zp = out_detail["quantization"]
            if scale == 0:
                scale = 0.00390625
            probs = (probs - zp) * scale

        # 965 vs 964 - background handling (Kaggle v3 has background at 0)
        background_prob = 0.0
        if len(probs) == len(self.labels) + 1:
            background_prob = float(probs[0])
            probs = probs[1:]  # drop background at idx 0
            logger.debug(f"965->964 drop bg prob {background_prob:.2%}")

        # Ensure probabilities
        if probs.min() < -0.05 or probs.max() > 1.2 or not (0.5 <= probs.sum() <= 1.5):
            # softmax for logits case
            m = np.max(probs)
            e = np.exp(np.clip(probs - m, -50, 50))
            s = e.sum()
            probs = e / s if s > 0 else np.ones_like(probs) / len(probs)

        probs = np.nan_to_num(probs, nan=0.0)
        if probs.sum() > 0:
            probs = probs / probs.sum()

        if background_prob > 0.5:
            top_idx = np.argsort(probs)[::-1][: self.top_k]
            top_k = [(self.labels[i] if i < len(self.labels) else f"species_{i}", float(probs[i])) for i in top_idx]
            return {
                "species": "unknown",
                "raw_species": "background",
                "confidence": background_prob,
                "top_k": top_k,
                "is_confident": False,
                "inference_ms": (time.time() - start) * 1000,
            }

        top_idx = np.argsort(probs)[::-1][: self.top_k]
        top_k = [(self.labels[i] if i < len(self.labels) else f"species_{i}", float(probs[i])) for i in top_idx]
        species, conf = top_k[0] if top_k else ("unknown", 0.0)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Species: {species} {conf:.2%} top3={top_k} {elapsed:.0f}ms")

        return {
            "species": species if conf >= self.threshold else "unknown",
            "raw_species": species,
            "confidence": float(conf),
            "top_k": top_k,
            "is_confident": conf >= self.threshold,
            "inference_ms": elapsed,
        }

    def classify_from_path(self, image_path: str) -> Dict:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"species": "unknown", "confidence": 0.0, "top_k": [], "is_confident": False}
        return self.classify(frame)
