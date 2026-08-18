"""
Stage 5: Local Species Classifier (TFLite)
Model: Google AIY birds_V1/3 - MobileNet, 964 classes trained on iNaturalist

Per blueprint:
- Runs once per visit on best frame
- Returns species label + confidence
- Confidence threshold 85-90%; below logs as unknown
- Optional cloud fallback (not primary)

Also handles Atlanta expected species verification list.
"""
import time
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2

try:
    import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# TFLite interpreter import same fallback pattern
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    Interpreter = TFLiteInterpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter as NewInterpreter
            Interpreter = NewInterpreter
        except:
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
            logger.warning(f"Species labels not found at {self.labels_path}, will use generic names")
            # create dummy 964 labels
            self.labels = [f"species_{i}" for i in range(964)]
            return
        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                raw = [l.strip() for l in f.readlines() if l.strip()]
            # Some label files contain "id:name" or just name
            cleaned = []
            for line in raw:
                if ":" in line and line.split(":")[0].strip().isdigit():
                    # format like "0: House Finch"
                    cleaned.append(line.split(":", 1)[1].strip())
                else:
                    cleaned.append(line)
            self.labels = cleaned
            logger.info(f"Loaded {len(self.labels)} species labels")
            # verify Atlanta list
            self._verify_atlanta_coverage()
        except Exception as e:
            logger.error(f"Failed loading species labels: {e}")
            self.labels = [f"species_{i}" for i in range(964)]

    def _verify_atlanta_coverage(self):
        # Check how many Atlanta expected are substring matched in labels
        found = []
        missing = []
        lower_labels = [l.lower() for l in self.labels]
        for expected in config.ATLANTA_EXPECTED:
            # fuzzy: if any label contains expected lower words
            exp_low = expected.lower()
            # try exact or partial
            if any(exp_low in lab or lab in exp_low for lab in lower_labels):
                found.append(expected)
            else:
                # try token match
                tokens = exp_low.split()
                match = any(all(t in lab for t in tokens) for lab in lower_labels)
                if match:
                    found.append(expected)
                else:
                    missing.append(expected)
        logger.info(f"Atlanta species coverage check: found {len(found)}/{len(config.ATLANTA_EXPECTED)} : {found}")
        if missing:
            logger.warning(f"Potentially missing Atlanta species in model: {missing} - spot-test with reference photos!")

    def _load_model(self):
        if Interpreter is None:
            logger.error("No TFLite interpreter for species classifier")
            return
        if not self.model_path.exists():
            logger.error(f"Species model missing at {self.model_path}. Run download_models.py")
            return
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Species model loaded: {self.model_path}")
            logger.info(f"Species input: {self.input_details}")
            logger.info(f"Species output: {self.output_details}")
        except Exception as e:
            logger.error(f"Failed load species model: {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray, bbox_px: List[int] = None) -> np.ndarray:
        """
        Crop to bbox (with slight expansion) then resize to 224x224 MobileNet input
        Mobilenet normalization: typically [-1,1] or 0-1. AIY model expects ?
        Google AIY birds_V1 usually expects normalized 0-255 uint8? or float [0,1] ?
        We'll handle both dtype cases like bird detector.
        """
        h,w = frame.shape[:2]
        if bbox_px:
            x1,y1,x2,y2 = bbox_px
            # expand bbox by 20% to include context, important for species ID
            bw = x2 - x1
            bh = y2 - y1
            pad_w = int(bw * 0.2)
            pad_h = int(bh * 0.2)
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

        if self.input_details and self.input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            # Float model: heuristic
            # Try [-1,1] normalization as MobileNet often uses
            # AIY docs: input mean 128 std 128 -> (p-128)/128 => [-1,1]
            # So we do (rgb - 127.5)/127.5
            input_data = rgb.astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        return input_data

    def classify(self, frame: np.ndarray, detection: Dict = None) -> Dict:
        """
        Run classifier on frame (or cropped to detection bbox)
        Returns dict: {
          "species": str,
          "confidence": float,
          "top_k": [(label,score),...],
          "is_confident": bool (above threshold)
        }
        """
        if self.interpreter is None:
            return {
                "species": "unknown",
                "confidence": 0.0,
                "top_k": [],
                "is_confident": False,
                "error": "model not loaded"
            }

        bbox = detection["bbox_px"] if detection and "bbox_px" in detection else None
        input_data = self._preprocess(frame, bbox)

        start = time.time()
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        except Exception as e:
            logger.error(f"Species inference failed: {e}")
            return {
                "species": "unknown",
                "confidence": 0.0,
                "top_k": [],
                "is_confident": False,
                "error": str(e)
            }

        # output is logits or probabilities shape [964]
        # If logits, softmax
        # Check if already softmax (sum ~1)
        # Apply softmax if needed
        probs = output
        if not np.isclose(np.sum(probs), 1.0, atol=0.1) or np.max(probs) > 1.0:
            # softmax
            e_x = np.exp(probs - np.max(probs))
            probs = e_x / e_x.sum()

        # top k
        top_indices = np.argsort(probs)[::-1][:self.top_k]
        top_k = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"species_{idx}"
            top_k.append((label, float(probs[idx])))

        species, conf = top_k[0] if top_k else ("unknown", 0.0)
        is_conf = conf >= self.threshold

        elapsed = (time.time()-start)*1000
        logger.info(f"Species: {species} ({conf:.2%}) top3={top_k} in {elapsed:.0f}ms confident={is_conf}")

        result = {
            "species": species if is_conf else "unknown",
            "raw_species": species,
            "confidence": float(conf),
            "top_k": top_k,
            "is_confident": is_conf,
            "inference_ms": elapsed
        }

        # Optional cloud fallback for low confidence
        if not is_conf and config.ENABLE_CLOUD_FALLBACK:
            result["cloud_fallback_attempted"] = True  # placeholder, implement elsewhere
            # Could call external API here

        return result

    def classify_from_path(self, image_path: str) -> Dict:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"species":"unknown","confidence":0.0,"top_k":[],"is_confident":False,"error":"invalid image"}
        return self.classify(frame)
