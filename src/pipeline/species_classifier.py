"""
Stage 5: Local Species Classifier (TFLite)
Model: Google AIY birds_V1/3 - MobileNet, 964 classes trained on iNaturalist
Kaggle v3 fix version: 3.tflite 3.4MB uint8 quantized, 965 outputs (964+background)

Fixed Feb 2024 changelog: out-of-order logit to labelmap mapping in tflite metadata.
Handles quantized output dequantization and background handling.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

try:
    import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as NewInterpreter
    Interpreter = NewInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        Interpreter = TFLiteInterpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            try:
                from ai_edge_litert import Interpreter as AltInterp
                Interpreter = AltInterp
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
            logger.warning(f"Species labels not found at {self.labels_path}, using generic names")
            self.labels = [f"species_{i}" for i in range(964)]
            return
        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                raw = [l.strip() for l in f.readlines() if l.strip()]
            cleaned = []
            for line in raw:
                if ":" in line and line.split(":", 1)[0].strip().isdigit():
                    cleaned.append(line.split(":", 1)[1].strip())
                else:
                    cleaned.append(line)
            self.labels = cleaned
            logger.info(f"Loaded {len(self.labels)} species labels")
            self._verify_atlanta_coverage()
        except Exception as e:
            logger.error(f"Failed loading species labels: {e}")
            self.labels = [f"species_{i}" for i in range(964)]

    def _verify_atlanta_coverage(self):
        found = []
        missing = []
        lower_labels = [l.lower() for l in self.labels]
        try:
            atl_list = config.ATLANTA_EXPECTED
        except:
            atl_list = []
        for expected in atl_list:
            exp_low = expected.lower()
            if any(exp_low in lab or lab in exp_low for lab in lower_labels):
                found.append(expected)
            else:
                tokens = exp_low.split()
                match = any(all(t in lab for t in tokens) for lab in lower_labels)
                if match:
                    found.append(expected)
                else:
                    missing.append(expected)
        if atl_list:
            logger.info(f"Atlanta coverage: found {len(found)}/{len(atl_list)} : {found}")
            if missing:
                logger.warning(f"Missing Atlanta species in model: {missing}")

    def _load_model(self):
        if Interpreter is None:
            logger.error("No TFLite interpreter")
            return
        model_candidates = [
            self.model_path,
            Path(str(self.model_path).replace("birds_V1_3.tflite", "mobilenet_v2_1.0_224_quant.tflite")),
            Path(__file__).resolve().parent.parent.parent / "models" / "mobilenet_v2_1.0_224_quant.tflite",
            Path(__file__).resolve().parent.parent.parent / "models" / "birds_V1_3.tflite",
        ]
        chosen = None
        for cand in model_candidates:
            if cand and cand.exists():
                chosen = cand
                break
        if chosen is None:
            logger.error(f"Species model missing at {self.model_path}")
            return
        if chosen != self.model_path:
            logger.warning(f"Using fallback {chosen}")
            self.model_path = chosen
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Species model loaded: {self.model_path}")
            logger.info(f"Species input: {self.input_details}")
            logger.info(f"Species output: {self.output_details}")
        except Exception as e:
            logger.error(f"Failed load species model {self.model_path}: {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray, bbox_px: List[int] = None) -> np.ndarray:
        h,w = frame.shape[:2]
        if bbox_px:
            x1,y1,x2,y2 = bbox_px
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
            # Quantized model expects uint8 RGB 0-255 directly mapping to [-1,1] via scale 0.0078125 zp 128
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            input_data = rgb.astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        return input_data

    def classify(self, frame: np.ndarray, detection: Dict = None) -> Dict:
        if self.interpreter is None:
            return {"species":"unknown","confidence":0.0,"top_k":[],"is_confident":False,"error":"model not loaded"}

        bbox = detection["bbox_px"] if detection and "bbox_px" in detection else None
        input_data = self._preprocess(frame, bbox)

        start = time.time()
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        except Exception as e:
            logger.error(f"Species inference failed: {e}")
            return {"species":"unknown","confidence":0.0,"top_k":[],"is_confident":False,"error":str(e)}

        # --- Dequantize if quantized ---
        probs = output
        out_detail = self.output_details[0]
        is_quantized = out_detail['dtype'] == np.uint8 or out_detail['dtype'] == np.int8
        if is_quantized:
            scale, zero_point = out_detail['quantization']
            if scale == 0:
                qp = out_detail.get('quantization_parameters', {})
                if qp and len(qp.get('scales',[]))>0:
                    scale = float(qp['scales'][0])
                    zero_point = int(qp['zero_points'][0]) if len(qp.get('zero_points',[]))>0 else 0
            # Fallback for case where quantization_parameters missing but still quantized
            if scale == 0:
                scale = 0.00390625
                zero_point = 0
            probs = (probs.astype(np.float32) - zero_point) * scale
            logger.debug(f"Dequantized scale={scale} zp={zero_point} min={probs.min():.4f} max={probs.max():.4f} sum={probs.sum():.4f} argmax={np.argmax(probs)}")

        # Log raw stats before background handling
        raw_argmax = int(np.argmax(probs))
        raw_max = float(np.max(probs))
        raw_sum = float(np.sum(probs))
        logger.info(f"Raw output: size={len(probs)} sum={raw_sum:.4f} max={raw_max:.4f} argmax={raw_argmax} (before background drop)")

        # --- Handle 965 vs 964 background ---
        # Kaggle v3 is 965 outputs: try to detect background position
        # Common: background at 0 (most likely) or at last index
        background_idx = None
        background_prob = 0.0
        if len(probs) == len(self.labels) + 1:
            # Check which end looks like background - background often has high prob for non-bird
            # If probs[0] > probs[-1], assume background at 0 else at end
            # We'll also check metadata hint: if raw_argmax == 0 or == len(probs)-1, treat as background
            if raw_argmax == 0:
                background_idx = 0
                background_prob = probs[0]
            elif raw_argmax == len(probs)-1:
                background_idx = len(probs)-1
                background_prob = probs[-1]
            else:
                # Default: assume background at 0 per AIY spec (v3 fix mentions logit to labelmap)
                background_idx = 0
                background_prob = probs[0]
            logger.info(f"965 vs 964 detected - background assumed at idx {background_idx} prob={background_prob:.4f}")
            # Remove background for label mapping
            if background_idx == 0:
                probs = probs[1:]
            else:
                probs = probs[:-1]
            logger.info(f"After dropping background: size={len(probs)} sum={np.sum(probs):.4f}")

        # --- Determine if already probabilities or logits ---
        # Quantized TFLite often outputs already-softmaxed probabilities (dequantized 0-1 sum ~1)
        # Logits will have sum far from 1 and/or negative values
        # Use lenient check: all values >=0 and <=1.2 and sum in [0.5, 1.5] -> treat as probabilities
        raw_sum = float(np.sum(probs))
        raw_min = float(np.min(probs))
        raw_max = float(np.max(probs))
        is_probabilities = (raw_min >= -0.05 and raw_max <= 1.2 and 0.5 <= raw_sum <= 1.5)
        if not is_probabilities:
            # Apply stable softmax (logits)
            max_val = np.max(probs)
            probs_shifted = probs - max_val
            probs_shifted = np.clip(probs_shifted, -50, 50)
            e_x = np.exp(probs_shifted)
            sum_ex = np.sum(e_x)
            if sum_ex > 0 and np.isfinite(sum_ex):
                probs = e_x / sum_ex
            else:
                probs = np.ones_like(probs) / len(probs)
            logger.info(f"Applied softmax to logits: raw_sum={raw_sum:.4f} raw_max={raw_max:.4f} -> new_sum={np.sum(probs):.4f} new_max={np.max(probs):.4f}")
        else:
            logger.info(f"Output already probabilities, skipping softmax raw_sum={raw_sum:.4f} raw_max={raw_max:.4f} raw_argmax={raw_argmax}")

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)

        # If background was high confidence, log as unknown
        if background_idx is not None and background_prob > 0.5:
            logger.info(f"Background class high prob {background_prob:.2%} >50% - logging as unknown (non-bird or low quality frame)")
            # Return unknown but keep top_k for debug
            top_indices = np.argsort(probs)[::-1][:self.top_k]
            top_k = [(self.labels[i] if i < len(self.labels) else f"species_{i}", float(probs[i])) for i in top_indices]
            return {
                "species": "unknown",
                "raw_species": f"background_{background_idx}",
                "confidence": float(background_prob),
                "top_k": top_k,
                "is_confident": False,
                "inference_ms": (time.time()-start)*1000,
                "background_prob": float(background_prob)
            }

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
        if not is_conf and config.ENABLE_CLOUD_FALLBACK:
            result["cloud_fallback_attempted"] = True
        return result

    def classify_from_path(self, image_path: str) -> Dict:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"species":"unknown","confidence":0.0,"top_k":[],"is_confident":False,"error":"invalid image"}
        return self.classify(frame)
