# ==========================================
# Member 4: LPR Pipeline Integration (End-to-End)
# ==========================================
# This module integrates the LPR sub-modules into a unified pipeline class
# that executes preprocessing, detection, type classification, character segmentation,
# feature extraction, and character classification.

from __future__ import annotations

from typing import Any
import cv2
import json
import os
import re
import threading
import joblib
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .preprocessing import (
    preprocess_scene_image,
    rectify_plate_crop,
    deskew_plate_with_angle,
    enhance_plate_crop_for_ocr,
    rotate_bound,
    crop_plate_body_after_deskew,
)
from .detection import detect_plate_yolo, detect_plate_contour_fallback
from .plate_classifier import classify_plate_type
from .segmentation import segment_plate
from .features import (
    extract_features as extract_resnet_features,
    extract_hog_features,
    extract_hog_legacy_features,
    extract_raw_features,
    extract_wavelet_features,
)
from .classifier import CHAR_CLASSES


FEATURE_EXTRACTORS = {
    "resnet": extract_resnet_features,
    "resnet18": extract_resnet_features,
    "hog": extract_hog_features,
    "hog_legacy": extract_hog_legacy_features,
    "wavelet": extract_wavelet_features,
    "raw": extract_raw_features,
}


PREFERRED_PLATE_RE = re.compile(r"^\d{2}[A-Z]\d{4,6}$")
VALID_PLATE_RE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{3,5}$")
LETTER_TO_DIGIT = {
    "B": "8",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "S": "5",
    "Z": "2",
    "T": "1",
}
DIGIT_TO_LETTER = {
    "0": "G",
    "1": "L",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def ensure_not_lfs_pointer(path: str | os.PathLike) -> None:
    """Fails fast when a model file is a Git LFS pointer instead of the model.

    Cloning without git-lfs leaves ~130-byte text pointers in models/, which
    joblib/torch would reject with a cryptic "invalid load key" error.
    """
    with open(path, "rb") as f:
        head = f.read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the model file. "
            "Install git-lfs and run `git lfs install && git lfs pull`."
        )


def _resize_plate_for_ocr(plate_crop, target_width=400):
    h, w = plate_crop.shape[:2]
    if w <= 0 or h <= 0 or w == target_width:
        return plate_crop
    new_h = max(1, int(h * target_width / w))
    interpolation = cv2.INTER_CUBIC if target_width > w else cv2.INTER_AREA
    return cv2.resize(plate_crop, (target_width, new_h), interpolation=interpolation)


def _normalise_feature_method(name):
    if not name:
        return "resnet"
    key = str(name).strip().lower()
    if "legacy" in key or "8x8" in key or "324" in key:
        return "hog_legacy"
    if "hog" in key:
        return "hog"
    if "wavelet" in key or "haar" in key:
        return "wavelet"
    if "raw" in key or "pixel" in key:
        return "raw"
    if "resnet" in key:
        return "resnet"
    return key


def _dedup_9char(text):
    if len(text) != 9:
        return text
    if text[3].isdigit() and text[3] == text[4]:
        alt = text[:3] + text[4:]
        if PREFERRED_PLATE_RE.match(alt) or VALID_PLATE_RE.match(alt):
            return alt
    return text


def _coerce_by_mask(text, letter_positions):
    chars = list(text)
    for idx, ch in enumerate(chars):
        should_be_letter = idx in letter_positions
        should_be_digit = idx not in letter_positions
        if idx < 2:
            should_be_letter = False
            should_be_digit = True
        if should_be_digit and ch.isalpha():
            chars[idx] = LETTER_TO_DIGIT.get(ch, ch)
        elif should_be_letter and ch.isdigit():
            chars[idx] = DIGIT_TO_LETTER.get(ch, ch)
    return "".join(chars)


def correct_plate_format(text: str) -> str:
    """
    Apply Vietnamese plate-format constraints after OCR.

    This mirrors the report's idea: use ML for character classification, then
    use deterministic plate grammar only to correct common digit/letter swaps.
    """
    text = "".join(ch for ch in text.upper() if ch.isalnum())
    variants = [
        text,
        _coerce_by_mask(text, {2}),
        _coerce_by_mask(text, {2, 3}),
    ]

    for candidate in variants:
        candidate = _dedup_9char(candidate)
        if PREFERRED_PLATE_RE.match(candidate):
            return candidate

    for candidate in variants:
        candidate = _dedup_9char(candidate)
        if VALID_PLATE_RE.match(candidate):
            return candidate
    return text


class LPRPipeline:
    """
    Unified License Plate Recognition Pipeline.
    """

    def __init__(self, models_dir: str | None = None) -> None:
        self.yolo_model = None
        self.yolo_model_path = None
        self.svm_model = None
        self.scaler = None
        self.feature_method = "resnet"
        self.classifier_name = "svm"
        self.char_classes = list(CHAR_CLASSES)
        self._lock = threading.Lock()

        if models_dir and os.path.exists(models_dir):
            self.load_svm_models(models_dir)

    def load_svm_models(self, models_dir: str) -> None:
        """Loads a saved OCR classifier, scaler, and metadata."""
        classifier_path = os.path.join(models_dir, "classifier.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        metadata_path = os.path.join(models_dir, "metadata.json")

        # Backward compatibility with the original notebook output.
        if not os.path.exists(classifier_path):
            classifier_path = os.path.join(models_dir, "svm_classifier.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(models_dir, "char_classes.json")

        if os.path.exists(classifier_path) and os.path.exists(scaler_path):
            ensure_not_lfs_pointer(classifier_path)
            ensure_not_lfs_pointer(scaler_path)
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            self.set_ocr_model(
                joblib.load(classifier_path), joblib.load(scaler_path), metadata
            )
            print(
                f"Loaded OCR model from {models_dir} "
                f"({self.feature_method} + {self.classifier_name})"
            )
        else:
            print(f"Model files not found in {models_dir}")

    def set_ocr_model(
        self, classifier: Any, scaler: Any, metadata: dict | None = None
    ) -> None:
        """Installs an already-loaded OCR classifier + scaler (local or MLflow)."""
        metadata = metadata or {}
        self.svm_model = classifier
        self.scaler = scaler
        self.char_classes = list(
            metadata.get("char_classes") or metadata.get("classes") or CHAR_CLASSES
        )
        self.feature_method = _normalise_feature_method(
            metadata.get("feature_method", self.feature_method)
        )
        self.classifier_name = metadata.get("classifier", self.classifier_name)

        n_features = getattr(scaler, "n_features_in_", None)
        declared_dim = metadata.get("feature_dim")
        if declared_dim and n_features and int(declared_dim) != n_features:
            raise ValueError(
                f"Model metadata declares feature_dim={declared_dim} but the "
                f"scaler expects {n_features} features"
            )
        # Old artifacts stored "hog" for the 8x8-cell (324-d) variant.
        if self.feature_method == "hog" and n_features == 324:
            self.feature_method = "hog_legacy"

    def load_yolo_model(self, yolo_model_path: str) -> None:
        """Loads the YOLOv8 model for detection."""
        if YOLO is None:
            print("ultralytics not installed; YOLO detection disabled")
            return
        ensure_not_lfs_pointer(yolo_model_path)
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model_path = yolo_model_path
        print(f"Loaded YOLO model: {yolo_model_path}")

    def _extract_char_features(self, char_imgs):
        feature_fn = FEATURE_EXTRACTORS.get(self.feature_method)
        if feature_fn is None:
            raise ValueError(f"Unsupported feature method: {self.feature_method}")
        return feature_fn(char_imgs)

    @staticmethod
    def _line_counts_from_info(char_info, plate_type):
        if plate_type != "2line":
            return None
        if not char_info or not all("y_center" in c for c in char_info):
            return None
        y_min = min(c["y_center"] for c in char_info)
        y_max = max(c["y_center"] for c in char_info)
        split_y = (y_min + y_max) / 2
        n_top = sum(1 for c in char_info if c["y_center"] < split_y)
        n_bottom = len(char_info) - n_top
        return n_top, n_bottom

    @staticmethod
    def _plate_format_score(pred_chars, plate_type, line_counts=None):
        text = "".join(pred_chars)
        score = 0.0

        if 7 <= len(text) <= 9:
            score += 3.0
        elif len(text) < 7:
            score -= 3.0
        elif len(text) > 9:
            score -= 2.0

        if plate_type == "2line" and line_counts:
            n_top, n_bottom = line_counts
            top = "".join(pred_chars[:n_top])
            bottom = "".join(pred_chars[n_top : n_top + n_bottom])

            if 3 <= len(top) <= 4:
                score += 1.0
            if len(bottom) >= len(top):
                score += 1.0
            if len(top) >= 2 and top[:2].isdigit():
                score += 3.0
            if any(ch.isalpha() for ch in top):
                score += 2.0
            if bottom.isdigit() and 4 <= len(bottom) <= 6:
                score += 4.0
            if bottom:
                score += 2.0 * sum(ch.isdigit() for ch in bottom) / len(bottom)
            score -= 1.5 * sum(ch.isalpha() for ch in bottom)
        else:
            if len(text) >= 3 and text[:2].isdigit() and text[2].isalpha():
                score += 5.0
            tail = text[3:]
            if tail:
                score += 3.0 * sum(ch.isdigit() for ch in tail) / len(tail)

        return score

    def _constrain_predictions(
        self, features_scaled, raw_preds, plate_type, line_counts=None
    ):
        if not hasattr(self.svm_model, "decision_function"):
            return raw_preds

        scores = self.svm_model.decision_function(features_scaled)
        if scores.ndim == 1:
            return raw_preds

        digit_indices = [i for i, ch in enumerate(self.char_classes) if ch.isdigit()]
        letter_indices = [i for i, ch in enumerate(self.char_classes) if ch.isalpha()]
        allowed_by_pos = []

        n_chars = len(raw_preds)
        if plate_type == "2line" and line_counts:
            n_top, n_bottom = line_counts
            for pos in range(n_chars):
                if pos < min(2, n_top):
                    allowed_by_pos.append(digit_indices)
                elif pos == 2:
                    allowed_by_pos.append(letter_indices)
                elif pos == 3 and n_top >= 4:
                    # Two-line Vietnamese plates can be:
                    # 1) Standard motorbikes: 2 digits + 1 letter + 1 digit (e.g. 59-S2, 29-H1)
                    # 2) Special/Commercial: 2 digits + 2 letters (e.g. 92CA, 51LD)
                    # Therefore, allow BOTH digits and letters at position 3.
                    allowed_by_pos.append(digit_indices + letter_indices)
                else:
                    allowed_by_pos.append(digit_indices)
        else:
            for pos in range(n_chars):
                if pos < 2:
                    allowed_by_pos.append(digit_indices)
                elif pos == 2:
                    allowed_by_pos.append(letter_indices)
                else:
                    allowed_by_pos.append(digit_indices)

        constrained = []
        for pos, pred in enumerate(raw_preds):
            allowed = allowed_by_pos[pos] if pos < len(allowed_by_pos) else None
            if not allowed or int(pred) in allowed:
                constrained.append(int(pred))
                continue
            best_allowed = max(allowed, key=lambda idx: scores[pos, idx])
            constrained.append(int(best_allowed))
        return constrained

    def _recognize_plate_crop_candidate(self, plate_crop):
        plate_type, ar = classify_plate_type(plate_crop)
        char_imgs, char_info, binary = segment_plate(plate_crop, plate_type)
        line_counts = self._line_counts_from_info(char_info, plate_type)
        candidate = {
            "plate_crop": plate_crop,
            "plate_type": plate_type,
            "aspect_ratio": ar,
            "char_images": char_imgs,
            "char_count": len(char_imgs),
            "binary": binary,
            "line_counts": line_counts,
            "success": False,
            "format_score": 0.0,
        }
        if len(char_imgs) == 0:
            return candidate
        if self.svm_model is None or self.scaler is None:
            return candidate

        features = self._extract_char_features(char_imgs)
        features_scaled = self.scaler.transform(features)
        raw_preds = self.svm_model.predict(features_scaled)
        preds = self._constrain_predictions(
            features_scaled, raw_preds, plate_type, line_counts
        )
        pred_chars = [self.char_classes[p] for p in preds]
        # Topological disambiguation for '1' vs '7':
        # Numeral '1' is a narrow vertical stroke (w/h <= 0.48 in 32x32 canvas),
        # whereas '7' possesses a wide horizontal top bar (w/h >= 0.55).
        for i, ch in enumerate(pred_chars):
            if ch == "7" and i < len(char_imgs):
                c_img = char_imgs[i]
                ys, xs = np.where(c_img > 50)
                if len(xs) > 0 and len(ys) > 0:
                    w = xs.max() - xs.min() + 1
                    h = ys.max() - ys.min() + 1
                    if w / float(h) <= 0.48:
                        pred_chars[i] = "1"
        raw_text = "".join(pred_chars)
        corrected_text = correct_plate_format(raw_text)
        candidate.update(
            {
                "features_shape": features.shape,
                "predictions": pred_chars,
                "raw_plate_string": raw_text,
                "plate_string": corrected_text,
                "format_score": self._plate_format_score(
                    list(corrected_text), plate_type, line_counts
                ),
                "success": True,
            }
        )
        return candidate

    @staticmethod
    def _candidate_key(candidate):
        text = candidate.get("plate_string") or ""
        preferred_bonus = 6.0 if PREFERRED_PLATE_RE.match(text) else 0.0
        valid_bonus = 2.0 if VALID_PLATE_RE.match(text) else 0.0
        return (
            1 if candidate.get("success") else 0,
            preferred_bonus + valid_bonus + candidate.get("format_score", 0.0),
            -abs(float(candidate.get("deskew_angle", 0.0) or 0.0)),
            len(candidate.get("char_images") or []),
        )

    @staticmethod
    def _add_crop_variant(variants, seen, crop, angle, enhance=True):
        if crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        if h < 18 or w < 35:
            return
        key = (h, w, round(float(angle), 1), bool(enhance))
        if key in seen:
            return
        seen.add(key)
        variants.append(
            (
                enhance_plate_crop_for_ocr(crop) if enhance else crop,
                float(angle),
            )
        )

    def _build_primary_ocr_crop_variants(
        self,
        plate_crop,
        deskewed_crop,
        deskew_angle,
        allow_full_crop_rescue=True,
    ):
        variants = []
        seen = set()
        # Keep the full resized plate crop in the primary candidate set. On
        # two-line plates, deskew cropping can occasionally lock onto only one
        # text row and turn a square plate into a false one-line crop.
        plate_ar = plate_crop.shape[1] / float(max(1, plate_crop.shape[0]))
        deskew_ar = deskewed_crop.shape[1] / float(max(1, deskewed_crop.shape[0]))
        if allow_full_crop_rescue and plate_ar <= 2.2 and deskew_ar >= 2.5:
            self._add_crop_variant(variants, seen, plate_crop, 0.0, enhance=False)
        self._add_crop_variant(
            variants, seen, deskewed_crop, deskew_angle, enhance=False
        )
        return variants

    def _build_fallback_ocr_crop_variants(
        self, plate_crop, deskewed_crop, deskew_angle
    ):
        variants = []
        seen = set()
        self._add_crop_variant(
            variants, seen, deskewed_crop, deskew_angle, enhance=True
        )
        self._add_crop_variant(variants, seen, plate_crop, 0.0, enhance=False)
        self._add_crop_variant(variants, seen, plate_crop, 0.0, enhance=True)

        if abs(deskew_angle) >= 1.0:
            opposite = crop_plate_body_after_deskew(
                rotate_bound(plate_crop, -deskew_angle)
            )
            self._add_crop_variant(variants, seen, opposite, -deskew_angle)

        # Small crops from traffic photos often get an imperfect skew estimate.
        # Try moderate residual rotations to avoid combinatorial explosion and excessive latency.
        for residual in (-10, -5, 5, 10):
            rotated = crop_plate_body_after_deskew(
                rotate_bound(deskewed_crop, residual)
            )
            self._add_crop_variant(variants, seen, rotated, deskew_angle + residual)
        return variants

    def _evaluate_ocr_crop_variants(self, variants):
        candidates = []
        for crop_variant, angle in variants:
            candidate_0 = self._recognize_plate_crop_candidate(crop_variant)
            candidate_0["deskew_angle"] = angle
            candidates.append((False, candidate_0))

            # Early exit: if candidate_0 already has high format score and valid format,
            # skip the expensive 180-degree flipped crop evaluation to save latency.
            text_0 = candidate_0.get("plate_string") or ""
            is_valid_format = bool(
                PREFERRED_PLATE_RE.match(text_0) or VALID_PLATE_RE.match(text_0)
            )
            if is_valid_format and candidate_0.get("format_score", 0.0) >= 12.0:
                continue

            candidate_180 = self._recognize_plate_crop_candidate(
                cv2.rotate(crop_variant, cv2.ROTATE_180)
            )
            candidate_180["deskew_angle"] = angle
            candidates.append((True, candidate_180))
        return max(candidates, key=lambda item: self._candidate_key(item[1]))

    @staticmethod
    def _needs_fallback(candidate):
        text = candidate.get("plate_string") or ""
        char_count = len(candidate.get("char_images") or [])
        if abs(float(candidate.get("deskew_angle", 0.0) or 0.0)) > 15.0:
            return True
        if char_count < 7:
            return True
        if PREFERRED_PLATE_RE.match(text) or VALID_PLATE_RE.match(text):
            return False
        return candidate.get("format_score", 0.0) < 10.0

    def recognize(
        self,
        image: np.ndarray,
        yolo_model_path: str | None = None,
        assume_plate_crop: bool = False,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Runs the End-to-End LPR Pipeline.

        Args:
            image: np.array (RGB scene image)
            yolo_model_path: path to YOLO weights (optional, overrides default)
            assume_plate_crop: if True, skip plate detection and read the whole image
            verbose: whether to print log output

        Returns:
            dict: result dictionary containing step outputs and success flag
        """
        result = {"success": False}

        # === Step 2: Preprocessing (Member 2) ===
        processed_img = preprocess_scene_image(image)

        # === Step 3: Plate Detection (Member 3) ===
        if assume_plate_crop:
            h, w = processed_img.shape[:2]
            plate_crop = processed_img
            result["bbox"] = (0, 0, w, h)
            result["detection_confidence"] = 1.0
        else:
            if yolo_model_path is not None:
                with self._lock:
                    if (
                        self.yolo_model is None
                        or self.yolo_model_path != yolo_model_path
                    ):
                        self.load_yolo_model(yolo_model_path)

            if self.yolo_model is not None:
                boxes = detect_plate_yolo(processed_img, self.yolo_model)
                if not boxes:
                    if verbose:
                        print(
                            "YOLO did not find a plate; falling back to contour detection"
                        )
                    boxes = [
                        (b[0], b[1], b[2], b[3], 0.5)
                        for b in detect_plate_contour_fallback(processed_img)
                    ]
            else:
                boxes = [
                    (b[0], b[1], b[2], b[3], 0.5)
                    for b in detect_plate_contour_fallback(processed_img)
                ]

            if not boxes:
                if verbose:
                    print("No license plate detected")
                return result

            # Pick detection box with highest confidence/priority
            boxes.sort(key=lambda b: b[4], reverse=True)
            x1, y1, x2, y2, conf = boxes[0]
            h, w = processed_img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            plate_crop = processed_img[y1:y2, x1:x2]

            result["bbox"] = (x1, y1, x2, y2)
            result["detection_confidence"] = conf

        if plate_crop.size == 0:
            if verbose:
                print("Detected plate crop is empty")
            return result

        plate_crop = rectify_plate_crop(plate_crop)
        plate_crop = _resize_plate_for_ocr(plate_crop)
        deskewed_crop, deskew_angle = deskew_plate_with_angle(plate_crop)
        result["deskew_angle"] = deskew_angle

        if self.svm_model is None or self.scaler is None:
            if verbose:
                print("OCR classifier/scaler has not been loaded")
            ocr_crop = enhance_plate_crop_for_ocr(deskewed_crop)
            result["plate_crop"] = ocr_crop
            plate_type, ar = classify_plate_type(ocr_crop)
            char_imgs, _, binary = segment_plate(ocr_crop, plate_type)
            result.update(
                {
                    "plate_type": plate_type,
                    "aspect_ratio": ar,
                    "char_images": char_imgs,
                    "binary": binary,
                }
            )
            return result

        # Try both possible 180-degree orientations and choose the one whose
        # OCR output looks most like a Vietnamese plate format.
        was_rotated, best = self._evaluate_ocr_crop_variants(
            self._build_primary_ocr_crop_variants(
                plate_crop,
                deskewed_crop,
                deskew_angle,
                allow_full_crop_rescue=not assume_plate_crop,
            )
        )
        if self._needs_fallback(best):
            fallback_rotated, fallback_best = self._evaluate_ocr_crop_variants(
                self._build_fallback_ocr_crop_variants(
                    plate_crop, deskewed_crop, deskew_angle
                )
            )
            if self._candidate_key(fallback_best) > self._candidate_key(best):
                was_rotated, best = fallback_rotated, fallback_best

        result.update(best)
        result["rotated_180"] = was_rotated
        result["deskew_angle"] = best.get("deskew_angle", deskew_angle)

        if verbose:
            print(
                f"Plate type: {result['plate_type']} (AR={result['aspect_ratio']:.2f})"
            )
            print(f"Segmented characters: {len(result.get('char_images') or [])}")
            print(f"{self.feature_method} features: {result.get('features_shape')}")
            print(f"Deskew angle: {result.get('deskew_angle', 0.0):.2f} deg")
            if result.get("raw_plate_string") and result.get(
                "raw_plate_string"
            ) != result.get("plate_string"):
                print(f"Raw OCR text: {result.get('raw_plate_string')}")
            print(f"Plate text: {result.get('plate_string', '')}")
            print(f"Format score: {result.get('format_score', 0.0):.2f}")

        return result
