import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGDetector(BaseDetector):
    """
    Sliding-window SVM detector using HOG features.
    MUST match training feature layout in hog_svm_model.pkl (feature length 3780).
    """

    def __init__(self, classifier, config: dict, default_params: dict | None = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("classifier must be sklearn Pipeline / SVM model")

        self.classifier = classifier

        # From config pkl
        self.window_size = tuple(config.get("window_size", (64, 128)))

        # Training-time HOG params (from config_svm.pkl)
        self.hog_params = {
            "orientations": int(config.get("orientations", 9)),
            "pixels_per_cell": tuple(config.get("pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("cells_per_block", (2, 2))),
            "block_norm": config.get("block_norm", "L2-Hys"),
            "transform_sqrt": bool(config.get("transform_sqrt", True)),
        }

        # Detection params (config can include these)
        base_params = default_params or {}
        self.step_size = int(base_params.get("step_size", config.get("step_size", 8)))
        self.min_confidence = float(
            base_params.get("min_confidence", config.get("min_confidence", 0.3))
        )
        self.nms_threshold = float(
            base_params.get("nms_threshold", config.get("nms_threshold", 0.15))
        )

        # Optional geometry constraints (if exist in config)
        self.min_box_area = int(config.get("min_box_area", 0))
        self.edge_margin = int(config.get("edge_margin", 0))
        self.min_aspect_ratio = float(config.get("min_aspect_ratio", 0.0))
        self.max_aspect_ratio = float(config.get("max_aspect_ratio", 999.0))

    def detect(
        self, image_bgr: np.ndarray, params: Optional[Dict] = None
    ) -> Tuple[List[List[float]], List[float]]:
        params = params or {}

        step_size = int(params.get("step_size", self.step_size))
        min_conf = float(params.get("min_confidence", self.min_confidence))
        nms_thr = float(params.get("nms_threshold", self.nms_threshold))

        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        h, w = gray.shape[:2]
        win_w, win_h = self.window_size

        boxes: List[List[float]] = []
        scores: List[float] = []

        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                patch = gray[y : y + win_h, x : x + win_w]
                if patch.shape != (win_h, win_w):
                    continue

                feat = self._extract_hog(patch).reshape(1, -1)

                try:
                    pred = int(self.classifier.predict(feat)[0])
                    # SVM margin as confidence
                    if hasattr(self.classifier, "decision_function"):
                        conf = float(self.classifier.decision_function(feat)[0])
                    else:
                        # fallback
                        conf = float(pred)
                except Exception:
                    continue

                if pred == 1 and conf >= min_conf:
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        boxes, scores = self._filter_boxes(boxes, scores, (h, w))
        return nms_xywh(boxes, scores, iou_thr=nms_thr)

    def _extract_hog(self, patch_gray: np.ndarray) -> np.ndarray:
        return hog(
            patch_gray,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
        )

    def _filter_boxes(
        self,
        boxes: List[List[float]],
        scores: List[float],
        image_shape: Tuple[int, int],
    ) -> Tuple[List[List[float]], List[float]]:
        if not boxes:
            return [], []

        h, w = image_shape
        out_b, out_s = [], []

        for (x, y, bw, bh), sc in zip(boxes, scores):
            # aspect constraints if set
            if self.min_aspect_ratio > 0 or self.max_aspect_ratio < 999:
                aspect = bw / (bh + 1e-6)
                if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                    continue

            if self.min_box_area > 0 and (bw * bh) < self.min_box_area:
                continue

            if self.edge_margin > 0:
                cx, cy = x + bw / 2, y + bh / 2
                if not (
                    self.edge_margin < cx < w - self.edge_margin
                    and self.edge_margin < cy < h - self.edge_margin
                ):
                    continue

            out_b.append([x, y, bw, bh])
            out_s.append(sc)

        return out_b, out_s
