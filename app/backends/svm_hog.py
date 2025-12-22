import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGDetector(BaseDetector):
    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("classifier must be sklearn Pipeline or SVM model")

        self.classifier = classifier
        self.config = config
        self.default_params = default_params or {}

        self.window_size = tuple(config.get("window_size", (64, 128)))

        # HOG params from your pkl config keys
        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": str(config.get("hog_block_norm", "L2-Hys")),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", True)),
        }

        # detection params (prefer config, fallback default_params)
        self.step_size = int(
            config.get("step_size", self.default_params.get("step_size", 8))
        )
        self.min_confidence = float(
            config.get("min_confidence", self.default_params.get("min_confidence", 0.0))
        )
        self.nms_threshold = float(
            config.get("nms_threshold", self.default_params.get("nms_threshold", 0.3))
        )

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:
        # allow runtime override (UI)
        p = dict(self.default_params)
        if params:
            p.update(params)

        step_size = int(p.get("step_size", self.step_size))
        min_conf = float(p.get("min_confidence", self.min_confidence))
        nms_thr = float(p.get("nms_threshold", self.nms_threshold))

        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        h, w = gray.shape
        win_w, win_h = self.window_size

        boxes: List[List[float]] = []
        scores: List[float] = []

        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                window = gray[y : y + win_h, x : x + win_w]
                if window.shape != (win_h, win_w):
                    continue

                feat = hog(
                    window,
                    orientations=self.hog_params["orientations"],
                    pixels_per_cell=self.hog_params["pixels_per_cell"],
                    cells_per_block=self.hog_params["cells_per_block"],
                    block_norm=self.hog_params["block_norm"],
                    transform_sqrt=self.hog_params["transform_sqrt"],
                    feature_vector=True,
                ).reshape(1, -1)

                pred = int(self.classifier.predict(feat)[0])
                conf = float(self.classifier.decision_function(feat)[0])

                if pred == 1 and conf >= min_conf:
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        return nms_xywh(boxes, scores, iou_thr=nms_thr)
