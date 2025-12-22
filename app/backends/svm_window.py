import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np

from .base import BaseDetector
from utils_detection import nms_xywh


class SVMWindowDetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        window_size: Tuple[int, int] = (64, 128),
        step_size: int = 8,
        min_confidence: float = 0.0,
        nms_threshold: float = 0.3,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Sanity check
        if not hasattr(self.model, "predict"):
            raise TypeError("Loaded pkl is not a sklearn model/pipeline")

        self.window_size = window_size
        self.step_size = step_size
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold

    def detect(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        win_w, win_h = self.window_size

        boxes: List[List[float]] = []
        scores: List[float] = []

        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                patch = image_bgr[y : y + win_h, x : x + win_w]

                # ⛔ NO FEATURE ENGINEERING HERE
                # Pipeline inside .pkl handles everything
                X = np.expand_dims(patch, axis=0)

                try:
                    pred = int(self.model.predict(X)[0])
                    conf = float(self.model.decision_function(X)[0])
                except Exception:
                    continue

                if pred == 1 and conf >= self.min_confidence:
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        boxes, scores = nms_xywh(boxes, scores, self.nms_threshold)
        return boxes, scores
