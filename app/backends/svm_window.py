import numpy as np
from typing import List, Tuple
from .base import BaseDetector
from utils_detection import nms_xywh


class SVMWindowDetector(BaseDetector):
    def __init__(
        self,
        model,
        config: dict,
    ):
        """
        model  : sklearn model / pipeline (from pkl)
        config : dict loaded from pkl
        """
        if not hasattr(model, "predict"):
            raise TypeError("model must be sklearn classifier or pipeline")

        self.model = model

        self.window_size = tuple(config.get("window_size", (64, 128)))
        self.step_size = int(config.get("step_size", 8))
        self.min_confidence = float(config.get("min_confidence", 0.0))
        self.nms_threshold = float(config.get("nms_threshold", 0.3))

    def detect(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        win_w, win_h = self.window_size

        boxes: List[List[float]] = []
        scores: List[float] = []

        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                patch = image_bgr[y : y + win_h, x : x + win_w]

                X = patch.reshape(1, -1)

                pred = int(self.model.predict(X)[0])
                conf = float(self.model.decision_function(X)[0])

                if pred == 1 and conf >= self.min_confidence:
                    boxes.append([x, y, win_w, win_h])
                    scores.append(conf)

        return nms_xywh(boxes, scores, self.nms_threshold)
