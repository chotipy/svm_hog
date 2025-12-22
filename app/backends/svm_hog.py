import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGDetector(BaseDetector):
    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        self.classifier = classifier
        self.config = config
        self.default_params = default_params or {}

        self.window_size = tuple(config.get("window_size", (64, 128)))

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": "L2-Hys",  # Default standar skimage
            "transform_sqrt": True,  # Biasanya True untuk robust detection
        }

        # 2. Parameter Deteksi
        self.step_size = int(config.get("step_size", 8))
        self.scale_factor = float(config.get("scale_factor", 1.25))
        self.min_confidence = float(config.get("min_confidence", 3.5))
        self.nms_threshold = float(config.get("nms_threshold", 0.1))

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:

        # Override parameter dari UI jika ada
        p = {
            "step_size": self.step_size,
            "min_confidence": self.min_confidence,
            "nms_threshold": self.nms_threshold,
            "scale_factor": self.scale_factor,
        }
        if params:
            p.update(params)

        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        orig_h, orig_w = gray.shape
        win_w, win_h = self.window_size

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        # 3. Multi-scale Detection (Image Pyramid)
        current_scale = 1.0
        while True:
            # Tentukan ukuran resize
            new_w = int(orig_w / current_scale)
            new_h = int(orig_h / current_scale)

            # Berhenti jika gambar lebih kecil dari window deteksi
            if new_w < win_w or new_h < win_h:
                break

            resized_img = cv2.resize(gray, (new_w, new_h))

            # 4. Sliding Window pada skala saat ini
            for y in range(0, new_h - win_h + 1, p["step_size"]):
                for x in range(0, new_w - win_w + 1, p["step_size"]):
                    window = resized_img[y : y + win_h, x : x + win_w]

                    # Ekstraksi Fitur HOG
                    feat = hog(
                        window,
                        orientations=self.hog_params["orientations"],
                        pixels_per_cell=self.hog_params["pixels_per_cell"],
                        cells_per_block=self.hog_params["cells_per_block"],
                        block_norm=self.hog_params["block_norm"],
                        transform_sqrt=self.hog_params["transform_sqrt"],
                        feature_vector=True,
                    ).reshape(1, -1)

                    score = float(self.classifier.decision_function(feat)[0])

                    if score >= p["min_confidence"]:
                        # Kembalikan koordinat ke skala gambar asli
                        abs_x = float(x * current_scale)
                        abs_y = float(y * current_scale)
                        abs_w = float(win_w * current_scale)
                        abs_h = float(win_h * current_scale)

                        all_boxes.append([abs_x, abs_y, abs_w, abs_h])
                        all_scores.append(score)

            current_scale *= p["scale_factor"]

            if p["scale_factor"] <= 1.0:
                break

        # 5. Non-Maximum Suppression
        if not all_boxes:
            return [], []

        return nms_xywh(all_boxes, all_scores, iou_thr=p["nms_threshold"])
