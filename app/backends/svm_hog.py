import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGDetector(BaseDetector):
    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        """
        Aligned with crowd_detection_hog_svm.ipynb
        """
        self.classifier = classifier
        self.config = config
        self.default_params = default_params or {}

        # Config stores size as (width, height) -> (64, 128)
        self.window_size = tuple(config.get("window_size", (64, 128)))

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": str(config.get("hog_block_norm", "L2-Hys")),
            "transform_sqrt": bool(
                config.get("hog_transform_sqrt", False)
            ),  # CHANGED to False
        }

        # Detection params
        self.step_size = int(config.get("step_size", 8))
        self.scale_factor = float(config.get("scale_factor", 1.25))
        self.min_confidence = float(
            config.get("min_confidence", 3.5)
        )  # Matches config pkl
        self.nms_threshold = float(config.get("nms_threshold", 0.1))

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:

        # Override params from UI
        p = {
            "step_size": self.step_size,
            "min_confidence": self.min_confidence,
            "nms_threshold": self.nms_threshold,
            "scale_factor": self.scale_factor,
        }
        if params:
            p.update(params)

        # 1. Preprocessing (Grayscale)
        # Notebook: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        orig_h, orig_w = gray.shape
        win_w, win_h = self.window_size  # (64, 128)

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        # 2. Multi-scale Sliding Window
        current_scale = 1.0

        # Guard clause for very small images
        if orig_w < win_w or orig_h < win_h:
            return [], []

        while True:
            # Calculate new dimensions
            new_w = int(orig_w / current_scale)
            new_h = int(orig_h / current_scale)

            # Stop if image is smaller than window
            if new_w < win_w or new_h < win_h:
                break

            # Resize the image (Image Pyramid)
            resized_img = cv2.resize(gray, (new_w, new_h))

            # Sliding Window
            # Use separate lists for x and y to avoid logic errors
            ys = range(0, new_h - win_h + 1, p["step_size"])
            xs = range(0, new_w - win_w + 1, p["step_size"])

            for y in ys:
                for x in xs:
                    # Crop the window
                    window = resized_img[y : y + win_h, x : x + win_w]

                    # Verify shape (edge case handling)
                    if window.shape[0] != win_h or window.shape[1] != win_w:
                        continue

                    # 3. Feature Extraction (Must match training!)
                    feat = hog(
                        window,
                        orientations=self.hog_params["orientations"],
                        pixels_per_cell=self.hog_params["pixels_per_cell"],
                        cells_per_block=self.hog_params["cells_per_block"],
                        block_norm=self.hog_params["block_norm"],
                        transform_sqrt=self.hog_params["transform_sqrt"],
                        feature_vector=True,
                        visualize=False,
                    ).reshape(1, -1)

                    # 4. Classification
                    # Use decision_function to get the raw score (distance from hyperplane)
                    # Pipeline handles Scaling automatically
                    score = float(self.classifier.decision_function(feat)[0])

                    if score >= p["min_confidence"]:
                        # Map coordinates back to original image size
                        abs_x = float(x * current_scale)
                        abs_y = float(y * current_scale)
                        abs_w = float(win_w * current_scale)
                        abs_h = float(win_h * current_scale)

                        all_boxes.append([abs_x, abs_y, abs_w, abs_h])
                        all_scores.append(score)

            current_scale *= p["scale_factor"]

            # Prevent infinite loop if scale factor is bad
            if p["scale_factor"] <= 1.0:
                break

        # 5. Non-Maximum Suppression
        if not all_boxes:
            return [], []

        return nms_xywh(all_boxes, all_scores, iou_thr=p["nms_threshold"])
