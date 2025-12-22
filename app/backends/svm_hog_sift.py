import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGSIFTDetector(BaseDetector):
    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("Classifier must be an sklearn Pipeline/SVM with predict()")

        self.classifier = classifier
        self.config = config

        raw_size = config.get("window_size", config.get("target_size", (64, 128)))
        self.window_size = tuple(raw_size)  # (64, 128) -> width, height

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": str(config.get("hog_block_norm", "L2-Hys")),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", False)),
        }

        self.sift_grid_size = tuple(
            config.get("sift_grid_size", (4, 8))
        )  # (4, 8) -> 32 cells

        self.sift = cv2.SIFT_create()

        dummy = np.zeros((self.window_size[1], self.window_size[0]), dtype=np.uint8)
        feat_dim = self._extract_features(dummy).shape[0]

        expected = getattr(
            getattr(classifier, "named_steps", {}).get("scaler", None),
            "n_features_in_",
            None,
        )

        if expected and feat_dim != expected:
            raise ValueError(
                f"Feature dimension mismatch! Code: {feat_dim}, Model: {expected}"
            )

        # 4. Runtime Params
        p = default_params or {}
        self.step_size = int(p.get("step_size", config.get("step_size", 8)))
        self.min_confidence = float(
            p.get("min_confidence", config.get("min_confidence", 0.5))
        )
        self.nms_threshold = float(
            p.get("nms_threshold", config.get("nms_threshold", 0.3))
        )
        self.scale_factor = 1.25  # Standard pyramid scale

    def detect(
        self,
        image_bgr: np.ndarray,
        params: Optional[dict] = None,
        verbose: bool = False,
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

        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        orig_h, orig_w = gray.shape
        win_w, win_h = self.window_size

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        current_scale = 1.0

        # --- Multi-scale Image Pyramid ---
        while True:
            new_w = int(orig_w / current_scale)
            new_h = int(orig_h / current_scale)

            if new_w < win_w or new_h < win_h:
                break

            resized_img = cv2.resize(gray, (new_w, new_h))

            # Sliding Window
            ys = range(0, new_h - win_h + 1, p["step_size"])
            xs = range(0, new_w - win_w + 1, p["step_size"])

            for y in ys:
                for x in xs:
                    window = resized_img[y : y + win_h, x : x + win_w]

                    # Validasi ketat ukuran window
                    if window.shape != (win_h, win_w):
                        continue

                    feat = self._extract_features(window).reshape(1, -1)
                    score = float(self.classifier.decision_function(feat)[0])

                    if score > p["min_confidence"]:
                        abs_x = float(x * current_scale)
                        abs_y = float(y * current_scale)
                        abs_w = float(win_w * current_scale)
                        abs_h = float(win_h * current_scale)

                        all_boxes.append([abs_x, abs_y, abs_w, abs_h])
                        all_scores.append(score)

            current_scale *= p["scale_factor"]
            if p["scale_factor"] <= 1.0:
                break

        return nms_xywh(all_boxes, all_scores, iou_thr=p["nms_threshold"])

    def _extract_features(self, patch: np.ndarray) -> np.ndarray:
        # 1. HOG
        hog_feat = hog(
            patch,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
            visualize=False,
        ).astype(np.float32)

        # 2. SIFT
        sift_feat = self._sift_grid_features(patch)

        # 3. Concatenate
        return np.concatenate([hog_feat, sift_feat])

    def _sift_grid_features(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape
        nx, ny = self.sift_grid_size  # (4, 8)

        # Grid parameters
        step_x = w / nx  # 16.0
        step_y = h / ny  # 16.0

        # Generate Keypoints at CENTERS of the grid cells
        # This prevents boundary issues
        kp_size = min(step_x, step_y)
        keypoints = []

        for y_idx in range(ny):
            for x_idx in range(nx):
                # Koordinat pusat cell
                cx = (x_idx + 0.5) * step_x
                cy = (y_idx + 0.5) * step_y
                keypoints.append(cv2.KeyPoint(float(cx), float(cy), float(kp_size)))

        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)

        _, descriptors = self.sift.compute(patch, keypoints)

        expected_rows = nx * ny  # 32
        expected_len = expected_rows * 128  # 4096

        if descriptors is None or descriptors.shape[0] != expected_rows:
            safe_desc = np.zeros((expected_rows, 128), dtype=np.float32)
            if descriptors is not None:
                m = min(descriptors.shape[0], expected_rows)
                safe_desc[:m] = descriptors[:m]
            descriptors = safe_desc

        return descriptors.flatten().astype(np.float32)
