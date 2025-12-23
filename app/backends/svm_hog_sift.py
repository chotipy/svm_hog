import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

try:
    from .base import BaseDetector
    from utils_detection import nms_xywh
except ImportError:
    from backends.base import BaseDetector
    from utils_detection import nms_xywh


class SVMHOGSIFTDetector(BaseDetector):
    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("Classifier harus berupa sklearn Pipeline/SVM")

        self.classifier = classifier
        self.config = config

        raw_size = config.get("window_size", config.get("target_size", (64, 128)))
        self.window_size = tuple(raw_size)

        # HOG Params
        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": str(config.get("hog_block_norm", "L2-Hys")),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", False)),
        }

        # SIFT Params
        self.sift_grid_size = tuple(config.get("sift_grid_size", (4, 8)))
        self.sift = cv2.SIFT_create()

        p = default_params or {}

        # AGGRESSIVE PARAMETERS to reduce overlapping boxes
        self.step_size = int(p.get("step_size", config.get("step_size", 16)))
        self.scale_factor = float(
            p.get("scale_factor", config.get("scale_factor", 1.25))
        )
        self.nms_threshold = float(
            p.get("nms_threshold", config.get("nms_threshold", 0.25))
        )
        self.min_confidence = float(
            p.get("min_confidence", config.get("min_confidence", 3.5))
        )

        dummy_patch = np.zeros(
            (self.window_size[1], self.window_size[0], 3), dtype=np.uint8
        )
        feat_dim = self._extract_features(dummy_patch).shape[0]

        expected = getattr(
            getattr(classifier, "named_steps", {}).get("scaler", None),
            "n_features_in_",
            None,
        )

        print(
            f"[INIT] HOG+SIFT Detector Ready. Feature Dim: {feat_dim} (Expected: {expected})"
        )

        if expected and feat_dim != expected:
            raise ValueError(
                f"CRITICAL ERROR: Dimensi fitur tidak cocok! Code={feat_dim}, Model={expected}"
            )

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:
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

        all_boxes = []
        all_scores = []
        current_scale = 1.0

        while True:
            new_w = int(orig_w / current_scale)
            new_h = int(orig_h / current_scale)

            if new_w < win_w or new_h < win_h:
                break

            resized_img = cv2.resize(gray, (new_w, new_h))

            ys = range(0, new_h - win_h + 1, p["step_size"])
            xs = range(0, new_w - win_w + 1, p["step_size"])

            for y in ys:
                for x in xs:
                    window = resized_img[y : y + win_h, x : x + win_w]

                    if window.shape != (win_h, win_w):
                        continue

                    feat = self._extract_features(window).reshape(1, -1)
                    score = float(self.classifier.decision_function(feat)[0])

                    if score > p["min_confidence"]:
                        abs_x = x * current_scale
                        abs_y = y * current_scale
                        abs_w = win_w * current_scale
                        abs_h = win_h * current_scale

                        all_boxes.append([abs_x, abs_y, abs_w, abs_h])
                        all_scores.append(score)

            current_scale *= p["scale_factor"]
            if p["scale_factor"] <= 1.0:
                break

        return nms_xywh(all_boxes, all_scores, iou_thr=p["nms_threshold"])

    def _extract_features(self, patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 3:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            patch_gray = patch

        # 1. HOG (skimage)
        hog_feat = hog(
            patch_gray,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
            visualize=False,
        ).astype(np.float32)

        # 2. SIFT (Dense Grid)
        sift_feat = self._sift_grid_features(patch_gray)

        # 3. Concatenate: MATCHES NOTEBOOK (HOG + SIFT)
        return np.concatenate([hog_feat, sift_feat])

    def _sift_grid_features(self, patch_gray: np.ndarray) -> np.ndarray:
        h, w = patch_gray.shape
        nx, ny = self.sift_grid_size

        step_x = w // nx
        step_y = h // ny

        # NOTEBOOK MATCH: keypoints.append(cv2.KeyPoint(x, y, step_x))
        kp_size = float(step_x)

        keypoints = []

        # --- VERIFIED LOOP ORDER: COLUMN-MAJOR ---
        # Notebook Line: for i in range(grid_size[0]): ... for j in range(grid_size[1]):
        for x_idx in range(nx):
            for y_idx in range(ny):

                cx = step_x * x_idx + step_x // 2
                cy = step_y * y_idx + step_y // 2

                keypoints.append(cv2.KeyPoint(float(cx), float(cy), kp_size))

        if patch_gray.dtype != np.uint8:
            patch_gray = patch_gray.astype(np.uint8)

        _, descriptors = self.sift.compute(patch_gray, keypoints)

        expected_rows = nx * ny

        if descriptors is None or descriptors.shape[0] != expected_rows:
            safe_desc = np.zeros((expected_rows, 128), dtype=np.float32)
            if descriptors is not None:
                m = min(descriptors.shape[0], expected_rows)
                safe_desc[:m] = descriptors[:m]
            descriptors = safe_desc

        return descriptors.flatten().astype(np.float32)


if __name__ == "__main__":
    pass
