import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog

from .base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGSIFTDetector(BaseDetector):
    """
    Sliding-window detector using:
    - HOG features (skimage)
    - SIFT descriptors on a fixed grid (cv2.SIFT)
    Feature dim must match training: 3780 (HOG) + 4096 (SIFT grid 4x8) = 7876
    """

    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("classifier must be sklearn Pipeline/SVM with predict()")

        self.classifier = classifier
        self.config = config

        self.window_size = tuple(config.get("window_size", (64, 128)))  # (w,h)

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": config.get("hog_block_norm", "L2-Hys"),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", True)),
        }

        # SIFT grid (training uses fixed grid)
        self.sift_grid_size = tuple(config.get("sift_grid_size", (4, 8)))  # (nx, ny)
        self.sift = cv2.SIFT_create()

        p = default_params or {}
        self.step_size = int(p.get("step_size", 8))
        self.min_confidence = float(p.get("min_confidence", 0.35))
        self.nms_threshold = float(p.get("nms_threshold", 0.3))

        # Optional geometry filters (kalau kamu mau)
        self.min_box_area = int(p.get("min_box_area", 0))  # 0 = off
        self.min_aspect_ratio = float(p.get("min_aspect_ratio", 0.0))
        self.max_aspect_ratio = float(p.get("max_aspect_ratio", 999.0))

        expected = getattr(
            getattr(self.classifier, "named_steps", {}).get("scaler", None),
            "n_features_in_",
            None,
        )
        if expected is not None:
            test = self._extract_features(
                np.zeros((self.window_size[1], self.window_size[0]), dtype=np.uint8)
            )
            if len(test) != expected:
                raise ValueError(
                    f"Feature dim mismatch: got {len(test)} but model expects {expected}"
                )

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:
        # allow runtime override
        if params:
            self.step_size = int(params.get("step_size", self.step_size))
            self.min_confidence = float(
                params.get("min_confidence", self.min_confidence)
            )
            self.nms_threshold = float(params.get("nms_threshold", self.nms_threshold))

        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        h, w = gray.shape
        win_w, win_h = self.window_size

        boxes: List[List[float]] = []
        scores: List[float] = []

        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                patch = gray[y : y + win_h, x : x + win_w]
                if patch.shape != (win_h, win_w):
                    continue

                try:
                    feat = self._extract_features(patch).reshape(1, -1)
                    pred = int(self.classifier.predict(feat)[0])
                    conf = float(self.classifier.decision_function(feat)[0])

                    if pred == 1 and conf >= self.min_confidence:
                        # xywh
                        boxes.append([float(x), float(y), float(win_w), float(win_h)])
                        scores.append(conf)
                except Exception:
                    # IMPORTANT: kalau mau debug, print error sekali-sekali
                    continue

        # optional geometry filter (kalau kamu aktifkan paramnya)
        if self.min_box_area > 0 or self.min_aspect_ratio > 0.0:
            boxes, scores = self._filter_boxes(boxes, scores)

        return nms_xywh(boxes, scores, iou_thr=self.nms_threshold)

    def _extract_features(self, patch_gray: np.ndarray) -> np.ndarray:
        hog_feat = hog(
            patch_gray,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
        )

        sift_feat = self._sift_grid_features(patch_gray)
        return np.concatenate(
            [hog_feat.astype(np.float32), sift_feat.astype(np.float32)], axis=0
        )

    def _sift_grid_features(self, patch_gray: np.ndarray) -> np.ndarray:
        win_h, win_w = patch_gray.shape[:2]
        nx, ny = self.sift_grid_size  # (cols, rows)
        n_points = nx * ny

        # Make fixed grid keypoints (center of each cell)
        xs = (np.arange(nx) + 0.5) * (win_w / nx)
        ys = (np.arange(ny) + 0.5) * (win_h / ny)

        # KeyPoint size: roughly cell size
        kp_size = float(min(win_w / nx, win_h / ny))

        keypoints = []
        for yy in ys:
            for xx in xs:
                keypoints.append(cv2.KeyPoint(float(xx), float(yy), kp_size))

        _, desc = self.sift.compute(patch_gray, keypoints)

        # We need exactly (n_points,128) -> 4096
        if desc is None or desc.shape[0] != n_points:
            out = np.zeros((n_points, 128), dtype=np.float32)
            if desc is not None:
                m = min(desc.shape[0], n_points)
                out[:m] = desc[:m]
            desc = out
        else:
            desc = desc.astype(np.float32)

        return desc.reshape(-1)  # 4096
