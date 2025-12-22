import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from skimage.feature import hog

from backends.base import BaseDetector
from utils_detection import nms_xywh


class SVMHOGSIFTDetector(BaseDetector):
    """
    Sliding-window SVM detector using HOG + SIFT features.
    MUST match training: total feature length 7876 = 3780 (HOG) + 4096 (SIFT).
    SIFT vector is fixed-size by pad/truncate (target_size=4096).
    """

    def __init__(self, classifier, config: dict, default_params: dict | None = None):
        if not hasattr(classifier, "predict"):
            raise TypeError("classifier must be sklearn Pipeline / SVM model")

        self.classifier = classifier
        self.window_size = tuple(config.get("window_size", (64, 128)))

        # HOG params (from model_config.pkl)
        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": config.get("hog_block_norm", "L2-Hys"),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", True)),
        }

        # SIFT params (from model_config.pkl)
        self.sift_target_size = int(config.get("target_size", 4096))
        self.sift_grid_size = int(config.get("sift_grid_size", 16))
        self.sift_max_keypoints = int(config.get("sift_max_keypoints", 32))

        self.sift = cv2.SIFT_create(
            nfeatures=int(config.get("sift_nfeatures", 0)),
            contrastThreshold=float(config.get("sift_contrast_threshold", 0.04)),
            edgeThreshold=float(config.get("sift_edge_threshold", 10)),
            sigma=float(config.get("sift_sigma", 1.6)),
        )

        # Detection params
        base_params = default_params or {}
        self.step_size = int(base_params.get("step_size", config.get("step_size", 8)))
        self.min_confidence = float(
            base_params.get("min_confidence", config.get("min_confidence", 0.35))
        )
        self.nms_threshold = float(
            base_params.get("nms_threshold", config.get("nms_threshold", 0.12))
        )

        # Optional geometry constraints
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

                feat = self._extract_hog_sift(patch).reshape(1, -1)

                try:
                    pred = int(self.classifier.predict(feat)[0])
                    conf = (
                        float(self.classifier.decision_function(feat)[0])
                        if hasattr(self.classifier, "decision_function")
                        else float(pred)
                    )
                except Exception:
                    continue

                if pred == 1 and conf >= min_conf:
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        boxes, scores = self._filter_boxes(boxes, scores, (h, w))
        return nms_xywh(boxes, scores, iou_thr=nms_thr)

    def _extract_hog_sift(self, patch_gray: np.ndarray) -> np.ndarray:
        hog_feat = hog(
            patch_gray,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
        )

        sift_feat = self._extract_sift_fixed(patch_gray)
        return np.concatenate([hog_feat, sift_feat])

    def _extract_sift_fixed(self, patch_gray: np.ndarray) -> np.ndarray:
        """
        Build fixed-length SIFT vector (target_size=4096) by:
        - create dense keypoints on grid (grid_size)
        - compute descriptors
        - take first max_keypoints descriptors (or pad zeros)
        - flatten -> pad/truncate to target_size
        """
        h, w = patch_gray.shape[:2]
        gs = self.sift_grid_size

        # Dense grid keypoints
        kps = []
        for yy in range(gs // 2, h, gs):
            for xx in range(gs // 2, w, gs):
                kps.append(cv2.KeyPoint(float(xx), float(yy), float(gs)))

        if not kps:
            return np.zeros(self.sift_target_size, dtype=np.float32)

        try:
            kps, desc = self.sift.compute(patch_gray, kps)
        except Exception:
            desc = None

        if desc is None or len(desc) == 0:
            return np.zeros(self.sift_target_size, dtype=np.float32)

        # Keep fixed number of descriptors
        desc = desc[: self.sift_max_keypoints]  # (N, 128)
        flat = desc.reshape(-1).astype(np.float32)

        # Pad/truncate to target_size (4096)
        if flat.size < self.sift_target_size:
            flat = np.pad(flat, (0, self.sift_target_size - flat.size))
        else:
            flat = flat[: self.sift_target_size]

        return flat

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
