import os
import pickle
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import Pipeline

from .base import BaseDetector
from utils_detection import nms_xywh


class IdentityScaler:
    def transform(self, X):
        return X


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class SVMWindowDetector(BaseDetector):
    def __init__(
        self,
        model_dir: str,
        clf_filename: str,
        scaler_or_bundle_filename: Optional[str],
        feature_type: str,
        config_filename: Optional[str] = None,
        default_window_size=(64, 128),
        default_step_size=8,
        default_min_conf=0.0,
        default_nms=0.3,
    ):
        self.model_dir = os.path.abspath(model_dir)

        clf_path = os.path.join(self.model_dir, clf_filename)
        self.clf = load_pickle(clf_path)
        self.is_pipeline = isinstance(self.clf, Pipeline)

        self.feature_type = feature_type
        self.window_size = default_window_size
        self.step_size = default_step_size
        self.min_confidence = default_min_conf
        self.nms_threshold = default_nms

        # HOG defaults
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)

        # SIFT defaults
        self.sift_grid_size = (4, 8)  # MUST match training
        self.sift = cv2.SIFT_create() if "sift" in feature_type else None

        if config_filename:
            cfg = load_pickle(os.path.join(self.model_dir, config_filename))
            if isinstance(cfg, dict):
                self.window_size = tuple(cfg.get("window_size", self.window_size))
                self.step_size = int(cfg.get("step_size", self.step_size))
                self.min_confidence = float(
                    cfg.get("min_confidence", self.min_confidence)
                )
                self.nms_threshold = float(cfg.get("nms_threshold", self.nms_threshold))
                self.sift_grid_size = tuple(
                    cfg.get("sift_grid_size", self.sift_grid_size)
                )

                self.hog_orientations = int(
                    cfg.get("hog_orientations", self.hog_orientations)
                )
                self.hog_pixels_per_cell = tuple(
                    cfg.get("hog_pixels_per_cell", self.hog_pixels_per_cell)
                )
                self.hog_cells_per_block = tuple(
                    cfg.get("hog_cells_per_block", self.hog_cells_per_block)
                )

        self.scaler = IdentityScaler()
        if scaler_or_bundle_filename and not self.is_pipeline:
            obj = load_pickle(os.path.join(self.model_dir, scaler_or_bundle_filename))
            if hasattr(obj, "transform"):
                self.scaler = obj

    def _extract_sift_grid(self, gray: np.ndarray) -> List[float]:
        gx, gy = self.sift_grid_size
        h, w = gray.shape

        cell_w = w // gx
        cell_h = h // gy

        feats: List[float] = []

        for iy in range(gy):
            for ix in range(gx):
                cell = gray[
                    iy * cell_h : (iy + 1) * cell_h,
                    ix * cell_w : (ix + 1) * cell_w,
                ]
                _, desc = self.sift.detectAndCompute(cell, None)
                if desc is not None:
                    feats.extend(desc.mean(axis=0))
                else:
                    feats.extend([0.0] * 128)

        return feats

    def _extract_features(self, patch_bgr: np.ndarray) -> np.ndarray:
        patch_bgr = cv2.resize(patch_bgr, self.window_size)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        feats: List[float] = []

        # HOG
        if "hog" in self.feature_type:
            hog_feat = hog(
                gray,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm="L2-Hys",
                transform_sqrt=True,
                feature_vector=True,
            )
            feats.extend(hog_feat.tolist())

        # SIFT GRID
        if "sift" in self.feature_type:
            feats.extend(self._extract_sift_grid(gray))

        return np.asarray(feats, dtype=np.float32)

    def detect(self, image_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        h, w = image_bgr.shape[:2]
        win_w, win_h = self.window_size

        boxes, scores = [], []

        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                patch = image_bgr[y : y + win_h, x : x + win_w]

                feat = self._extract_features(patch)
                X = [feat]  # ALWAYS 2D

                if not self.is_pipeline:
                    X = self.scaler.transform(X)

                pred = int(self.clf.predict(X)[0])
                conf = float(self.clf.decision_function(X)[0])

                if pred == 1 and conf >= self.min_confidence:
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        return nms_xywh(boxes, scores, self.nms_threshold)
