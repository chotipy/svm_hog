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
        """
        Detector Hybrid: HOG (Shape) + SIFT (Texture) + SVM (Classifier)
        """
        if not hasattr(classifier, "predict"):
            raise TypeError("Classifier harus berupa sklearn Pipeline/SVM")

        self.classifier = classifier
        self.config = config

        raw_size = config.get("window_size", config.get("target_size", (64, 128)))
        self.window_size = tuple(raw_size)

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": str(config.get("hog_block_norm", "L2-Hys")),
            "transform_sqrt": bool(
                config.get("hog_transform_sqrt", False)
            ),  # Default False
        }

        # SIFT Params
        self.sift_grid_size = tuple(
            config.get("sift_grid_size", (4, 8))
        )  # (cols, rows)
        self.sift = cv2.SIFT_create()

        p = default_params or {}
        self.step_size = int(p.get("step_size", config.get("step_size", 8)))
        self.min_confidence = float(p.get("min_confidence", -1.0))
        self.nms_threshold = float(p.get("nms_threshold", 0.3))
        self.scale_factor = 1.1  # Image pyramid scale

        dummy_patch = np.zeros(
            (self.window_size[1], self.window_size[0], 3), dtype=np.uint8
        )
        feat_dim = self._extract_features(dummy_patch).shape[0]

        # Cek ekspektasi model (biasanya ada di scaler)
        expected = getattr(
            getattr(classifier, "named_steps", {}).get("scaler", None),
            "n_features_in_",
            None,
        )

        print(f"[INIT] HOG+SIFT Detector Ready.")
        print(f"       Window Size: {self.window_size}")
        print(f"       Feature Dim: {feat_dim} (HOG + SIFT)")

        if expected and feat_dim != expected:
            raise ValueError(
                f"CRITICAL ERROR: Dimensi fitur tidak cocok! Code={feat_dim}, Model={expected}"
            )

    def detect(
        self, image_bgr: np.ndarray, params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[float]]:
        # Override params dari UI
        p = {
            "step_size": self.step_size,
            "min_confidence": self.min_confidence,
            "nms_threshold": self.nms_threshold,
            "scale_factor": self.scale_factor,
        }
        if params:
            p.update(params)

        # Preprocessing
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        orig_h, orig_w = gray.shape
        win_w, win_h = self.window_size

        all_boxes = []
        all_scores = []

        current_scale = 1.0

        # --- Multi-scale Sliding Window ---
        while True:
            new_w = int(orig_w / current_scale)
            new_h = int(orig_h / current_scale)

            if new_w < win_w or new_h < win_h:
                break

            resized_img = cv2.resize(gray, (new_w, new_h))

            # Loop Sliding Window
            ys = range(0, new_h - win_h + 1, p["step_size"])
            xs = range(0, new_w - win_w + 1, p["step_size"])

            for y in ys:
                for x in xs:
                    window = resized_img[y : y + win_h, x : x + win_w]

                    if window.shape != (win_h, win_w):
                        continue

                    # 1. Extract
                    feat = self._extract_features(window).reshape(1, -1)

                    # 2. Predict (Decision Function = Jarak Hyperplane)
                    score = float(self.classifier.decision_function(feat)[0])

                    if score > p["min_confidence"]:
                        # Restore coordinate to original scale
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
        # Input patch bisa BGR atau Grayscale, kita butuh keduanya
        if patch.ndim == 3:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            patch_gray = patch

        # 1. HOG Extraction (skimage)
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

        # 2. SIFT Extraction (Grid Dense SIFT)
        sift_feat = self._sift_grid_features(patch_gray)

        # 3. Concatenate: HOG + SIFT (Urutan standar)
        return np.concatenate([hog_feat, sift_feat])

    def _sift_grid_features(self, patch_gray: np.ndarray) -> np.ndarray:
        h, w = patch_gray.shape
        nx, ny = self.sift_grid_size  # (4, 8)

        step_x = w // nx
        step_y = h // ny

        # Keypoint size ~ ukuran cell
        kp_size = float(min(step_x, step_y))

        keypoints = []

        # --- CRITICAL FIX: LOOP ORDER ---
        # Kebanyakan notebook training melakukan loop X dulu (Kolom), baru Y (Baris)
        # untuk membentuk urutan fitur per kolom.
        for x_idx in range(nx):  # Col 0, Col 1, ...
            for y_idx in range(ny):  # Row 0, Row 1, ... di dalam Col tsb

                # Koordinat Center dari cell
                cx = step_x * x_idx + step_x // 2
                cy = step_y * y_idx + step_y // 2

                keypoints.append(cv2.KeyPoint(float(cx), float(cy), kp_size))

        # OpenCV SIFT butuh uint8
        if patch_gray.dtype != np.uint8:
            patch_gray = patch_gray.astype(np.uint8)

        _, descriptors = self.sift.compute(patch_gray, keypoints)

        expected_rows = nx * ny  # 32

        if descriptors is None or descriptors.shape[0] != expected_rows:
            # Padding zero jika ada yang kurang
            safe_desc = np.zeros((expected_rows, 128), dtype=np.float32)
            if descriptors is not None:
                m = min(descriptors.shape[0], expected_rows)
                safe_desc[:m] = descriptors[:m]
            descriptors = safe_desc

        return descriptors.flatten().astype(np.float32)


if __name__ == "__main__":
    import pickle
    import joblib
    import os

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(BASE_DIR, "models", "hog+svm+sift")

    PKL_MODEL = os.path.join(MODEL_DIR, "hog_sift_svm_model.pkl")
    PKL_CONFIG = os.path.join(MODEL_DIR, "model_config.pkl")

    if os.path.exists(PKL_MODEL) and os.path.exists(PKL_CONFIG):
        print(f"Loading test model from: {MODEL_DIR}")
        with open(PKL_MODEL, "rb") as f:
            clf = joblib.load(f)
        with open(PKL_CONFIG, "rb") as f:
            cfg = pickle.load(f)

        try:
            detector = SVMHOGSIFTDetector(clf, cfg)
            print("Detector berhasil diinisialisasi!")
            print("   Sekarang coba jalankan di Streamlit.")
        except ValueError as e:
            print(f"ERROR DIMENSI: {e}")
            print("Cek kembali notebook.")
    else:
        print("Pastikan path benar.")
