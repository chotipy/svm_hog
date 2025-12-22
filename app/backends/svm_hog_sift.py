import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import hog
import sys

try:
    from utils_detection import nms_xywh
except ImportError:

    def nms_xywh(boxes, scores, iou_thr):
        return boxes, scores


from .base import BaseDetector


class SVMHOGSIFTDetector(BaseDetector):

    def __init__(self, classifier, config: dict, default_params: Optional[dict] = None):
        if not hasattr(classifier, "predict"):
            raise TypeError(
                "Classifier harus berupa sklearn Pipeline/SVM dengan method predict()"
            )

        self.classifier = classifier
        self.config = config

        raw_size = config.get("window_size", config.get("target_size", (64, 128)))
        self.window_size = tuple(raw_size)  # Format: (width, height)

        self.hog_params = {
            "orientations": int(config.get("hog_orientations", 9)),
            "pixels_per_cell": tuple(config.get("hog_pixels_per_cell", (8, 8))),
            "cells_per_block": tuple(config.get("hog_cells_per_block", (2, 2))),
            "block_norm": config.get("hog_block_norm", "L2-Hys"),
            "transform_sqrt": bool(config.get("hog_transform_sqrt", True)),
        }

        # SIFT grid params
        self.sift_grid_size = tuple(
            config.get("sift_grid_size", (4, 8))
        )  # (cols, rows)

        # Inisialisasi SIFT detector
        self.sift = cv2.SIFT_create()

        # Parameter Deteksi Runtime
        p = default_params or {}
        self.step_size = int(p.get("step_size", config.get("step_size", 8)))
        self.min_confidence = float(
            p.get("min_confidence", config.get("min_confidence", 0.35))
        )
        self.nms_threshold = float(
            p.get("nms_threshold", config.get("nms_threshold", 0.3))
        )

        # Geometry filters
        self.min_box_area = int(p.get("min_box_area", 0))

        expected_features = getattr(
            getattr(self.classifier, "named_steps", {}).get("scaler", None),
            "n_features_in_",
            getattr(self.classifier, "n_features_in_", None),
        )

        # Test extract satu patch kosong untuk memastikan dimensi cocok
        dummy_patch = np.zeros(
            (self.window_size[1], self.window_size[0]), dtype=np.uint8
        )
        test_features = self._extract_features(dummy_patch)

        if expected_features is not None:
            if len(test_features) != expected_features:
                raise ValueError(
                    f"CRITICAL ERROR: Dimensi fitur tidak cocok!\n"
                    f"Model mengharapkan: {expected_features} fitur.\n"
                    f"Ekstraktor menghasilkan: {len(test_features)} fitur.\n"
                    f"Cek konfigurasi HOG dan SIFT grid size."
                )
        print(
            f"Detector initialized. Window: {self.window_size}, Feature Dim: {len(test_features)}"
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        params: Optional[dict] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[float]], List[float]]:

        # Override parameter jika ada
        current_step = self.step_size
        current_conf = self.min_confidence
        current_nms = self.nms_threshold

        if params:
            current_step = int(params.get("step_size", current_step))
            current_conf = float(params.get("min_confidence", current_conf))
            current_nms = float(params.get("nms_threshold", current_nms))

        # Convert to Grayscale
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        h_img, w_img = gray.shape
        win_w, win_h = self.window_size

        # Validasi ukuran gambar input
        if h_img < win_h or w_img < win_w:
            if verbose:
                print(
                    f"Warning: Gambar terlalu kecil ({w_img}x{h_img}) untuk window ({win_w}x{win_h})"
                )
            return [], []

        boxes: List[List[float]] = []
        scores: List[float] = []

        # Sliding Window Loop
        # Menggunakan list comprehension atau loop biasa
        for y in range(0, h_img - win_h + 1, current_step):
            for x in range(0, w_img - win_w + 1, current_step):
                patch = gray[y : y + win_h, x : x + win_w]

                # Validasi ukuran patch (penting untuk HOG)
                if patch.shape != (win_h, win_w):
                    continue

                # Extract features
                feat = self._extract_features(patch).reshape(1, -1)

                # Predict
                conf = float(self.classifier.decision_function(feat)[0])

                if conf > current_conf:
                    # Simpan format xywh
                    boxes.append([float(x), float(y), float(win_w), float(win_h)])
                    scores.append(conf)

        if verbose and len(boxes) > 0:
            print(f"Found {len(boxes)} candidates before NMS.")

        # Apply NMS
        final_boxes, final_scores = nms_xywh(boxes, scores, iou_thr=current_nms)

        return final_boxes, final_scores

    def _extract_features(self, patch_gray: np.ndarray) -> np.ndarray:
        # 1. HOG Extraction
        try:
            hog_feat = hog(
                patch_gray,
                orientations=self.hog_params["orientations"],
                pixels_per_cell=self.hog_params["pixels_per_cell"],
                cells_per_block=self.hog_params["cells_per_block"],
                block_norm=self.hog_params["block_norm"],
                transform_sqrt=self.hog_params["transform_sqrt"],
                feature_vector=True,
            )
        except Exception as e:
            # Fallback jika patch error (sangat jarang)
            print(f"HOG Error: {e}")
            return np.zeros(1, dtype=np.float32)

        # 2. SIFT Extraction
        sift_feat = self._sift_grid_features(patch_gray)

        # 3. Concatenate
        # Pastikan tipe data float32 agar efisien dan kompatibel dengan sklearn scaler
        return np.concatenate(
            [hog_feat.astype(np.float32), sift_feat.astype(np.float32)], axis=0
        )

    def _sift_grid_features(self, patch_gray: np.ndarray) -> np.ndarray:
        win_h, win_w = patch_gray.shape[:2]
        nx, ny = self.sift_grid_size
        n_points = nx * ny  # Total 32 titik

        # Buat grid keypoints (titik tengah setiap sel)
        # Tambah +0.5 agar titik ada di tengah cell, bukan di pojok
        step_x = win_w / nx
        step_y = win_h / ny

        xs = (np.arange(nx) + 0.5) * step_x
        ys = (np.arange(ny) + 0.5) * step_y

        # KeyPoint size: kira-kira seukuran cell
        kp_size = float(min(step_x, step_y))

        keypoints = []
        for yy in ys:
            for xx in xs:
                keypoints.append(cv2.KeyPoint(float(xx), float(yy), kp_size))

        # Compute SIFT descriptors
        # patch_gray harus uint8
        _, desc = self.sift.compute(patch_gray, keypoints)

        # Validasi Output SIFT: Kita butuh tepat (n_points * 128) fitur
        # SIFT descriptor panjangnya 128 float
        target_shape = (n_points, 128)

        if desc is None:
            # Jika SIFT gagal total (misal gambar flat color), return zero vector
            desc = np.zeros(target_shape, dtype=np.float32)
        elif desc.shape[0] != n_points:
            # Jika keypoint ada yang terbuang (out of frame), padding dengan nol
            # Ini critical fix agar tidak crash saat reshape
            fixed_desc = np.zeros(target_shape, dtype=np.float32)
            # Salin yang berhasil didapat
            m = min(desc.shape[0], n_points)
            fixed_desc[:m, :] = desc[:m, :]
            desc = fixed_desc

        # Flatten menjadi 1D array (4096,)
        return desc.reshape(-1).astype(np.float32)
