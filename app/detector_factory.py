# app/detector_factory.py

import os
import pickle
import sys
import numpy as np

from model_registry import BackendType
from backends.opencv_hog import OpenCVHOGDetector
from backends.svm_hog import SVMHOGDetector
from backends.svm_hog_sift import SVMHOGSIFTDetector


def load_pkl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PKL not found: {path}")

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)

    with open(path, "rb") as f:
        return pickle.load(f)


def build_detector(cfg):
    ok, missing = cfg.validate()
    if not ok:
        raise FileNotFoundError(
            "Missing required model files:\n- " + "\n- ".join(missing)
        )

    backend = cfg.backend.value if hasattr(cfg.backend, "value") else str(cfg.backend)

    if backend == BackendType.OPENCV.value:
        return OpenCVHOGDetector()

    if backend == BackendType.SVM_HOG.value:
        classifier = load_pkl(cfg.files.classifier)
        config = load_pkl(cfg.files.config)
        return SVMHOGDetector(
            classifier=classifier, config=config, default_params=cfg.default_params
        )

    if backend == BackendType.SVM_HOG_SIFT.value:
        classifier = load_pkl(cfg.files.classifier)
        config = load_pkl(cfg.files.config)
        return SVMHOGSIFTDetector(
            classifier=classifier, config=config, default_params=cfg.default_params
        )

    raise ValueError(f"Unknown backend: {backend}")
