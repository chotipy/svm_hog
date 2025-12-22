import os
import pickle

from model_registry import BackendType, ModelConfig
from backends.opencv_hog import OpenCVHOGDetector
from backends.svm_hog import SVMHOGDetector
from backends.svm_hog_sift import SVMHOGSIFTDetector


def load_pkl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PKL not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_detector(cfg: ModelConfig):
    ok, missing = cfg.validate()
    if not ok:
        raise FileNotFoundError("Missing required model files:\n- " + "\n- ".join(missing))

    if cfg.backend == BackendType.OPENCV:
        return OpenCVHOGDetector()

    if cfg.backend == BackendType.SVM_HOG:
        classifier = load_pkl(cfg.files.classifier)
        config = load_pkl(cfg.files.config)
        return SVMHOGDetector(classifier=classifier, config=config, default_params=cfg.default_params)

    if cfg.backend == BackendType.SVM_HOG_SIFT:
        classifier = load_pkl(cfg.files.classifier)
        config = load_pkl(cfg.files.config)
        return SVMHOGSIFTDetector(classifier=classifier, config=config, default_params=cfg.default_params)

    raise ValueError(f"Unknown backend: {cfg.backend}")
