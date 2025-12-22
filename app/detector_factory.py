import os
import pickle
from backends.opencv_hog import OpenCVHOGDetector
from backends.svm_window import SVMWindowDetector


def load_pkl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PKL not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_detector(cfg):
    if cfg.backend == "opencv":
        return OpenCVHOGDetector()

    if cfg.backend == "svm":
        model = load_pkl(os.path.join(cfg.model_dir, cfg.model_file))
        config = load_pkl(os.path.join(cfg.model_dir, cfg.config_file))
        return SVMWindowDetector(model=model, config=config)

    raise ValueError(f"Unknown backend: {cfg.backend}")
