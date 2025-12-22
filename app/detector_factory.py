import os
import pickle
from backends.opencv_hog import ImprovedHOGDetector
from backends.svm_window import SVMWindowDetector


def load_pkl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PKL not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_detector(cfg):
    if cfg.backend == "opencv":
        return ImprovedHOGDetector()

    if cfg.backend == "svm":
        if cfg.model_file is None or cfg.config_file is None:
            raise ValueError("SVM backend requires model_file and config_file")

        model_path = os.path.join(cfg.model_dir, cfg.model_file)
        config_path = os.path.join(cfg.model_dir, cfg.config_file)

        model = load_pkl(model_path)
        config = load_pkl(config_path)

        return SVMWindowDetector(model=model, config=config)

    # =========================
    raise ValueError(f"Unknown backend: {cfg.backend}")
