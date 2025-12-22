import os
import pickle
from backends.opencv_hog import OpenCVHOGDetector
from backends.svm_window import SVMWindowDetector


def build_detector(cfg):
    if cfg.backend == "opencv":
        pkl_path = os.path.join(cfg.model_dir, "opencv_hog.pkl")
        with open(pkl_path, "rb") as f:
            config = pickle.load(f)
        return OpenCVHOGDetector(config)

    elif cfg.backend in ("svm_hog", "svm_hog_sift"):
        model_path = os.path.join(cfg.model_dir, "svm_model.pkl")
        config_path = os.path.join(cfg.model_dir, "svm_config.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        return SVMWindowDetector(model=model, config=config)

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")
