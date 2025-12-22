from enum import Enum
from dataclasses import dataclass
import os
from typing import Optional

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


class ModelKey(Enum):
    OPENCV = "OpenCV"
    SVM_HOG = "SVM + HOG"
    SVM_HOG_SIFT = "SVM + HOG + SIFT"


@dataclass(frozen=True)
class ModelConfig:
    key: ModelKey
    model_dir: Optional[str]
    backend: str
    feature_type: str

    clf_filename: Optional[str] = None
    scaler_or_bundle_filename: Optional[str] = None
    config_filename: Optional[str] = None

    window_size: tuple[int, int] = (64, 128)
    step_size: int = 8
    min_confidence: float = 0.0
    nms_threshold: float = 0.3


MODEL_CONFIGS = {
    ModelKey.OPENCV: ModelConfig(
        key=ModelKey.OPENCV,
        model_dir=os.path.join(MODEL_DIR, "opencv-hog"),
        backend="opencv",
        feature_type="hog",
    ),
    ModelKey.SVM_HOG: ModelConfig(
        key=ModelKey.SVM_HOG,
        model_dir=os.path.join(MODEL_DIR, "hog+svm"),
        backend="svm",
        feature_type="hog",
        clf_filename="hog_svm_model.pkl",
        scaler_or_bundle_filename=None,
        config_filename="model_config_svm.pkl",
        window_size=(64, 128),
        step_size=8,
        min_confidence=0.2,
        nms_threshold=0.25,
    ),
    ModelKey.SVM_HOG_SIFT: ModelConfig(
        key=ModelKey.SVM_HOG_SIFT,
        model_dir=os.path.join(MODEL_DIR, "hog+svm+sift"),
        backend="svm",
        feature_type="hog+sift",
        clf_filename="hog_sift_svm_model.pkl",
        scaler_or_bundle_filename=None,
        config_filename="model_config.pkl",
        window_size=(64, 128),
        step_size=8,
        min_confidence=0.2,
        nms_threshold=0.20,
    ),
}
