from dataclasses import dataclass
from enum import Enum
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelKey(Enum):
    OPENCV_HOG = "OpenCV HOG"
    SVM_HOG = "SVM + HOG"
    SVM_HOG_SIFT = "SVM + HOG + SIFT"


@dataclass
class ModelConfig:
    backend: str
    model_dir: str
    model_file: str | None = None
    config_file: str | None = None


MODEL_CONFIGS = {
    ModelKey.OPENCV_HOG: ModelConfig(
        backend="opencv",
        model_dir=os.path.join(MODELS_DIR, "opencv-hog"),
        config_file="opencv_hog.pkl",
    ),
    ModelKey.SVM_HOG: ModelConfig(
        backend="svm",
        model_dir=os.path.join(MODELS_DIR, "hog+svm"),
        model_file="hog_svm_model.pkl",
        config_file="model_config_svm.pkl",
    ),
    ModelKey.SVM_HOG_SIFT: ModelConfig(
        backend="svm",
        model_dir=os.path.join(MODELS_DIR, "hog+svm+sift"),
        model_file="hog_sift_svm_model.pkl",
        config_file="model_config.pkl",
    ),
}
