from enum import Enum
from dataclasses import dataclass
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class ModelKey(Enum):
    OPENCV_HOG = "OpenCV HOG"
    SVM_HOG = "SVM + HOG"
    SVM_HOG_SIFT = "SVM + HOG + SIFT"


@dataclass
class ModelConfig:
    key: ModelKey
    backend: str
    model_dir: str


MODEL_CONFIGS = {
    ModelKey.OPENCV_HOG: ModelConfig(
        key=ModelKey.OPENCV_HOG,
        backend="opencv",
        model_dir=os.path.join(MODEL_DIR, "opencv_hog"),
    ),
    ModelKey.SVM_HOG: ModelConfig(
        key=ModelKey.SVM_HOG,
        backend="svm_hog",
        model_dir=os.path.join(MODEL_DIR, "svm_hog"),
    ),
    ModelKey.SVM_HOG_SIFT: ModelConfig(
        key=ModelKey.SVM_HOG_SIFT,
        backend="svm_hog_sift",
        model_dir=os.path.join(MODEL_DIR, "svm_hog_sift"),
    ),
}
