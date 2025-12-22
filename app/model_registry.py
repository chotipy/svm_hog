from dataclasses import dataclass
from enum import Enum
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelKey(Enum):
    OPENCV_HOG = "OpenCV HOG"
    SVM_HOG = "SVM + HOG"
    SVM_HOG_SIFT = "SVM + HOG + SIFT"


class BackendType(Enum):
    """Backend implementation types"""

    OPENCV = "opencv"
    SVM_HOG = "svm_hog"
    SVM_HOG_SIFT = "svm_hog_sift"


@dataclass
class ModelFiles:
    detector: str | None = None
    classifier: str | None = None
    config: str | None = None

    def validate(self, required: list[str]) -> tuple[bool, list[str]]:
        """Check if required files exist"""
        missing: list[str] = []
        for field in required:
            path = getattr(self, field, None)
            if path is None:
                missing.append(f"{field}: not set")
            elif not os.path.isfile(path):
                missing.append(f"{field}: {path}")
        return (len(missing) == 0), missing


@dataclass
class ModelConfig:
    key: ModelKey
    backend: BackendType
    model_dir: str
    files: ModelFiles
    default_params: dict | None = None

    def get_required_files(self) -> list[str]:
        if self.backend == BackendType.OPENCV:
            return []  # built-in: no files required
        if self.backend in (BackendType.SVM_HOG, BackendType.SVM_HOG_SIFT):
            return ["classifier", "config"]
        return []

    def validate(self) -> tuple[bool, list[str]]:
        required = self.get_required_files()
        if not required:
            return True, []
        return self.files.validate(required)


MODEL_CONFIGS: dict[ModelKey, ModelConfig] = {
    ModelKey.OPENCV_HOG: ModelConfig(
        key=ModelKey.OPENCV_HOG,
        backend=BackendType.OPENCV,
        model_dir="",  # built-in
        files=ModelFiles(),
    ),
    ModelKey.SVM_HOG: ModelConfig(
        key=ModelKey.SVM_HOG,
        backend=BackendType.SVM_HOG,
        model_dir=os.path.join(MODELS_DIR, "hog+svm"),
        files=ModelFiles(
            classifier=os.path.join(MODELS_DIR, "hog+svm", "hog_svm_model.pkl"),
            config=os.path.join(MODELS_DIR, "hog+svm", "model_config_svm.pkl"),
        ),
    ),
    ModelKey.SVM_HOG_SIFT: ModelConfig(
        key=ModelKey.SVM_HOG_SIFT,
        backend=BackendType.SVM_HOG_SIFT,
        model_dir=os.path.join(MODELS_DIR, "hog+svm+sift"),
        files=ModelFiles(
            classifier=os.path.join(
                MODELS_DIR, "hog+svm+sift", "hog_sift_svm_model.pkl"
            ),
            config=os.path.join(MODELS_DIR, "hog+svm+sift", "model_config.pkl"),
        ),
    ),
}
